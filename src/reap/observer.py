from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dataclasses import dataclass
import logging
import pathlib

from reap.metrics import (
    ttm_online,
    get_routed_characteristic_activation,
    ca_dist_online,
    OnlineStatsTracker,
    get_distance_fn,
)
from reap.pruning_metrics import initialize_pruning_state, update_pruning_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_ACT_FN_MAP = {
    "silu": F.silu,
    "swiglu": F.silu,  # SwiGLU: act(gate) * up; act_fn returns the gate activation
    "gelu": F.gelu,
    "relu": F.relu,
}


def _resolve_act_fn(config: Any) -> callable:
    """Return the expert-MLP gate activation from ``config.hidden_act``.

    All supported MoE targets (Qwen3, Llama4, LFM2, Mixtral) use SwiGLU/silu.
    The ``* up`` half of SwiGLU is applied by the caller.
    """
    name = str(getattr(config, "hidden_act", None) or "silu").lower()
    try:
        return _ACT_FN_MAP[name]
    except KeyError:
        raise ValueError(f"Unsupported expert activation: {name!r}") from None


def compute_fused_expert_activations(
    module: nn.Module,
    adapter,
    flat_input: torch.Tensor,
    num_experts: int,
    top_k: int,
    act_fn: callable,
):
    """Phase-1 bmm "grouped" (routed-only) activation computation for fused
    expert layouts (``gate_up_proj`` / ``down_proj`` stacked on dim 0).

    Shared by :class:`MoETransformerObserver` and
    :class:`reap.layerwise_observer.LayerwiseMoEObserver` so the two observers
    cannot drift apart. This is the bridge implementation replaced by the FREA
    Triton kernel in the kernels epic (#13).

    Returns ``(activations, selected_experts, router_logits)`` where
    ``activations[e]`` is non-zero only at positions routed to expert ``e``.
    """
    router = getattr(module, adapter.router_attr())
    router_logits = router(flat_input)  # (total_tokens, num_experts)
    # Some routers (Qwen3.5/3.6 ``Qwen3_5MoeTopKRouter``) return a tuple
    # ``(logits, scores, indices)``; unwrap to the logits tensor. (The logits
    # are already softmax-normalized there, which is fine: top-k selection is
    # unchanged and the router-logit similarity merging metric is only used
    # when ``record_pruning_metrics_only=False``.)
    if isinstance(router_logits, tuple):
        router_logits = router_logits[0]
    _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    selected_experts = selected_experts.to(flat_input.device)
    exps = module.experts
    gup = exps.gate_up_proj  # (E, 2*I, H): rows [0:I]=gate, [I:2I]=up
    dp = exps.down_proj        # (E, H, I)
    inter = gup.shape[1] // 2
    activations = torch.zeros(
        (num_experts, *flat_input.shape),
        device=flat_input.device,
        dtype=flat_input.dtype,
    )
    for e in range(num_experts):
        mask = (selected_experts == e).any(dim=-1)  # (total_tokens,)
        if not bool(mask.any()):
            continue
        xe = flat_input[mask]                       # (n_e, H)
        g = F.linear(xe, gup[e, :inter])             # (n_e, I) gate
        u = F.linear(xe, gup[e, inter:])            # (n_e, I) up
        h = act_fn(g) * u                            # SwiGLU
        activations[e, mask] = F.linear(h, dp[e])   # (n_e, H) down
    return activations, selected_experts, router_logits


class BaseTransformerObserverHookConfig:
    state_attr_name: str = "hook_state"
    hook_attr_name: str = "hooks"
    module_name_to_hook_regex: Optional[str] = None
    module_class_name_to_hook_regex: Optional[str] = None


class BaseTransformerObserver(ABC):
    def __init__(
        self,
        model,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
    ):
        self.model = model
        self.hook_config = hook_config
        self.hooks = []
        self.state: dict[Any, Any] = {}
        self._hook_model()
        logger.info(
            "%s initialized for %s.",
            self.__class__.__name__,
            self.model.__class__.__name__,
        )

    @abstractmethod
    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        """
        Factory method to create a hook function for the given module.
        This method should be implemented by subclasses to define how the
        hook function should behave.
        """
        raise NotImplementedError("Subclasses must implement _hook_factory method.")

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return self.state

    def close_hooks(self):
        """Close all hooks registered to the model."""
        self.reset()  # Reset the state before closing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("All hooks closed for %s.", self.model.__class__.__name__)

    def reset(self):
        """Reset the observer state."""
        del self.state
        gc.collect()
        self.state = {}
        logger.debug("Observer state reset for %s.", self.model.__class__.__name__)

    def save_state(self, file_path: str | pathlib.Path):
        self._move_state_tensors_to_cpu()
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.report_state()
        with open(file_path, "wb") as f:
            torch.save(state_dict, f)
        logger.info("State saved to %s", file_path)

    def _move_state_tensors_to_cpu(self):
        """
        Move all tensors in the state dictionary to CPU.
        This is useful before saving the state to avoid GPU memory issues.
        """
        for layer_number, layer_state in self.state.items():
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    self.state[layer_number][key] = value.cpu()

    def _validate_hook_config(self):
        if self.hook_config is None:
            return
        if (
            self.hook_config.module_name_to_hook_regex is None
            and self.hook_config.module_class_name_to_hook_regex is None
        ):
            raise ValueError(
                "At least one of 'module_n`ame_to_hook_regex' or "
                "'module_type_to_hook_regex' must be provided in the hook config."
            )
        if (
            self.hook_config.module_name_to_hook_regex is not None
            and self.hook_config.module_class_name_to_hook_regex is not None
        ):
            logger.warning(
                "Both 'module_name_to_hook_regex' and 'module_type_to_hook_regex' are "
                "provided. Both conditions must be satisfied to hook the module."
            )

    def _hook_model(self):
        for name, module in self.model.named_modules():
            hook_module = False
            if (
                self.hook_config.module_name_to_hook_regex
                and re.search(self.hook_config.module_name_to_hook_regex, name)
            ) or (
                self.hook_config.module_class_name_to_hook_regex
                and module.__class__.__name__
                == self.hook_config.module_class_name_to_hook_regex
            ):
                hook_module = True
            if hook_module:
                layer_number = int(re.search(r"\d+", name).group(0))
                hook_fn = self._hook_factory(module, layer_number)
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                logger.info("Hooked module: %s at layer %d", name, layer_number)
        if len(self.hooks) == 0:
            raise ValueError(
                "No modules matched the provided hook configuration. "
                "Check your hook configuration settings."
            )

    @classmethod
    def _get_registry_for_cls(cls) -> dict[str, type[BaseTransformerObserver]]:
        """Helper to get the registry from the specific class 'cls'."""
        if not hasattr(cls, "_architecture_registry") or not isinstance(
            cls._architecture_registry, dict
        ):
            raise AttributeError(
                f"Class {cls.__name__} must define its own "
                "`_architecture_registry: dict[str, type] = {{}}` "
                f"to use the common registration/creation methods."
            )
        return cls._architecture_registry

    @classmethod
    def register_implementation(cls, *arch_names: str):
        """
        Class method decorator to register a concrete observer implementation.
        'cls' is the class on which this decorator's factory is called (e.g.,
        MoEExpertObserver) 'sub_cls' is the class being decorated
        (e.g., Llama4MoEExpertObserver).
        """

        def decorator(sub_cls: type[BaseTransformerObserver]):
            registry = cls._get_registry_for_cls()

            for name in arch_names:
                if name in registry:
                    raise RuntimeError(
                        f"Architecture {name} already registered with "
                        f"{registry[name].__name__} for {cls.__name__}."
                    )
                registry[name] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def create_from_registry(
        cls,
        model: nn.Module,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
        return_rank_0_only: bool = True,
        **kwargs: Any,
    ) -> BaseTransformerObserver:
        registry = cls._get_registry_for_cls()
        model_cls_name = model.__class__.__name__

        specific_observer_cls = registry.get(model_cls_name)

        if specific_observer_cls:
            return specific_observer_cls(
                model,
                hook_config=hook_config,
                return_rank_0_only=return_rank_0_only,
                **kwargs,
            )
        else:
            raise ValueError(
                "Unsupported architecture for "
                f"{cls.__name__}: {model_cls_name}. "
                "Registered architectures in "
                f"{cls.__name__}._architecture_registry: "
                f"{list(registry.keys())}"
            )


# --- MoE Transformer Observer ---------------------------------------------------------


@dataclass
class MoETransformerObserverConfig(BaseTransformerObserverHookConfig):
    module_name_to_hook_regex: Optional[str] = None
    module_class_name_to_hook_regex: Optional[str] = None
    fused_experts: bool = False
    distance_measure: str = "angular"
    renormalize_router_weights: bool = False
    record_pruning_metrics_only: bool = False


class MoETransformerObserver(BaseTransformerObserver):
    """MoE Transformer Observer for all methods including both pruning and merging."""

    def __init__(self, model, hook_config=None, adapter=None):
        self._current_attention_mask: Optional[torch.Tensor] = None
        self.adapter = adapter
        super().__init__(model, hook_config)

    @contextmanager
    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]):
        """Temporarily set the attention mask for the current forward pass.

        Use this as a context manager around each forward pass when using
        batched inputs with padding, to ensure padding tokens are excluded
        from statistics.

        Args:
            attention_mask: Tensor of shape (batch_size, seq_len) with 1 for real
                tokens and 0 for padding tokens. Can be None for unbatched inputs.
        """
        previous_attention_mask = self._current_attention_mask
        self._current_attention_mask = attention_mask
        try:
            yield
        finally:
            self._current_attention_mask = previous_attention_mask

    def clear_attention_mask(self):
        """Clear the attention mask after forward pass."""
        self._current_attention_mask = None

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }

    def _initialize_state(self, hidden_dim: int, num_experts: int):
        # hidden_dim is taken from the hook input (args[0]) rather than the
        # module output, because some MoE blocks (e.g. Qwen3.5/3.6
        # ``Qwen3_5MoeSparseMoeBlock``) return a single tensor instead of a
        # ``(hidden_states, router_logits)`` tuple, so ``output[0]`` is not a
        # reliable shape source.
        device = "cpu"
        layer_state = initialize_pruning_state(num_experts, device=device)

        if not self.hook_config.record_pruning_metrics_only:
            # per routed token normalized states
            layer_state["ttm_similarity_matrix"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=(num_experts, num_experts),
                device=device,
                dtype=torch.float32,
            )
            layer_state["routed_characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=(num_experts, hidden_dim),
                device=device,
                dtype=torch.float32,
            )
            # HC-SMoE
            layer_state["characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # SubMoE
            layer_state["online_characteristic_activation_dist"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # per total token normalized states -> MC-SMoE
            layer_state["router_logit_similiarity"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )

        return layer_state

    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        distance_fn = get_distance_fn("cosine") # always use cosine for online dist. metrics

        # Read num_experts / top_k from the adapter-driven layer config.
        if self.adapter is None:
            raise RuntimeError(
                "MoETransformerObserver requires an adapter (pass adapter=...)"
            )
        layers = self.adapter.layers(self.model)
        layer = layers[layer_number]
        layer_cfg = self.adapter.get_layer_config(layer, self.model.config)
        num_experts = layer_cfg.num_experts
        top_k = layer_cfg.top_k

        @torch.no_grad()
        def _hook_fn(module, args, output):
            input = args[0]  # (batch_size, seq_len, hidden_dim)
            device = input.device
            batch_size, sequence_length, hidden_dim = input.shape
            if layer_number not in self.state:
                self.state[layer_number] = self._initialize_state(hidden_dim, num_experts)
            flat_input = input.view(-1, hidden_dim)  # total_seq_len, hidden

            attention_mask = self._current_attention_mask
            if attention_mask is not None:
                # Flatten mask to match flat_input: (batch_size * seq_len,)
                flat_mask = attention_mask.view(-1).bool().to(device)
            else:
                # No mask provided - treat all tokens as valid
                flat_mask = None

            if self.hook_config.fused_experts:
                # Fused experts (Llama4 / LFM2 / Qwen3.5/3.6): per-expert
                # activations computed from the stacked gate_up_proj / down_proj
                # weights via the shared helper (Phase-1 bmm "grouped" pattern,
                # routed-only) -- a bridge until the FREA kernel (docs/kernels/
                # Phase 3) replaces it.
                act_fn = _resolve_act_fn(self.model.config)
                activations, selected_experts, router_logits = (
                    compute_fused_expert_activations(
                        module, self.adapter, flat_input, num_experts, top_k, act_fn,
                    )
                )

            else:  # loop based MoE execution
                if not isinstance(output, tuple) or len(output) < 2:
                    raise ValueError(
                        f"Expected output of module {module.__class__.__name__} "
                        f"at layer {layer_number} to be a tuple of at least length "
                        f"2 for the non-fused (loop) observer path, got "
                        f"{type(output).__name__}."
                    )
                *_, router_logits = output  # (total_tokens, num_experts)
                _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
                activations = torch.zeros(
                    (num_experts, *flat_input.shape),
                    device=device,
                    dtype=flat_input.dtype,
                )
                for idx, expert in enumerate(module.experts):
                    activations[idx] = expert(flat_input).to(
                        device
                    )  # (num_experts, total_seq_len, hidden_dim)

            del flat_input
            
            pruning_batch = update_pruning_state(
                self.state[layer_number],
                activations=activations,
                selected_experts=selected_experts,
                router_logits=router_logits,
                num_experts=num_experts,
                valid_token_mask=flat_mask,
                renormalize_router_weights=self.hook_config.renormalize_router_weights,
            )

            # Merging critera
            if not self.hook_config.record_pruning_metrics_only:
                ttm_similarity_matrix = ttm_online(
                    pruning_batch.activations,
                    pruning_batch.selected_experts,
                    distance_callable=distance_fn,
                    num_experts=num_experts,
                    pairwise_expert_frequency=pruning_batch.pairwise_expert_frequency,
                )

                # ttm_similarity_matrix with pairwise frequency counts
                self.state[layer_number]["ttm_similarity_matrix"].update(
                    ttm_similarity_matrix, pruning_batch.pairwise_expert_frequency
                )
                del ttm_similarity_matrix

                routed_characteristic_activation = get_routed_characteristic_activation(
                    pruning_batch.activations,
                    pruning_batch.selected_experts,
                    pruning_batch.expert_frequency,
                    device,
                    hidden_dim,
                    num_experts,
                )

                # routed_characteristic_activation with expert frequency counts
                expert_freq_expanded = pruning_batch.expert_frequency.unsqueeze(-1).expand(
                    (-1, hidden_dim)
                )
                self.state[layer_number]["routed_characteristic_activation"].update(
                    routed_characteristic_activation, expert_freq_expanded
                )
                del expert_freq_expanded, routed_characteristic_activation

                online_characteristic_activation_dist = ca_dist_online(
                    pruning_batch.activations,
                    distance_callable=distance_fn,
                ).to(device="cpu")

                # online_characteristic_activation_dist with expert frequency counts
                self.state[layer_number]["online_characteristic_activation_dist"].update(
                    online_characteristic_activation_dist, pruning_batch.num_tokens
                )
                del online_characteristic_activation_dist

                # router logit similarity -> must align with distance_fn shape expectations
                # dim 0 "batch" dim, dims 1,2 expert pairwise, dim 3 token logits
                router_logit_sim = (
                    distance_fn(
                        pruning_batch.router_logits.permute(1, 0).view(
                            1, num_experts, 1, -1
                        ),  # 1, num_experts, 1, logits
                        pruning_batch.router_logits.permute(1, 0).view(
                            1, 1, num_experts, -1
                        ),  # 1, 1, num_experts, logits
                    )
                    .squeeze()
                    .to(device="cpu")
                )  # yields (num_experts, num_experts)

                # router_logit_similarity with total tokens count
                self.state[layer_number]["router_logit_similiarity"].update(
                    router_logit_sim, pruning_batch.num_tokens
                )
                del router_logit_sim

                # characteristic_activation with total tokens count
                self.state[layer_number]["characteristic_activation"].update(
                    pruning_batch.activations.mean(dim=1), pruning_batch.num_tokens
                )

            # --- CLEAN UP -------------------------------------------------------------
            del (
                activations,
                selected_experts,
                router_logits,
                pruning_batch,
            )
            gc.collect()

        return _hook_fn

