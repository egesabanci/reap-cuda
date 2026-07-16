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
from reap.pruning_metrics import (
    initialize_pruning_state,
    resolve_compute_device,
)
from reap.kernels.backend import select_observe_backend
from reap.kernels.observe import observe_moe_batch
from reap.kernels.weight_cache import free_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_ACT_FN_MAP = {
    "silu": F.silu,
    "swiglu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


def _resolve_act_fn(config: Any) -> callable:
    """Return the expert-MLP gate activation from ``config.hidden_act``."""
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
    """Layout-normalized fused expert activations (routed-only).

    Uses F4 weight stacks so Llama4 bmm and Qwen/LFM2 linear layouts both work.
    Kept for tests / external callers; observers go through :func:`observe_moe_batch`.
    """
    from reap.kernels.router import extract_router_logits, f5_router
    from reap.kernels.weight_cache import get_stacked_expert_weights
    from reap.kernels.bmm import (
        routed_expert_activations_grouped,
        materialize_sparse_activations,
    )

    router = getattr(module, adapter.router_attr(), None) or getattr(
        module, "router", None
    ) or getattr(module, "gate", None)
    if router is None:
        raise ValueError("No router on MoE module")
    router_logits = extract_router_logits(router, flat_input)
    pairs = f5_router(router_logits, top_k, norm_topk_prob=False)
    stacked = get_stacked_expert_weights(module, adapter, device=flat_input.device)
    pair_out = routed_expert_activations_grouped(
        flat_input,
        pairs,
        stacked["W_gate"],
        stacked["W_up"],
        stacked["W_down"],
        act_fn=act_fn,
    )
    activations = materialize_sparse_activations(
        pair_out,
        pairs.pair_token_idx,
        pairs.pair_expert_idx,
        num_experts,
        flat_input.shape[0],
        flat_input.shape[1],
        device=flat_input.device,
        dtype=flat_input.dtype,
    )
    return activations, pairs.selected_experts, router_logits


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
        raise NotImplementedError("Subclasses must implement _hook_factory method.")

    def report_state(self) -> dict[str, Any]:
        return self.state

    def close_hooks(self):
        self.reset()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        free_cache()
        logger.debug("All hooks closed for %s.", self.model.__class__.__name__)

    def reset(self):
        del self.state
        gc.collect()
        self.state = {}
        free_cache()
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
        for layer_number, layer_state in self.state.items():
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    self.state[layer_number][key] = value.cpu()
                elif isinstance(value, OnlineStatsTracker):
                    value.to(torch.device("cpu"))

    def _validate_hook_config(self):
        if self.hook_config is None:
            return
        if (
            self.hook_config.module_name_to_hook_regex is None
            and self.hook_config.module_class_name_to_hook_regex is None
        ):
            raise ValueError(
                "At least one of 'module_name_to_hook_regex' or "
                "'module_class_name_to_hook_regex' must be provided in the hook config."
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
        raise ValueError(
            "Unsupported architecture for "
            f"{cls.__name__}: {model_cls_name}. "
            "Registered architectures in "
            f"{cls.__name__}._architecture_registry: "
            f"{list(registry.keys())}"
        )


@dataclass
class MoETransformerObserverConfig(BaseTransformerObserverHookConfig):
    module_name_to_hook_regex: Optional[str] = None
    module_class_name_to_hook_regex: Optional[str] = None
    fused_experts: bool = False
    distance_measure: str = "angular"
    renormalize_router_weights: bool = False
    record_pruning_metrics_only: bool = True
    observe_backend: str = "auto"


class MoETransformerObserver(BaseTransformerObserver):
    """MoE Transformer Observer for pruning and merging metrics."""

    def __init__(self, model, hook_config=None, adapter=None):
        self._current_attention_mask: Optional[torch.Tensor] = None
        self.adapter = adapter
        super().__init__(model, hook_config)

    @contextmanager
    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]):
        previous_attention_mask = self._current_attention_mask
        self._current_attention_mask = attention_mask
        try:
            yield
        finally:
            self._current_attention_mask = previous_attention_mask

    def clear_attention_mask(self):
        self._current_attention_mask = None

    def report_state(self) -> dict[str, Any]:
        return {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }

    def _initialize_state(self, hidden_dim: int, num_experts: int, device=None):
        device = resolve_compute_device(device)
        layer_state = initialize_pruning_state(num_experts, device=device)

        if not self.hook_config.record_pruning_metrics_only:
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
            layer_state["characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            layer_state["online_characteristic_activation_dist"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            layer_state["router_logit_similiarity"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )

        return layer_state

    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        distance_fn = get_distance_fn("cosine")

        if self.adapter is None:
            raise RuntimeError(
                "MoETransformerObserver requires an adapter (pass adapter=...)"
            )
        layers = self.adapter.layers(self.model)
        layer = layers[layer_number]
        layer_cfg = self.adapter.get_layer_config(layer, self.model.config)
        num_experts = layer_cfg.num_experts
        top_k = layer_cfg.top_k
        fused = layer_cfg.fused_experts
        backend = select_observe_backend(
            getattr(self.hook_config, "observe_backend", "auto")
        )
        act_fn = _resolve_act_fn(self.model.config)
        compute_routed_ca = not self.hook_config.record_pruning_metrics_only

        @torch.no_grad()
        def _hook_fn(module, args, output):
            input = args[0]
            device = input.device
            batch_size, sequence_length, hidden_dim = input.shape
            if layer_number not in self.state:
                self.state[layer_number] = self._initialize_state(
                    hidden_dim, num_experts, device=device
                )

            flat_input = input.view(-1, hidden_dim)
            attention_mask = self._current_attention_mask
            flat_mask = (
                attention_mask.view(-1).bool().to(device)
                if attention_mask is not None
                else None
            )

            batch_out = observe_moe_batch(
                self.state[layer_number],
                module,
                self.adapter,
                flat_input,
                num_experts=num_experts,
                top_k=top_k,
                act_fn=act_fn,
                valid_token_mask=flat_mask,
                renormalize_router_weights=self.hook_config.renormalize_router_weights,
                backend=backend,
                record_pruning_metrics_only=self.hook_config.record_pruning_metrics_only,
                compute_routed_ca=compute_routed_ca,
                fused=fused,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )

            if not self.hook_config.record_pruning_metrics_only:
                activations = batch_out["activations"]
                selected_experts = batch_out["selected_experts"]
                router_logits = batch_out["router_logits"]
                expert_frequency = batch_out["expert_frequency"]
                pairwise = batch_out["pairwise_expert_frequency"]
                num_tokens = batch_out["num_tokens"]

                ttm_similarity_matrix = ttm_online(
                    activations,
                    selected_experts,
                    distance_callable=distance_fn,
                    num_experts=num_experts,
                    pairwise_expert_frequency=pairwise,
                )
                self.state[layer_number]["ttm_similarity_matrix"].update(
                    ttm_similarity_matrix, pairwise
                )
                del ttm_similarity_matrix

                routed_ca = get_routed_characteristic_activation(
                    activations,
                    selected_experts,
                    expert_frequency,
                    device,
                    hidden_dim,
                    num_experts,
                )
                expert_freq_expanded = expert_frequency.unsqueeze(-1).expand(
                    (-1, hidden_dim)
                )
                self.state[layer_number]["routed_characteristic_activation"].update(
                    routed_ca, expert_freq_expanded
                )
                del expert_freq_expanded, routed_ca

                online_ca_dist = ca_dist_online(
                    activations, distance_callable=distance_fn
                )
                self.state[layer_number]["online_characteristic_activation_dist"].update(
                    online_ca_dist, num_tokens
                )
                del online_ca_dist

                router_logit_sim = distance_fn(
                    router_logits.permute(1, 0).view(1, num_experts, 1, -1),
                    router_logits.permute(1, 0).view(1, 1, num_experts, -1),
                ).squeeze()
                self.state[layer_number]["router_logit_similiarity"].update(
                    router_logit_sim, num_tokens
                )
                del router_logit_sim

                self.state[layer_number]["characteristic_activation"].update(
                    activations.mean(dim=1), num_tokens
                )

            # Keep GPU free of large transients; do not force CPU on metrics.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return _hook_fn
