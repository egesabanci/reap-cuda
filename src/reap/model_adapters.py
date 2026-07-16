"""Model adapter helpers for HuggingFace-backed MoE models.

This module describes HuggingFace-style model layouts through plain Python
attribute access. It is intentionally import-light — no ``transformers`` or
``torch`` imports at module top level beyond what is needed for type stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture-neutral MoE layer metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeLayerConfig:
    """Architecture-agnostic MoE layer metadata.

    Attributes:
        num_experts: Total routed experts in this MoE layer.
        top_k: Number of experts selected per token.
        norm_topk_prob: Whether selected router scores are renormalised.
        adapter_name: Adapter identifier (e.g. ``"qwen3_moe"``).
        fused_experts: Experts share a single fused weight (Llama4-style).
        use_expert_bias: Layer uses per-expert bias terms (LFM2 / ERNIE).
        weight_convention: ``"linear"`` = ``(E, out, in)`` / F.linear;
            ``"bmm"`` = ``(E, in, out)`` / bmm (Llama4 fused).
    """

    num_experts: int
    top_k: int
    norm_topk_prob: bool
    adapter_name: str = "qwen3_moe"
    fused_experts: bool = False
    use_expert_bias: bool = False
    weight_convention: str = "linear"  # "linear" | "bmm"


def _patch_router_after_slice(router: Any, keep_indices: list[int], top_k: int | None = None) -> None:
    """Slice router weights / live counters after expert pruning."""
    n = len(keep_indices)
    if hasattr(router, "weight") and router.weight is not None:
        router.weight.data = router.weight.data[keep_indices, ...]
    if getattr(router, "bias", None) is not None:
        router.bias.data = router.bias.data[keep_indices]
    if hasattr(router, "out_features"):
        router.out_features = n
    if hasattr(router, "num_experts"):
        router.num_experts = n
    if hasattr(router, "top_k") and top_k is not None:
        router.top_k = min(int(top_k), n)
    if hasattr(router, "e_score_correction_bias"):
        router.e_score_correction_bias.data = router.e_score_correction_bias.data[
            keep_indices
        ]


def _patch_fused_experts_count(experts: Any, n: int) -> None:
    if hasattr(experts, "num_experts"):
        experts.num_experts = n


# ---------------------------------------------------------------------------
# Attribute-path helpers (port of reap-mlx model_adapters helpers)
# ---------------------------------------------------------------------------


def _lookup_attr_path(root: Any, path: tuple[str, ...]) -> Any | None:
    current = root
    for attr in path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _config_value(
    config: Any,
    *keys: str,
    default: Any = None,
) -> Any:
    """Read the first non-None value for *keys from a config.

    Supports both dict-like configs (``.get(key)``) and attribute-based configs
    such as HuggingFace ``PretrainedConfig`` (which has no ``.get()``).
    """
    if config is None:
        return default
    getter = getattr(config, "get", None)
    use_get = callable(getter)
    for key in keys:
        value = getter(key) if use_get else getattr(config, key, None)
        if value is not None:
            return value
    return default


def _live_value(module: Any, *attrs: str) -> Any:
    for attr in attrs:
        value = getattr(module, attr, None)
        if value is not None:
            return value
    return None


def _live_or_config_value(
    module: Any,
    live_attrs: tuple[str, ...],
    config: Mapping[str, Any] | None,
    config_keys: tuple[str, ...],
    *,
    default: Any = None,
) -> Any:
    value = _live_value(module, *live_attrs)
    if value is not None:
        return value
    return _config_value(config, *config_keys, default=default)


def _positive_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


# ---------------------------------------------------------------------------
# Layer discovery (HF convention)
# ---------------------------------------------------------------------------


def get_model_layers(model: Any) -> Sequence[Any]:
    """Return decoder layers from a HuggingFace causal LM model."""
    # HF convention: model.model.layers
    layers = _lookup_attr_path(model, ("model", "layers"))
    if layers is not None:
        return layers
    # Fallback: model.layers (some non-causal-LM models)
    layers = _lookup_attr_path(model, ("layers",))
    if layers is not None:
        return layers
    raise ValueError(
        "Could not find model layers. Expected model.model.layers or model.layers."
    )


def get_shared_expert(moe: nn.Module) -> nn.Module | None:
    """Return the shared expert module if this MoE block exposes one."""
    for attr in ("shared_experts", "shared_expert"):
        shared = getattr(moe, attr, None)
        if shared is not None:
            return shared
    return None


# ---------------------------------------------------------------------------
# Config update helpers
# ---------------------------------------------------------------------------


def update_qwen3_moe_config(
    config: MutableMapping[str, Any],
    *,
    num_experts: int,
    top_k: int,
) -> MutableMapping[str, Any]:
    """Update a Qwen3-MoE config dict after expert pruning."""
    num_experts = _positive_int(num_experts, "num_experts")
    top_k = min(_positive_int(top_k, "top_k"), num_experts)

    config["num_experts"] = num_experts
    config["num_experts_per_tok"] = top_k
    if "top_k" in config:
        config["top_k"] = top_k
    return config


# ===================================================================
# Adapter classes
# ===================================================================


class Qwen3MoeModelAdapter:
    """Qwen3-MoE adapter for HuggingFace-style model objects.

    Targets ``Qwen3MoeSparseMoeBlock`` modules found under
    ``layer.mlp``.  This adapter covers Qwen3-MoE, Qwen3.5-MoE, and
    Qwen3.6-MoE (subject to layout validation in item 4).
    """

    adapter_name = "qwen3_moe"

    # -- concrete attribute accessors (replaces MODEL_ATTRS dict) --------
    def moe_block_attr(self) -> str:
        return "mlp"

    def hook_regex(self) -> str:
        return "Qwen3MoeSparseMoeBlock"

    def experts_attr(self) -> str:
        return "experts"

    def router_attr(self) -> str:
        return "gate"

    def num_experts_config_attr(self) -> str:
        return "num_experts"

    def weight_convention(self) -> str:
        """``linear`` = F.linear ``(out, in)``; ``bmm`` = matmul ``(in, out)``."""
        return "linear"

    def expert_weight_attrs(self, moe: Any | None = None) -> dict[str, Any]:
        """Per-expert weight attribute-name layout used by merge/permute/kernels.

        When *moe* is provided, ``fused`` reflects the live module layout so
        transformers>=5.x Qwen3 fused stacks are reported correctly.
        """
        fused = False
        if moe is not None:
            fused = self._is_fused_experts(getattr(moe, self.experts_attr(), None))
        if fused:
            return {
                "experts": self.experts_attr(),
                "gate": self.router_attr(),
                "fused": True,
                "gate_proj": "gate_up_proj",
                "up_proj": "gate_up_proj",
                "down_proj": "down_proj",
                "weight_convention": self.weight_convention(),
            }
        return {
            "experts": self.experts_attr(),
            "gate": self.router_attr(),
            "fused": False,
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
            "weight_convention": self.weight_convention(),
        }

    # -- layout inspection ------------------------------------------------
    def layers(self, model: Any) -> Sequence[Any]:
        return get_model_layers(model)

    def identify_moe_layers(self, model: Any) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.layers(model))
            if self.is_moe_layer(layer)
        ]

    def is_moe_layer(self, layer: Any) -> bool:
        mlp = getattr(layer, "mlp", None)
        return mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate")

    def get_moe(self, layer: Any) -> Any:
        if not self.is_moe_layer(layer):
            raise ValueError("Layer does not expose a Qwen3-style MoE mlp.experts.")
        return layer.mlp

    def get_dense_mlp(self, layer: Any) -> Any:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("Layer does not expose an mlp module.")
        return mlp

    @staticmethod
    def _is_fused_experts(experts) -> bool:
        """True if experts is a fused module (stacked params on dim 0),
        False if it is an iterable ModuleList."""
        if experts is None:
            return False
        if isinstance(experts, nn.ModuleList):
            return False
        # Fused Qwen3MoeExperts (transformers>=5.x): gate_up_proj/down_proj
        # are nn.Parameter tensors stacked on the expert axis.
        return hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")

    def get_layer_config(
        self,
        layer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> MoeLayerConfig:
        moe = self.get_moe(layer)

        experts = getattr(moe, self.experts_attr(), None)
        # Prefer live stack / ModuleList length over stale counters.
        if self._is_fused_experts(experts) and hasattr(experts, "gate_up_proj"):
            num_experts = int(experts.gate_up_proj.shape[0])
        elif isinstance(experts, nn.ModuleList):
            num_experts = len(experts)
        else:
            num_experts = _positive_int(
                _live_or_config_value(
                    moe,
                    ("num_experts",),
                    config,
                    (self.num_experts_config_attr(), "num_experts"),
                ),
                "num_experts",
            )

        router = getattr(moe, self.router_attr(), None)
        live_top = _live_value(moe, "top_k", "num_experts_per_tok")
        if live_top is None and router is not None:
            live_top = _live_value(router, "top_k", "num_experts_per_tok")
        if live_top is None:
            live_top = _config_value(config, "num_experts_per_tok", "top_k")
        top_k = _positive_int(live_top, "top_k")

        norm_topk_prob = bool(
            _live_or_config_value(
                moe,
                ("norm_topk_prob",),
                config,
                ("norm_topk_prob",),
                default=False,
            )
        )
        return MoeLayerConfig(
            num_experts=num_experts,
            top_k=min(top_k, num_experts),
            norm_topk_prob=norm_topk_prob,
            adapter_name=self.adapter_name,
            fused_experts=self._is_fused_experts(experts),
            weight_convention=self.weight_convention(),
        )

    # -- expert slicing (PyTorch, replaces prune.py non-fused branch) -----
    def slice_experts(
        self,
        moe: nn.Module,
        keep_indices: list[int],
    ) -> None:
        """Slice MoE experts and router so the live module stays runnable."""
        n = len(keep_indices)
        all_experts = getattr(moe, self.experts_attr())
        if self._is_fused_experts(all_experts):
            all_experts.gate_up_proj.data = all_experts.gate_up_proj.data[keep_indices]
            all_experts.down_proj.data = all_experts.down_proj.data[keep_indices]
            _patch_fused_experts_count(all_experts, n)
        else:
            retained = nn.ModuleList([all_experts[i] for i in keep_indices])
            setattr(moe, self.experts_attr(), retained)

        if hasattr(moe, "num_experts"):
            moe.num_experts = n

        router = getattr(moe, self.router_attr())
        # Preserve current top_k if present, clamp to retained count.
        prev_top = getattr(router, "top_k", None)
        if prev_top is None:
            prev_top = getattr(moe, "top_k", None)
        _patch_router_after_slice(router, keep_indices, top_k=prev_top)

    # -- config patching ---------------------------------------------------
    def update_config(
        self,
        config: MutableMapping[str, Any],
        num_experts: int,
        top_k: int,
    ) -> None:
        """Patch config dict / PretrainedConfig after pruning."""
        setattr(config, self.num_experts_config_attr(), num_experts)
        top_k = min(top_k, num_experts)
        if hasattr(config, "num_experts_per_tok"):
            config.num_experts_per_tok = top_k
        if hasattr(config, "top_k"):
            config.top_k = top_k


class Qwen3_5MoeModelAdapter(Qwen3MoeModelAdapter):
    """Qwen3.5/3.6-MoE adapter for the transformers>=5.x ``qwen3_5_moe`` family.

    Shares the fused expert layout of :class:`Qwen3MoeModelAdapter` (stacked
    ``gate_up_proj`` / ``down_proj`` ``nn.Parameter`` tensors) but targets the
    ``Qwen3_5MoeSparseMoeBlock`` module class, whose router is a
    ``Qwen3_5MoeTopKRouter`` (returning a ``(logits, scores, indices)`` tuple;
    the observer unwraps it) and whose block additionally carries a
    ``shared_expert`` + ``shared_expert_gate``. The shared expert is **not** a
    routed expert and must survive pruning unchanged; :meth:`slice_experts`
    only rewrites ``experts`` (``gate_up_proj``/``down_proj``) and the ``gate``
    router, so the shared expert is preserved automatically.
    """

    adapter_name = "qwen3_5_moe"

    def hook_regex(self) -> str:
        return "Qwen3_5MoeSparseMoeBlock"


class Llama4MoeModelAdapter:
    """Llama4-MoE adapter for HuggingFace-style fused expert blocks.

    Llama4 uses fused ``gate_up_proj`` / ``down_proj`` arrays stacked
    on dim 0 and accessed via ``layer.feed_forward``.
    """

    adapter_name = "llama4_moe"

    def moe_block_attr(self) -> str:
        return "feed_forward"

    def hook_regex(self) -> str:
        return "Llama4TextMoe"

    def experts_attr(self) -> str:
        return "experts"

    def router_attr(self) -> str:
        # HF Llama4TextMoe uses ``self.router`` (not ``.gate``).
        return "router"

    def num_experts_config_attr(self) -> str:
        return "num_local_experts"

    def weight_convention(self) -> str:
        # gate_up_proj is (E, H, 2I); down_proj is (E, I, H) — bmm convention.
        return "bmm"

    def expert_weight_attrs(self, moe: Any | None = None) -> dict[str, Any]:
        """Fused expert layout: bmm convention stacked gate_up / down."""
        return {
            "experts": self.experts_attr(),
            "gate": self.router_attr(),
            "fused": True,
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "down_proj": "down_proj",
            "weight_convention": self.weight_convention(),
        }

    def layers(self, model: Any) -> Sequence[Any]:
        return get_model_layers(model)

    def identify_moe_layers(self, model: Any) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.layers(model))
            if self.is_moe_layer(layer)
        ]

    def is_moe_layer(self, layer: Any) -> bool:
        ff = getattr(layer, "feed_forward", None)
        if ff is None or not hasattr(ff, "experts"):
            return False
        # Real HF Llama4 uses .router; some mocks alias .gate = .router.
        return hasattr(ff, "router") or hasattr(ff, "gate")

    def get_moe(self, layer: Any) -> Any:
        if not self.is_moe_layer(layer):
            raise ValueError(
                "Layer does not expose Llama4-style feed_forward.experts."
            )
        return layer.feed_forward

    def get_dense_mlp(self, layer: Any) -> Any:
        ff = getattr(layer, "feed_forward", None)
        if ff is None:
            raise ValueError("Layer does not expose a feed_forward module.")
        return ff

    def get_layer_config(
        self,
        layer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> MoeLayerConfig:
        moe = self.get_moe(layer)
        experts = getattr(moe, self.experts_attr(), None)
        if experts is not None and hasattr(experts, "gate_up_proj"):
            num_experts = int(experts.gate_up_proj.shape[0])
        else:
            num_experts = _positive_int(
                _live_or_config_value(
                    moe,
                    ("num_experts",),
                    config,
                    ("num_local_experts", "num_experts"),
                ),
                "num_experts",
            )
        live_top = _live_value(moe, "top_k", "num_experts_per_tok")
        if live_top is None:
            live_top = _config_value(config, "num_experts_per_tok", "top_k")
        top_k = _positive_int(live_top, "top_k")
        return MoeLayerConfig(
            num_experts=num_experts,
            top_k=min(top_k, num_experts),
            norm_topk_prob=False,
            adapter_name=self.adapter_name,
            fused_experts=True,
            weight_convention=self.weight_convention(),
        )

    def slice_experts(self, moe: nn.Module, keep_indices: list[int]) -> None:
        """Slice fused Llama4 expert weights on dim 0."""
        n = len(keep_indices)
        moe.experts.gate_up_proj.data = moe.experts.gate_up_proj.data[keep_indices]
        moe.experts.down_proj.data = moe.experts.down_proj.data[keep_indices]
        _patch_fused_experts_count(moe.experts, n)
        moe.num_experts = n
        router = getattr(moe, self.router_attr(), None) or getattr(moe, "gate", None)
        if router is None:
            raise ValueError("Llama4 MoE block has neither .router nor .gate")
        prev_top = getattr(moe, "top_k", None)
        _patch_router_after_slice(router, keep_indices, top_k=prev_top)
        if hasattr(moe, "top_k") and prev_top is not None:
            moe.top_k = min(int(prev_top), n)

    def update_config(
        self,
        config: MutableMapping[str, Any],
        num_experts: int,
        top_k: int,
    ) -> None:
        setattr(config, self.num_experts_config_attr(), num_experts)
        top_k = min(top_k, num_experts)
        if hasattr(config, "num_experts_per_tok"):
            config.num_experts_per_tok = top_k


class Lfm2MoeModelAdapter:
    """Liquid LFM2.5 MoE adapter for HuggingFace fused expert blocks.

    Targets ``Lfm2MoeSparseMoeBlock`` modules found under
    ``layer.feed_forward``. The router is a plain ``nn.Linear`` at
    ``moe.gate`` (weight shape ``(num_experts, hidden_size)``). Experts are
    fused into a single ``Lfm2MoeExperts`` module exposing stacked
    ``gate_up_proj`` ``(E, 2*I, H)`` and ``down_proj`` ``(E, H, I)`` tensors
    — no per-expert ``nn.Linear``, not iterable (no ``__getitem__``/``__len__``).
    Dense layers (the first ``num_dense_layers``) use ``Lfm2MoeMLP`` and
    expose neither ``gate`` nor ``experts``.

    Requires ``transformers>=5.2`` (``Lfm2MoeForCausalLM`` is not present in
    older releases — see issue #4 / the LFM2 runtime note).
    """

    adapter_name = "lfm2_moe"

    def moe_block_attr(self) -> str:
        return "feed_forward"

    def hook_regex(self) -> str:
        return "Lfm2MoeSparseMoeBlock"

    def experts_attr(self) -> str:
        return "experts"

    def router_attr(self) -> str:
        # LFM2 router is an ``nn.Linear`` at ``.gate`` (not ``.router`` like
        # Llama4). Weight shape: ``(num_experts, hidden_size)``.
        return "gate"

    def num_experts_config_attr(self) -> str:
        return "num_experts"

    def weight_convention(self) -> str:
        # LFM2 matches Qwen fused Linear layout: gate_up (E, 2I, H).
        return "linear"

    def expert_weight_attrs(self, moe: Any | None = None) -> dict[str, Any]:
        """Fused layout: gate+up stacked in ``gate_up_proj`` (Linear convention)."""
        return {
            "experts": self.experts_attr(),
            "gate": self.router_attr(),
            "fused": True,
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "down_proj": "down_proj",
            "weight_convention": self.weight_convention(),
        }

    # -- layout inspection ------------------------------------------------
    def layers(self, model: Any) -> Sequence[Any]:
        return get_model_layers(model)

    def identify_moe_layers(self, model: Any) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.layers(model))
            if self.is_moe_layer(layer)
        ]

    def is_moe_layer(self, layer: Any) -> bool:
        ff = getattr(layer, "feed_forward", None)
        if ff is None or not hasattr(ff, "experts"):
            return False
        # Guard against Llama4-style blocks (which also have feed_forward.experts
        # but use .router). The LFM2 MoE block class is Lfm2MoeSparseMoeBlock.
        return type(ff).__name__ == "Lfm2MoeSparseMoeBlock"

    def get_moe(self, layer: Any) -> Any:
        if not self.is_moe_layer(layer):
            raise ValueError(
                "Layer does not expose LFM2-style feed_forward.experts "
                "(Lfm2MoeSparseMoeBlock)."
            )
        return layer.feed_forward

    def get_dense_mlp(self, layer: Any) -> Any:
        ff = getattr(layer, "feed_forward", None)
        if ff is None:
            raise ValueError("Layer does not expose a feed_forward module.")
        return ff

    def get_layer_config(
        self,
        layer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> MoeLayerConfig:
        moe = self.get_moe(layer)

        num_experts = _positive_int(
            _live_or_config_value(
                moe,
                ("num_experts",),
                config,
                ("num_experts",),
            ),
            "num_experts",
        )
        top_k = _positive_int(
            _live_or_config_value(
                moe,
                ("top_k", "num_experts_per_tok"),
                config,
                ("num_experts_per_tok", "top_k"),
            ),
            "top_k",
        )
        norm_topk_prob = bool(
            _live_or_config_value(
                moe,
                ("norm_topk_prob",),
                config,
                ("norm_topk_prob",),
                default=False,
            )
        )
        use_expert_bias = bool(
            _live_or_config_value(
                moe,
                ("use_expert_bias",),
                config,
                ("use_expert_bias",),
                default=False,
            )
        )
        experts = getattr(moe, self.experts_attr(), None)
        if experts is not None and hasattr(experts, "gate_up_proj"):
            num_experts = int(experts.gate_up_proj.shape[0])
        return MoeLayerConfig(
            num_experts=num_experts,
            top_k=min(top_k, num_experts),
            norm_topk_prob=norm_topk_prob,
            adapter_name=self.adapter_name,
            fused_experts=True,
            use_expert_bias=use_expert_bias,
            weight_convention=self.weight_convention(),
        )

    # -- expert slicing (fused, dim 0 = expert axis) ---------------------
    def slice_experts(self, moe: nn.Module, keep_indices: list[int]) -> None:
        """Slice fused LFM2 expert weights on dim 0 (expert axis)."""
        n = len(keep_indices)
        exps = getattr(moe, self.experts_attr())
        exps.gate_up_proj.data = exps.gate_up_proj.data[keep_indices]
        exps.down_proj.data = exps.down_proj.data[keep_indices]
        _patch_fused_experts_count(exps, n)
        if hasattr(moe, "num_experts"):
            moe.num_experts = n
        gate = getattr(moe, self.router_attr())
        prev_top = getattr(moe, "top_k", None) or getattr(gate, "top_k", None)
        _patch_router_after_slice(gate, keep_indices, top_k=prev_top)
        # LFM2 per-expert bias (config.use_expert_bias=True).
        if hasattr(moe, "expert_bias") and moe.expert_bias is not None:
            moe.expert_bias.data = moe.expert_bias.data[keep_indices]

    def update_config(
        self,
        config: MutableMapping[str, Any],
        num_experts: int,
        top_k: int,
    ) -> None:
        setattr(config, self.num_experts_config_attr(), num_experts)
        top_k = min(top_k, num_experts)
        if hasattr(config, "num_experts_per_tok"):
            config.num_experts_per_tok = top_k


class MixtralMoeModelAdapter(Qwen3MoeModelAdapter):
    """Mixtral / PhiMoE-style adapter (``layer.block_sparse_moe``)."""

    adapter_name = "mixtral_moe"

    def moe_block_attr(self) -> str:
        return "block_sparse_moe"

    def hook_regex(self) -> str:
        return "MixtralSparseMoeBlock"

    def num_experts_config_attr(self) -> str:
        return "num_local_experts"

    def is_moe_layer(self, layer: Any) -> bool:
        block = getattr(layer, "block_sparse_moe", None)
        return block is not None and hasattr(block, "experts") and hasattr(block, "gate")

    def get_moe(self, layer: Any) -> Any:
        if not self.is_moe_layer(layer):
            raise ValueError(
                "Layer does not expose Mixtral-style block_sparse_moe.experts."
            )
        return layer.block_sparse_moe

    def get_dense_mlp(self, layer: Any) -> Any:
        block = getattr(layer, "block_sparse_moe", None)
        if block is not None:
            return block
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("Layer does not expose block_sparse_moe or mlp.")
        return mlp


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

_QWEN_FAMILY_TYPES = frozenset(
    {
        "qwen2_moe",
        "qwen3_moe",
        "qwen3_vl_moe",
    }
)
_MIXTRAL_FAMILY_TYPES = frozenset({"mixtral", "phimoe", "phi_moe"})
_LFM2_FAMILY_TYPES = frozenset({"lfm2_moe"})


def infer_model_adapter(
    model: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Infer the model adapter from config or live model layout.

    Returns ``None`` when no supported MoE architecture can be detected.
    When a model object is provided, live layout inspection wins over
    config ``model_type`` so dense models with a MoE-looking tag are not
    mis-classified.
    """
    model_type = str(_config_value(config, "model_type") or "")
    architectures = _config_value(config, "architectures", default=()) or ()

    if model is not None:
        try:
            layers = get_model_layers(model)
        except ValueError:
            layers = ()

        # LFM2: fused experts under feed_forward, MoE block class is
        # Lfm2MoeSparseMoeBlock (router attr is .gate). Must be checked before
        # the Llama4 fallback, since both expose feed_forward.experts.
        if any(
            type(getattr(layer, "feed_forward", None)).__name__
            == "Lfm2MoeSparseMoeBlock"
            for layer in layers
        ):
            return Lfm2MoeModelAdapter()

        if any(
            getattr(getattr(layer, "feed_forward", None), "experts", None)
            is not None
            for layer in layers
        ):
            return Llama4MoeModelAdapter()

        if any(
            getattr(getattr(layer, "block_sparse_moe", None), "experts", None)
            is not None
            for layer in layers
        ):
            return MixtralMoeModelAdapter()

        if any(
            type(getattr(layer, "mlp", None)).__name__
            == "Qwen3_5MoeSparseMoeBlock"
            for layer in layers
        ):
            return Qwen3_5MoeModelAdapter()

        if any(
            getattr(getattr(layer, "mlp", None), "experts", None)
            is not None
            for layer in layers
        ):
            return Qwen3MoeModelAdapter()

        # Live model has no supported MoE layout.
        return None

    # Config-only inference (no model object).
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or any(
        "Qwen3_5Moe" in str(a) for a in architectures
    ):
        return Qwen3_5MoeModelAdapter()
    if model_type in _LFM2_FAMILY_TYPES or any(
        str(a).startswith("Lfm2") and "Moe" in str(a) for a in architectures
    ):
        return Lfm2MoeModelAdapter()
    if model_type in _MIXTRAL_FAMILY_TYPES or any(
        "Mixtral" in str(a) or "PhiMoE" in str(a) for a in architectures
    ):
        return MixtralMoeModelAdapter()
    if model_type in _QWEN_FAMILY_TYPES or any(
        "Qwen" in str(a) and "Moe" in str(a) for a in architectures
    ):
        return Qwen3MoeModelAdapter()
    if model_type == "llama4" or any(
        "Llama4" in str(a) for a in architectures
    ):
        return Llama4MoeModelAdapter()

    return None


__all__ = [
    "Lfm2MoeModelAdapter",
    "Llama4MoeModelAdapter",
    "MixtralMoeModelAdapter",
    "MoeLayerConfig",
    "Qwen3_5MoeModelAdapter",
    "Qwen3MoeModelAdapter",
    "get_model_layers",
    "get_shared_expert",
    "infer_model_adapter",
    "update_qwen3_moe_config",
]
