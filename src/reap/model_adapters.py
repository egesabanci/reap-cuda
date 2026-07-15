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
    """

    num_experts: int
    top_k: int
    norm_topk_prob: bool
    adapter_name: str = "qwen3_moe"
    fused_experts: bool = False
    use_expert_bias: bool = False


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

    def expert_weight_attrs(self) -> dict[str, Any]:
        """Per-expert weight attribute-name layout used by merge/permute.

        Non-fused layout: each expert in a ModuleList exposes separate
        ``gate_proj`` / ``up_proj`` / ``down_proj`` linears.
        """
        return {
            "experts": self.experts_attr(),
            "gate": self.router_attr(),
            "fused": False,
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
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
                (self.num_experts_config_attr(), "num_experts"),
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
        return MoeLayerConfig(
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            adapter_name=self.adapter_name,
            fused_experts=False,
        )

    # -- expert slicing (PyTorch, replaces prune.py non-fused branch) -----
    def slice_experts(
        self,
        moe: nn.Module,
        keep_indices: list[int],
    ) -> None:
        """Slice the non-fused MoE module to only retain *keep_indices*."""
        all_experts = getattr(moe, self.experts_attr())
        retained = nn.ModuleList([all_experts[i] for i in keep_indices])
        setattr(moe, self.experts_attr(), retained)

        router = getattr(moe, self.router_attr())
        router.weight.data = router.weight.data[keep_indices, :]
        if getattr(router, "bias", None) is not None:
            router.bias.data = router.bias.data[keep_indices]
        router.out_features = len(keep_indices)
        if hasattr(router, "e_score_correction_bias"):
            router.e_score_correction_bias.data = (
                router.e_score_correction_bias.data[keep_indices]
            )

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
        return "gate"

    def num_experts_config_attr(self) -> str:
        return "num_local_experts"

    def expert_weight_attrs(self) -> dict[str, Any]:
        """Fused expert layout: gate+up stacked into ``gate_up_proj`` and
        ``down_proj``, both ``(num_experts, ...)`` tensors on a single module.
        """
        return {
            "experts": self.experts_attr(),
            "gate": self.router_attr(),
            "fused": True,
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "down_proj": "down_proj",
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
        return ff is not None and hasattr(ff, "experts") and hasattr(ff, "gate")

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

        num_experts = _positive_int(
            _live_or_config_value(
                moe,
                ("num_experts",),
                config,
                ("num_local_experts", "num_experts"),
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
        return MoeLayerConfig(
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=False,
            adapter_name=self.adapter_name,
            fused_experts=True,
        )

    def slice_experts(self, moe: nn.Module, keep_indices: list[int]) -> None:
        """Slice fused Llama4 expert weights on dim 0."""
        moe.experts.gate_up_proj.data = moe.experts.gate_up_proj.data[keep_indices]
        moe.experts.down_proj.data = moe.experts.down_proj.data[keep_indices]
        moe.num_experts = len(keep_indices)
        moe.router.weight.data = moe.router.weight.data[keep_indices]
        moe.router.out_features = len(keep_indices)
        if hasattr(moe.router, "num_experts"):
            moe.router.num_experts = len(keep_indices)

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
            getattr(getattr(layer, "mlp", None), "experts", None)
            is not None
            for layer in layers
        ):
            return Qwen3MoeModelAdapter()

        # Live model has no supported MoE layout.
        return None

    # Config-only inference (no model object).
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
    "Llama4MoeModelAdapter",
    "MixtralMoeModelAdapter",
    "MoeLayerConfig",
    "Qwen3MoeModelAdapter",
    "get_model_layers",
    "get_shared_expert",
    "infer_model_adapter",
    "update_qwen3_moe_config",
]
