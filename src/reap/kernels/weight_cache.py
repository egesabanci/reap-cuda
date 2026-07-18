"""F4: layout-normalized stacked expert weight cache (model-agnostic).

All stacked tensors are returned in **Linear** convention so kernels can use
``F.linear`` / ``(E, out, in)``:

* ``W_gate``: ``(E, I, H)``
* ``W_up``:   ``(E, I, H)``
* ``W_down``: ``(E, H, I)``

Llama4's native bmm layout ``(E, H, 2I)`` / ``(E, I, H)`` is transposed once
at cache build so FREA never needs architecture branches.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_CACHED_MOE_ID: int | None = None
_CACHED_STACK: dict[str, Any] | None = None
# Full-observer path hooks every MoE each batch. Keep at most one entry so
# stacked weights do not accumulate across layers (tight-VRAM OOM guard).


def free_cache(moe: nn.Module | None = None) -> None:
    """Drop the stacked-weight cache (after a layer finishes calibrating)."""
    global _CACHED_MOE_ID, _CACHED_STACK
    if moe is None or id(moe) == _CACHED_MOE_ID:
        _CACHED_MOE_ID = None
        _CACHED_STACK = None


def cache_size() -> int:
    """Number of MoE modules currently holding stacked weights (tests)."""
    return 1 if _CACHED_STACK is not None else 0


def _source_expert_weight(moe: nn.Module, attrs: dict[str, Any]) -> torch.Tensor:
    """Return one canonical source expert-weight tensor for this MoE layer.

    Used to resolve caller-omitted ``device`` / ``dtype`` to the source-native
    representation before cache matching. Fused and non-fused layouts are both
    handled; the returned tensor is the first expert's gate/up weight (any of
    gate/up/down would carry the same source dtype/device for a given layer).
    """
    if attrs["fused"]:
        exps = getattr(moe, attrs["experts"])
        return exps.gate_up_proj
    experts = getattr(moe, attrs["experts"])
    return getattr(experts[0], attrs["gate_proj"]).weight


def get_stacked_expert_weights(
    moe: nn.Module,
    adapter: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Return contiguous stacked expert weights in Linear convention.

    Single-entry cache per process — at most one MoE's stacked weights are
    retained so the full-observer path cannot pin every layer's stacks.
    """
    global _CACHED_MOE_ID, _CACHED_STACK

    # Resolve caller omissions against the canonical source-weight
    # representation of this MoE layer, so that an omitted device/dtype can
    # never wildcard-match a previously cached converted representation.
    attrs = adapter.expert_weight_attrs(moe)
    convention = attrs.get("weight_convention") or getattr(
        adapter, "weight_convention", lambda: "linear"
    )()
    if callable(convention):
        convention = convention()

    source_weight = _source_expert_weight(moe, attrs)
    requested_device = (
        torch.device(device) if device is not None else source_weight.device
    )
    requested_dtype = dtype if dtype is not None else source_weight.dtype

    if _CACHED_STACK is not None and id(moe) == _CACHED_MOE_ID:
        cached_device = _CACHED_STACK["_resolved_device"]
        cached_dtype = _CACHED_STACK["_resolved_dtype"]
        # A cache hit requires an exact match on the fully resolved
        # representation (device + dtype). Omitted values are resolved to the
        # source-native representation above, so a converted entry cannot be
        # reused by a source-native lookup.
        if cached_device == requested_device and cached_dtype == requested_dtype:
            return _CACHED_STACK
        # Representation mismatch (device or dtype changed) — rebuild.
        _CACHED_MOE_ID = None
        _CACHED_STACK = None

    # Evict stale entry before building a new stack (OOM guard for full observe).
    if _CACHED_STACK is not None:
        _CACHED_MOE_ID = None
        _CACHED_STACK = None

    if attrs["fused"]:
        exps = getattr(moe, attrs["experts"])
        gate_up = exps.gate_up_proj  # Qwen/LFM2: (E, 2I, H); Llama4: (E, H, 2I)
        down = exps.down_proj
        if convention == "bmm":
            # Llama4: gate_up (E, H, 2I) -> split last dim, transpose to (E, I, H)
            e, h, two_i = gate_up.shape
            i = two_i // 2
            W_gate = gate_up[..., :i].transpose(-1, -2).contiguous()  # (E, I, H)
            W_up = gate_up[..., i:].transpose(-1, -2).contiguous()
            # down is (E, I, H) bmm -> need (E, H, I) for F.linear
            W_down = down.transpose(-1, -2).contiguous()
        else:
            # Linear: gate_up (E, 2I, H)
            i = gate_up.shape[1] // 2
            W_gate = gate_up[:, :i, :].contiguous()
            W_up = gate_up[:, i:, :].contiguous()
            W_down = down.contiguous()
    else:
        experts = getattr(moe, attrs["experts"])  # ModuleList
        W_gate = torch.stack(
            [getattr(e, attrs["gate_proj"]).weight for e in experts]
        )
        W_up = torch.stack([getattr(e, attrs["up_proj"]).weight for e in experts])
        W_down = torch.stack(
            [getattr(e, attrs["down_proj"]).weight for e in experts]
        )

    if requested_device != source_weight.device or requested_dtype != source_weight.dtype:
        W_gate = W_gate.to(device=requested_device, dtype=requested_dtype)
        W_up = W_up.to(device=requested_device, dtype=requested_dtype)
        W_down = W_down.to(device=requested_device, dtype=requested_dtype)

    # Detach from autograd to drop memory overhead (observer runs under no_grad).
    W_gate = W_gate.detach()
    W_up = W_up.detach()
    W_down = W_down.detach()

    stacked = {
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
        "fused": bool(attrs["fused"]),
        "weight_convention": "linear",  # always normalized
        "_resolved_device": W_gate.device,
        "_resolved_dtype": W_gate.dtype,
    }
    _CACHED_MOE_ID = id(moe)
    _CACHED_STACK = stacked
    return stacked


def apply_swiglu(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    act_fn=F.silu,
) -> torch.Tensor:
    """Single-expert SwiGLU MLP: weights already Linear ``(out, in)``."""
    g = F.linear(x, W_gate)
    u = F.linear(x, W_up)
    return F.linear(act_fn(g) * u, W_down)
