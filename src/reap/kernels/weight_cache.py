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

_STACK_CACHE: dict[int, dict[str, Any]] = {}


def free_cache(moe: nn.Module | None = None) -> None:
    """Drop the stacked-weight cache (after a layer finishes calibrating)."""
    if moe is None:
        _STACK_CACHE.clear()
    else:
        _STACK_CACHE.pop(id(moe), None)


def get_stacked_expert_weights(
    moe: nn.Module,
    adapter: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Return contiguous stacked expert weights in Linear convention.

    Cached on ``id(moe)`` for the layer's calibration lifetime.
    """
    key = id(moe)
    if key in _STACK_CACHE:
        cached = _STACK_CACHE[key]
        if device is None or cached["W_gate"].device == device:
            return cached

    attrs = adapter.expert_weight_attrs(moe)
    convention = attrs.get("weight_convention") or getattr(
        adapter, "weight_convention", lambda: "linear"
    )()
    if callable(convention):
        convention = convention()

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

    if device is not None or dtype is not None:
        W_gate = W_gate.to(device=device or W_gate.device, dtype=dtype or W_gate.dtype)
        W_up = W_up.to(device=device or W_up.device, dtype=dtype or W_up.dtype)
        W_down = W_down.to(device=device or W_down.device, dtype=dtype or W_down.dtype)

    stacked = {
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
        "fused": bool(attrs["fused"]),
        "weight_convention": "linear",  # always normalized
    }
    _STACK_CACHE[key] = stacked
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
