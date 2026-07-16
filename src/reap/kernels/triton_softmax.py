"""F5 Triton fused online-softmax over expert dim (optional accelerator).

Top-k and pair sorting stay in PyTorch for correctness across all ``top_k``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from reap.kernels.triton_utils import (
    log_triton_fallback,
    next_power_of_2,
    prefer_triton_for,
    record_triton_ok,
    triton_runtime_available,
)

_COMPONENT = "softmax"


def softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax in fp32, matching ``F.softmax(..., dtype=torch.float32)``.

    Uses a Triton kernel on CUDA when available; otherwise pure PyTorch.
    """
    if logits.ndim != 2:
        raise ValueError(f"softmax_rows expects 2D logits, got {tuple(logits.shape)}")
    if logits.numel() == 0:
        return logits.to(dtype=torch.float32)

    if prefer_triton_for(logits, min_numel=16) and triton_runtime_available():
        try:
            out = _softmax_triton(logits)
            record_triton_ok(_COMPONENT)
            return out
        except Exception as exc:  # pragma: no cover - device specific
            log_triton_fallback(_COMPONENT, str(exc))
    return F.softmax(logits, dim=-1, dtype=torch.float32)


def _softmax_triton(logits: torch.Tensor) -> torch.Tensor:
    import triton
    import triton.language as tl

    t, e = logits.shape
    # Work in fp32 for stability (matches pruning_metrics).
    x = logits.contiguous()
    out = torch.empty(t, e, device=x.device, dtype=torch.float32)

    # Cap block size for large expert counts (256/512 experts).
    block_e = min(next_power_of_2(e), 1024)
    if block_e < e:
        # Fall back when E does not fit one block (rare for current MoEs ≤256).
        # Multi-tile online softmax is possible but not required for targets.
        log_triton_fallback(_COMPONENT, f"E={e} > BLOCK_E={block_e}")
        return F.softmax(logits, dim=-1, dtype=torch.float32)

    @triton.jit
    def _kernel(
        X_ptr,
        Y_ptr,
        T,
        E,
        stride_xt,
        stride_xe,
        stride_yt,
        stride_ye,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= T:
            return
        offs_e = tl.arange(0, BLOCK_E)
        mask = offs_e < E
        x_ptrs = X_ptr + row * stride_xt + offs_e * stride_xe
        x = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(x, axis=0)
        x = x - row_max
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        y = num / den
        y_ptrs = Y_ptr + row * stride_yt + offs_e * stride_ye
        tl.store(y_ptrs, y, mask=mask)

    grid = (t,)
    _kernel[grid](
        x,
        out,
        t,
        e,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_E=block_e,
        num_warps=4 if e <= 128 else 8,
    )
    return out
