"""F2 Triton scatter-reduce over routed pair outputs (optional).

Computes per-pair L2 norms and scatters into per-expert sum / weighted-sum /
weight-sum / max buffers. Accumulates **fp64** to match the PyTorch path and
the documented state schema. Welford online means stay in PyTorch
(``OnlineStatsTracker``) for exact parity with the existing path.
"""

from __future__ import annotations

import torch

from reap.kernels.triton_utils import (
    log_triton_fallback,
    next_power_of_2,
    prefer_triton_for,
    record_triton_ok,
    triton_runtime_available,
)

_COMPONENT = "f2_reduce"


def scatter_pair_stats(
    pair_out: torch.Tensor,
    pair_expert_idx: torch.Tensor,
    pair_router_w: torch.Tensor,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    """Reduce pair activations into per-expert batch statistics.

    Returns dict with keys:
      ``ean_sum``, ``weighted_ean_sum``, ``weighted_freq``, ``batch_max``
    all on ``pair_out.device``. Sums are ``float64``; ``batch_max`` is ``float32``.
    """
    device = pair_out.device
    if pair_out.numel() == 0:
        z64 = torch.zeros(num_experts, device=device, dtype=torch.float64)
        z32 = torch.zeros(num_experts, device=device, dtype=torch.float32)
        return {
            "ean_sum": z64,
            "weighted_ean_sum": z64.clone(),
            "weighted_freq": z64.clone(),
            "batch_max": z32,
        }

    if (
        triton_runtime_available()
        and prefer_triton_for(pair_out, min_numel=16)
        and pair_out.shape[-1] >= 16
    ):
        try:
            out = _scatter_triton(
                pair_out, pair_expert_idx, pair_router_w, num_experts
            )
            record_triton_ok(_COMPONENT)
            return out
        except Exception as exc:  # pragma: no cover
            log_triton_fallback(_COMPONENT, str(exc))

    return _scatter_pytorch(pair_out, pair_expert_idx, pair_router_w, num_experts)


def _scatter_pytorch(
    pair_out: torch.Tensor,
    pair_expert_idx: torch.Tensor,
    pair_router_w: torch.Tensor,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    device = pair_out.device
    ean_norm = torch.linalg.norm(pair_out.float(), dim=-1)
    w = pair_router_w.to(device=device, dtype=torch.float32)
    idx = pair_expert_idx.to(device)

    ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    weighted_ean = torch.zeros(num_experts, device=device, dtype=torch.float64)
    weighted_freq = torch.zeros(num_experts, device=device, dtype=torch.float64)
    ean_sum.index_add_(0, idx, ean_norm.to(torch.float64))
    weighted_ean.index_add_(0, idx, (ean_norm * w).to(torch.float64))
    weighted_freq.index_add_(0, idx, w.to(torch.float64))

    pair_raw_max = pair_out.float().amax(dim=-1)
    batch_max = torch.full(
        (num_experts,), float("-inf"), device=device, dtype=torch.float32
    )
    batch_max.scatter_reduce_(0, idx, pair_raw_max, reduce="amax", include_self=True)
    batch_max = torch.where(
        torch.isfinite(batch_max), batch_max, torch.zeros_like(batch_max)
    )
    return {
        "ean_sum": ean_sum,
        "weighted_ean_sum": weighted_ean,
        "weighted_freq": weighted_freq,
        "batch_max": batch_max,
    }


def _scatter_triton(
    pair_out: torch.Tensor,
    pair_expert_idx: torch.Tensor,
    pair_router_w: torch.Tensor,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    import triton
    import triton.language as tl

    n, h = pair_out.shape
    y = pair_out.contiguous()
    idx = pair_expert_idx.to(device=y.device, dtype=torch.int64).contiguous()
    w = pair_router_w.to(device=y.device, dtype=torch.float32).contiguous()

    # fp64 accumulators match PyTorch path and docs (ean_sum (E,) fp64).
    ean_sum_f = torch.zeros(num_experts, device=y.device, dtype=torch.float64)
    weighted_ean_f = torch.zeros(num_experts, device=y.device, dtype=torch.float64)
    weighted_freq_f = torch.zeros(num_experts, device=y.device, dtype=torch.float64)
    batch_max = torch.full(
        (num_experts,), float("-inf"), device=y.device, dtype=torch.float32
    )

    block_h = min(next_power_of_2(h), 128)

    @triton.jit
    def _reduce_kernel(
        Y_ptr,
        IDX_ptr,
        W_ptr,
        EAN_ptr,
        WEAN_ptr,
        WFREQ_ptr,
        MAX_ptr,
        N,
        H,
        stride_yn,
        stride_yh,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return
        acc = 0.0
        row_max = -float("inf")
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            mask = offs_h < H
            vals = tl.load(
                Y_ptr + pid * stride_yn + offs_h * stride_yh, mask=mask, other=0.0
            ).to(tl.float32)
            acc += tl.sum(vals * vals, axis=0)
            row_max = tl.maximum(
                row_max, tl.max(tl.where(mask, vals, -float("inf")), axis=0)
            )
        norm = tl.sqrt(acc)
        e_id = tl.load(IDX_ptr + pid)
        ww = tl.load(W_ptr + pid)
        # Cast to fp64 for atomic_add into fp64 buffers (cc ≥ 6.0).
        norm64 = norm.to(tl.float64)
        ww64 = ww.to(tl.float64)
        tl.atomic_add(EAN_ptr + e_id, norm64)
        tl.atomic_add(WEAN_ptr + e_id, norm64 * ww64)
        tl.atomic_add(WFREQ_ptr + e_id, ww64)
        tl.atomic_max(MAX_ptr + e_id, row_max)

    grid = (n,)
    _reduce_kernel[grid](
        y,
        idx,
        w,
        ean_sum_f,
        weighted_ean_f,
        weighted_freq_f,
        batch_max,
        n,
        h,
        y.stride(0),
        y.stride(1),
        BLOCK_H=block_h,
        num_warps=2,
    )
    batch_max = torch.where(
        torch.isfinite(batch_max), batch_max, torch.zeros_like(batch_max)
    )
    return {
        "ean_sum": ean_sum_f,
        "weighted_ean_sum": weighted_ean_f,
        "weighted_freq": weighted_freq_f,
        "batch_max": batch_max,
    }
