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


def _validate_inputs(
    pair_out: torch.Tensor,
    pair_expert_idx: torch.Tensor,
    pair_router_w: torch.Tensor,
    num_experts: int,
) -> None:
    """Validate the public F2 input contract before any Triton launch.

    Raises ``ValueError`` / ``TypeError`` with actionable messages for
    malformed structural, dtype, device, or index-domain inputs. This guard
    runs *before* the empty-result short-circuit and Triton eligibility so
    that invalid data never reaches unchecked atomic writes.
    """
    if not isinstance(num_experts, int) or isinstance(num_experts, bool):
        raise TypeError(f"num_experts must be int, got {type(num_experts).__name__}")
    if num_experts < 0:
        raise ValueError(f"num_experts must be non-negative, got {num_experts}")

    if not isinstance(pair_out, torch.Tensor):
        raise TypeError(f"pair_out must be a Tensor, got {type(pair_out).__name__}")
    if pair_out.ndim != 2:
        raise ValueError(f"pair_out must be 2-D, got {pair_out.ndim}-D")
    n = pair_out.shape[0]
    if pair_out.shape[1] == 0:
        raise ValueError("pair_out hidden dimension must be non-zero")
    if not torch.is_floating_point(pair_out):
        raise TypeError(f"pair_out must be floating-point, got dtype={pair_out.dtype}")

    if not isinstance(pair_expert_idx, torch.Tensor):
        raise TypeError(
            f"pair_expert_idx must be a Tensor, got {type(pair_expert_idx).__name__}"
        )
    if pair_expert_idx.ndim != 1:
        raise ValueError(f"pair_expert_idx must be 1-D, got {pair_expert_idx.ndim}-D")
    if pair_expert_idx.shape[0] != n:
        raise ValueError(
            f"pair_expert_idx length {pair_expert_idx.shape[0]} != pair_out rows {n}"
        )
    # Accept only index dtypes that PyTorch index_add_ / Triton safely support.
    # Rejects bool, uint8, int8, int16, float, complex, etc.
    if pair_expert_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"pair_expert_idx must be int32 or int64, got {pair_expert_idx.dtype}"
        )

    if not isinstance(pair_router_w, torch.Tensor):
        raise TypeError(
            f"pair_router_w must be a Tensor, got {type(pair_router_w).__name__}"
        )
    if pair_router_w.ndim != 1:
        raise ValueError(f"pair_router_w must be 1-D, got {pair_router_w.ndim}-D")
    if pair_router_w.shape[0] != n:
        raise ValueError(
            f"pair_router_w length {pair_router_w.shape[0]} != pair_out rows {n}"
        )

    if n > 0:
        if num_experts == 0:
            raise ValueError("num_experts must be > 0 when pair_out is non-empty")
        # Device consistency: all three tensors must agree.
        dev = pair_out.device
        if pair_expert_idx.device != dev:
            raise ValueError(
                f"pair_expert_idx device {pair_expert_idx.device} != pair_out device {dev}"
            )
        if pair_router_w.device != dev:
            raise ValueError(
                f"pair_router_w device {pair_router_w.device} != pair_out device {dev}"
        )


def _validate_index_domain(
    pair_expert_idx: torch.Tensor,
    num_experts: int,
) -> None:
    """Ensure all expert indices are within ``[0, num_experts)``.

    Uses a single combined validity reduction (one CUDA sync) on the valid
    path. Only on the exceptional invalid path do we compute min/max for a
    descriptive error message. This prevents arbitrary-address atomic
    writes in the Triton kernel and out-of-bounds ``index_add_`` on the
    PyTorch path.
    """
    if pair_expert_idx.numel() == 0:
        return
    # One .all().item() sync instead of two min/max .item() syncs.
    in_range = (pair_expert_idx >= 0) & (pair_expert_idx < num_experts)
    if not bool(in_range.all()):
        lo = int(pair_expert_idx.min().item())
        hi = int(pair_expert_idx.max().item())
        raise ValueError(
            f"pair_expert_idx values [{lo}, {hi}] out of range [0, {num_experts})"
        )


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
    _validate_inputs(pair_out, pair_expert_idx, pair_router_w, num_experts)

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

    # Validate index domain for ALL paths (not just Triton) to prevent
    # silent index_add_ out-of-bounds errors on the PyTorch path too.
    _validate_index_domain(pair_expert_idx, num_experts)

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

    from reap.kernels.triton_utils import (
        device_compute_capability,
        supports_fp64_atomics,
    )

    if not supports_fp64_atomics(device=pair_out.device):
        raise RuntimeError(
            f"Device compute capability {device_compute_capability(pair_out.device)} "
            "below minimum (6,0) for fp64 atomics"
        )

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
    # Occupancy: more warps when H is large; still 1D grid (safe, low risk).
    num_warps = 4 if h >= 512 else 2
    if n >= 2048:
        num_warps = max(num_warps, 4)

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
        num_warps=num_warps,
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
