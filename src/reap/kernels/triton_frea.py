"""FREA Triton kernel: per-expert SwiGLU MLP over routed token segments.

Layout-agnostic: expects Linear-convention weights from F4
(``W_gate/up: (E, I, H)``, ``W_down: (E, H, I)``).

Only supports SiLU gate activation (all current MoE targets). Other ``act_fn``
values fall back to the PyTorch grouped path. Block sizes auto-scale to the
active GPU's per-block shared memory so L4/T4 and large-datacenter GPUs both
work without hardcoding SKUs.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from reap.kernels.bmm import routed_expert_activations_grouped
from reap.kernels.router import RouterPairOutputs
from reap.kernels.triton_utils import (
    device_shared_memory_bytes,
    disable_component,
    is_component_disabled,
    log_triton_fallback,
    next_power_of_2,
    prefer_triton_for,
    record_triton_ok,
    triton_runtime_available,
)

# tl.dot on recent Triton wants M/N/K multiples of 16 for tensor cores.
_MIN_DOT = 16
_COMPONENT = "frea"


def _is_silu(act_fn: Callable) -> bool:
    return act_fn is F.silu or getattr(act_fn, "__name__", "") in {"silu", "swish"}


def estimate_frea_shared_bytes(block_n: int, block_h: int, block_i: int) -> int:
    """Estimate dynamic shared memory for the SwiGLU tile kernel.

    Matches observed Triton requirements: two live weight tiles (gate/up) as
    ``BLOCK_I × BLOCK_H`` fp32 plus the ``x`` tile ``BLOCK_N × BLOCK_H`` fp32
    (e.g. 128/128/16 → 139 264 B on Ada default 99 KiB → too large; 64/64/16 fits).
    """
    return int(2 * block_h * block_i * 4 + block_n * block_h * 4 + 4096)


def choose_frea_block_sizes(
    h: int,
    i_dim: int,
    *,
    device: torch.device | None = None,
    block_n: int = 16,
) -> tuple[int, int, int] | None:
    """Pick largest power-of-two tiles that fit device shared memory.

    Returns ``(block_h, block_i, block_n)`` or ``None`` if no feasible config.
    """
    smem = device_shared_memory_bytes(device)
    if smem is None:
        return None

    # Prefer larger tiles when the device allows (throughput).
    candidates_h = [
        b
        for b in (128, 64, 32, 16)
        if b >= _MIN_DOT and b <= max(_MIN_DOT, next_power_of_2(h))
    ]
    candidates_i = [
        b
        for b in (128, 64, 32, 16)
        if b >= _MIN_DOT and b <= max(_MIN_DOT, next_power_of_2(i_dim))
    ]
    if not candidates_h:
        candidates_h = [_MIN_DOT]
    if not candidates_i:
        candidates_i = [_MIN_DOT]

    safety = 2048
    for bh in candidates_h:
        for bi in candidates_i:
            need = estimate_frea_shared_bytes(block_n, bh, bi)
            if need + safety <= smem:
                return bh, bi, block_n
    return None


def _triton_frea_supported(
    flat_input: torch.Tensor,
    W_gate: torch.Tensor,
    *,
    act_fn: Callable,
) -> tuple[bool, str]:
    disabled = is_component_disabled(_COMPONENT)
    if disabled:
        return False, f"disabled: {disabled}"
    if not _is_silu(act_fn):
        return False, "non-SiLU activation"
    if not triton_runtime_available():
        return False, "triton runtime unavailable"
    if not prefer_triton_for(flat_input, min_numel=16):
        return False, "input not CUDA float16/bfloat16/float32 or too small"
    if not (W_gate.is_cuda and prefer_triton_for(W_gate)):
        return False, "weights not on CUDA"
    _e, i_dim, h = W_gate.shape
    if h < _MIN_DOT or i_dim < _MIN_DOT:
        return False, f"dims H={h}, I={i_dim} below Triton tl.dot floor {_MIN_DOT}"
    blocks = choose_frea_block_sizes(h, i_dim, device=flat_input.device)
    if blocks is None:
        limit = device_shared_memory_bytes(flat_input.device)
        return (
            False,
            f"no FREA tile config fits shared mem (device limit={limit}B, H={h}, I={i_dim})",
        )
    return True, ""


def frea_triton_activations(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    *,
    act_fn: Callable = F.silu,
) -> torch.Tensor:
    """Compute ``(n_pairs, H)`` routed expert outputs via Triton when possible."""
    ok, reason = _triton_frea_supported(flat_input, W_gate, act_fn=act_fn)
    if not ok:
        log_triton_fallback(_COMPONENT, reason)
        # Permanent disable when shared-mem infeasible (avoids N failed launches).
        if "shared mem" in reason or reason.startswith("disabled:"):
            if not is_component_disabled(_COMPONENT) and "shared mem" in reason:
                disable_component(_COMPONENT, reason)
        return routed_expert_activations_grouped(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )
    try:
        out = _frea_triton_impl(flat_input, router_pairs, W_gate, W_up, W_down)
        record_triton_ok(_COMPONENT)
        return out
    except Exception as exc:  # pragma: no cover - device/compile specific
        msg = str(exc)
        log_triton_fallback(_COMPONENT, msg)
        # Memoize hard resource failures (shared mem / out of resource).
        if "shared memory" in msg.lower() or "out of resource" in msg.lower():
            disable_component(_COMPONENT, msg)
        return routed_expert_activations_grouped(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )


def _frea_triton_impl(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    import triton
    import triton.language as tl

    pair_token_idx = router_pairs.pair_token_idx.contiguous()
    expert_offsets = router_pairs.expert_offsets.contiguous()
    e, i_dim, h = W_gate.shape
    if W_up.shape != (e, i_dim, h) or W_down.shape != (e, h, i_dim):
        raise ValueError(
            f"Weight shape mismatch: gate={tuple(W_gate.shape)} "
            f"up={tuple(W_up.shape)} down={tuple(W_down.shape)}"
        )

    n_pairs = int(pair_token_idx.numel())
    out = torch.empty(n_pairs, h, device=flat_input.device, dtype=flat_input.dtype)
    if n_pairs == 0:
        return out

    routed_x = flat_input.index_select(0, pair_token_idx).contiguous()
    Wg = W_gate.contiguous()
    Wu = W_up.contiguous()
    Wd = W_down.contiguous()

    blocks = choose_frea_block_sizes(h, i_dim, device=flat_input.device)
    if blocks is None:
        raise RuntimeError("FREA block selection failed after support check")
    block_h, block_i, block_n = blocks

    @triton.jit
    def _expert_swiglu_kernel(
        X_ptr,
        WG_ptr,
        WU_ptr,
        WD_ptr,
        Y_ptr,
        N,
        H,
        I,
        stride_xn,
        stride_xh,
        stride_wgi,
        stride_wgh,
        stride_wui,
        stride_wuh,
        stride_wdh,
        stride_wdi,
        stride_yn,
        stride_yh,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            mask_nh = mask_n[:, None] & mask_h[None, :]
            y_ptrs = Y_ptr + offs_n[:, None] * stride_yn + offs_h[None, :] * stride_yh
            tl.store(
                y_ptrs, tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32), mask=mask_nh
            )

        for i0 in range(0, I, BLOCK_I):
            offs_i = i0 + tl.arange(0, BLOCK_I)
            mask_i = offs_i < I

            g_acc = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float32)
            u_acc = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float32)

            for h0 in range(0, H, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                mask_ih = mask_i[:, None] & mask_h[None, :]

                x = tl.load(
                    X_ptr + offs_n[:, None] * stride_xn + offs_h[None, :] * stride_xh,
                    mask=mask_nh,
                    other=0.0,
                ).to(tl.float32)
                wg = tl.load(
                    WG_ptr + offs_i[:, None] * stride_wgi + offs_h[None, :] * stride_wgh,
                    mask=mask_ih,
                    other=0.0,
                ).to(tl.float32)
                wu = tl.load(
                    WU_ptr + offs_i[:, None] * stride_wui + offs_h[None, :] * stride_wuh,
                    mask=mask_ih,
                    other=0.0,
                ).to(tl.float32)
                g_acc += tl.dot(x, tl.trans(wg))
                u_acc += tl.dot(x, tl.trans(wu))

            act = g_acc * (1.0 / (1.0 + tl.exp(-g_acc))) * u_acc

            for h0 in range(0, H, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                mask_hi = mask_h[:, None] & mask_i[None, :]

                wd = tl.load(
                    WD_ptr + offs_h[:, None] * stride_wdh + offs_i[None, :] * stride_wdi,
                    mask=mask_hi,
                    other=0.0,
                ).to(tl.float32)
                y_delta = tl.dot(act, tl.trans(wd))
                y_ptrs = (
                    Y_ptr + offs_n[:, None] * stride_yn + offs_h[None, :] * stride_yh
                )
                y_prev = tl.load(y_ptrs, mask=mask_nh, other=0.0)
                tl.store(y_ptrs, y_prev + y_delta, mask=mask_nh)

    # num_stages=2 keeps pipeline staging modest vs default 3–4 on shared mem.
    num_warps = 4 if h <= 1024 else 8
    for expert_id in range(e):
        start = int(expert_offsets[expert_id].item())
        end = int(expert_offsets[expert_id + 1].item())
        n_e = end - start
        if n_e <= 0:
            continue

        x_e = routed_x[start:end]
        y_fp32 = torch.zeros(n_e, h, device=out.device, dtype=torch.float32)
        grid = (triton.cdiv(n_e, block_n),)
        _expert_swiglu_kernel[grid](
            x_e,
            Wg[expert_id],
            Wu[expert_id],
            Wd[expert_id],
            y_fp32,
            n_e,
            h,
            i_dim,
            x_e.stride(0),
            x_e.stride(1),
            Wg.stride(1),
            Wg.stride(2),
            Wu.stride(1),
            Wu.stride(2),
            Wd.stride(1),
            Wd.stride(2),
            y_fp32.stride(0),
            y_fp32.stride(1),
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            BLOCK_I=block_i,
            num_warps=num_warps,
            num_stages=2,
        )
        out[start:end].copy_(y_fp32.to(dtype=out.dtype))

    return out


def frea_activations_auto(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    *,
    act_fn: Callable = F.silu,
    use_triton: bool | None = None,
) -> torch.Tensor:
    """Public FREA entry used by ``frea.frea_activations``."""
    if use_triton is None:
        use_triton = triton_runtime_available() and flat_input.is_cuda
    if use_triton:
        return frea_triton_activations(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )
    return routed_expert_activations_grouped(
        flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
    )
