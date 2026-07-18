"""FREA Triton kernel: per-expert SwiGLU MLP over routed token segments.

Layout-agnostic: expects Linear-convention weights from F4
(``W_gate/up: (E, I, H)``, ``W_down: (E, H, I)``).

Throughput vs memory on shared-mem-bound GPUs
---------------------------------------------
* **auto** (default): one-shot empirical probe (Triton vs cuBLAS PyTorch) per
  ``(device, H, I)``, memoize the winner for the rest of the process.
* **triton**: force Triton when tiles fit (uses device SM default + opt-in;
  on L4 that is typically up to 128×64 for large H/I — **not** 128×128).
* **pytorch**: force grouped ``F.linear`` path (max throughput on many L4/T4s).

Set via :func:`set_frea_backend`, env ``REAP_FREA_BACKEND``, or CLI
``--frea-backend``. Ops guide: ``docs/frea-throughput.md``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Literal

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

logger = logging.getLogger(__name__)

# tl.dot on recent Triton wants M/N/K multiples of 16 for tensor cores.
_MIN_DOT = 16
_COMPONENT = "frea"
# Prefer tiles at least this large for static profitability (when probe is off).
_PROFITABLE_BLOCK_FLOOR = 128

FreaBackend = Literal["auto", "triton", "pytorch"]
FREA_BACKENDS: tuple[str, ...] = ("auto", "triton", "pytorch")

# Process-level override (CLI / set_frea_backend). Env wins if set.
_FREA_BACKEND: str = "auto"
# Memo: (device_type, device_index, dtype, h, i_dim) -> "triton" | "pytorch"
_PROBE_CHOICE: dict[tuple[str, int | None, str, int, int], str] = {}
# Prefer opt-in SM budget per device after a successful large-tile launch; clear on failure.
# Keyed by str(device) so different GPUs track opt-in independently.
_USE_SMEM_OPTIN: dict[str, bool | None] = {}


def set_frea_backend(mode: str) -> str:
    """Set process-wide FREA backend: ``auto`` | ``triton`` | ``pytorch``."""
    global _FREA_BACKEND
    m = (mode or "auto").lower().strip()
    if m not in FREA_BACKENDS:
        raise ValueError(f"Unknown frea backend {mode!r}; expected one of {FREA_BACKENDS}")
    _FREA_BACKEND = m
    return m


def get_frea_backend() -> str:
    env = os.environ.get("REAP_FREA_BACKEND", "").strip().lower()
    if env in FREA_BACKENDS:
        return env
    return _FREA_BACKEND


def reset_frea_probe_cache() -> None:
    """Clear profitability probe memo and SM opt-in state (tests)."""
    _PROBE_CHOICE.clear()
    _USE_SMEM_OPTIN.clear()


def _is_silu(act_fn: Callable) -> bool:
    return act_fn is F.silu or getattr(act_fn, "__name__", "") in {"silu", "swish"}


def estimate_frea_shared_bytes(block_n: int, block_h: int, block_i: int) -> int:
    """Estimate dynamic shared memory for the SwiGLU tile kernel."""
    return int(2 * block_h * block_i * 4 + block_n * block_h * 4 + 4096)


def _smem_budget(device: torch.device | None, *, prefer_optin: bool) -> int | None:
    return device_shared_memory_bytes(device, prefer_optin=prefer_optin)


def _device_key(device: torch.device | None) -> str:
    """Stable string key for per-device state maps."""
    if device is None:
        return "cpu"
    return str(device)


def _get_smem_optin(device: torch.device | None) -> bool | None:
    """Get the per-device SM opt-in state (None = not yet determined)."""
    return _USE_SMEM_OPTIN.get(_device_key(device))


def _set_smem_optin(device: torch.device | None, value: bool | None) -> None:
    """Set the per-device SM opt-in state."""
    _USE_SMEM_OPTIN[_device_key(device)] = value


def choose_frea_block_sizes(
    h: int,
    i_dim: int,
    *,
    device: torch.device | None = None,
    block_n: int = 16,
    prefer_optin: bool | None = None,
) -> tuple[int, int, int] | None:
    """Pick largest power-of-two tiles that fit device shared memory.

    Tries the opt-in SM budget first when larger than the default (e.g. L4
    99 KiB opt-in vs 48 KiB default; A100/L40S often ~164 KiB opt-in). Falls
    back to the default per-block limit. 128×128 (~140 KiB) only fits when the
    device reports enough opt-in budget — **not** on L4/T4-class.
    """
    if prefer_optin is None:
        prefer_optin = _get_smem_optin(device) is not False

    for use_optin in ((True, False) if prefer_optin else (False,)):
        smem = _smem_budget(device, prefer_optin=use_optin)
        if smem is None:
            continue
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


def _tile_profitable(blocks: tuple[int, int, int] | None) -> bool:
    """Static heuristic: small auto-tiles are usually slower than cuBLAS on L4."""
    if blocks is None:
        return False
    bh, bi, _bn = blocks
    return bh >= _PROFITABLE_BLOCK_FLOOR or bi >= _PROFITABLE_BLOCK_FLOOR


def _triton_frea_supported(
    flat_input: torch.Tensor,
    W_gate: torch.Tensor,
    *,
    act_fn: Callable,
    require_profitable_tiles: bool = False,
) -> tuple[bool, str]:
    dev_scope = _device_key(flat_input.device)
    disabled = is_component_disabled(_COMPONENT, scope=dev_scope)
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
        limit = device_shared_memory_bytes(flat_input.device, prefer_optin=True)
        return (
            False,
            f"no FREA tile config fits shared mem (device limit={limit}B, H={h}, I={i_dim})",
        )
    if require_profitable_tiles and not _tile_profitable(blocks):
        return (
            False,
            f"auto-tiled blocks {blocks[0]}x{blocks[1]} below profitability floor "
            f"{_PROFITABLE_BLOCK_FLOOR} (prefer cuBLAS PyTorch)",
        )
    return True, ""


def _probe_key(flat_input: torch.Tensor, W_gate: torch.Tensor) -> tuple[str, int | None, str, int, int]:
    dev = flat_input.device
    idx = dev.index if dev.type == "cuda" else None
    _e, i_dim, h = W_gate.shape
    return (dev.type, idx, str(flat_input.dtype), int(h), int(i_dim))


def _run_probe(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    *,
    act_fn: Callable,
) -> str:
    """Time Triton vs PyTorch once; return ``triton`` or ``pytorch``.

    Does **not** memoize on tiny ``n_pairs`` (would stick the wrong choice for
    the whole run when the first batch is sparse). Includes a warm-up Triton
    launch so JIT compile time does not bias the probe toward PyTorch.
    """
    key = _probe_key(flat_input, W_gate)
    if key in _PROBE_CHOICE:
        return _PROBE_CHOICE[key]

    n_pairs = int(router_pairs.pair_token_idx.numel())
    # Too few pairs: launch overhead dominates — skip probe *without* memoizing
    # so a later denser batch can still probe.
    if n_pairs < 16:
        logger.info(
            "FREA profitability probe deferred: n_pairs=%d < 16 (not memoized)",
            n_pairs,
        )
        return "pytorch"

    ok, reason = _triton_frea_supported(
        flat_input, W_gate, act_fn=act_fn, require_profitable_tiles=False
    )
    # If Triton is structurally unsupported (CPU, disabled, non-SiLU, etc.),
    # return PyTorch immediately — no timing, no CUDA sync, no memoization.
    # The support check is cheap and deterministic per scope, so re-checking on
    # later calls is harmless.
    if not ok:
        logger.info(
            "FREA profitability probe: triton unsupported (%s) -> pytorch",
            reason,
        )
        return "pytorch"

    dev = flat_input.device
    t_tr = float("inf")
    if ok:
        try:
            # Warm-up (JIT + first-touch); not timed. Stream-ordered before
            # the timed launch, so no explicit synchronize is needed when CUDA
            # events will provide timing.
            _ = _frea_triton_impl(flat_input, router_pairs, W_gate, W_up, W_down)

            # Use CUDA events on the launch device for timing instead of
            # global torch.cuda.synchronize() so unrelated streams are not
            # disrupted. Fall back to wall-clock + device-scoped sync if events fail.
            use_events = dev.type == "cuda"
            if use_events:
                try:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record(stream=torch.cuda.current_stream(dev))
                    _ = _frea_triton_impl(
                        flat_input, router_pairs, W_gate, W_up, W_down
                    )
                    end_evt.record(stream=torch.cuda.current_stream(dev))
                    end_evt.synchronize()
                    t_tr = start_evt.elapsed_time(end_evt) / 1000.0  # ms -> s
                except Exception:
                    use_events = False
            if not use_events:
                # Wall-clock fallback: narrow device-scoped sync only.
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                _ = _frea_triton_impl(
                    flat_input, router_pairs, W_gate, W_up, W_down
                )
                torch.cuda.synchronize(dev)
                t_tr = time.perf_counter() - t0
        except Exception as exc:  # pragma: no cover
            logger.debug("FREA probe Triton failed: %s", exc)
            t_tr = float("inf")

    # Warm-up PyTorch path too (allocator / autotune noise). Stream-ordered
    # before the timed launch, so no explicit synchronize needed when events
    # will provide timing.
    _ = routed_expert_activations_grouped(
        flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
    )

    # Time PyTorch path with CUDA events when possible.
    use_events_py = dev.type == "cuda"
    if use_events_py:
        try:
            start_py = torch.cuda.Event(enable_timing=True)
            end_py = torch.cuda.Event(enable_timing=True)
            start_py.record(stream=torch.cuda.current_stream(dev))
            _ = routed_expert_activations_grouped(
                flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
            )
            end_py.record(stream=torch.cuda.current_stream(dev))
            end_py.synchronize()
            t_py = start_py.elapsed_time(end_py) / 1000.0  # ms -> s
        except Exception:
            use_events_py = False
    if not use_events_py:
        # Wall-clock fallback: narrow device-scoped sync only.
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        _ = routed_expert_activations_grouped(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )
        torch.cuda.synchronize(dev)
        t_py = time.perf_counter() - t0

    # Prefer Triton on ties / near-ties (memory win); require clear win for pytorch.
    if t_tr <= t_py * 1.05:
        choice = "triton" if t_tr < float("inf") else "pytorch"
    else:
        choice = "pytorch"

    _PROBE_CHOICE[key] = choice
    logger.info(
        "FREA profitability probe: triton=%.4fs pytorch=%.4fs -> %s (reason=%s)",
        t_tr if t_tr < float("inf") else -1.0,
        t_py,
        choice,
        reason if not ok else "ok",
    )
    return choice


def frea_triton_activations(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    *,
    act_fn: Callable = F.silu,
    backend: str | None = None,
) -> torch.Tensor:
    """Compute ``(n_pairs, H)`` via Triton and/or PyTorch per *backend* policy."""
    mode = (backend or get_frea_backend()).lower().strip()
    if mode not in FREA_BACKENDS:
        mode = "auto"

    if mode == "pytorch":
        log_triton_fallback(_COMPONENT, "frea_backend=pytorch")
        return routed_expert_activations_grouped(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )

    if mode == "auto":
        # Empirical probe is the default profitability signal (#25).
        # Skip probe when REAP_FREA_PROBE=0 → static tile floor (#26).
        probe_env = os.environ.get("REAP_FREA_PROBE", "1").strip().lower()
        if probe_env in {"0", "false", "no", "off"}:
            ok, reason = _triton_frea_supported(
                flat_input, W_gate, act_fn=act_fn, require_profitable_tiles=True
            )
            if not ok:
                log_triton_fallback(_COMPONENT, reason)
                return routed_expert_activations_grouped(
                    flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
                )
        else:
            choice = _run_probe(
                flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
            )
            if choice == "pytorch":
                log_triton_fallback(_COMPONENT, "probe chose pytorch")
                return routed_expert_activations_grouped(
                    flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
                )
            # else triton — fall through

    # mode == "triton" or auto after probe chose triton
    ok, reason = _triton_frea_supported(
        flat_input, W_gate, act_fn=act_fn, require_profitable_tiles=False
    )
    if not ok:
        log_triton_fallback(_COMPONENT, reason)
        dev_scope = _device_key(flat_input.device)
        if "shared mem" in reason and not is_component_disabled(_COMPONENT, scope=dev_scope):
            disable_component(_COMPONENT, reason, scope=dev_scope)
        return routed_expert_activations_grouped(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )
    try:
        out = _frea_triton_impl(flat_input, router_pairs, W_gate, W_up, W_down)
        record_triton_ok(_COMPONENT)
        return out
    except Exception as exc:  # pragma: no cover
        msg = str(exc)
        log_triton_fallback(_COMPONENT, msg)
        dev_scope = _device_key(flat_input.device)
        if "shared memory" in msg.lower() or "out of resource" in msg.lower():
            # Disable opt-in path for this device and retry once with default SM tiles.
            if _get_smem_optin(flat_input.device) is not False:
                _set_smem_optin(flat_input.device, False)
                try:
                    out = _frea_triton_impl(
                        flat_input, router_pairs, W_gate, W_up, W_down
                    )
                    record_triton_ok(_COMPONENT)
                    return out
                except Exception as exc2:
                    msg = str(exc2)
                    log_triton_fallback(_COMPONENT, msg)
            disable_component(_COMPONENT, msg, scope=dev_scope)
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
    # Prefer larger token tiles when H/I tiles are small (#29 occupancy), but
    # re-check shared-mem with the larger BLOCK_N (estimate depends on it).
    if block_h <= 64 and block_i <= 64 and n_pairs >= 32:
        candidate_n = 32
        smem = _smem_budget(
            flat_input.device, prefer_optin=_get_smem_optin(flat_input.device) is not False
        )
        if smem is not None:
            need = estimate_frea_shared_bytes(candidate_n, block_h, block_i)
            if need + 2048 <= smem:
                block_n = candidate_n

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

            act = tl.sigmoid(g_acc) * g_acc * u_acc

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

    num_warps = 4 if h <= 1024 else 8
    if block_h <= 64:
        num_warps = max(num_warps, 8)

    # Bulk-transfer CSR offsets to host once (avoids O(E) scalar .item() syncs).
    offsets_host = expert_offsets.detach().to("cpu", dtype=torch.long).tolist()

    for expert_id in range(e):
        start = offsets_host[expert_id]
        end = offsets_host[expert_id + 1]
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

    # Successful launch with large tiles implies opt-in path is usable for this device.
    if block_h >= 128 and block_i >= 128 and _get_smem_optin(flat_input.device) is None:
        _set_smem_optin(flat_input.device, True)
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
    frea_backend: str | None = None,
) -> torch.Tensor:
    """Public FREA entry used by ``frea.frea_activations``."""
    if use_triton is False:
        return routed_expert_activations_grouped(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )
    if use_triton is None:
        use_triton = triton_runtime_available() and flat_input.is_cuda
    if use_triton:
        return frea_triton_activations(
            flat_input,
            router_pairs,
            W_gate,
            W_up,
            W_down,
            act_fn=act_fn,
            backend=frea_backend,
        )
    return routed_expert_activations_grouped(
        flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
    )
