"""Safe Triton capability detection, usage accounting, and launch helpers.

Importing this module never requires ``triton`` unless the package is present;
kernel modules import triton lazily inside functions that only run on CUDA.

Hardware checks (shared memory, dtype, profitability) are device-agnostic:
they query the active CUDA device properties rather than hardcoding SKUs.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import lru_cache
from typing import Any

import torch

logger = logging.getLogger(__name__)

_TRITON_IMPORT_ERROR: str | None = None
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _HAS_TRITON_PKG = True
except Exception as exc:  # pragma: no cover - environment dependent
    _HAS_TRITON_PKG = False
    _TRITON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Per-component launch accounting (ok / fallback). Reset per observe run.
_USAGE: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "fallback": 0})
_FALLBACK_WARNED: set[str] = set()
# Memoized permanent disable reasons for a component (process-local).
_DISABLED: dict[str, str] = {}


def triton_package_available() -> bool:
    return _HAS_TRITON_PKG


def triton_import_error() -> str | None:
    return _TRITON_IMPORT_ERROR


@lru_cache(maxsize=1)
def triton_runtime_available() -> bool:
    """True when Triton can actually launch kernels on this process."""
    if not _HAS_TRITON_PKG:
        return False
    if not torch.cuda.is_available():
        return False
    if os.environ.get("REAP_DISABLE_TRITON", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
    }:
        return False
    try:
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception as exc:  # pragma: no cover
        logger.debug("Triton runtime unavailable: %s", exc)
        return False


def _cuda_device_index(device: torch.device | int | None = None) -> int:
    if device is None:
        return 0
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    return int(device)


def device_shared_memory_bytes(
    device: torch.device | int | None = None,
    *,
    prefer_optin: bool = False,
) -> int | None:
    """Return per-block dynamic shared-memory limit (bytes) for *device*.

    When *prefer_optin* is True and the device exposes a larger
    ``shared_memory_per_block_optin``, that value is returned so FREA can try
    bigger tiles. Limits are **device-reported**, not hardcoded — examples:

    * L4 (AD104): default ~48 KiB, opt-in ~99 KiB (128×128 FREA still too big)
    * A100/L40S-class: opt-in often ~164 KiB (128×128 can fit)

    Triton typically opts into dynamic SM via the launcher when the kernel
    needs more than the static default. On launch failure, callers should
    fall back to the default limit (see FREA safe-retry).
    """
    if not torch.cuda.is_available():
        return None
    try:
        idx = _cuda_device_index(device)
        props = torch.cuda.get_device_properties(idx)
        base = int(props.shared_memory_per_block)
        optin = getattr(props, "shared_memory_per_block_optin", None)
        if prefer_optin and optin is not None and int(optin) > base:
            return int(optin)
        return base
    except Exception as exc:  # pragma: no cover
        logger.debug("shared_memory probe failed: %s", exc)
        return None


def shared_mem_feasible(
    required_bytes: int,
    *,
    device: torch.device | int | None = None,
    safety_margin: int = 2048,
    prefer_optin: bool = False,
) -> tuple[bool, str]:
    """Whether *required_bytes* fits the device's per-block shared memory."""
    limit = device_shared_memory_bytes(device, prefer_optin=prefer_optin)
    if limit is None:
        return False, "no CUDA shared-memory limit (no device)"
    if required_bytes + safety_margin > limit:
        return (
            False,
            f"shared mem required~{required_bytes}B + margin > device limit {limit}B",
        )
    return True, ""


def prefer_triton_for(
    tensor: torch.Tensor,
    *,
    min_numel: int | None = None,
) -> bool:
    """Whether *tensor* is eligible for a Triton launch (dtype/device/size).

    Does **not** check shared-memory feasibility for a specific kernel — each
    kernel's ``_supported`` gate must call :func:`shared_mem_feasible` with its
    estimated requirement. Optional *min_numel* avoids launch overhead on tiny
    shapes (profitability).
    """
    if not triton_runtime_available():
        return False
    if not tensor.is_cuda:
        return False
    if tensor.numel() <= 0:
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return min_numel is None or tensor.numel() >= min_numel


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def is_component_disabled(component: str) -> str | None:
    """Return disable reason if *component* was memoized as permanently failed."""
    return _DISABLED.get(component)


def disable_component(component: str, reason: str) -> None:
    """Memoize a permanent failure so we stop re-attempting the same launch."""
    if component not in _DISABLED:
        _DISABLED[component] = reason
        logger.warning(
            "Triton %s disabled for this process: %s", component, reason
        )


def record_triton_ok(component: str) -> None:
    _USAGE[component]["ok"] += 1


def log_triton_fallback(component: str, reason: str) -> None:
    """Account a fallback; WARN once per component, DEBUG thereafter."""
    _USAGE[component]["fallback"] += 1
    if component not in _FALLBACK_WARNED:
        _FALLBACK_WARNED.add(component)
        logger.warning(
            "Triton %s fallback → PyTorch (%s); further fallbacks at DEBUG",
            component,
            reason,
        )
    logger.debug("Triton %s fallback → PyTorch (%s)", component, reason)


def reset_triton_usage() -> None:
    """Clear ok/fallback counters (call at start of an observe run)."""
    _USAGE.clear()
    _FALLBACK_WARNED.clear()
    # Keep _DISABLED: once a kernel can't launch on this device, stay off.


def clear_triton_disable_memo() -> None:
    """Clear permanent disable memo (tests only)."""
    _DISABLED.clear()


def triton_usage_snapshot() -> dict[str, dict[str, int]]:
    return {k: dict(v) for k, v in _USAGE.items()}


def format_triton_usage_summary() -> str:
    if not _USAGE and not _DISABLED:
        return "no Triton kernel attempts this run"
    parts: list[str] = []
    keys = sorted(set(_USAGE) | set(_DISABLED))
    for name in keys:
        stats = _USAGE.get(name, {"ok": 0, "fallback": 0})
        ok = stats.get("ok", 0)
        fb = stats.get("fallback", 0)
        disabled = _DISABLED.get(name)
        if disabled:
            parts.append(f"{name}: {ok} Triton / {fb} PyTorch (disabled: {disabled})")
        else:
            parts.append(f"{name}: {ok} Triton / {fb} PyTorch")
    return "; ".join(parts)


def log_triton_usage_summary() -> None:
    """Emit INFO summary so the f2 performance contract is verifiable."""
    summary = format_triton_usage_summary()
    logger.info("Triton usage summary: %s", summary)
    # Explicit warn if a heavy kernel never succeeded but fell back.
    for name, stats in _USAGE.items():
        if stats.get("fallback", 0) > 0 and stats.get("ok", 0) == 0:
            logger.warning(
                "Triton %s never launched successfully (%d fallbacks); "
                "backend label may overstate Triton coverage",
                name,
                stats["fallback"],
            )


def get_triton() -> Any:
    """Return the ``triton`` module or raise ImportError."""
    if not _HAS_TRITON_PKG:
        raise ImportError(
            f"triton is not available ({_TRITON_IMPORT_ERROR}). "
            "Install with: pip install -e '.[cuda]'"
        )
    import triton

    return triton


def get_triton_language() -> Any:
    if not _HAS_TRITON_PKG:
        raise ImportError("triton is not available")
    import triton.language as tl

    return tl
