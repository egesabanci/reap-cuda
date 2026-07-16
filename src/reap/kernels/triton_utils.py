"""Safe Triton capability detection and launch helpers.

Importing this module never imports ``triton`` unless it is available; kernel
modules import triton lazily inside functions that only run on CUDA.
"""

from __future__ import annotations

import logging
import os
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
    if os.environ.get("REAP_DISABLE_TRITON", "").strip() in {"1", "true", "TRUE", "yes"}:
        return False
    try:
        # Touch a device to fail early on broken driver installs.
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception as exc:  # pragma: no cover
        logger.debug("Triton runtime unavailable: %s", exc)
        return False


def prefer_triton_for(tensor: torch.Tensor) -> bool:
    """Whether *tensor* is eligible for a Triton launch."""
    return (
        triton_runtime_available()
        and tensor.is_cuda
        and tensor.numel() > 0
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def log_triton_fallback(component: str, reason: str) -> None:
    logger.debug("Triton %s fallback → PyTorch (%s)", component, reason)


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
