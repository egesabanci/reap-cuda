"""Backend selection for expert-activation observation."""

from __future__ import annotations

import os
from typing import Literal

import torch

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

ObserveBackend = Literal["auto", "loop", "bmm", "frea", "f2"]
OBSERVE_BACKENDS = ("auto", "loop", "bmm", "frea", "f2")


def triton_available() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()


def select_observe_backend(
    requested: str | None = None,
    *,
    prefer_triton: bool = True,
) -> str:
    """Resolve observation backend.

    * ``auto``: ``f2`` on CUDA+Triton, else ``bmm`` (GPU or MPS/CPU PyTorch).
    * ``frea`` / ``f2``: Triton when available, otherwise the pure-PyTorch
      grouped-bmm path (same math; still GPU-resident).
    * ``loop``: legacy full ``(E,T,H)`` path (parity oracle).
    * ``bmm``: grouped routed-only PyTorch (parity oracle for Triton).
    """
    req = (requested or os.environ.get("REAP_OBSERVE_BACKEND") or "auto").lower()
    if req not in OBSERVE_BACKENDS:
        raise ValueError(
            f"Unknown observe backend {req!r}; expected one of {OBSERVE_BACKENDS}"
        )
    if req == "auto":
        if prefer_triton and triton_available():
            return "f2"
        return "bmm"
    if req in ("frea", "f2") and not triton_available():
        # Still correct: PyTorch FREA/F2 fallbacks are GPU-resident grouped bmm.
        return req
    return req
