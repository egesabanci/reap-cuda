"""Backend selection for expert-activation observation."""

from __future__ import annotations

import os
from typing import Literal

from reap.kernels.triton_utils import (
    triton_import_error,
    triton_package_available,
    triton_runtime_available,
)

ObserveBackend = Literal["auto", "loop", "bmm", "frea", "f2"]
OBSERVE_BACKENDS = ("auto", "loop", "bmm", "frea", "f2")


def triton_available() -> bool:
    """True when custom Triton kernels can run (CUDA + triton package)."""
    return triton_runtime_available()


def triton_status() -> dict[str, object]:
    """Diagnostics for CLI / logs."""
    return {
        "package": triton_package_available(),
        "runtime": triton_runtime_available(),
        "import_error": triton_import_error(),
        "disabled_env": os.environ.get("REAP_DISABLE_TRITON", ""),
    }


def select_observe_backend(
    requested: str | None = None,
    *,
    prefer_triton: bool = True,
) -> str:
    """Resolve observation backend.

    * ``auto``: ``f2`` when Triton runtime is up, else ``bmm``.
    * ``frea`` / ``f2``: keep the name so dispatch can try Triton, with
      automatic PyTorch fallback inside the kernel wrappers.
    * ``loop``: legacy path (parity oracle).
    * ``bmm``: pure-PyTorch grouped routed matmul (parity oracle for Triton).
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
    return req
