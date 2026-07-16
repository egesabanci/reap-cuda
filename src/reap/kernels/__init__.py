"""Model-agnostic MoE observation kernels (bmm / FREA / F2) with Triton optional.

Public entry: :func:`observe_moe_batch` selects backend via
:func:`reap.kernels.backend.select_observe_backend`.
"""

from __future__ import annotations

from reap.kernels.backend import (
    OBSERVE_BACKENDS,
    select_observe_backend,
)
from reap.kernels.observe import observe_moe_batch
from reap.kernels.weight_cache import free_cache, get_stacked_expert_weights

__all__ = [
    "OBSERVE_BACKENDS",
    "free_cache",
    "get_stacked_expert_weights",
    "observe_moe_batch",
    "select_observe_backend",
]
