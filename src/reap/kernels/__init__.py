"""Model-agnostic MoE observation kernels (bmm / FREA / F2) with Triton.

Public entry: :func:`observe_moe_batch` selects backend via
:func:`reap.kernels.backend.select_observe_backend`.

Triton kernels (when CUDA + ``triton`` are available):

* F5 softmax — ``triton_softmax.softmax_rows``
* FREA SwiGLU — ``triton_frea.frea_triton_activations``
* F2 scatter — ``triton_reduce.scatter_pair_stats``

Every path falls back to pure PyTorch automatically. Set
``REAP_DISABLE_TRITON=1`` to force PyTorch.
"""

from __future__ import annotations

from reap.kernels.backend import (
    OBSERVE_BACKENDS,
    select_observe_backend,
    triton_available,
    triton_status,
)
from reap.kernels.observe import observe_moe_batch
from reap.kernels.weight_cache import free_cache, get_stacked_expert_weights

__all__ = [
    "OBSERVE_BACKENDS",
    "free_cache",
    "get_stacked_expert_weights",
    "observe_moe_batch",
    "select_observe_backend",
    "triton_available",
    "triton_status",
]
