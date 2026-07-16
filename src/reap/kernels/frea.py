"""FREA: fused routed expert activation.

Dispatches to Triton SwiGLU kernels when CUDA+Triton are available and shapes
qualify; otherwise uses the pure-PyTorch grouped bmm path (parity oracle).

Backend policy (``auto`` / ``triton`` / ``pytorch``) is controlled via
``frea_backend`` / ``REAP_FREA_BACKEND`` / :func:`set_frea_backend`.
"""

from __future__ import annotations

from typing import Callable

import torch.nn.functional as F

from reap.kernels.bmm import routed_expert_activations_grouped
from reap.kernels.router import RouterPairOutputs
from reap.kernels.triton_frea import frea_activations_auto, get_frea_backend
from reap.kernels.triton_utils import triton_runtime_available


def frea_activations(
    flat_input,
    router_pairs: RouterPairOutputs,
    W_gate,
    W_up,
    W_down,
    *,
    act_fn: Callable = F.silu,
    use_triton: bool | None = None,
    frea_backend: str | None = None,
):
    """Return routed pair outputs ``(n_pairs, H)``."""
    backend = frea_backend or get_frea_backend()
    if use_triton is None:
        use_triton = (
            triton_runtime_available()
            and flat_input.is_cuda
            and backend != "pytorch"
        )
    if use_triton or backend == "auto":
        return frea_activations_auto(
            flat_input,
            router_pairs,
            W_gate,
            W_up,
            W_down,
            act_fn=act_fn,
            use_triton=True if backend != "pytorch" else False,
            frea_backend=backend,
        )
    return routed_expert_activations_grouped(
        flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
    )
