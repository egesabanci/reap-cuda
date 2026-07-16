"""FREA: fused routed expert activation.

Dispatches to Triton SwiGLU kernels when CUDA+Triton are available and shapes
qualify; otherwise uses the pure-PyTorch grouped bmm path (parity oracle).
"""

from __future__ import annotations

from typing import Callable

import torch.nn.functional as F

from reap.kernels.bmm import routed_expert_activations_grouped
from reap.kernels.router import RouterPairOutputs
from reap.kernels.triton_frea import frea_activations_auto
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
):
    """Return routed pair outputs ``(n_pairs, H)``."""
    if use_triton is None:
        use_triton = triton_runtime_available() and flat_input.is_cuda
    if use_triton:
        return frea_activations_auto(
            flat_input,
            router_pairs,
            W_gate,
            W_up,
            W_down,
            act_fn=act_fn,
            use_triton=True,
        )
    return routed_expert_activations_grouped(
        flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
    )
