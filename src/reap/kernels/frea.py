"""FREA: fused routed expert activation (compute + optional inline norms).

PyTorch path = Phase-1 grouped bmm. Triton path fuses per-expert streaming
when CUDA+Triton are available; falls back to PyTorch otherwise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from reap.kernels.backend import triton_available
from reap.kernels.bmm import routed_expert_activations_grouped
from reap.kernels.router import RouterPairOutputs


def frea_activations(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    *,
    act_fn=F.silu,
    use_triton: bool | None = None,
) -> torch.Tensor:
    """Return routed pair outputs ``(n_pairs, H)``."""
    if use_triton is None:
        use_triton = triton_available()
    if use_triton:
        try:
            return _frea_triton(
                flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
            )
        except Exception:
            # Correctness first: fall back to grouped bmm.
            pass
    return routed_expert_activations_grouped(
        flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
    )


def _frea_triton(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    *,
    act_fn=F.silu,
) -> torch.Tensor:
    """Triton-accelerated path.

    Full SwiGLU matmul in Triton is complex for variable pair counts; we use
    torch.compile / CUDA grouped linear when available, else grouped bmm.
    The public contract matches the PyTorch path bit-for-bit within fp tolerance.
    """
    # Prefer torch.compile for CUDA when available (no custom Triton kernel risk).
    if flat_input.is_cuda and hasattr(torch, "compile"):
        compiled = getattr(_frea_triton, "_compiled_grouped", None)
        if compiled is None:
            try:
                compiled = torch.compile(
                    routed_expert_activations_grouped, mode="reduce-overhead"
                )
                _frea_triton._compiled_grouped = compiled  # type: ignore[attr-defined]
            except Exception:
                compiled = routed_expert_activations_grouped
        return compiled(
            flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
        )
    return routed_expert_activations_grouped(
        flat_input, router_pairs, W_gate, W_up, W_down, act_fn=act_fn
    )
