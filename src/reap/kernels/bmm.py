"""Phase 1: grouped routed-only expert activations (parity oracle).

Processes only ``(token, expert)`` pairs that top-k selected. Peak memory is
``O(max_pairs_per_expert * H)`` — never materializes ``(E, T, H)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from reap.kernels.router import RouterPairOutputs
from reap.kernels.weight_cache import apply_swiglu


def routed_expert_activations_grouped(
    flat_input: torch.Tensor,
    router_pairs: RouterPairOutputs,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    *,
    act_fn=F.silu,
) -> torch.Tensor:
    """Compute expert outputs for sorted routed pairs.

    Args:
        flat_input: ``(T, H)`` — original token axis (before padding filter).
            When F5 filtered padding, ``pair_token_idx`` still indexes the
            **unfiltered** token positions if mask was applied only to pairs;
            our F5 filters selected_experts and reindexes pairs to filtered
            tokens, so pair_token_idx is into the filtered sequence...

    Actually F5 with mask rewrites selected_experts to filtered tokens and
    builds pair_token_idx over the filtered length. Callers must pass
    ``flat_input`` already filtered OR pair indices into original flat_input.

    Contract used by :func:`observe_moe_batch`:
        * ``flat_input`` is the full ``(T, H)`` tensor.
        * When a mask is present, F5 builds pairs only for valid tokens and
          ``pair_token_idx`` indexes into the **original** ``flat_input``.

    Returns:
        ``out`` of shape ``(n_pairs, H)`` aligned with sorted pair arrays.
    """
    device = flat_input.device
    pair_token_idx = router_pairs.pair_token_idx
    pair_expert_idx = router_pairs.pair_expert_idx
    expert_offsets = router_pairs.expert_offsets
    e = W_gate.shape[0]
    h = flat_input.shape[-1]
    n_pairs = pair_token_idx.numel()

    out = torch.empty(n_pairs, h, device=device, dtype=flat_input.dtype)
    if n_pairs == 0:
        return out

    # Gather once then segment by expert.
    routed_x = flat_input.index_select(0, pair_token_idx)  # (n_pairs, H)

    # Bulk-transfer CSR offsets to host once (avoids O(E) scalar .item() syncs).
    offsets_host = expert_offsets.detach().to("cpu", dtype=torch.long).tolist()

    for expert_id in range(e):
        start = offsets_host[expert_id]
        end = offsets_host[expert_id + 1]
        if start == end:
            continue
        xe = routed_x[start:end]
        out[start:end] = apply_swiglu(
            xe, W_gate[expert_id], W_up[expert_id], W_down[expert_id], act_fn=act_fn
        )
    return out


def materialize_sparse_activations(
    out: torch.Tensor,
    pair_token_idx: torch.Tensor,
    pair_expert_idx: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Optional ``(E, T, H)`` materialization for merge-path metrics only."""
    activations = torch.zeros(
        (num_experts, num_tokens, hidden_dim), device=device, dtype=dtype
    )
    if out.numel() == 0:
        return activations
    # Last write wins if a token routes to the same expert twice (rare with unique topk).
    activations[pair_expert_idx, pair_token_idx] = out.to(dtype)
    return activations
