"""F5: fused router stage — softmax + topk + expert-sorted pair indices.

Pure-PyTorch implementation (GPU-resident). Optional Triton path accelerates
the softmax when CUDA+Triton are available; topk/sort stay in PyTorch for
correctness across all top_k.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from reap.kernels.backend import triton_available


@dataclass
class RouterPairOutputs:
    selected_experts: torch.Tensor  # (T, k)
    router_weights_full: torch.Tensor  # (T, E) softmax (+ optional renorm)
    pair_token_idx: torch.Tensor  # (T*k,)
    pair_expert_idx: torch.Tensor  # (T*k,)
    pair_router_w: torch.Tensor  # (T*k,)
    expert_offsets: torch.Tensor  # (E+1,)
    pair_perm: torch.Tensor  # (T*k,) sort-by-expert permutation


def unwrap_router_logits(router_out) -> torch.Tensor:
    """Normalize heterogeneous router return types to raw logits ``(T, E)``."""
    if isinstance(router_out, tuple):
        # Qwen3/Qwen3.5: (logits, scores, indices) — element 0 is raw logits.
        return router_out[0]
    return router_out


def extract_router_logits(
    router_module,
    flat_input: torch.Tensor,
    *,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    hidden_dim: int | None = None,
) -> torch.Tensor:
    """Call routers that accept flat or sequence-shaped hidden states."""
    try:
        result = router_module(flat_input)
    except (TypeError, ValueError):
        if flat_input.ndim != 2 or batch_size is None:
            raise
        result = router_module(
            flat_input.view(batch_size, sequence_length, hidden_dim or flat_input.shape[-1])
        )
    return unwrap_router_logits(result)


def f5_router_pytorch(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool = False,
    valid_token_mask: torch.Tensor | None = None,
) -> RouterPairOutputs:
    """Build routed pair tensors on the same device as *router_logits*."""
    device = router_logits.device
    t, e = router_logits.shape
    k = min(top_k, e)

    # Softmax in fp32 for stability (matches pruning_metrics).
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    selected_vals, selected_experts = torch.topk(routing_weights, k, dim=-1)
    selected_experts = selected_experts.to(device)

    if norm_topk_prob and selected_experts.numel() > 0:
        # Renormalize full distribution by top-k mass (matches existing code).
        topk_sum = selected_vals.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(routing_weights.dtype).eps
        )
        routing_weights = routing_weights / topk_sum
        routing_weights = torch.clamp(
            routing_weights, min=torch.finfo(routing_weights.dtype).eps
        )
        selected_vals = torch.gather(routing_weights, 1, selected_experts)

    # Optional padding filter: zero out invalid tokens' pairs by masking weights
    # and leaving expert ids; update_pruning_state_routed filters via mask.
    pair_token_idx = torch.arange(t, device=device).repeat_interleave(k)
    pair_expert_idx = selected_experts.reshape(-1)
    pair_router_w = selected_vals.reshape(-1).to(torch.float32)

    if valid_token_mask is not None:
        mask = valid_token_mask.reshape(-1).bool().to(device)
        keep = mask[pair_token_idx]
        pair_token_idx = pair_token_idx[keep]
        pair_expert_idx = pair_expert_idx[keep]
        pair_router_w = pair_router_w[keep]
        # Also filter selected_experts / routing for downstream consumers that
        # use the token axis.
        selected_experts = selected_experts[mask]
        routing_weights = routing_weights[mask]

    # Sort pairs by expert for coalesced grouped matmul (CSR offsets).
    if pair_expert_idx.numel() == 0:
        pair_perm = pair_expert_idx
        expert_offsets = torch.zeros(e + 1, device=device, dtype=torch.long)
    else:
        pair_perm = torch.argsort(pair_expert_idx, stable=True)
        pair_token_idx = pair_token_idx[pair_perm]
        pair_expert_idx = pair_expert_idx[pair_perm]
        pair_router_w = pair_router_w[pair_perm]
        counts = torch.bincount(pair_expert_idx, minlength=e)
        expert_offsets = torch.zeros(e + 1, device=device, dtype=torch.long)
        expert_offsets[1:] = torch.cumsum(counts, dim=0)

    return RouterPairOutputs(
        selected_experts=selected_experts,
        router_weights_full=routing_weights,
        pair_token_idx=pair_token_idx,
        pair_expert_idx=pair_expert_idx,
        pair_router_w=pair_router_w,
        expert_offsets=expert_offsets,
        pair_perm=pair_perm,
    )


def f5_router(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool = False,
    valid_token_mask: torch.Tensor | None = None,
) -> RouterPairOutputs:
    """F5 entry: currently PyTorch on all devices (Triton optional soft-path)."""
    # Triton softmax is a micro-optimization; keep one code path for parity.
    _ = triton_available  # reserved for future fused softmax kernel
    return f5_router_pytorch(
        router_logits,
        top_k,
        norm_topk_prob=norm_topk_prob,
        valid_token_mask=valid_token_mask,
    )
