"""F2: fuse saliency reductions over routed pair activations (GPU-resident)."""

from __future__ import annotations

from typing import Any, Optional

import torch

from reap.kernels.router import RouterPairOutputs
from reap.pruning_metrics import update_pruning_state_routed


def f2_accumulate(
    layer_state: dict[str, Any],
    *,
    pair_out: torch.Tensor,
    router_pairs: RouterPairOutputs,
    router_logits: torch.Tensor,
    num_experts: int,
    valid_token_mask: Optional[torch.Tensor] = None,
    renormalize_router_weights: bool = False,
    compute_routed_ca: bool = False,
) -> None:
    """Accumulate all prune-path metrics from pair outputs (no ``(E,T,H)``)."""
    update_pruning_state_routed(
        layer_state,
        pair_out=pair_out,
        pair_expert_idx=router_pairs.pair_expert_idx,
        pair_token_idx=router_pairs.pair_token_idx,
        pair_router_w=router_pairs.pair_router_w,
        selected_experts=router_pairs.selected_experts,
        router_logits=router_logits
        if valid_token_mask is None
        else router_logits,  # caller filters logits if needed
        num_experts=num_experts,
        valid_token_mask=None,  # pairs already filtered by F5
        renormalize_router_weights=renormalize_router_weights,
        compute_routed_ca=compute_routed_ca,
        router_weights_full=router_pairs.router_weights_full,
    )
