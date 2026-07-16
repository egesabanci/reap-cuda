"""F2: fuse saliency reductions over routed pair activations (GPU-resident).

Reductions go through ``update_pruning_state_routed``, which uses Triton
scatter-reduce when eligible and PyTorch otherwise. Welford means stay in
``OnlineStatsTracker`` for bit-stable multi-batch accumulation.
"""

from __future__ import annotations

from typing import Any, Optional

from reap.kernels.router import RouterPairOutputs
from reap.pruning_metrics import update_pruning_state_routed


def f2_accumulate(
    layer_state: dict[str, Any],
    *,
    pair_out,
    router_pairs: RouterPairOutputs,
    router_logits,
    num_experts: int,
    valid_token_mask: Optional[Any] = None,
    renormalize_router_weights: bool = False,
    compute_routed_ca: bool = False,
) -> None:
    """Accumulate all prune-path metrics from pair outputs (no ``(E,T,H)``)."""
    del valid_token_mask  # pairs already filtered by F5
    update_pruning_state_routed(
        layer_state,
        pair_out=pair_out,
        pair_expert_idx=router_pairs.pair_expert_idx,
        pair_token_idx=router_pairs.pair_token_idx,
        pair_router_w=router_pairs.pair_router_w,
        selected_experts=router_pairs.selected_experts,
        router_logits=router_logits,
        num_experts=num_experts,
        valid_token_mask=None,
        renormalize_router_weights=renormalize_router_weights,
        compute_routed_ca=compute_routed_ca,
        router_weights_full=router_pairs.router_weights_full,
    )
