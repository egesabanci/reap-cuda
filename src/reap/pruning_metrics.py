"""GPU-resident REAP pruning saliency accumulation.

All hot-path tensors stay on the compute device. Host transfer happens only
when the observer saves state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from reap.metrics import OnlineStatsTracker

# CLI prune_method -> observer state key
PRUNE_METHOD_KEY_MAP: dict[str, str] = {
    "frequency": "expert_frequency",
    "ean_sum": "ean_sum",
    "ean_mean": "ean_mean",
    "weighted_frequency_sum": "weighted_expert_frequency_sum",
    "weighted_ean_sum": "weighted_ean_sum",
    "weighted_ean_sum_l2": "weighted_ean_sum",  # alias
    "reap": "reap",
    "reap_l2": "reap",  # alias (l2 is already the ean_norm)
    "max_activations": "max_activations",
    "ean_ca": "routed_characteristic_activation",
}

PRUNING_STATE_KEYS = frozenset(
    {
        "total_tokens",
        "expert_frequency",
        "pairwise_expert_frequency",
        "ean_sum",
        "ean_mean",
        "reap",
        "weighted_ean_sum",
        "weighted_expert_frequency_sum",
        "max_activations",
    }
)

MERGING_CRITERIA_KEYS = frozenset(
    {
        "ttm_similarity_matrix",
        "characteristic_activation",
        "online_characteristic_activation_dist",
        "router_logit_similiarity",  # sic: codebase misspelling
    }
)


def resolve_compute_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PreparedPruningBatch:
    activations: torch.Tensor
    selected_experts: torch.Tensor
    router_logits: torch.Tensor
    num_tokens: torch.Tensor
    expert_frequency: torch.Tensor
    pairwise_expert_frequency: torch.Tensor


def initialize_pruning_state(
    num_experts: int,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Create per-layer pruning state on the compute device (GPU when available)."""
    device = resolve_compute_device(device)
    layer_state: dict[str, Any] = {}
    layer_state["total_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
    layer_state["expert_frequency"] = torch.zeros(
        num_experts, device=device, dtype=torch.long
    )
    layer_state["pairwise_expert_frequency"] = torch.zeros(
        num_experts, num_experts, dtype=torch.long, device=device
    )
    layer_state["ean_sum"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float64, requires_grad=False
    )
    layer_state["weighted_ean_sum"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float64, requires_grad=False
    )
    layer_state["ean_mean"] = OnlineStatsTracker(
        shape=(num_experts,),
        count_shape=(num_experts,),
        device=device,
        dtype=torch.float32,
    )
    layer_state["reap"] = OnlineStatsTracker(
        shape=(num_experts,),
        count_shape=(num_experts,),
        device=device,
        dtype=torch.float32,
    )
    layer_state["weighted_expert_frequency_sum"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float64, requires_grad=False
    )
    layer_state["max_activations"] = torch.zeros(
        (num_experts,), device=device, dtype=torch.float32, requires_grad=False
    )
    return layer_state


def move_pruning_state_to_device(
    layer_state: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move all tensors / trackers in a layer state onto *device*."""
    for key, value in list(layer_state.items()):
        if isinstance(value, torch.Tensor):
            layer_state[key] = value.to(device)
        elif isinstance(value, OnlineStatsTracker):
            value.to(device)
    return layer_state


def _prepare_pruning_batch(
    *,
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    valid_token_mask: Optional[torch.Tensor] = None,
) -> PreparedPruningBatch:
    device = activations.device
    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1]).to(device)
    router_logits = router_logits.to(device)

    if valid_token_mask is not None:
        valid_token_mask = valid_token_mask.reshape(-1).bool().to(device)
        activations = activations[:, valid_token_mask, :]
        selected_experts = selected_experts[valid_token_mask]
        router_logits = router_logits[valid_token_mask]

    if activations.shape[0] != num_experts:
        raise ValueError(
            f"Expected activations for {num_experts} experts, got {activations.shape[0]}"
        )
    if router_logits.shape[1] != num_experts:
        raise ValueError(
            f"Expected router logits for {num_experts} experts, got {router_logits.shape[1]}"
        )
    if activations.shape[1] != selected_experts.shape[0]:
        raise ValueError(
            "Activations and selected expert token counts do not match: "
            f"{activations.shape[1]} vs {selected_experts.shape[0]}"
        )
    if router_logits.shape[0] != selected_experts.shape[0]:
        raise ValueError(
            "Router logits and selected expert token counts do not match: "
            f"{router_logits.shape[0]} vs {selected_experts.shape[0]}"
        )

    num_tokens = torch.tensor(
        selected_experts.shape[0], device=device, dtype=torch.long
    )
    if selected_experts.numel() == 0:
        expert_frequency = torch.zeros(num_experts, device=device, dtype=torch.long)
    else:
        expert_frequency = torch.bincount(
            selected_experts.reshape(-1), minlength=num_experts
        ).to(device)
    # Matches historical REAP semantics: freq_i + freq_j (not co-routing counts).
    pairwise_expert_frequency = expert_frequency.unsqueeze(0) + expert_frequency.unsqueeze(
        1
    )

    return PreparedPruningBatch(
        activations=activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_tokens=num_tokens,
        expert_frequency=expert_frequency,
        pairwise_expert_frequency=pairwise_expert_frequency,
    )


def update_pruning_state(
    layer_state: dict[str, Any],
    *,
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    valid_token_mask: Optional[torch.Tensor] = None,
    renormalize_router_weights: bool = False,
    selected_router_weights: Optional[torch.Tensor] = None,
) -> PreparedPruningBatch:
    """Accumulate pruning saliency from a dense ``(E, T, H)`` activation tensor.

    Preferred for the legacy ``loop`` backend. Prefer
    :func:`update_pruning_state_routed` for bmm/FREA/F2 (no ``(E,T,H)``).
    All reductions stay on the activations device.

    When *selected_router_weights* is supplied (shape ``(T, top_k)``), the
    dense update uses these native-router-selected weights directly for the
    router-weighted saliency statistics (``weighted_frequency_sum``,
    ``weighted_ean_sum``, and the router-weighted REAP samples). Their pair
    ordering must exactly align with *selected_experts*. The logits/softmax
    derivation is retained only as the backward-compatible fallback for
    callers that do not supply native selected weights (standard softmax
    routers); a supplied native-router result must not be overridden.
    Unweighted metrics (``expert_frequency``, EAN, EAN mean, max activation,
    pair frequency, and token count) preserve their existing behavior.
    """
    pruning_batch = _prepare_pruning_batch(
        activations=activations,
        selected_experts=selected_experts,
        router_logits=router_logits,
        num_experts=num_experts,
        valid_token_mask=valid_token_mask,
    )

    device = pruning_batch.activations.device
    move_pruning_state_to_device(layer_state, device)

    # Validate optional native-router selected weights against the prepared
    # batch before any metric is touched so a contract violation cannot corrupt
    # the layer state. ``selected_router_weights`` arrives in (T, top-k) order
    # aligned with ``pruning_batch.selected_experts``.
    use_native_selected_weights = False
    if selected_router_weights is not None:
        # Apply the same ``valid_token_mask`` used inside
        # ``_prepare_pruning_batch`` to ``selected_router_weights`` so its
        # token axis matches the already-filtered ``pruning_batch.selected_experts``
        # before the shape comparison below. The mask is validated against the
        # original (pre-mask) token dimension shared with ``selected_experts``;
        # no reshaping, broadcasting, or fallback is introduced. The unmasked
        # path (``valid_token_mask is None``) is left untouched.
        if valid_token_mask is not None:
            mask = valid_token_mask.reshape(-1).bool().to(
                selected_router_weights.device
            )
            if selected_router_weights.shape[0] != mask.shape[0]:
                raise ValueError(
                    "valid_token_mask and selected_router_weights token counts "
                    f"do not match: mask={mask.shape[0]} vs "
                    f"selected_router_weights={selected_router_weights.shape[0]}"
                )
            selected_router_weights = selected_router_weights[mask]
        if selected_router_weights.shape != pruning_batch.selected_experts.shape:
            raise ValueError(
                "selected_router_weights shape "
                f"{tuple(selected_router_weights.shape)} != selected_experts "
                f"{tuple(pruning_batch.selected_experts.shape)}"
            )
        if selected_router_weights.device != device:
            raise ValueError(
                "selected_router_weights must share the activations device "
                f"({device}); got {selected_router_weights.device}"
            )
        if not torch.is_floating_point(selected_router_weights):
            raise ValueError(
                "selected_router_weights must be a floating-point tensor"
            )
        selected_router_weights = selected_router_weights.to(device)
        use_native_selected_weights = True

    layer_state["total_tokens"] = layer_state["total_tokens"] + pruning_batch.num_tokens
    layer_state["expert_frequency"] = (
        layer_state["expert_frequency"] + pruning_batch.expert_frequency
    )
    layer_state["pairwise_expert_frequency"] = (
        layer_state["pairwise_expert_frequency"] + pruning_batch.pairwise_expert_frequency
    )

    ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_expert_frequency_sum = torch.zeros(
        num_experts, device=device, dtype=torch.float64
    )
    batch_max = torch.zeros(num_experts, device=device, dtype=torch.float32)

    # Router-weighted statistics. When native-router selected weights are
    # supplied, they are authoritative and are gathered per (token, top-k) pair
    # to align exactly with the dense activations. Otherwise derive weights
    # from logits via softmax (backward-compatible fallback for standard
    # softmax routers; must not override a supplied native-router result).
    if use_native_selected_weights:
        # routing_weights_per_pair: (T, top_k) aligned with selected_experts.
        routing_weights_per_pair = selected_router_weights.to(torch.float32)
        if renormalize_router_weights and pruning_batch.selected_experts.numel() > 0:
            pair_sum = routing_weights_per_pair.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(routing_weights_per_pair.dtype).eps
            )
            routing_weights_per_pair = routing_weights_per_pair / pair_sum
    else:
        routing_weights = F.softmax(
            pruning_batch.router_logits, dim=1, dtype=torch.float
        ).to(device)
        if renormalize_router_weights and pruning_batch.selected_experts.numel() > 0:
            topk_weights = torch.gather(
                routing_weights,
                1,
                pruning_batch.selected_experts,
            )
            routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)
            routing_weights = torch.clamp(
                routing_weights, min=torch.finfo(routing_weights.dtype).eps
            )
        routing_weights_per_pair = torch.gather(
            routing_weights, 1, pruning_batch.selected_experts
        ).to(torch.float32)

    for i in range(num_experts):
        # ``expert_mask`` is (T,) True for tokens that route to expert i in any
        # top-k slot. ``slot_mask`` is (T, top_k) True for the (token, slot)
        # pairs that select expert i, so gathering per-pair weights matches the
        # routed CSR construction exactly (no broadcasting across unselected
        # experts).
        slot_mask = pruning_batch.selected_experts == i
        active_mask = slot_mask.any(dim=-1)
        if not active_mask.any():
            continue

        selected_activations = pruning_batch.activations[i, active_mask, :]
        active_pair_weights = routing_weights_per_pair[active_mask][slot_mask[active_mask]]
        ean_norm = torch.linalg.norm(selected_activations.float(), dim=-1)
        ean_sum[i] = ean_norm.sum().to(torch.float64)
        ean_mean[i] = ean_norm.mean().to(torch.float32)
        weighted_expert_frequency_sum[i] = active_pair_weights.sum().to(torch.float64)
        weighted_ean_sum[i] = (ean_norm * active_pair_weights).sum().to(torch.float64)
        reap[i] = (ean_norm * active_pair_weights).mean().to(torch.float32)
        batch_max[i] = selected_activations.float().max()

    layer_state["ean_sum"] = layer_state["ean_sum"] + ean_sum
    layer_state["ean_mean"].update(ean_mean, pruning_batch.expert_frequency)
    layer_state["weighted_ean_sum"] = layer_state["weighted_ean_sum"] + weighted_ean_sum
    layer_state["reap"].update(reap, pruning_batch.expert_frequency)
    layer_state["weighted_expert_frequency_sum"] = (
        layer_state["weighted_expert_frequency_sum"] + weighted_expert_frequency_sum
    )
    layer_state["max_activations"] = torch.maximum(
        layer_state["max_activations"], batch_max
    )

    return pruning_batch


def update_pruning_state_routed(
    layer_state: dict[str, Any],
    *,
    pair_out: torch.Tensor,
    pair_expert_idx: torch.Tensor,
    pair_token_idx: torch.Tensor,
    pair_router_w: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    valid_token_mask: Optional[torch.Tensor] = None,
    renormalize_router_weights: bool = False,
    compute_routed_ca: bool = False,
    router_weights_full: Optional[torch.Tensor] = None,
) -> None:
    """Accumulate prune metrics from routed pair tensors only (no ``(E,T,H)``).

    Pair arrays must be filtered to valid tokens already (F5). ``selected_experts``
    and ``router_logits`` should share the same filtered token axis.
    """
    device = pair_out.device if pair_out.numel() else selected_experts.device
    move_pruning_state_to_device(layer_state, device)

    selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1]).to(device)
    router_logits = router_logits.to(device)
    pair_expert_idx = pair_expert_idx.to(device)
    pair_router_w = pair_router_w.to(device=device, dtype=torch.float32)

    if valid_token_mask is not None:
        # Safety: if mask provided, filter selected_experts / logits.
        m = valid_token_mask.reshape(-1).bool().to(device)
        selected_experts = selected_experts[m]
        router_logits = router_logits[m]

    num_tokens = torch.tensor(selected_experts.shape[0], device=device, dtype=torch.long)
    if selected_experts.numel() == 0:
        expert_frequency = torch.zeros(num_experts, device=device, dtype=torch.long)
    else:
        expert_frequency = torch.bincount(
            selected_experts.reshape(-1), minlength=num_experts
        ).to(device)
    pairwise = expert_frequency.unsqueeze(0) + expert_frequency.unsqueeze(1)

    layer_state["total_tokens"] = layer_state["total_tokens"] + num_tokens
    layer_state["expert_frequency"] = layer_state["expert_frequency"] + expert_frequency
    layer_state["pairwise_expert_frequency"] = (
        layer_state["pairwise_expert_frequency"] + pairwise
    )

    if pair_out.numel() == 0:
        return

    # F2 scatter: Triton atomics when eligible, else PyTorch index_add_/scatter_reduce.
    from reap.kernels.triton_reduce import scatter_pair_stats

    stats = scatter_pair_stats(
        pair_out, pair_expert_idx, pair_router_w, num_experts
    )
    ean_sum = stats["ean_sum"]
    weighted_ean_sum = stats["weighted_ean_sum"]
    weighted_freq = stats["weighted_freq"]
    batch_max_raw = stats["batch_max"]

    # Per-expert batch means for OnlineStatsTracker / Welford (PyTorch; exact).
    ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
    reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
    nonzero = expert_frequency > 0
    ean_mean[nonzero] = (
        ean_sum[nonzero] / expert_frequency[nonzero].to(torch.float64)
    ).to(torch.float32)
    reap[nonzero] = (
        weighted_ean_sum[nonzero] / expert_frequency[nonzero].to(torch.float64)
    ).to(torch.float32)

    layer_state["ean_sum"] = layer_state["ean_sum"] + ean_sum
    layer_state["ean_mean"].update(ean_mean, expert_frequency)
    layer_state["weighted_ean_sum"] = layer_state["weighted_ean_sum"] + weighted_ean_sum
    layer_state["reap"].update(reap, expert_frequency)
    layer_state["weighted_expert_frequency_sum"] = (
        layer_state["weighted_expert_frequency_sum"] + weighted_freq
    )
    layer_state["max_activations"] = torch.maximum(
        layer_state["max_activations"], batch_max_raw
    )

    if compute_routed_ca:
        h = pair_out.shape[-1]
        ca = torch.zeros(num_experts, h, device=device, dtype=torch.float64)
        ca.index_add_(0, pair_expert_idx, pair_out.double())
        freq = expert_frequency.to(torch.float64).clamp_min(1).unsqueeze(-1)
        ca = (ca / freq).nan_to_num(0)
        if "routed_characteristic_activation" not in layer_state:
            from reap.metrics import OnlineStatsTracker as OST

            layer_state["routed_characteristic_activation"] = OST(
                shape=(num_experts, h),
                count_shape=(num_experts, h),
                device=device,
                dtype=torch.float32,
            )
        freq_exp = expert_frequency.unsqueeze(-1).expand(-1, h)
        layer_state["routed_characteristic_activation"].update(
            ca.to(torch.float32), freq_exp
        )
