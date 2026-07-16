"""Unified MoE observation step used by standard + layerwise observers."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from reap.kernels.bmm import materialize_sparse_activations
from reap.kernels.f2 import f2_accumulate
from reap.kernels.frea import frea_activations
from reap.kernels.router import (
    extract_router_logits,
    f5_router,
    f5_router_from_module,
    prefers_native_router,
)
from reap.kernels.weight_cache import free_cache, get_stacked_expert_weights
from reap.pruning_metrics import update_pruning_state


def _loop_activations(
    moe: nn.Module,
    adapter: Any,
    flat_input: torch.Tensor,
    num_experts: int,
    top_k: int,
    act_fn: Callable,
    fused: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy full-expert activation path (parity oracle).

    Routing uses the model router when softmax+topk would be wrong
    (``prefers_native_router``); otherwise top-k on raw logits as historically.
    """
    device = flat_input.device

    if prefers_native_router(moe, adapter):
        router_logits, pairs = f5_router_from_module(
            moe, adapter, flat_input, top_k=top_k, valid_token_mask=None
        )
        selected_experts = pairs.selected_experts
    else:
        router = (
            getattr(moe, adapter.router_attr(), None)
            or getattr(moe, "router", None)
            or getattr(moe, "gate", None)
        )
        if router is None:
            raise ValueError("Cannot find router on MoE module")
        router_logits = extract_router_logits(router, flat_input)
        _, selected_experts = torch.topk(
            router_logits, min(top_k, router_logits.shape[-1]), dim=-1
        )
        selected_experts = selected_experts.to(device)

    activations = torch.zeros(
        (num_experts, *flat_input.shape), device=device, dtype=flat_input.dtype
    )

    if fused:
        stacked = get_stacked_expert_weights(moe, adapter, device=device)
        W_gate, W_up, W_down = stacked["W_gate"], stacked["W_up"], stacked["W_down"]
        for e in range(num_experts):
            mask = (selected_experts == e).any(dim=-1)
            if not bool(mask.any()):
                continue
            xe = flat_input[mask]
            g = F.linear(xe, W_gate[e])
            u = F.linear(xe, W_up[e])
            activations[e, mask] = F.linear(act_fn(g) * u, W_down[e])
    else:
        for idx, expert in enumerate(moe.experts):
            activations[idx] = expert(flat_input).to(device)

    free_cache(moe)
    return activations, selected_experts, router_logits


def observe_moe_batch(
    layer_state: dict[str, Any],
    moe: nn.Module,
    adapter: Any,
    flat_input: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    act_fn: Callable = F.silu,
    valid_token_mask: Optional[torch.Tensor] = None,
    renormalize_router_weights: bool = False,
    backend: str = "bmm",
    record_pruning_metrics_only: bool = True,
    compute_routed_ca: bool = False,
    fused: bool | None = None,
    batch_size: int | None = None,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Run one MoE observation step into *layer_state*.

    Returns a small dict of tensors needed by merge-criteria metrics when
    ``record_pruning_metrics_only=False``:

    * ``activations`` ``(E, T_valid, H)`` (only if merge metrics needed)
    * ``selected_experts`` ``(T_valid, k)``
    * ``router_logits`` ``(T_valid, E)``
    * ``num_tokens``
    """
    device = flat_input.device
    if fused is None:
        fused = adapter.expert_weight_attrs(moe).get("fused", False)

    if valid_token_mask is not None:
        mask = valid_token_mask.reshape(-1).bool().to(device)
    else:
        mask = None

    need_dense = not record_pruning_metrics_only
    from reap.kernels.triton_utils import triton_runtime_available

    if backend == "loop":
        activations, selected_experts, router_logits = _loop_activations(
            moe, adapter, flat_input, num_experts, top_k, act_fn, fused
        )
        pruning_batch = update_pruning_state(
            layer_state,
            activations=activations,
            selected_experts=selected_experts,
            router_logits=router_logits,
            num_experts=num_experts,
            valid_token_mask=valid_token_mask,
            renormalize_router_weights=renormalize_router_weights,
        )
        return {
            "activations": pruning_batch.activations,
            "selected_experts": pruning_batch.selected_experts,
            "router_logits": pruning_batch.router_logits,
            "num_tokens": pruning_batch.num_tokens,
            "expert_frequency": pruning_batch.expert_frequency,
            "pairwise_expert_frequency": pruning_batch.pairwise_expert_frequency,
        }

    # --- Routed backends: bmm / frea / f2 ---------------------------------
    router = (
        getattr(moe, adapter.router_attr(), None)
        or getattr(moe, "router", None)
        or getattr(moe, "gate", None)
    )
    if router is None:
        raise ValueError("Cannot find router on MoE module")

    # Model-agnostic: use native router when softmax+topk would be wrong.
    if prefers_native_router(moe, adapter):
        router_logits_full, router_pairs = f5_router_from_module(
            moe,
            adapter,
            flat_input,
            top_k=top_k,
            valid_token_mask=valid_token_mask,
        )
    else:
        router_logits_full = extract_router_logits(
            router,
            flat_input,
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_dim=flat_input.shape[-1],
        )
        router_pairs = f5_router(
            router_logits_full,
            top_k,
            norm_topk_prob=renormalize_router_weights,
            valid_token_mask=valid_token_mask,
        )

    stacked = get_stacked_expert_weights(moe, adapter, device=device)
    use_triton = backend in ("frea", "f2") and triton_runtime_available()
    # FREA sub-backend from process policy (CLI / set_frea_backend / env).
    try:
        from reap.kernels.triton_frea import get_frea_backend

        frea_backend = get_frea_backend()
    except Exception:
        frea_backend = "auto"
    pair_out = frea_activations(
        flat_input,
        router_pairs,
        stacked["W_gate"],
        stacked["W_up"],
        stacked["W_down"],
        act_fn=act_fn,
        use_triton=use_triton,
        frea_backend=frea_backend if use_triton else "pytorch",
    )

    if mask is not None:
        router_logits = router_logits_full[mask]
    else:
        router_logits = router_logits_full

    f2_accumulate(
        layer_state,
        pair_out=pair_out,
        router_pairs=router_pairs,
        router_logits=router_logits,
        num_experts=num_experts,
        valid_token_mask=None,
        renormalize_router_weights=False,
        compute_routed_ca=compute_routed_ca,
    )

    # Drop F4 stack for this MoE (full path also limited to 1 entry in cache).
    free_cache(moe)

    result: dict[str, Any] = {
        "selected_experts": router_pairs.selected_experts,
        "router_logits": router_logits,
        "num_tokens": torch.tensor(
            router_pairs.selected_experts.shape[0], device=device, dtype=torch.long
        ),
        "expert_frequency": layer_state["expert_frequency"],
        "pairwise_expert_frequency": layer_state["pairwise_expert_frequency"],
    }

    if need_dense:
        t_valid = router_pairs.selected_experts.shape[0]
        if mask is not None:
            valid_pos = torch.full(
                (flat_input.shape[0],), -1, device=device, dtype=torch.long
            )
            valid_pos[mask] = torch.arange(int(mask.sum().item()), device=device)
            pair_tok_f = valid_pos[router_pairs.pair_token_idx]
            keep = pair_tok_f >= 0
            activations = materialize_sparse_activations(
                pair_out[keep],
                pair_tok_f[keep],
                router_pairs.pair_expert_idx[keep],
                num_experts,
                t_valid,
                flat_input.shape[-1],
                device=device,
                dtype=flat_input.dtype,
            )
        else:
            activations = materialize_sparse_activations(
                pair_out,
                router_pairs.pair_token_idx,
                router_pairs.pair_expert_idx,
                num_experts,
                t_valid,
                flat_input.shape[-1],
                device=device,
                dtype=flat_input.dtype,
            )
        result["activations"] = activations
        if router_pairs.selected_experts.numel():
            result["expert_frequency"] = torch.bincount(
                router_pairs.selected_experts.reshape(-1), minlength=num_experts
            ).to(device)
        else:
            result["expert_frequency"] = torch.zeros(
                num_experts, device=device, dtype=torch.long
            )
        result["pairwise_expert_frequency"] = (
            result["expert_frequency"].unsqueeze(0)
            + result["expert_frequency"].unsqueeze(1)
        )

    return result
