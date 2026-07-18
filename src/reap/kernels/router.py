"""F5: fused router stage — softmax + topk + expert-sorted pair indices.

Softmax uses a Triton kernel on CUDA when available; top-k and pair CSR
construction stay in PyTorch for correctness across all ``top_k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from reap.kernels.triton_softmax import softmax_rows


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
            flat_input.view(
                batch_size, sequence_length, hidden_dim or flat_input.shape[-1]
            )
        )
    return unwrap_router_logits(result)


def f5_router_pytorch(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool = False,
    valid_token_mask: torch.Tensor | None = None,
    use_triton_softmax: bool = True,
) -> RouterPairOutputs:
    """Build routed pair tensors on the same device as *router_logits*."""
    device = router_logits.device
    t, e = router_logits.shape
    k = min(top_k, e)

    if use_triton_softmax:
        routing_weights = softmax_rows(router_logits)
    else:
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

    selected_vals, selected_experts = torch.topk(routing_weights, k, dim=-1)
    selected_experts = selected_experts.to(device)

    if norm_topk_prob and selected_experts.numel() > 0:
        topk_sum = selected_vals.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(routing_weights.dtype).eps
        )
        routing_weights = routing_weights / topk_sum
        routing_weights = torch.clamp(
            routing_weights, min=torch.finfo(routing_weights.dtype).eps
        )
        selected_vals = selected_vals / topk_sum

    pair_token_idx = torch.arange(t, device=device).repeat_interleave(k)
    pair_expert_idx = selected_experts.reshape(-1)
    pair_router_w = selected_vals.reshape(-1).to(torch.float32)

    if valid_token_mask is not None:
        mask = valid_token_mask.reshape(-1).bool().to(device)
        keep = mask[pair_token_idx]
        pair_token_idx = pair_token_idx[keep]
        pair_expert_idx = pair_expert_idx[keep]
        pair_router_w = pair_router_w[keep]
        selected_experts = selected_experts[mask]
        routing_weights = routing_weights[mask]

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
    """F5 entry: Triton softmax when eligible, PyTorch top-k + CSR always."""
    return f5_router_pytorch(
        router_logits,
        top_k,
        norm_topk_prob=norm_topk_prob,
        valid_token_mask=valid_token_mask,
        use_triton_softmax=True,
    )


def _resolve_router(moe, adapter):
    return (
        getattr(moe, adapter.router_attr(), None)
        or getattr(moe, "router", None)
        or getattr(moe, "gate", None)
    )


def prefers_native_router(moe, adapter) -> bool:
    """True when softmax+topk would mis-route this MoE (model-agnostic).

    Structural signals (no architecture name required):
    * adapter exposes ``prefer_native_router`` / ``prefer_native_router()``
    * adapter name is a known non-softmax family (e.g. ``lfm2_moe``)
    * MoE has an ``expert_bias`` buffer (LFM2-style)
    * router has ``use_expert_bias=True``

    Does **not** treat a bare ``expert_bias`` kwarg in the signature as enough
    (optional kwargs on softmax routers would false-positive).
    """
    flag = getattr(adapter, "prefer_native_router", None)
    if callable(flag):
        try:
            if flag():
                return True
        except TypeError:
            pass
    elif flag:
        return True
    if getattr(adapter, "adapter_name", "") == "lfm2_moe":
        return True
    if getattr(moe, "expert_bias", None) is not None:
        return True
    router = _resolve_router(moe, adapter)
    if router is None:
        return False
    return bool(getattr(router, "use_expert_bias", False))


def _router_weight_activation(moe, adapter) -> str:
    """Heuristic for full (T, E) weight matrix used by merge metrics."""
    explicit = getattr(adapter, "router_activation", None)
    if callable(explicit):
        try:
            return str(explicit())
        except TypeError:
            pass
    elif isinstance(explicit, str):
        return explicit
    if prefers_native_router(moe, adapter):
        # sigmoid+bias families (LFM2); merge path uses sigmoid on logits.
        return "sigmoid"
    return "softmax"


def f5_router_from_module(
    moe,
    adapter,
    flat_input: torch.Tensor,
    *,
    top_k: int,
    valid_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, RouterPairOutputs]:
    """Build router pairs by calling the model's own router module.

    Used when routing semantics differ from softmax+topk (e.g. sigmoid +
    expert_bias → top-k on scores → gather → renorm → scale). The model router
    returns ``(logits, routing_weights, selected_experts)``; we reuse its exact
    routing and only build the CSR pair structure that FREA/F2 expect.

    Returns ``(router_logits_full, RouterPairOutputs)`` where
    ``router_logits_full`` is the raw ``(T, E)`` logits.
    """
    import inspect

    device = flat_input.device
    router = _resolve_router(moe, adapter)
    if router is None:
        raise ValueError("Cannot find router on MoE module")

    kw: dict = {}
    try:
        sig = inspect.signature(router.forward)
    except (TypeError, ValueError):
        sig = None
    expert_bias = getattr(moe, "expert_bias", None)
    has_expert_bias = expert_bias is not None and (
        (sig is not None and "expert_bias" in sig.parameters)
        or bool(getattr(router, "use_expert_bias", False))
    )
    if has_expert_bias:
        kw["expert_bias"] = expert_bias

    try:
        out = router(flat_input, **kw)
    except TypeError:
        if "expert_bias" in kw:
            kw.pop("expert_bias")
            out = router(flat_input, **kw)
        else:
            raise
    if not (isinstance(out, tuple) and len(out) >= 3):
        raise ValueError(
            "f5_router_from_module expects a router returning "
            "(logits, weights, selected_experts); got type that does not."
        )
    router_logits_full, routing_weights, selected_experts = out[0], out[1], out[2]

    t, e = router_logits_full.shape
    selected_experts = selected_experts.to(device)
    routing_weights = routing_weights.to(device)
    # Use the router's actual top-k width (may differ from the requested top_k).
    if selected_experts.ndim == 1:
        selected_experts = selected_experts.unsqueeze(-1)
        routing_weights = routing_weights.unsqueeze(-1)
    k = int(selected_experts.shape[-1])
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            f"router weights shape {tuple(routing_weights.shape)} != "
            f"selected_experts shape {tuple(selected_experts.shape)}"
        )

    # Full (T, E) weights for merge-criteria metrics (prune-only ignores these).
    act = _router_weight_activation(moe, adapter)
    if act == "sigmoid":
        router_weights_full = torch.sigmoid(router_logits_full.float()).to(device)
    else:
        router_weights_full = F.softmax(
            router_logits_full, dim=-1, dtype=torch.float32
        ).to(device)

    pair_token_idx = torch.arange(t, device=device).repeat_interleave(k)
    pair_expert_idx = selected_experts.reshape(-1)
    pair_router_w = routing_weights.reshape(-1).to(torch.float32)

    if valid_token_mask is not None:
        mask = valid_token_mask.reshape(-1).bool().to(device)
        keep = mask[pair_token_idx]
        pair_token_idx = pair_token_idx[keep]
        pair_expert_idx = pair_expert_idx[keep]
        pair_router_w = pair_router_w[keep]
        selected_experts = selected_experts[mask]
        router_weights_full = router_weights_full[mask]

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

    pairs = RouterPairOutputs(
        selected_experts=selected_experts,
        router_weights_full=router_weights_full,
        pair_token_idx=pair_token_idx,
        pair_expert_idx=pair_expert_idx,
        pair_router_w=pair_router_w,
        expert_offsets=expert_offsets,
        pair_perm=pair_perm,
    )
    return router_logits_full, pairs
