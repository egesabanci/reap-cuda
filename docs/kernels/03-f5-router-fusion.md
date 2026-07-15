# 03 — Phase 2: F5 Fused Router (softmax + topk + gather-index builder)

> **Concern:** collapse the router stage — `Linear → softmax → topk →
> where/mask → gather` — from ~6 small kernels into **one** Triton kernel that
> also emits the **sorted `(token, expert)` pair index** FREA consumes. F5 is
> the prerequisite data-layout kernel for FREA (Phase 3) and F2 (Phase 4).

## 1. The current router stage

Across the observer the router is computed in at least three places, each
slightly differently:

### Standard observer, non-fused branch (`src/reap/observer.py:377`)
```python
*_, router_logits = output  # (total_tokens, num_experts)   <- already computed by the model
_, selected_experts = torch.topk(router_logits, top_k, dim=-1)
```
Here `router_logits` comes **free** from the model's own forward (`output`
tuple), so no extra `Linear` — but `topk` is a separate kernel, and the
downstream mask `(selected_experts == i)` (`src/reap/pruning_metrics.py:178`)
is rebuilt **E times**.

### Standard observer, fused branch (`src/reap/observer.py:358`)
```python
router_logits = module.router(flat_input)   # (total_tokens, num_experts)  <- REDUNDANT
_, selected_experts = torch.topk(router_logits, top_k, dim=-1)
```
This **re-runs the router `Linear`** even though `output` already contains
router scores (`_, router_scores = output`). Wasted matmul + a launch.

### Layerwise observer (`src/reap/layerwise_observer.py`, `_process_moe_activations`)
```python
router_logits = extract_router_logits(router, flat_input)   # explicit Linear
_, selected_experts = torch.topk(router_logits, top_k, dim=-1)
```

### The mask rebuild in the reduction (`src/reap/pruning_metrics.py:178`)
```python
for i in range(num_experts):
    active_mask = (pruning_batch.selected_experts == i).any(dim=-1).to(device)
```
This constructs `E` boolean masks of shape `(T,)` — **E × T** boolean work
per layer, all to recover information `topk` already had.

## 2. What F5 produces

One Triton kernel, input `router_logits` `(T, E)` (from the model forward — no
redundant `Linear`), output:

| Output | Shape | Meaning |
|---|---|---|
| `selected_experts` | `(T, top_k)` | top-k expert ids per token (API-compatible with existing code) |
| `router_weights` | `(T, top_k)` | softmax-normalized top-k weights (renorm-aware) |
| `pair_token_idx` | `(T*top_k,)` | token index for each routed pair |
| `pair_expert_idx` | `(T*top_k,)` | expert index for each routed pair |
| `pair_perm` | `(T*top_k,)` | permutation sorting pairs **by expert** (so FREA tiles are coalesced) |
| `expert_offsets` | `(E+1,)` | CSR-style start/end of each expert's pair block in the sorted order |

`expert_offsets` turns the downstream per-expert work into a
`segment reduce` / `segment matmul` — no `where`, no mask rebuild.

## 3. The kernel structure (Triton)

```
grid: (cdiv(T, BLOCK_T),)
program p over token block [t0, t0+BLOCK_T):

  1. Load router_logits[t0:t0+BLOCK_T, :]   -> (BLOCK_T, E)   [E up to 256; tile E if needed]
  2. Online softmax over E (numerically stable):
        m = -inf; s = 0
        for e in tiles of E:
            x = logits[t, e]
            m_new = max(m, x.max())
            s = s * exp(m - m_new) + exp(x - m_new).sum()
            m = m_new
        weights = exp(x - m) / s            # (BLOCK_T, E)
  3. Top-k (k=8, small): bitonic select in SRAM over the E candidates.
        emit (expert_id, weight) for the k winners per token.
  4. Renormalize if config.norm_topk_prob:  w_k = w_k / sum(w_k)
  5. Write:
        selected_experts[t, :]      <- k expert ids
        router_weights[t, :]        <- k weights
        pair_token_idx[t*k : t*k+k] <- t (repeated)
        pair_expert_idx[t*k : t*k+k]<- k ids
```

The **sort-by-expert** (`pair_perm`, `expert_offsets`) is a second pass: a
small radix/counting sort over `pair_expert_idx` (E buckets, T*top_k items),
done on-device. This is cheap (one histogram + prefix sum) and is what lets
FREA load each expert's weights once and process all its routed tokens
contiguously.

## 4. Why fuse

| Kernel in the current chain | Replaced by |
|---|---|
| `module.router(flat_input)` (fused branch, redundant) | dropped — use `output`'s logits |
| `F.softmax(router_logits, dim=1)` (in `update_pruning_state` via `routing_weights`) | fused into F5 |
| `torch.topk(router_logits, top_k)` | fused into F5 |
| `torch.gather(routing_weights, 1, selected_experts)` for renorm | fused into F5 |
| `(selected_experts == i).any(-1)` × E (mask rebuild) | eliminated by `expert_offsets` |
| `torch.gather(flat_input, 0, router_indices)` (fused branch) | moved into FREA, which gathers by `pair_token_idx` |

**~6 kernels → 1**, plus the E-mask rebuilds vanish entirely.

## 5. Numerical-stability notes

- **Softmax** uses the online max/sum form (matches `torch.softmax` within
  fp32 tolerance; the existing code does
  `F.softmax(router_logits, dim=1, dtype=torch.float)` (`pruning_metrics.py`)
  so F5 must upcast logits to fp32 before the exp, then downcast the weights).
- **Renormalization** (`src/reap/observer.py` / `pruning_metrics.py`):
  ```python
  # pruning_metrics.py
  if renormalize_router_weights and selected_experts.numel() > 0:
      topk_weights = torch.gather(routing_weights, 1, selected_experts)
      routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)
      routing_weights = torch.clamp(routing_weights, min=eps)
  ```
  F5 applies this **only** when `config.norm_topk_prob` is true (gated by
  `adapter.get_layer_config(...).norm_topk_prob`, see `src/reap/args.py`
  `ObserverArgs.renormalize_router_weights` and `src/reap/main.py` `_setup_observer`
  which already wires `renormalize_router_weights = model.config.norm_topk_prob
  and obs_args.renormalize_router_weights`).

## 6. Fallback (Mac / no-Triton)

Pure-PyTorch `f5_router_pytorch(router_logits, top_k, norm_topk_prob)`:
- `torch.softmax` + `torch.topk` + a `torch.argsort(pair_expert_idx)` for the
  sort-by-expert + `torch.searchsorted` for `expert_offsets`.
- This is the path Phase 1's bmm baseline also uses (it needs the same
  `pair_*` layout). So F5's fallback is effectively part of the Phase-1
  reference.

## 7. Parity contract

`tests/test_kernel_parity_f5.py`:
- F5 (Triton) vs `f5_router_pytorch`: `selected_experts` identical (ids),
  `router_weights` allclose(atol=1e-6), `expert_offsets` identical, on a tiny
  random `(T=512, E=8, k=2)` logits tensor.
- The downstream consumed metrics (via F2) must be unchanged — covered by the
  end-to-end FREA parity test (`04-frea-kernel.md`).

## 8. Expected improvement

- Router stage: **~6 kernels → 1** per layer. Absolute latency is small (a few
  ms for T=2.1 M) but it removes a serialization point and the E×T mask work.
- **The real win is data layout**: `expert_offsets` + `pair_perm` make FREA a
  clean segment-matmul instead of a gather-scatter. Without F5, FREA would
  have to sort inside its own kernel (messy) or rebuild masks (defeating the
  purpose).

## 9. Acceptance

- `f5_router` (Triton) and `f5_router_pytorch` (fallback) produce identical
  `selected_experts` / `expert_offsets` and allclose `router_weights`.
- The fused-branch redundant `module.router(flat_input)` call is removed from
  `observer.py:358` (use `output`'s logits) — small cleanup.
- No `(selected_experts == i)` mask construction remains on the FREA path
  (F2 consumes `expert_offsets` directly).