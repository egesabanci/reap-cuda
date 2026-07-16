# 05 — Phase 4: F2 Saliency Accumulation

> **Status: LANDED (hybrid)**  
> Orchestration: `kernels/f2.py` → `pruning_metrics.update_pruning_state_routed`  
> Triton scatter: `triton_reduce.scatter_pair_stats`  
> Welford means: always `OnlineStatsTracker` (PyTorch) for multi-batch fidelity

> **Concern:** turn pair outputs `(n_pairs, H)` into prune-consumed `(E,)` stats
> without a second Python expert loop or rereading `(E,T,H)`.

## What is fused vs not

| Step | Implementation |
|---|---|
| Pair L2 norms | PyTorch or inside Triton reduce |
| Scatter sum / weighted sum / weight sum | Triton `atomic_add` **or** `index_add_` |
| Max activation | Triton `atomic_max` **or** `scatter_reduce` amax |
| Cross-batch `ean_mean` / `reap` | **Welford in PyTorch** (exact match to historical OnlineStatsTracker) |
| `routed_characteristic_activation` | Optional pair scatter when `compute_routed_ca` |
| `pairwise_expert_frequency` | Still `freq_i + freq_j` from selected_experts (not co-routing) |

Design docs that described full in-kernel Welford are **aspirational**; shipping
code keeps Welford in Python for parity and simplicity.

## Flow

```txt
pair_out, pair_expert_idx, pair_router_w
  → scatter_pair_stats  (Triton|PyTorch)
  → ean_mean_e = ean_sum_e / freq_e   (batch)
  → OnlineStatsTracker.update(mean, freq)
  → max_activations = max(prev, batch_max)
```

## Integration

All non-`loop` backends call `f2_accumulate` after FREA. Backend name `f2`
mainly signals “prefer Triton for FREA + reduce”; reduce also runs after `bmm`.

## Expected impact

| | Effect |
|---|---|
| vs dense reduce | No `(E,T,H)` read; no E× mask rebuild |
| vs PyTorch scatter alone | Triton helps when `n_pairs` large; often modest % of total observe |
| Combined with FREA | Projected total observe **~20–30×** vs loop (unmeasured) |

## Tests

- Routed metrics covered by `test_kernel_parity_bmm.py`
- `tests/test_triton_kernels.py::TestScatterReduce`

## Note on `max_activations`

Historical dense path used `selected_activations.max()` (raw max over n_e×H).
Routed path uses max over pair-wise amax(H) then scatter amax — same family of
statistic for ranking.
