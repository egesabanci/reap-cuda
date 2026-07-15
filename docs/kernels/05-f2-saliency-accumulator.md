# 05 — Phase 4: F2 Fused Online Saliency Accumulator

> **Concern:** generalize FREA (Phase 3) so that **one kernel per layer**
> produces **every metric `prune.py` consumes**, with the reductions fused
> in-register. F2 = FREA + the full `update_pruning_state` reduction set, so
> the post-kernel Python reduction passes disappear entirely.

## 1. What F2 removes on top of FREA

FREA (Phase 3) fuses the **compute** (gate/up/down + SiLU) and the **norm**
(`ean_norm = ||y||_2`), but still emits per-pair scalars that a follow-up pass
reduces. F2 moves the **entire** `update_pruning_state` reduction set
(`src/reap/pruning_metrics.py:133`) into the kernel, so the observer emits the
final `(E,)` / `(E,H)` state buffers directly — no `index_add_`, no
`OnlineStatsTracker.update`, no Python loop.

## 2. The full reduction set F2 must produce

From `src/reap/pruning_metrics.py:178` (the reduce loop) +
`src/reap/pruning_metrics.py:22` (`initialize_pruning_state`):

| State key | Dtype/Shape | Per-pair contribution | Aggregation |
|---|---|---|---|
| `total_tokens` | scalar long | `+ num_tokens` (valid tokens) | add (once/batch) |
| `expert_frequency` | (E,) long | `+ 1` per routed pair | atomic add |
| `pairwise_expert_frequency` | (E, E) long | co-routing counts | atomic add (see §4) |
| `ean_sum` | (E,) fp64 | `+ ean_norm` | atomic add |
| `ean_mean` | OnlineStatsTracker (E,) | `ean_norm.mean()` per expert | Welford (see §3) |
| `reap` | OnlineStatsTracker (E,) | `(ean_norm * w).mean()` per expert | Welford (see §3) |
| `weighted_ean_sum` | (E,) fp64 | `+ ean_norm * w` | atomic add |
| `weighted_expert_frequency_sum` | (E,) fp64 | `+ w` | atomic add |
| `max_activations` | (E,) fp32 | `max(|y|)` over pairs×H | atomic max |
| `routed_characteristic_activation` | (E,H) fp32 *(ean_ca only)* | `+ y` then `/= freq` | atomic add + deferred div |

## 3. The Welford subtlety (`ean_mean`, `reap`)

`ean_mean` and `reap` are **not** simple sums — they are `OnlineStatsTracker`
(`src/reap/metrics.py:218`) running **means** over *batches*, with per-batch
`new_count = expert_frequency` (the count of routed tokens for that expert in
this batch):

```python
# pruning_metrics.py:205
layer_state["ean_mean"].update(ean_mean.to("cpu"), pruning_batch.expert_frequency.to("cpu"))
layer_state["reap"].update(reap.to("cpu"), pruning_batch.expert_frequency.to("cpu"))
```

And `OnlineStatsTracker.update` (`src/reap/metrics.py:258`) is a Welford /
Kahan-summed running mean over the *sequence of per-batch means*. So F2 must:

1. For the current batch, compute the per-expert **batch mean**
   `ean_mean_e = sum_pairs(ean_norm) / freq_e` (and
   `reap_e = sum_pairs(ean_norm * w) / freq_e`).
2. Update the cross-batch Welford mean using `freq_e` as the batch count.

F2 keeps the `(E,)` Welford state (`mean`, `count`, `mean_compensation` —
exactly `OnlineStatsTracker`'s fields, `src/reap/metrics.py:246–256`) in HBM
and does the Welford update in the kernel's epilogue (one program per expert,
after streaming all pairs). This reproduces `OnlineStatsTracker.update`
bit-for-bit.

> **Why this matters**: `ean_mean` and `reap` are the two consumed keys that are
> *means over batch-means*, not raw sums. Getting the Welford math identical is
> the trickiest part of the parity test.

## 4. `pairwise_expert_frequency` (E, E)

This counts, per token, how often each expert pair co-routes. The existing
code (`pruning_metrics.py` `_prepare_pruning_batch`) computes it from
`selected_experts` (T, top_k) — an `(E, E)` histogram of co-occurrence. This is
**independent of the expert activations**, so F2 handles it in a small separate
kernel (or a fused epilogue) that consumes `selected_experts` directly:

```python
# per token t: for i in range(top_k): for j in range(top_k): pair_freq[s[i], s[j]] += 1
```

top_k=8 → 64 pairs/token. A Triton kernel over tokens with an (E,E) atomic-add
histogram handles this cheaply. It does **not** need activations, so it can run
in parallel with FREA's compute.

## 5. Method-gated computation

`prune.py` reads exactly one saliency key per run (`prune_args.prune_method`,
`src/reap/prune.py:62–73`). F2 computes the **union** cheaply:
- The `(E,)` accumulators (`ean_sum`, `weighted_ean_sum`, `reap`, `ean_mean`,
  `max_activations`, `weighted_expert_frequency_sum`, `expert_frequency`) are
  all tiny (E scalars) — computing all of them costs ~nothing extra over
  computing one.
- The `(E,H)` `routed_characteristic_activation` (1 MB) is only needed for
  `ean_ca`; gate it behind `prune_method == "ean_ca"` to avoid the atomic-add
  bandwidth when unused.

A `--compute-routed-ca` flag (default = method == ean_ca) controls this.

## 6. Kernel shape

F2 = FREA with the epilogue extended:

```
program p (expert e, pair block [s, t)):
    # ... FREA's compute: load W_gate/up/down once, stream pairs, get y, ean_norm ...
    # Accumulate in-register:
    freq_e, ean_sum_e, wean_e, reap_e, wfreq_e (scalars)
    max_e (scalar abs-max over y)
    ca_e[H_blk] (only if ean_ca)
    for pair: ... (same as FREA)
    # Epilogue: one Welford update for ean_mean / reap using freq_e as batch count
    welford_update(ean_mean_tracker[e], ean_sum_e / freq_e, freq_e)
    welford_update(reap_tracker[e],     reap_e / freq_e,   freq_e)
    # Atomic scatter of the sums:
    atomic_add(expert_frequency[e], freq_e)
    atomic_add(ean_sum[e], ean_sum_e)
    atomic_add(weighted_ean_sum[e], wean_e)
    atomic_add(weighted_expert_frequency_sum[e], wfreq_e)
    atomic_max(max_activations[e], max_e)
    if ean_ca: atomic_add(routed_characteristic_activation[e,:], ca_e)
```

`pairwise_expert_frequency` is a separate small kernel (§4).

## 7. What the observer becomes after F2

`src/reap/observer.py::_hook_factory` non-fused branch collapses to:

```python
# After F5 + F2:
f2_observe(                          # one Triton kernel per layer
    self.state[layer_number],
    flat_input,                      # (T, H)
    f5_router_outputs,               # selected_experts, expert_offsets, pair_*, router_weights
    W_gate, W_up, W_down,             # stacked weights from F4
    layer_cfg,                       # num_experts, top_k, norm_topk_prob
    compute_routed_ca = (prune_method == "ean_ca"),
)
# No update_pruning_state call. No (E,T,H) tensor. No Python reduce loop.
```

`update_pruning_state` / `initialize_pruning_state` remain for the **fallback**
(loop/bmm) path and for the merge path (which needs the merging-criteria
metrics F2 does not produce — those are a separate concern, see
`01-f3-dead-metric-audit.md`).

## 8. Fallback

`f2_observe_pytorch` = the grouped-bmm (Phase 1) + a Python epilogue that
calls `update_pruning_state_routed` (the pair-tensor variant from
`02-bmm-baseline.md` §4). On Mac, this is the path that runs. F2-Triton must
match it.

## 9. Parity contract

`tests/test_kernel_parity_f2.py` (EC2):
- F2-Triton vs `f2_observe_pytorch` on a tiny Qwen3-MoE: **all** consumed
  metrics bit-for-bit, including the Welford `ean_mean` / `reap` after
  multiple batches (the cross-batch mean must match — test with ≥3 batches of
  different sizes to exercise the Welford path).
- `pairwise_expert_frequency` (E,E) identical.

## 10. Expected improvement (incremental over FREA)

| Metric | FREA + Python reduce | F2 |
|---|---|---|
| Post-compute Python reduce kernels / layer | ~6 (`index_add_`, `OnlineStatsTracker.update`, mask) | **0** |
| `(E,)` scatter passes / layer | ~6 | fused |
| `ean_mean`/`reap` Welford | Python (with CPU `.to("cpu")` round-trips — see `pruning_metrics.py:205`) | in-kernel (no CPU hop) |

The Python reduce passes are a measurable fraction of layer time today (each
involves a `.to("cpu")` round-trip, `pruning_metrics.py:204–213`). F2 removes
them. Combined with FREA, F2 delivers the **~20–30×** observer speedup
(`08-expected-improvements.md`).

## 11. Acceptance

- F2-Triton matches the PyTorch fallback on all consumed metrics across
  multi-batch calibration (Welford fidelity).
- `update_pruning_state` is no longer called on the `--observe-backend f2`
  path.
- `pairwise_expert_frequency` produced by the separate kernel matches the
  existing `_prepare_pruning_batch` output.
- The merge path (which forces `record_pruning_metrics_only=False`) is
  unaffected — it still uses the standard observer with merging-criteria
  metrics (F2 is prune-path only).