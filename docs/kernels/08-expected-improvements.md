# 08 — Expected Improvements (Performance & Memory)

> **Concern:** the concrete, quantified deltas the kernel suite delivers
> against the loop baseline (`00-cost-model.md` §7). All numbers are for
> **Qwen3-30B-A3B** (E=128, top_k=8, H=2048, I=768, 48 layers) on a single
> **NVIDIA L40S (46 GB, ~864 GB/s HBM)**, unless noted. 256-expert variants
> (Qwen3.5/3.6 large) are called out where the factor changes.

## 1. Performance — observer/calibration phase

The observer forward is the bottleneck (the prune/merge selection itself is
trivial: a per-layer `torch.topk` over an `(E,)` vector, `src/reap/prune.py:79`).

### Per-layer, per-forward (T = 8192 tokens)

| Metric | Loop (current) | bmm (Ph.1) | FREA (Ph.3) | F2 (Ph.4) |
|---|---|---|---|---|
| Expert matmul launches | 384 | 3 (grouped) | **1 fused** | 1 fused |
| Reduce-loop kernels | ~6 | ~6 (scatter) | ~6 | **0** (fused) |
| Total kernels / layer | ~390 | ~9 | ~7 | **~1** |
| Expert FLOPs | T × 1.21 GFLOP | T × 75.7 MFLOP | T × 75.7 MFLOP | T × 75.7 MFLOP |
| FLOP waste vs routed | 16× (E=128) | 1× | 1× | 1× |
| Router-stage kernels | ~6 | ~6 | 1 (F5) | 1 (F5) |

### Per whole-calibration run (1024 batches, T_total ≈ 8.4 M tokens)

| Metric | Loop | bmm | FREA | F2 |
|---|---|---|---|---|
| Expert matmul launches | ~18.9 M | ~147 k | **~49 k** | ~49 k |
| Expert FLOPs | ~153 PFLOP | ~9.8 PFLOP | ~9.8 PFLOP | ~9.8 PFLOP |
| Launch tax (5 µs/launch) | ~94 s | ~0.7 s | ~0.25 s | ~0.25 s |

### Wall-clock speedup (observer-only, vs loop = 1×)

| Backend | E=128 (Qwen3-30B-A3B) | E=256 (Qwen3.5/3.6 large) |
|---|---|---|
| bmm (Phase 1) | ~10–15× | ~15–20× |
| + F5 (Phase 2) | small extra (router fuse) | small extra |
| FREA (Phase 3) | ~15–25× | ~20–40× |
| F2 (Phase 4) | ~20–30× | ~30–40× |

**Why the E=256 column is higher**: the FLOP waste ratio is `E / top_k` =
128/8 = 16× for E=128, but 256/8 = 32× for E=256. The memory benefit scales
similarly (the eliminated `(E,T,H)` tensor doubles).

**Why the speedup exceeds the FLOP ratio**: the L40S is **memory-bandwidth
bound** for these small expert matmuls, not compute-bound. Eliminating the
8.6 GB `(E,T,H)` HBM round-trip (write in the compute loop, read in the reduce
loop) is worth more than the FLOP cut. The ~864 GB/s HBM means an 8.6 GB
transient costs ~10 ms just in bandwidth, per layer, per forward — that
vanishes entirely under FREA/F2.

## 2. Memory — peak VRAM during calibration

### Standard observer (whole model on GPU)

The standard path is **not** the target for 30B+ on a single L40S (it needs
the whole model, ~60 GB bf16 > 46 GB). Numbers here are for completeness;
the real target is layerwise (below).

| Per-layer transient | Loop | bmm | FREA/F2 |
|---|---|---|---|
| `(E, T, H)` activation tensor | 8.6 GB (E=128) / 17 GB (E=256) | ~0 (grouped, ~MB) | **0** |
| Stat buffers `(E,)` + `(E,H)` | ~1 MB | ~1 MB | ~1 MB |
| F4 weight cache (bf16) | n/a | 1.2 GB/layer | 1.2 GB/layer |

### Layerwise observer (one block on GPU — the real target)

This is the memory story that **enables** 30B+ on a single 46 GB GPU. With
layerwise (`src/reap/layerwise_observer.py`), only one decoder block is on GPU
at a time; hidden states are cached on CPU between blocks.

| Component | Loop (current) | FREA/F2 |
|---|---|---|
| One decoder block weights (bf16) | ~1.2 GB | ~1.2 GB |
| Hidden-state CPU cache (all blocks) | on CPU (not VRAM) | on CPU |
| Per-layer `(E,T,H)` activation transient | **8.6 GB** (E=128) / **17 GB** (E=256) | **0** |
| F4 weight cache | n/a | 1.2 GB (freed after block) |
| Stat buffers | ~1 MB | ~1 MB |
| **Peak VRAM (one block)** | ~10 GB (E=128) / ~18 GB (E=256) | **~2.4 GB** (E=128) / **~3.6 GB** (E=256) |

**The 8.6 GB activation transient is the dominant VRAM consumer in the current
layerwise path.** FREA removes it. Result: Qwen3-30B-A3B calibration on a
single L40S has **~20× headroom** (peak ~2.4 GB vs 46 GB). For a 256-expert
model, the current path's 17 GB transient is the first thing to OOM on a
larger batch; FREA keeps peak ~3.6 GB.

### Merging-criteria trackers (F3, prune path only)

With `record_pruning_metrics_only=True` (Phase 0 default flip,
`01-f3-dead-metric-audit.md`), the prune path drops the 4 merging-criteria
`OnlineStatsTracker` allocations per layer:

| Tracker | Shape | Saved on prune path |
|---|---|---|
| `ttm_similarity_matrix` | (E, E) | ~64 KB (E=128) |
| `characteristic_activation` | (E, H) | 1 MB |
| `online_characteristic_activation_dist` | (E, E) | ~64 KB |
| `router_logit_similiarity` | (E, E) | ~64 KB |

Small in absolute terms, but these also carry **Welford update FLOPs** and the
`ttm_online` / `ca_dist_online` distance-pass kernels in `observer.py`'s
`_hook_factory` (the code after `update_pruning_state` at
`src/reap/observer.py` ~line 390+), which F3 drops entirely on the prune path.

## 3. The combined story (what lands when)

| Phase | Lands | Perf delta | Memory delta | Mac-runnable |
|---|---|---|---|---|
| 0 (F3) | default flip + contract test | small (drops dead trackers) | -4 trackers/layer | ✅ |
| 1 (bmm) | pure-PyTorch grouped bmm | **~10–15×** observer | **-8.6 GB transient** | ✅ (MPS) |
| 2 (F5) | Triton router fuse | small + enables FREA coalescing | removes E×T masks | ❌ (fallback ✅) |
| 3 (FREA) | Triton fused routed activation | **~15–25×** observer | **0 transient** | ❌ (fallback ✅) |
| 4 (F2) | Triton fused saliency | **~20–30×** observer | 0 + no reduce passes | ❌ (fallback ✅) |
| 5 (F4) | weight stack cache | enables FREA tile coalescing | +1.2 GB/layer (freed) | ✅ |

### Cumulative end-state (Phase 4, layerwise, Qwen3-30B-A3B, single L40S)

- **Observer wall-clock**: ~20–30× faster than today.
- **Peak VRAM**: ~2.4 GB (one block) vs ~10 GB today → fits 30B on a single
  L40S with ~20× headroom; fits a 256-expert model without OOM.
- **Launches**: ~49 k vs ~18.9 M per calibration run.
- **Expert FLOPs**: ~9.8 PFLOP (routed) vs ~153 PFLOP (all-pairs) — 16× less
  (32× for E=256).

## 4. Caveats & assumptions

- **Wall-clock numbers are projections**, not measurements. The FLOP and
  launch counts are exact (from `00-cost-model.md`); the wall-clock multipliers
  assume the L40S is memory-bound for these small matmuls, which the Phase-1
  bench (`07-validation-strategy.md` §5) will confirm. If a phase under- or
  over-performs, update this table with the measured value.
- **bf16 vs fp32**: the observer does fp32 accumulation
  (`pruning_metrics.py` uses fp64 for `ean_sum`/`weighted_ean_sum` and fp32
  for `reap`/`ean_mean` via `OnlineStatsTracker`). FREA/F2 preserve these
  dtypes for the accumulators; the matmuls run in bf16 (matching the model
  weights) with fp32 accumulation. This matches the existing loop's behavior
  (HF `Qwen3MoeMLP` runs in the model's dtype).
- **`ean_ca` path**: when `prune_method == "ean_ca"`, F2 also produces the
  `(E, H)` `routed_characteristic_activation` (1 MB atomic-add buffer). This
  adds ~1 MB HBM traffic per layer but no FLOPs; the speedup is unchanged.
- **Merge path is unaffected**: F2 is prune-path only. Merge keeps the
  merging-criteria metrics (it forces `record_pruning_metrics_only=False`,
  `merge_pipeline.py` `main()`). A future **layerwise merge** (issue #3) would
  need FREA for the compute but a *separate* kernel for the merging-criteria
  reductions (ttm/ca_dist/characteristic_activation) — out of scope for this
  kernel suite.
- **Fused Qwen3.5/3.6** (issue #4): FREA's layout-agnostic design (via F4
  stacked weights) means the kernel works for fused layouts **once the adapter
  detects them**. The adapter change (runtime fused detection) is issue #4's
  scope, not a kernel concern.