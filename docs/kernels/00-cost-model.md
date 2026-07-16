# 00 — Cost Model: Loop Baseline (Historical Bottleneck)

> **Concern:** quantify the **legacy loop observer** so kernel wins are measured
> against a fixed baseline.  
> **Current code:** default observe path is **not** this loop. Use
> `--observe-backend loop` only for parity. Production defaults:
> `auto` → `bmm` (no Triton) or `f2` (Triton runtime). See [README.md](README.md).

## 1. Two observer entrypoints (still true)

| Observer | File | CLI | Memory mode |
|---|---|---|---|
| Standard | `observer.py` (`MoETransformerObserver`) | `reap prune full` / `reap merge full` | whole model on GPU |
| Layerwise | `layerwise_observer.py` | `reap prune layerwise` / `reap merge layerwise` | one decoder block on GPU |

Both call **`reap.kernels.observe.observe_moe_batch`**. Backend `loop` exercises
the dense/legacy branch; `bmm` / `frea` / `f2` use routed-only paths.

## 2. Legacy loop bottleneck (what kernels replace)

When `backend == "loop"` (or older pre-kernel code), experts are executed in a
Python loop. For **non-fused** ModuleList experts:

```python
# Conceptual legacy path (still available via --observe-backend loop)
for idx, expert in enumerate(module.experts):
    activations[idx] = expert(flat_input)  # ALL tokens, every expert
```

That is **E × T × (3 matmuls)** work even though saliency only needs
**top_k × T** routed pairs. Masking happens later in reduce.

**Fused** layouts historically also materialize large activation buffers; current
routed backends never build full `(E, T, H)` for prune-only observe.

## 3. Quantified cost (Qwen3-30B-A3B-class)

Per layer, per forward with T tokens (e.g. batch 4 × seq 2048 → T = 8192):

| Quantity | Formula | Value (E=128, top_k=8, H=2048, I=768) |
|---|---|---|
| Expert matmul launches / layer | E × 3 | **384** |
| Expert matmul launches / forward (48 layers) | 384 × 48 | **18,432** |
| Expert FLOPs / layer | T × E × 3 × 2 × H × I | T × ~1.21 GFLOP |
| **Routed** expert FLOPs / layer | T × top_k × 3 × 2 × H × I | T × ~75.7 MFLOP |
| **Waste ratio** | E / top_k | **16×** (32× for E=256) |
| `(E,T,H)` fp32 size, T=8192 | E × T × H × 4 | **~8.6 GB / layer** |

Across a large calib run, launch tax alone can be minutes of overhead on L40S
before counting matmul time.

## 4. What the modern path does instead

| Legacy | Current (`bmm` / `frea` / `f2`) |
|---|---|
| All experts × all tokens | F5 pairs only (`T × top_k`) |
| `(E, T, H)` materialize | `(n_pairs, H)` only |
| Python reduce over E masks | F2 scatter + Welford |
| Scattered ModuleList reads | F4 stacked weights |

See [README.md](README.md) impact table and [08-expected-improvements.md](08-expected-improvements.md).

## 5. Saliency reductions

Legacy dense path: `pruning_metrics.update_pruning_state` over `(E,T,H)`.

Current routed path: `update_pruning_state_routed` over pair tensors
(optionally via Triton scatter in `triton_reduce.py`).

Consumed prune keys remain routed-only (`frequency`, `ean_*`, `reap`,
`max_activations`, …). Merge criteria are separate (F3).

## 6. Baseline numbers used by the rest of this guide

| Metric | Loop (baseline) |
|---|---|
| Expert matmul launches / forward | 18,432 |
| Expert FLOPs / forward | T × 58.1 TFLOP (E=128) |
| Peak activation transient / layer | ~8.6 GB (T=8192, E=128) |
| Expert loops / layer | 2 (compute + reduce) |

Improvement tables in `08-expected-improvements.md` are deltas against these
**loop** numbers, not against an already-routed baseline.
