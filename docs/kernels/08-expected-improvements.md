# 08 — Expected Improvements (Performance & Memory)

> **Concern:** projected deltas vs the **loop** baseline in `00-cost-model.md`.  
> **Numbers are projections** unless marked measured. Update this file after EC2
> benches. Implementation status: [README.md](README.md).

Reference: **Qwen3-30B-A3B-class** (E=128, top_k=8, H=2048, I=768, 48 layers),
single L40S-class GPU for layerwise. Wall-clock assumes memory/launch bound
expert MLPs.

## 1. What is implemented

| Backend | Expert compute | Reduce | Default when |
|---|---|---|---|
| `loop` | all tokens × experts | dense `update_pruning_state` | parity only |
| `bmm` | routed grouped PyTorch | routed scatter (PyTorch/Triton) | no Triton runtime |
| `frea` | Triton SwiGLU if eligible else bmm | same | explicit |
| `f2` / `auto` | same as frea | prefer Triton scatter | Triton runtime OK |

## 2. Per-layer expert work (T = 8192)

| Metric | Loop | bmm / FREA |
|---|---|---|
| Expert FLOPs | T × 1.21 GFLOP | T × 75.7 MFLOP (**16× less**) |
| Matmul launches (order) | ~384 | O(active experts) or 1 Triton grid/expert |
| `(E,T,H)` transient | ~8.6 GB | **~0** (pair buffers ~MB) |
| F4 cache | n/a | ~1.2 GB bf16 while layer active |

## 3. Projected wall-clock (observer expert+reduce)

| Path | E=128 | E=256 |
|---|---|---|
| bmm | ~10–15× vs loop | ~15–20× |
| FREA Triton (+ F5) | ~15–25× | ~20–40× |
| + F2 Triton reduce | ~20–30× | ~30–40× |

**Not included:** full HF forward (attention + original MoE), data load, cluster,
save. Observer still recomputes experts after the model forward.

## 4. Memory — layerwise peak (conceptual)

| Component | Loop | bmm/FREA/F2 |
|---|---|---|
| One block weights | ~1.2 GB | ~1.2 GB |
| `(E,T,H)` | **~8.6 / 17 GB** | **0** |
| F4 cache | — | ~1.2 GB |
| Stats | ~1 MB | ~1 MB |
| **Peak (order)** | ~10–18 GB | **~2–4 GB** + system |

## 5. F3 (prune-only metrics)

Default `record_pruning_metrics_only=True` drops merge trackers and ttm/ca_dist
passes on prune. Small absolute memory; less CPU/GPU distance work.

## 6. How to measure (EC2)

```bash
uv pip install -e '.[cuda]'
uv run reap kernels

# Compare backends on same calib subset
for b in loop bmm frea f2; do
  # time + torch.cuda.max_memory_allocated around observe-only runs
  reap prune layerwise --observe-backend $b --observe-only \
    --batches-per-category 8 --batch-size 1 ...
done
```

Paste measured numbers back into this file when available.

## 7. Caveats

- Projections assume L40S-like HBM pressure on small expert matmuls.
- bf16 matmul + fp32 accum: CUDA Triton vs bmm use looser atol in tests.
- Double expert execution (model + observer) caps end-to-end speedup until
  fused into the HF forward.
