# GPU and Observation Backends

REAP CUDA targets **GPU-first calibration**: saliency and expert activation work
should run on the device holding the MoE weights, not bounce to CPU each batch.

This page is about **activation / kernel device policy** and observe backends.
**Where model weights are loaded and saved** is
[residency.md](residency.md). **FREA throughput vs memory** (probe, CLI) is
[frea-throughput.md](frea-throughput.md).

## Device policy

| Data | Device |
| --- | --- |
| Model weights | Per `--residency` → `plan_load` ([residency.md](residency.md)) |
| Active block (layerwise schedule) | CUDA if available else CPU |
| Hidden-state replay cache | **CPU** (layerwise memory trade-off) |
| Pruning state / OnlineStatsTracker | Activation device (prefer CUDA) |
| Saved `.pt` / checkpoints | CPU / disk (stream save when residency allows) |
| Clustering (scipy/sklearn) | CPU |
| Expert matmuls in observe backends | Same as activations (GPU) |

### Triton hardware gates (model / SKU agnostic)

- `prefer_triton_for` — CUDA + dtype + optional min numel.
- **FREA tiles** — auto-scale to live `shared_memory_per_block` and, when larger,
  **`shared_memory_per_block_optin`**. Typical measured limits:
  - **L4/T4-class:** default **48 KiB**, opt-in **99 KiB** → max FREA tiles often
    **128×64** for large H/I; **128×128 does not fit**.
  - **A100/L40S-class:** opt-in often **~164 KiB** → **128×128** can fit.
- Fail once on SM OOM → disable opt-in / memoize → no launch spam.
- **`--frea-backend auto`** — profitability **probe** (Triton vs cuBLAS) per
  shape; on L4 the probe correctly picks PyTorch for LFM2-scale shapes. See
  [frea-throughput.md](frea-throughput.md).
- End of observe: INFO **Triton usage summary** (`frea: N Triton / M PyTorch; …`).
- **F2** accumulates saliency sums in **fp64** (matches docs / PyTorch path).

Native routers (sigmoid + `expert_bias`, etc.) use `f5_router_from_module` when
`prefers_native_router` detects **structural** signals (bias buffer,
`use_expert_bias`, adapter flag / name) — not a SKU list.

## Backend selection (coarse)

```python
# reap.kernels.backend.select_observe_backend
auto  -> "f2" if (triton importable and torch.cuda.is_available()) else "bmm"
bmm   -> grouped routed PyTorch
frea  -> FREA path (+ --frea-backend policy)
f2    -> FREA + F2 scatter reduce
loop  -> legacy / parity
```

```bash
reap prune layerwise --observe-backend bmm
export REAP_OBSERVE_BACKEND=bmm
export REAP_DISABLE_TRITON=1   # all Triton kernels off
```

## FREA sub-backend (fine)

When the coarse backend uses FREA (`auto`→`f2`, `frea`, or `f2`):

| `--frea-backend` | Meaning |
| --- | --- |
| `auto` (default) | Probe once; memoize faster of Triton vs PyTorch |
| `triton` | Force Triton when tiles fit |
| `pytorch` | Force cuBLAS grouped path |

```bash
reap prune full --observe-backend auto --frea-backend auto    # default
reap prune full --frea-backend pytorch                        # L4 throughput
reap prune full --frea-backend triton                         # force kernel
export REAP_FREA_BACKEND=pytorch
export REAP_FREA_PROBE=0   # auto without timing; static tile floor instead
```

Details, env vars, and the L4 tradeoff story: **[frea-throughput.md](frea-throughput.md)**.

## Pipeline of a routed observe step

```txt
hidden flat_input (T, H)
  -> router                          # F5 softmax+topk OR native module router
  -> top-k pairs + CSR indices
  -> stacked W_gate/up/down          # F4 (layout-normalized; max 1 cache entry)
  -> pair outputs (n_pairs, H)       # FREA (Triton|PyTorch per --frea-backend)
  -> scatter saliency                # F2 fp64 reduce
```

No full expert loop over all tokens; no persistent `(E, T, H)` on prune path.

## F4 weight cache

- Non-fused: `torch.stack` of Linear weights → `(E, I, H)` / `(E, H, I)`
- Fused linear (Qwen/LFM2): split `gate_up_proj` on dim 1
- Fused bmm (Llama4): transpose native `(E, H, 2I)` into Linear form
- **At most one MoE** in `_STACK_CACHE` (full-observer OOM guard); free after
  each `observe_moe_batch` / layerwise block / observer close

## F5 / native router

| Case | Path |
| --- | --- |
| Softmax MoEs (Qwen, Mixtral, …) | `f5_router` — Triton softmax when eligible + PyTorch topk/CSR |
| Non-softmax / bias routers (e.g. LFM2) | `f5_router_from_module` — call model router; rebuild CSR |

## FREA / F2 / Triton

| Name | Implementation |
| --- | --- |
| F5 softmax | Triton online row-softmax when `E ≤ 1024` and CUDA; else `F.softmax` |
| FREA | Triton tiled SwiGLU when supported; else grouped `F.linear`; policy via `--frea-backend` |
| F2 | Triton fp64 atomic scatter; Welford means stay PyTorch |
| Force all PyTorch | `REAP_DISABLE_TRITON=1` or `--observe-backend bmm` |

```bash
reap kernels   # package/runtime + auto coarse backend (not per-kernel probe)
```

Fallbacks log **WARNING** once per component, then DEBUG. Run summary at INFO.

## Why double compute?

HF forward already runs the MoE block. The observer **recomputes** routed
expert outputs for metrics. Invasive fusion into the model forward is future work.

## Recommended EC2 settings

```bash
# Bring-up (safe, pure PyTorch observe math)
reap prune layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --observe-backend bmm \
  --batches-per-category 8 --batch-size 1 --observe-only

# Production on L40S-class (large SM): auto FREA often picks Triton
reap prune layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --observe-backend auto \
  --frea-backend auto \
  --residency auto \
  --compression-ratio 0.5

# L4 / g6.xlarge, small MoE that fits VRAM: GPU residency + FREA auto (probe → pytorch)
reap prune full \
  --model LiquidAI/LFM2.5-8B-A1B \
  --residency gpu_full \
  --observe-backend auto \
  --frea-backend auto \
  --dataset-path /data/datasets/evol-codealpaca-calib-200 \
  --artifacts-dir /data/reap-artifacts

# Force Triton FREA only if you need to experiment (slower on L4; opt-in → ~128×64 tiles)
# REAP_FREA_BACKEND=triton reap prune full ...
```

## Related

- [frea-throughput.md](frea-throughput.md) — probe, tiles, env vars
- [residency.md](residency.md) — weight load/save
- [calibration.md](calibration.md) — offline `--dataset-path`, composite specs
- [kernels/README.md](kernels/README.md)
- [observation-and-metrics.md](observation-and-metrics.md)
- [layerwise.md](layerwise.md)
- Field reports: `run-findings.md` … `run-findings-4.md` (repo root)
