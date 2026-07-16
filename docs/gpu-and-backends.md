# GPU and Observation Backends

REAP CUDA targets **GPU-first calibration**: saliency and expert activation work
should run on the device holding the MoE weights, not bounce to CPU each batch.

This page is about **activation / kernel device policy** and observe backends.
**Where model weights are loaded and saved** (host vs GPU vs disk offload) is
documented separately: **[residency.md](residency.md)** (`--residency`).

## Device policy

| Data | Device |
| --- | --- |
| Model weights | Per `--residency` → `plan_load` ([residency.md](residency.md)): typically `device_map="auto"` (+ optional offload) or `"cpu"` |
| Active block (layerwise schedule) | CUDA if available else CPU |
| Hidden-state replay cache | **CPU** (layerwise memory trade-off) |
| Pruning state / OnlineStatsTracker | Activation device (prefer CUDA) |
| Saved `.pt` / checkpoints | CPU / disk (weights via stream save from GPU when residency allows) |
| Clustering (scipy/sklearn) | CPU |
| Expert matmuls in observe backends | Same as activations (GPU) |

### Triton hardware gates (model / SKU agnostic)

- `prefer_triton_for` checks CUDA + dtype + optional min numel.
- **FREA** auto-scales tiles to device shared mem (default + Ampere/Ada **opt-in**
  ~164 KiB when available so 128×128 can fit on L4/T4).
- **`--frea-backend auto`** (default): one-shot **profitability probe** (Triton vs
  cuBLAS PyTorch) per shape; memoize winner. Force with `triton` / `pytorch`,
  or env `REAP_FREA_BACKEND` / `REAP_FREA_PROBE=0` for static tile-floor gate.
- Shared-mem failures are memoized; end of observe logs **Triton usage summary**.
- F2 scatter accumulates **fp64**; slightly higher `num_warps` on large H.

Native routers (sigmoid + `expert_bias`, etc.) use
`f5_router_from_module` when `prefers_native_router` detects structural signals —
not a hard-coded model list.

## Backend selection

```python
# reap.kernels.backend.select_observe_backend
auto  -> "f2" if (triton importable and torch.cuda.is_available()) else "bmm"
bmm   -> grouped routed PyTorch
frea  -> FREA path (compile optional)
f2    -> FREA compute + routed reductions
loop  -> legacy / parity
```

Override:

```bash
reap prune layerwise --observe-backend bmm
# or
export REAP_OBSERVE_BACKEND=bmm
```

## Pipeline of a routed observe step

```txt
hidden flat_input (T, H)
  -> router logits (T, E)          # F5 / extract_router_logits
  -> top-k + pair CSR indices      # F5
  -> stacked W_gate/up/down        # F4 (layout-normalized)
  -> pair outputs (n_pairs, H)     # grouped bmm / FREA
  -> scatter saliency              # F2 / update_pruning_state_routed
```

No full expert loop over all tokens; no persistent `(E, T, H)` on prune path.

## F4 weight cache

- Non-fused: `torch.stack` of Linear weights → `(E, I, H)` / `(E, H, I)`
- Fused linear (Qwen/LFM2): split `gate_up_proj` on dim 1
- Fused bmm (Llama4): transpose native `(E, H, 2I)` into Linear form
- Cached by `id(moe)`; free after layerwise block (and on observer close)

## F5 router

- Softmax in fp32 for stability
- Top-k on probabilities (monotone vs logits)
- Optional renorm when `renormalize_router_weights` and model `norm_topk_prob`
- Emits expert-sorted pair indices + CSR `expert_offsets` for coalesced work

## FREA / F2 / Triton

| Name | Implementation |
| --- | --- |
| F5 softmax | Triton online row-softmax when `E ≤ 1024` and CUDA; else `F.softmax` |
| FREA | Triton per-expert tiled SwiGLU (`triton_frea.py`) when `H,I ≥ 16` + SiLU; else grouped `F.linear` |
| F2 | Triton atomic scatter of norms/weights (`triton_reduce.py`); Welford means stay PyTorch |
| Force PyTorch | `REAP_DISABLE_TRITON=1` or `--observe-backend bmm` |

```bash
reap kernels   # print Triton package/runtime + auto backend
```

All Triton launches are **optional**: failures or unsupported shapes fall back
to pure PyTorch automatically (debug log: `Triton … fallback → PyTorch`).

## Why double compute?

HF forward already runs the MoE block. The observer **recomputes** routed
expert outputs for metrics. Invasive fusion into the model forward is a future
optimization, not the current API surface.

## Recommended EC2 settings

```bash
# First bring-up on L40S 46GB
reap prune layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --batches-per-category 8 \
  --batch-size 1 \
  --observe-backend bmm \
  --observe-only

# Production-ish
reap prune layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend auto \
  --batches-per-category 256
```

Watch `torch.cuda.max_memory_allocated()`: peak should track **one block + F4
stack**, not an extra multi-GB `(E,T,H)` per layer.

## Related

- [residency.md](residency.md) — weight load/save (not saliency device)
- [kernels/README.md](kernels/README.md)
- [observation-and-metrics.md](observation-and-metrics.md)
- [layerwise.md](layerwise.md)
