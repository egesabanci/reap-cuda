# GPU and Observation Backends

REAP CUDA targets **GPU-first calibration**: saliency and expert activation work
should run on the device holding the MoE weights, not bounce to CPU each batch.

## Device policy

| Data | Device |
| --- | --- |
| Model (full mode) | `device_map="auto"` (CUDA multi-GPU OK) |
| Active block (layerwise) | CUDA if available else CPU |
| Hidden-state replay cache | **CPU** (layerwise memory trade-off) |
| Pruning state / OnlineStatsTracker | Activation device (prefer CUDA) |
| Saved `.pt` / checkpoints | CPU / disk |
| Clustering (scipy/sklearn) | CPU |
| Expert matmuls in observe backends | Same as activations (GPU) |

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

## FREA / F2 reality check

| Name | Implementation today |
| --- | --- |
| FREA | Grouped `F.linear` per active expert; optional `torch.compile` on CUDA |
| F2 | GPU `index_add_` / scatter_reduce + Welford trackers |
| Custom Triton kernels | Design docs under `docs/kernels/`; not required for correctness |

Expect **algorithmic** wins (routed FLOPs, no 8 GB activation blob) immediately;
wall-clock vs a hand-tuned Triton suite is a separate optimization track.

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

- [kernels/README.md](kernels/README.md)
- [observation-and-metrics.md](observation-and-metrics.md)
- [layerwise.md](layerwise.md)
