# Layerwise Mode

Layerwise calibration enables 30B+ MoE observation on a **single mid-size GPU**
(e.g. L40S 46 GB) by keeping only one decoder block on device at a time.

## Components

| Piece | Role |
| --- | --- |
| `LayerwiseMoEObserver` | Orchestrates block loop |
| `ReplayCache` | Stores first-block inputs / kwargs on CPU |
| `extract_model_components` / `find_decoder_blocks` | Locate blocks |
| `observe_moe_batch` | Shared metrics path |

## Algorithm

1. **Seed replay cache**  
   Register a pre-hook on block 0; run full model far enough to capture
   `hidden_states` (+ masks) entering the first block; raise a sentinel to stop.
   Tensors stored on **CPU**.

2. **For each block index**  
   - Offload previous block to CPU  
   - Load current block to CUDA  
   - For each cached batch: materialize inputs on GPU, forward the block  
   - On MoE blocks: hook captures MoE input → `observe_moe_batch`  
   - Store block outputs on CPU as next-layer inputs  
   - Free F4 weight cache for the MoE module  

3. **Report / save** state (CPU for disk)

4. **Prune path only:** delete CPU model, reload with `device_map="auto"`, slice,
   save.

## Memory profile (order of magnitude)

| Item | Layerwise | Full |
| --- | --- | --- |
| Weights on GPU | ~1 block (~1–2 GB bf16 for 30B-class) | entire model (~60 GB) |
| Activation transient (routed backends) | MB-scale stats | same, but all layers present |
| Activation transient (old dense loop) | up to multi-GB per layer | multi-GB × concurrency |
| F4 cache | ~1.2 GB/layer while active | same while hooked layer runs |
| Replay cache | CPU RAM (all batches × seq × hidden) | N/A |

Use `--batch-group-size` to limit how many batches are cached at once (CPU RAM).

## CLI

```bash
reap prune layerwise [options]
reap merge layerwise [options]
```

Layerwise-specific flags:

| Flag | Meaning |
| --- | --- |
| `--batch-group-size` | Process groups of batches through all blocks |
| `--save-intermediate` | Dump per-block metrics while running |
| `--low-cpu-mem` / `--no-low-cpu-mem` | HF load flag for CPU model |

## When to use which mode

| Situation | Prefer |
| --- | --- |
| Single L40S / 48 GB, 30B MoE | **layerwise** observe |
| Multi-GPU 80 GB+ | full (simpler, faster wall-clock) |
| Merge on large model | layerwise observe + CPU merge weights |
| Debugging hooks | full + tiny model |

## Limitations

- Hidden states on CPU add PCIe / host bandwidth cost.
- Prune mutate still needs a full-model GPU load (or multi-GPU map).
- Block `forward` signatures differ across families; kwargs are filtered by
  inspect.signature and forced `use_cache=False`.
- Dense / hybrid layers (LFM2 early layers) forward without MoE metrics.

## Related

- [pipeline.md](pipeline.md)
- [gpu-and-backends.md](gpu-and-backends.md)
- [architecture.md](architecture.md)
