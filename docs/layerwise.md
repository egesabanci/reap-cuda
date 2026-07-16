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

4. **Prune path only:** delete observe model, reload with **`plan_load("gpu_full")`**
   (`device_map="auto"`), slice, **`stream_save_pretrained`**.

## Weight load (residency)

Layerwise **no longer pins the full model with `device_map="cpu"` by default**.
When residency resolves to `layerwise`, observe load uses:

| Setting | Value |
| --- | --- |
| `device_map` | `"auto"` |
| `offload_folder` | `artifacts/<model>/<dataset>/.offload` |
| `low_cpu_mem_usage` | from `--low-cpu-mem` (default on) |

That keeps host RAM from holding every parameter on small instances. If
`--residency auto` decides the model fits VRAM and host is tight, the layerwise
CLI may **delegate** to the full pipeline (`gpu_full`) instead of block replay.

See **[residency.md](residency.md)** for modes, heuristics, and delegation.

## Memory profile (order of magnitude)

| Item | Layerwise | Full |
| --- | --- | --- |
| Weights on GPU (observe) | ~1 block (~1–2 GB bf16 for 30B-class) | entire model (~60 GB) |
| Weights on host | Offload/disk + working set (not full pin) | accelerate map / none for pure GPU |
| Activation transient (routed backends) | MB-scale stats | same, but all layers present |
| Activation transient (old dense loop) | up to multi-GB per layer | multi-GB × concurrency |
| F4 cache | ~1.2 GB/layer while active | same while hooked layer runs |
| Replay cache | CPU RAM (all batches × seq × hidden) | N/A |
| Mutate/save | Full model via `gpu_full` plan | Already loaded |

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
| `--low-cpu-mem` / `--no-low-cpu-mem` | HF low-CPU-mem load flag |
| `--residency` | `auto` \| `gpu_full` \| `layerwise` \| `cpu_full` (weight policy) |

## When to use which mode

| Situation | Prefer |
| --- | --- |
| Single L40S / 48 GB, 30B MoE | **layerwise** observe (`--residency auto`) |
| Multi-GPU 80 GB+ | full (simpler, faster wall-clock) |
| ~8B MoE, 16 GiB host RAM, 24 GiB GPU | **full + `--residency gpu_full`** (or auto → gpu_full) |
| Merge on large model | layerwise observe + residency auto |
| Debugging hooks | full + tiny model |

## Limitations

- Hidden states on CPU add PCIe / host bandwidth cost.
- Prune mutate still needs a full-model GPU load (or multi-GPU map).
- Block `forward` signatures differ across families; kwargs are filtered by
  inspect.signature and forced `use_cache=False`.
- Dense / hybrid layers (LFM2 early layers) forward without MoE metrics.

## Related

- [residency.md](residency.md)
- [pipeline.md](pipeline.md)
- [gpu-and-backends.md](gpu-and-backends.md)
- [architecture.md](architecture.md)
