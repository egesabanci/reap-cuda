# Architecture

REAP CUDA is organized so **model layout**, **observation**, **saliency**,
**pruning/merging**, and **CLI** stay independent. Heavy PyTorch / transformers
work is confined to the phases that need it.

## Module responsibilities

| Module / package | Responsibility | Notes |
| --- | --- | --- |
| `reap.cli` | Typer CLI; builds dataclasses; calls `run()` APIs | No model I/O |
| `reap.args` | Dataclass config surface (shared by CLI + legacy HF parsers) | Includes `ReapArgs.residency` |
| `reap.residency` | Weight load/save policy: auto/gpu_full/layerwise/cpu_full, stream save | Orthogonal to observe backends |
| `reap.pipeline` | Full-model `record_activations`, results dirs, smoke helper | Hooks standard observer |
| `reap.model_adapters` | Layout discovery, layer config, `slice_experts`, config patch | Layout-agnostic contract |
| `reap.observer` | Full-model forward hooks + MoE observation | Uses `kernels.observe` |
| `reap.layerwise_observer` | Block load/offload, replay cache, same metrics | One block GPU |
| `reap.layerwise_model_utils` | Block discovery, device moves, cleanup | Layerwise helpers |
| `reap.pruning_metrics` | GPU-resident saliency state + dense/routed updates | Prune contract |
| `reap.metrics` | Distance fns, Welford `OnlineStatsTracker`, merge helpers | Shared math |
| `reap.kernels` | Backend select, F4 cache, F5/native router, bmm/FREA/F2, FREA probe, `observe_moe_batch` | Model-agnostic tensors |
| `reap.prune` / `layerwise_prune` | Orchestrate observe → rank → slice → save | `run()` + legacy `main()` |
| `reap.merge_pipeline` / `layerwise_merge` | Observe (full metrics) → cluster → merge → save | Merge-only metrics |
| `reap.cluster` / `restricted_cluster` | Hierarchical / k-means / MC-SMoE clustering | CPU (scipy/sklearn) |
| `reap.merge` / `permute` | Expert weight merge methods + optional permutation | In-place tensors |
| `reap.data` | HF datasets → tokenized batches | Registered processors |
| `reap.eval` | lm-eval harness (HF backend) | Optional `[eval]` |

## Package layout

```txt
src/reap/
  cli/                 # Typer: prune|merge × full|layerwise
  kernels/             # observe backends (bmm/frea/f2), F4/F5
  residency.py         # weight residency + stream save
  model_adapters.py
  observer.py
  layerwise_observer.py
  pruning_metrics.py
  metrics.py
  prune.py
  layerwise_prune.py
  merge_pipeline.py
  layerwise_merge.py
  cluster.py
  merge.py
  permute.py
  data.py
  pipeline.py
  args.py
  eval.py
```

## Data flow (prune, full mode)

```txt
Typer / HfArgumentParser
  -> ReapArgs (incl. residency), ModelArgs, DatasetArgs, ...
  -> resolve_residency / preflight; maybe delegate to layerwise_prune
  -> create_results_directory(artifacts/<model>/<dataset>/)
  -> load_causal_lm(plan_load(gpu_full|cpu_full)) + tokenizer
  -> load_category_batches / load_composite_category_batches
       (hub | --dataset-path | composite @path)
  -> MoETransformerObserver hooks
       for each batch: model(**batch)
         hook -> observe_moe_batch (backend)
              -> update pruning state (GPU)
  -> report_state / save .pt
  -> per-layer topk lowest saliency -> slice_experts
  -> update_config(num_experts, top_k)
  -> stream_save_pretrained + tokenizer
  -> optional smoke_test / lm-eval
```

## Data flow (prune, layerwise mode)

```txt
resolve_residency (cli_prefers_layerwise=True)
  -> maybe delegate to prune.run if gpu_full|cpu_full
Load with plan_load(layerwise): device_map=auto + offload_folder
  -> capture first-block inputs (pre-hook) into ReplayCache (CPU)
  -> for each decoder block:
       move block to CUDA
       for each cached batch: forward block
         MoE hook -> observe_moe_batch
       offload block to CPU
  -> retain the auto+disk-offloaded model
  -> in-place prune + staged stream_save_pretrained (no gpu_full reload)
```

Hidden states between blocks are cached on **CPU** to fit large models; saliency
and expert matmuls for the active block stay on **GPU**. Weight placement is
governed by [residency.md](residency.md); block schedule by [layerwise.md](layerwise.md).

## Adapter boundary

Adapters **describe layout**; they do not accumulate metrics or run clustering.

Each adapter provides:

- `layers(model)`, `identify_moe_layers`, `is_moe_layer`, `get_moe`
- `hook_regex()` for observer module matching
- `router_attr()`, `experts_attr()`, `num_experts_config_attr()`
- `get_layer_config` → `MoeLayerConfig` (experts, top_k, fused, weight_convention, …)
- `expert_weight_attrs(moe?)` for merge/kernels
- `weight_convention()` → `"linear"` \| `"bmm"`
- `slice_experts(moe, keep_indices)` — must leave the module **forward-runnable**
- `update_config(config, num_experts, top_k)`

Kernels never branch on architecture names for matmul: F4 normalizes stacks to
Linear `(E, I, H)` / `(E, H, I)`. See [gpu-and-backends.md](gpu-and-backends.md)
and [model-adapters.md](model-adapters.md).

## Observer boundary

- **Standard observer**: registers forward hooks on MoE blocks; runs during a
  normal full forward.
- **Layerwise observer**: does not keep the full model on GPU; replays blocks
  with a CPU `ReplayCache`.
- Both call **`observe_moe_batch`** so fused/non-fused and backends cannot drift.
- Hooks recompute expert activations for metrics (the HF forward already ran
  experts once). Routed backends minimize that second pass to top-k pairs.

## Metrics boundary

| Path | State keys | Consumer |
| --- | --- | --- |
| Prune-only | frequency, EAN, REAP, max_activations, … | `prune.py` |
| Merge criteria | ttm, CA, router logit sim, … | `cluster` / `merge_pipeline` |

Default `record_pruning_metrics_only=True` skips merge criteria. Merge entrypoints
force `False`. Contract tests live in
`tests/test_pruning_metrics_only_contract.py`.

## Invariants

1. After prune, every MoE layer has the **same** retained expert count (global
   config `num_experts`).
2. `top_k` / `num_experts_per_tok` is clamped to `min(top_k, retained)`.
3. Shared experts (if any) are never in the routed expert index set.
4. Observer state is moved to CPU only when saving; live accumulation prefers
   the activation device.
5. `pairwise_expert_frequency` is `freq_i + freq_j` (historical REAP semantics),
   not co-routing counts.
6. Weight residency is resolved once per `run()` entry; cross-delegation uses
   `_residency_resolved` so `auto` cannot recurse between full and layerwise.
7. Default save path strips accelerate hooks and does **not** force
   `model.to("cpu")` before `save_pretrained`.

## Design non-goals

- Not a training framework.
- Not a vLLM / TensorRT serving stack.
- Full hand-written Triton matmul kernels are optional acceleration; the
  correctness path is pure PyTorch GPU (see `docs/kernels/`).
