# Pipeline

End-to-end execution for prune and merge. Orchestration lives in `run()` APIs;
the Typer CLI only builds dataclasses and invokes them.

**Before load**, every `run()` resolves **weight residency** (`ReapArgs.residency`
/ `--residency`) and may **delegate** full↔layerwise. Details:
[residency.md](residency.md).

## Residency gate (all prune/merge entrypoints)

| Step | What happens |
| --- | --- |
| Validate | `validate_residency(reap_args.residency)` |
| Estimate | Optional `estimate_model_bytes_from_config(model_name)` |
| Resolve | `resolve_residency(..., cli_prefers_layerwise=…)` → concrete mode + reason |
| Preflight | `preflight_or_warn` if size vs host/GPU looks unsafe |
| Delegate | Full path ↔ layerwise path if resolved mode disagrees with entrypoint |

| Entrypoint | Prefers layerwise in `auto` | Delegates when resolved… |
| --- | --- | --- |
| `prune.run` / `merge_pipeline.run` | No | `layerwise` → layerwise `run` |
| `layerwise_prune.run` / `layerwise_merge.run` | Yes | `gpu_full` / `cpu_full` → full `run` |

Delegation passes `_residency_resolved=` so the peer does not re-resolve `auto`
and bounce forever.

## Prune pipelines

### Full (`reap prune full` → `reap.prune.run`)

| Phase | What happens | Primary modules |
| --- | --- | --- |
| 0. Residency | Resolve / preflight / maybe delegate to layerwise | `residency` |
| 1. Setup | Seed, create `artifacts/<model>/<dataset>/` | `pipeline.create_results_directory` |
| 2. Load | `load_causal_lm` with `plan_load(gpu_full|cpu_full)` | `residency`, transformers / accelerate |
| 3. Calibrate | Tokenized batches from HF hub, `--dataset-path`, composite (`load_composite_category_batches`), or `combined` cache | `data`, `pipeline.record_activations` |
| 4. Observe | Forward hooks on every MoE block; saliency on GPU | `observer`, `kernels.observe` |
| 5. Persist stats | Optional `.pt` under category dirs | `observer.save_state` |
| 6. Rank | Per-layer lowest saliency → experts to drop | `prune._resolve_saliency` |
| 7. Slice | `adapter.slice_experts` + `update_config` | `model_adapters` |
| 8. Save | `stream_save_pretrained` (hooks stripped; no full CPU dump) | `residency`, `prune` |
| 9. Smoke / eval | Optional generate smoke; optional lm-eval | `pipeline.smoke_test`, `eval` |

If `--observe-only`, stop after phase 5.

Load plan when staying on full path:

| Resolved residency | `device_map` | Notes |
| --- | --- | --- |
| `gpu_full` | `"auto"` | Default for GPU hosts; stream save |
| `cpu_full` | `"cpu"` | Only when host RAM is safe |

### Layerwise (`reap prune layerwise` → `reap.layerwise_prune.run`)

| Phase | Difference from full |
| --- | --- |
| Residency | May delegate to full path if resolved `gpu_full` / `cpu_full` |
| Load (observe) | `plan_load("layerwise")`: `device_map="auto"` **+ disk offload** under `artifacts/.../.offload` — **not** a full host pin |
| Observe | `LayerwiseMoEObserver.record_all_blocks` — one block on GPU |
| Prune | Delete observe model; **reload** with `plan_load("gpu_full")`; slice; stream save (needs VRAM for full model at mutate/save) |

Older docs said layerwise always used `device_map="cpu"`. That caused host OOM on
small-RAM GPU instances; offload + optional delegation replaces that default.

Observer cache path: `artifacts/.../layerwise/<output_file_name>` (or `all/` for
composite / combined).

## Merge pipelines

### Full (`reap merge full` → `reap.merge_pipeline.run`)

| Phase | What happens |
| --- | --- |
| Residency | Same gate as prune; may delegate to layerwise merge |
| Load | `load_causal_lm` via `gpu_full` or `cpu_full` plan |
| Observe | Forces `record_pruning_metrics_only=False` (merge metrics required) |
| Cluster | Per-layer labels from similarity + optional frequency penalty |
| Merge | In-place `MoEExpertMerger` per layer |
| Save | `stream_save_pretrained` + `clusters.pkl` + `reap_args.yaml` |

### Layerwise (`reap merge layerwise` → `reap.layerwise_merge.run`)

Residency may delegate to full merge when resolved `gpu_full` / `cpu_full`.
Otherwise: layerwise observe with merge metrics (auto + offload load); clustering
and merge mutate weights in place on the loaded instance; save uses the stream
path when applicable.

## Artifact layout

```txt
artifacts/
  <model_short_name>/
    <dataset_short_or_composite_hash>/
      all/ | layerwise/ | <category>/
        observations_*.pt          # observer state
      pruned_models/
        <method>-seed_*-<ratio>/
          config.json
          *.safetensors
          tokenizer files
          reap_args.yaml           # layerwise prune dumps this
      merged_models/
        <merge_desc>/
          <cluster_desc>/
            model files
            clusters/clusters.pkl
            reap_args.yaml
```

## Compression budget

- **Prune**: remove `n_experts_to_prune` experts, or
  `int(total_experts * compression_ratio)` when `n_experts_to_prune` is unset.
  At least one expert must remain per layer (`n_prune = min(..., num_experts - 1)`).
- **Merge**: target `num_clusters = int(experts_per_layer * (1 - compression_ratio))`
  retained super-experts after merge (skipped layers get identity clusters).

## Failure modes to know

| Symptom | Likely cause |
| --- | --- |
| Host OOM on load (small RAM + large model) | Full CPU pin; set `--residency gpu_full` or `auto` ([residency.md](residency.md)) |
| OOM on full prune | 30B does not fit VRAM; use layerwise observe + multi-GPU or larger instance for prune reload |
| OOM on layerwise observe | Batch/seq too large; lower `--batch-size` / `--model-max-length` |
| OOM on layerwise mutate/save | Full `gpu_full` reload needs VRAM; observe-only on small GPU then mutate elsewhere |
| Smoke fails after prune | Fixed for fused Qwen if using current `slice_experts`; check live `num_experts` |
| Missing saliency key | Wrong `--prune-method` or pruning-only cache used with `ean_ca` |
| Merge KeyError on ttm/CA | Observations recorded with pruning-only metrics |

## Related

- [calibration.md](calibration.md) — datasets, offline path, composite specs
- [residency.md](residency.md)
- [observation-and-metrics.md](observation-and-metrics.md)
- [pruning.md](pruning.md)
- [merging.md](merging.md)
- [cli.md](cli.md)
