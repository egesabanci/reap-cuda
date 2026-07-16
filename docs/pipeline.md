# Pipeline

End-to-end execution for prune and merge. Orchestration lives in `run()` APIs;
the Typer CLI only builds dataclasses and invokes them.

## Prune pipelines

### Full (`reap prune full` → `reap.prune.run`)

| Phase | What happens | Primary modules |
| --- | --- | --- |
| 1. Setup | Seed, create `artifacts/<model>/<dataset>/` | `pipeline.create_results_directory` |
| 2. Load | `AutoModelForCausalLM(..., device_map="auto")`, tokenizer | transformers / accelerate |
| 3. Calibrate | Tokenized batches from HF (or composite / combined cache) | `data`, `pipeline.record_activations` |
| 4. Observe | Forward hooks on every MoE block; saliency on GPU | `observer`, `kernels.observe` |
| 5. Persist stats | Optional `.pt` under category dirs | `observer.save_state` |
| 6. Rank | Per-layer lowest saliency → experts to drop | `prune._resolve_saliency` |
| 7. Slice | `adapter.slice_experts` + `update_config` | `model_adapters` |
| 8. Save | Strip accelerate hooks; `save_pretrained` (GPU-streamed) | `prune` |
| 9. Smoke / eval | Optional generate smoke; optional lm-eval | `pipeline.smoke_test`, `eval` |

If `--observe-only`, stop after phase 5.

### Layerwise (`reap prune layerwise` → `reap.layerwise_prune.run`)

| Phase | Difference from full |
| --- | --- |
| Load | Model on **CPU** (`device_map="cpu"`, `low_cpu_mem_usage`) |
| Observe | `LayerwiseMoEObserver.record_all_blocks` — one block on GPU |
| Prune | **Reload** full model with `device_map="auto"` then slice (needs VRAM for full model at mutate/save) |

Observer cache path: `artifacts/.../layerwise/<output_file_name>` (or `all/` for
composite / combined).

## Merge pipelines

### Full (`reap merge full` → `reap.merge_pipeline.run`)

| Phase | What happens |
| --- | --- |
| Observe | Forces `record_pruning_metrics_only=False` (merge metrics required) |
| Cluster | Per-layer labels from similarity + optional frequency penalty |
| Merge | In-place `MoEExpertMerger` per layer |
| Save | `save_pretrained` + `clusters.pkl` + `reap_args.yaml` |

### Layerwise (`reap merge layerwise` → `reap.layerwise_merge.run`)

Calibration uses the layerwise observer with merge metrics; clustering/merge
still mutate a full model instance held on CPU (weights updated in place).

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
| OOM on full prune | 30B does not fit; use layerwise observe + multi-GPU or larger instance for prune reload |
| OOM on layerwise observe | Batch/seq too large; lower `--batch-size` / `--model-max-length` |
| Smoke fails after prune | Fixed for fused Qwen if using current `slice_experts`; check live `num_experts` |
| Missing saliency key | Wrong `--prune-method` or pruning-only cache used with `ean_ca` |
| Merge KeyError on ttm/CA | Observations recorded with pruning-only metrics |

## Related

- [observation-and-metrics.md](observation-and-metrics.md)
- [pruning.md](pruning.md)
- [merging.md](merging.md)
- [cli.md](cli.md)
