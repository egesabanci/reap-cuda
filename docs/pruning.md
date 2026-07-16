# Pruning

Pruning removes low-saliency routed experts **in place**, then patches config
and saves a HuggingFace-compatible checkpoint.

Implementation: `src/reap/prune.py`, `src/reap/layerwise_prune.py`.

## Algorithm (per MoE layer)

1. Load saliency vector `s ∈ R^E` from observer state (`_resolve_saliency`).
2. Optionally protect super/outlier experts by setting their score to `+inf`.
3. Choose `n = min(n_experts_to_prune, E - 1)` lowest scores (`torch.topk`,
   `largest=False`).
4. `keep = {0..E-1} \ prune_set`.
5. `adapter.slice_experts(moe, keep)`.
6. After all layers: `adapter.update_config(config, retained, top_k)`.

All MoE layers share the same retained count (global `num_experts`).

## Saliency methods

Higher is more important. Defaults prefer `--prune-method reap`.

| Method | Definition (routed tokens) |
| --- | --- |
| `reap` | Online mean of mean(`‖y‖₂ * w`) over batches (Welford) |
| `frequency` | Assignment counts |
| `ean_sum` | Sum of `‖y‖₂` |
| `ean_mean` | Online mean of mean norms |
| `weighted_ean_sum` | Sum of `‖y‖₂ * w` |
| `weighted_frequency_sum` | Sum of router weights |
| `max_activations` | Max activation element over routed outputs |
| `ean_ca` | Norm of routed characteristic activation (needs merge metrics / CA) |

Aliases: `reap_l2` → `reap`, `weighted_ean_sum_l2` → `weighted_ean_sum`.

## Compression controls

| Flag | Meaning |
| --- | --- |
| `--compression-ratio R` | Remove `int(E * R)` experts |
| `--n-experts-to-prune N` | Absolute count (wins if set) |
| `--preserve-super-experts` | Protect super-experts (first ~75% layers) |
| `--preserve-outliers` | Protect outliers across all layers |

Super-expert identification reuses merge helper thresholds on
`max_activations` (`get_super_expert_indices`).

## Config and live module updates

After slicing:

- `config.num_experts` (or `num_local_experts`) = retained
- `config.num_experts_per_tok` = `min(top_k, retained)`
- Live `experts.num_experts`, router `top_k` / `num_experts` updated so **reload
  is not required** for a correct forward (smoke test)

Shared experts remain.

## Saving

1. `remove_hook_from_module(model, recurse=True)` so accelerate does not
   materialize a full CPU state dict.
2. `model.save_pretrained` streams shards (CUDA tensors via safetensors).
3. Tokenizer saved alongside.

## Layerwise caveat

Observe is memory-efficient; **mutate/save still reloads the full model** with
`device_map="auto"`. Plan VRAM for that step separately from calibration.

## Smoke test

Optional generate with chat template (`pipeline.smoke_test`). Enable with
`--smoke-test` on full prune (default on for `reap prune full`).

## Related

- [observation-and-metrics.md](observation-and-metrics.md)
- [model-adapters.md](model-adapters.md)
- [pipeline.md](pipeline.md)
