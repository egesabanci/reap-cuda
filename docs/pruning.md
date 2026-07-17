# Pruning

Pruning removes low-saliency routed experts **in place**, then patches config
and saves a HuggingFace-compatible checkpoint.

Implementation: `src/reap/prune.py`, `src/reap/layerwise_prune.py`.

## Algorithm (per MoE layer)

1. Load saliency vector `s ∈ R^E` from observer state (`_resolve_saliency`).
2. Build an explicit candidate set containing only unprotected experts.
3. Compute each layer's capacity `E - protected - 1`, then choose one global
   prune count bounded by the smallest capacity so every MoE layer retains the
   same expert count.
4. Select the lowest-saliency experts only from each unprotected candidate set.
5. `keep = {0..E-1} \ prune_set`; protected indices are an invariant of `keep`.
6. `adapter.slice_experts(moe, keep)`, then clamp router top-k while patching
   config after all layers.

If protection makes the requested compression impossible, REAP reduces the
prune count uniformly and logs a warning rather than pruning a protected expert.

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

Implemented as `stream_save_pretrained` in `reap.residency` (called from
`prune_model`):

1. `remove_hook_from_module(model, recurse=True)` so accelerate does not
   materialize a full CPU state dict.
2. **No** `model.to("cpu")` — safetensors streams CUDA tensors shard-wise.
3. `model.save_pretrained` writes shards under the pruned model dir.
4. Tokenizer saved alongside.

This avoids host OOM after a successful GPU prune on small-RAM instances.
See [residency.md](residency.md#stream-save).

## Layerwise caveat

Observe and mutation are memory-efficient under residency `layerwise`: REAP
reuses the accelerate auto+disk-offloaded model for expert slicing and staged
stream save. It does **not** reload with `plan_load("gpu_full")`; logs report
the actual device map and CUDA peak allocation for the observe/mutation phases.
Do not confuse disk offload with `cpu_full`, which intentionally pins all
weights in host RAM.

## Smoke test

Optional generate with chat template (`pipeline.smoke_test`). It runs before
publication: a failure raises and leaves no newly published checkpoint or
staging directory. `reap prune full` defaults it on; layerwise exposes the
same flag but defaults it off for offloaded-model cost control.

## Related

- [residency.md](residency.md)
- [observation-and-metrics.md](observation-and-metrics.md)
- [model-adapters.md](model-adapters.md)
- [pipeline.md](pipeline.md)
