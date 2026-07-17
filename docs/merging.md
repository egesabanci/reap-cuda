# Merging

Merging clusters experts by similarity and fuses each cluster into a
super-expert (weights averaged or TIES/… fused), keeping a smaller effective
expert set without pure deletion.

Implementation: `merge_pipeline.py`, `layerwise_merge.py`, `cluster.py`,
`merge.py`, `permute.py`.

## Pipeline summary

```txt
resolve --residency (may delegate full ↔ layerwise)
  -> load via residency plan
  -> observe (record_pruning_metrics_only=False)
  -> cluster labels per layer
  -> MoEExpertMerger.merge_experts (in place)
  -> stream_save_pretrained + clusters.pkl
```

Merge entrypoints **force** full merge-criteria metrics even if the CLI default
is pruning-only. Weight placement uses the same `--residency` policy as prune
([residency.md](residency.md)).

## Expert similarity (`--expert-sim`)

| Value | Observer key / source |
| --- | --- |
| `ttm` / `dynamic_ttm` | `ttm_similarity_matrix` |
| `characteristic_activation` | `characteristic_activation` |
| `routed_characteristic_activation` | routed CA |
| `router_logits` | `router_logit_similiarity` (sic) |
| `online_characteristic_activation_dist` | online CA dist |

Vector similarities may be converted with `--distance`
(`angular` / `euclidean` / `jsd` / `cka` / `cosine`).

## Clustering (`--cluster-method`)

| Method | Notes |
| --- | --- |
| `agglomerative` | Default; linkage via `--linkage` |
| `kmeans` | On characteristic activations |
| `mc_smoe` | Multi-criterion style path |

Optional:

- frequency penalty on distances
- `max_cluster_size` (restricted hierarchical)
- singleton super / outlier experts
- `multi_layer` joint clustering

Target cluster count is resolved before clustering:

```txt
--num-clusters N                 # explicit surviving experts; takes precedence
# otherwise
num_clusters = int(experts_per_layer * (1 - compression_ratio))
```

Exactly one effective control is required. `N` must be in
`[1, experts_per_layer]`; ratios must be finite and in `[0, 1)`. Unsupported
methods (including `spectral`) fail before model loading.

## Skip layers

`--skip-first` / `--skip-last` assign **identity** cluster labels (no merge) to
boundary layers. Validated so skips cannot remove all layers.

## Merge methods (`--merge-method`)

| Method | Behavior |
| --- | --- |
| `frequency_weighted_average` | Default; weight by expert frequency |
| `average` | Uniform weights |
| `ties` | TIES-style; `select_top_k` density |
| `multislerp` | Multi-SLERP; optional `dom_as_base` |
| `sce` / `karcher` / `submoe` | Additional fusion schemes |

Optional `--permute {direct,wm}` aligns intermediate neurons before merge on
non-fused expert layouts. Both preserve each expert's forward function while
reordering the intermediate-neuron axis. `direct` intentionally rejects fused
layouts with an actionable `NotImplementedError`; use `wm` where fused
permutation support is required.

Fused vs non-fused weight access uses `expert_weight_attrs` (live fused detection
for Qwen).

## Super-experts

`get_super_expert_indices` uses high `max_activations` quantiles. Used for
singleton clustering and for prune preservation flags.

## Artifacts

```txt
merged_models/<merge_name>/<cluster_desc>/
  config + weights
  clusters/clusters.pkl
  reap_args.yaml
  eval/   # if --eval
```

## Layerwise merge notes

- Calibration is block-wise on GPU when residency resolves to `layerwise`.
- Weight load uses residency plan (auto + disk offload preferred over full CPU
  pin); see [residency.md](residency.md). Explicit `cpu_full` still pins CPU.
- Ensure merge metrics were observed (not pruning-only cache).

## Related

- [residency.md](residency.md)
- [observation-and-metrics.md](observation-and-metrics.md)
- [pruning.md](pruning.md)
- [cli.md](cli.md)
