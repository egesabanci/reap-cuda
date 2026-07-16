# Merging

Merging clusters experts by similarity and fuses each cluster into a
super-expert (weights averaged or TIES/… fused), keeping a smaller effective
expert set without pure deletion.

Implementation: `merge_pipeline.py`, `layerwise_merge.py`, `cluster.py`,
`merge.py`, `permute.py`.

## Pipeline summary

```txt
observe (record_pruning_metrics_only=False)
  -> cluster labels per layer
  -> MoEExpertMerger.merge_experts (in place)
  -> save model + clusters.pkl
```

Merge entrypoints **force** full merge-criteria metrics even if the CLI default
is pruning-only.

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
| `spectral` | Supported in cluster utilities |
| `mc_smoe` | Multi-criterion style path |

Optional:

- frequency penalty on distances
- `max_cluster_size` (restricted hierarchical)
- singleton super / outlier experts
- `multi_layer` joint clustering

Target cluster count:

```txt
num_clusters = int(experts_per_layer * (1 - compression_ratio))
```

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

Optional `--permute {direct,wm}` aligns intermediate neurons before merge.

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

- Calibration is block-wise on GPU.
- Model weights for merge are loaded on CPU; merger mutates CPU tensors.
- Ensure merge metrics were observed (not pruning-only cache).

## Related

- [observation-and-metrics.md](observation-and-metrics.md)
- [pruning.md](pruning.md)
- [cli.md](cli.md)
