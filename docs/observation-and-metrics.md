# Observation and Metrics

Observation collects per-expert statistics during calibration. Those statistics
drive prune ranking and merge clustering.

## Observers

| Class | File | Mode |
| --- | --- | --- |
| `MoETransformerObserver` | `observer.py` | Full model forward hooks |
| `LayerwiseMoEObserver` | `layerwise_observer.py` | Block replay |

Both call `reap.kernels.observe.observe_moe_batch` so backends and fused math
cannot diverge.

### Standard hook path

1. Match modules by exact class name (`adapter.hook_regex()`).
2. On each MoE forward, read `hidden_states` from hook `args[0]`.
3. Optionally apply `attention_mask` via `set_attention_mask` context.
4. Run `observe_moe_batch` with configured backend.
5. If merge metrics enabled, update ttm / CA / router-logit trackers from dense
   or sparse activations.

### Layerwise path

See [layerwise.md](layerwise.md). Metrics code path is the same
`observe_moe_batch` call after capturing MoE inputs on the active block.

## Observation backends

Selected by `--observe-backend` / `ObserverArgs.observe_backend`:

| Backend | Behavior |
| --- | --- |
| `auto` | `f2` if CUDA+Triton importable, else `bmm` |
| `loop` | Legacy path; fused still uses layout-normalized weights |
| `bmm` | Grouped routed-only matmuls (parity oracle) |
| `frea` | FREA entry (grouped + optional `torch.compile` on CUDA) |
| `f2` | Same compute + routed scatter reductions |

Details: [gpu-and-backends.md](gpu-and-backends.md), [kernels/](kernels/README.md).

## Prune-path state keys

Initialized by `initialize_pruning_state` (compute device):

| Key | Shape | Meaning |
| --- | --- | --- |
| `total_tokens` | scalar | Valid tokens seen |
| `expert_frequency` | `(E,)` | Count of top-k assignments |
| `pairwise_expert_frequency` | `(E,E)` | `freq_i + freq_j` (historical) |
| `ean_sum` | `(E,)` fp64 | Sum of L2 norms of routed outputs |
| `ean_mean` | `(E,)` | Welford mean of per-batch means of norms |
| `weighted_ean_sum` | `(E,)` fp64 | Sum of `norm * router_weight` |
| `reap` | `(E,)` | Welford mean of per-batch mean(`norm * w`) |
| `weighted_expert_frequency_sum` | `(E,)` | Sum of router weights on routes |
| `max_activations` | `(E,)` | Max element over routed activations |

`report_state` converts `OnlineStatsTracker` fields to `.mean`.

### CLI prune method → key

| `--prune-method` | State key |
| --- | --- |
| `frequency` | `expert_frequency` |
| `ean_sum` / `ean_mean` | same names |
| `weighted_frequency_sum` | `weighted_expert_frequency_sum` |
| `weighted_ean_sum` (+ `_l2` alias) | `weighted_ean_sum` |
| `reap` (+ `reap_l2` alias) | `reap` |
| `max_activations` | `max_activations` |
| `ean_ca` | derived from `routed_characteristic_activation` |

Mapping: `pruning_metrics.PRUNE_METHOD_KEY_MAP`. Higher score ⇒ more important
⇒ **kept**; prune takes `topk(..., largest=False)`.

## Merge-criteria state keys

Allocated only when `record_pruning_metrics_only=False`:

| Key | Role |
| --- | --- |
| `ttm_similarity_matrix` | Token-to-token style expert distance (online) |
| `routed_characteristic_activation` | Mean routed activation per expert |
| `characteristic_activation` | Mean activation over tokens (path-dependent if sparse) |
| `online_characteristic_activation_dist` | Pairwise CA distances |
| `router_logit_similiarity` | Pairwise router logit distance (**misspelled** key — do not rename without migration) |

Default for prune CLI is **pruning-only** (`True`). Merge commands force
`False`.

## Device residency (saliency tensors)

This section is about **metrics / saliency tensors**, not model weight placement.
For weight load/save policy (`--residency`), see [residency.md](residency.md).

- Create state on the **activation device** (CUDA when the block/model is there).
- Reductions stay on that device (no per-batch `.cpu()` in hot path).
- `save_state` / report for disk moves tensors to CPU.
- `OnlineStatsTracker.to(device)` follows incoming batch device.

## Routed vs dense activations

| Backend | Expert FLOPs | `(E,T,H)` tensor |
| --- | --- | --- |
| Historical non-fused loop | All experts × all tokens | Yes |
| `bmm` / `frea` / `f2` | Routed pairs only | No (unless merge needs dense rebuild) |
| Merge + routed backend | Sparse rebuild for ttm/CA | Temporary dense for merge math |

For merge quality A/B against historical full-expert CA, use `--observe-backend loop`
with non-fused layouts, or accept routed/sparse semantics for fused models.

## Attention masks

- Full observer: `with observer.set_attention_mask(mask): model(**batch)`
- Layerwise: mask recovered from replay kwargs / 2D padding / 4D causal last row
- Padding positions are excluded from frequency and EAN aggregates

## Related

- [residency.md](residency.md) — where **weights** live
- [pruning.md](pruning.md)
- [merging.md](merging.md)
- [kernels/01-f3-dead-metric-audit.md](kernels/01-f3-dead-metric-audit.md)
