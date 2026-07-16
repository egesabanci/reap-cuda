# 01 — Phase 0: F3 Dead-Metric Audit + Prune-Only Default

> **Status: LANDED**  
> `ObserverArgs.record_pruning_metrics_only` defaults to **`True`**.  
> Contract tests: `tests/test_pruning_metrics_only_contract.py`.  
> Merge entrypoints force `False`.

> **Concern:** make “prune path consumes only routed-token metrics” load-bearing
> so FREA/F2 routed-only math cannot silently diverge from consumers.

## Why this comes first

FREA/F2 compute saliency from **routed `(token, top_k)` pairs only**. That is
valid **iff** `prune.py` never reads all-token merge metrics
(`ttm_similarity_matrix`, `characteristic_activation`, …).

## Current default

```python
# src/reap/args.py — ObserverArgs
record_pruning_metrics_only: bool = field(default=True, ...)
```

| Path | Behavior |
|---|---|
| `reap prune *` | Default **True** (pruning metrics only) |
| `reap merge *` | Forces **False** in `merge_pipeline.run` / `layerwise_merge.run` |
| Typer `--all-metrics` | Sets pruning-only to False on prune commands |

## Who allocates merge trackers

When `record_pruning_metrics_only=False`, observers allocate:

- `ttm_similarity_matrix`
- `routed_characteristic_activation`
- `characteristic_activation`
- `online_characteristic_activation_dist`
- `router_logit_similiarity`  (**sic** — misspelled key; do not rename without migration)

## Contract tests

`tests/test_pruning_metrics_only_contract.py`:

1. Pruning-only state has **no** merge-criteria keys and **all** prune keys.
2. Prune metrics match full-path on shared keys (same seeded model, `loop` backend).

## Acceptance (done)

- [x] `record_pruning_metrics_only` default `True`
- [x] Contract tests pass
- [x] Merge forces full metrics

## Unlocks

FREA/F2 correctness on prune path; less Welford work and no ttm/ca_dist on
default prune calibration.
