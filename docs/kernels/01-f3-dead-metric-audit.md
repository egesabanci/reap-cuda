# 01 — Phase 0: F3 Dead-Metric Audit + `record_pruning_metrics_only=True` Default

> **Concern:** make the "prune path consumes only routed-token metrics"
> property **load-bearing in code** before any kernel is written. This is a
> prerequisite (not a kernel) — but FREA and F2 are *only correct* because of it,
> and a future change that re-adds an all-token consumer would silently make the
> kernels wrong.

## Why this must come first

FREA (Phase 3) and F2 (Phase 4) compute saliency from **routed `(token, top_k)`
pairs only**. That is valid **iff** `prune.py` never reads an all-token metric.
Today that is true *by inspection* (see `00-cost-model.md` §6), but it is not
*enforced*. If someone wires, say, `ttm_similarity_matrix` into a new prune
method, FREA would produce correct activations but **wrong saliency** with no
error.

Phase 0 turns the inspection into:
1. an **audit** that proves the consumer set is routed-only, and
2. a **default flip** + **contract test** that fails loudly if the contract is
   ever broken.

## 2. The current default and what honors it

`src/reap/args.py` (`ObserverArgs.record_pruning_metrics_only`, default `False`):

```python
# src/reap/args.py — ObserverArgs
record_pruning_metrics_only: bool = field(
    default=False,
    metadata={
        "help": ("Whether to only record pruning metrics during observation to reduce "
                 "memory usage and wall-clock time.")
    },
)
```

Who sets/forces it:
- `src/reap/merge_pipeline.py` `main()` **forces** `record_pruning_metrics_only
  = False` (merge needs the merging-criteria metrics):
  ```python
  # merge_pipeline.py main()
  if obs_args.record_pruning_metrics_only:
      logger.info("Merging requires merging-criteria metrics; forcing "
                  "record_pruning_metrics_only=False.")
      obs_args.record_pruning_metrics_only = False
  ```
- `src/reap/observer.py:277` **reads** it to decide whether to allocate the
  merging-criteria `OnlineStatsTracker`s:
  ```python
  # src/reap/observer.py:277
  if not self.hook_config.record_pruning_metrics_only:
      layer_state["ttm_similarity_matrix"] = OnlineStatsTracker(...)
      layer_state["routed_characteristic_activation"] = OnlineStatsTracker(...)
      layer_state["characteristic_activation"] = OnlineStatsTracker(...)
      layer_state["online_characteristic_activation_dist"] = OnlineStatsTracker(...)
      layer_state["router_logit_similiarity"] = OnlineStatsTracker(...)
  ```
- The prune entrypoints (`reap.prune`, `reap.layerwise_prune`) **pass it
  through** but do not force a value — so the default (`False`) means the prune
  path **currently allocates the merging-criteria trackers it never reads**.

## 3. The audit (mechanical, no code changes)

For each of the four merging-criteria keys, prove there is **no read on the
prune path**:

```bash
# Writers (must be only observer.py + pruning_metrics.py):
git grep -n 'ttm_similarity_matrix'           -- src/reap
git grep -n 'characteristic_activation'       -- src/reap      # note: NOT routed_characteristic_activation
git grep -n 'online_characteristic_activation_dist' -- src/reap
git grep -n 'router_logit_similiarity'        -- src/reap      # sic: the key is misspelled in the codebase

# Consumers (must be only merge_pipeline.py + cluster.py):
git grep -n 'ttm_similarity_matrix'           -- src/reap/merge_pipeline.py src/reap/cluster.py
git grep -n 'characteristic_activation'       -- src/reap/merge_pipeline.py src/reap/cluster.py
git grep -n 'router_logit_similiarity'        -- src/reap/merge_pipeline.py src/reap/cluster.py
```

**Expected result of the audit:**
- **Writes**: `src/reap/observer.py:279,285,292,299,306` and the
  `update_pruning_state` post-loop in `src/reap/observer.py` (the `ttm_online`,
  `get_routed_characteristic_activation`, `ca_dist_online` calls right after
  `update_pruning_state` in `_hook_factory`).
- **Reads**: `src/reap/merge_pipeline.py` `cluster()` (the `expert_similarity_scores`
  dict referencing `ttm_similarity_matrix`, `characteristic_activation`,
  `routed_characteristic_activation`, `router_logit_similiarity`,
  `online_characteristic_activation_dist`) and `src/reap/cluster.py`.
- **`src/reap/prune.py` must return zero matches** for all four keys. ✅ (this
  is the contract).

> ⚠️ **Name spellings to watch**: the codebase key is `router_logit_similiarity`
> (sic — misspelled "similarity"). Any audit grep must use the misspelling or
> it will miss real references. Do not "fix" the spelling without a migration
> of saved `.pt` observer files.

## 4. The default flip

`src/reap/args.py`:

```diff
  record_pruning_metrics_only: bool = field(
-     default=False,
+     default=True,
      metadata={ ... },
  )
```

Rationale:
- The prune path never reads the merging-criteria metrics, so allocating them
  is pure waste (4 extra `OnlineStatsTracker`s per layer, each (E,E) or (E,H),
  plus their Welford update FLOPs).
- The merge path **forces** `False` already, so it is unaffected.
- The standard-observer merge E2E test (`tests/test_merge_pipeline.py`) passes
  `ObserverArgs(record_pruning_metrics_only=False)` explicitly, so it is
  unaffected.

## 5. The contract test

New file `tests/test_pruning_metrics_only_contract.py`:

```python
"""Contract: on the prune path (record_pruning_metrics_only=True) the observer
produces ONLY routed-token metrics and NEVER the all-token merging criteria.

This is the correctness precondition for the FREA / F2 kernels (docs/kernels/).
If this test ever fails, the kernels would silently produce wrong saliency.
"""
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.args import ObserverArgs
from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig

MERGING_CRITERIA_KEYS = {
    "ttm_similarity_matrix",
    "characteristic_activation",            # NB: not routed_characteristic_activation
    "online_characteristic_activation_dist",
    "router_logit_similiarity",             # sic: codebase misspelling
}
PRUNING_KEYS = {
    "total_tokens", "expert_frequency", "pairwise_expert_frequency",
    "ean_sum", "ean_mean", "reap", "weighted_ean_sum",
    "weighted_expert_frequency_sum", "max_activations",
}


def _observe(record_pruning_metrics_only: bool):
    cfg = Qwen3MoeConfig(vocab_size=32, hidden_size=8, intermediate_size=8,
        moe_intermediate_size=8, num_hidden_layers=2, num_attention_heads=1,
        num_key_value_heads=1, num_experts=4, num_experts_per_tok=1,
        norm_topk_prob=False)
    model = Qwen3MoeForCausalLM(cfg).eval()
    batch = {"input_ids": torch.tensor([[1,2,3,0],[4,5,6,7]],dtype=torch.long),
             "attention_mask": torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.long)}
    adapter = infer_model_adapter(model, model.config)
    hc = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=False,
        record_pruning_metrics_only=record_pruning_metrics_only)
    obs = MoETransformerObserver(model, hook_config=hc, adapter=adapter)
    with obs.set_attention_mask(batch["attention_mask"]):
        _ = model(**batch)
    state = obs.report_state()
    obs.close_hooks()
    return state


def test_pruning_only_excludes_merging_criteria():
    state = _observe(record_pruning_metrics_only=True)
    keys = set(state[0].keys())
    assert MERGING_CRITERIA_KEYS.isdisjoint(keys), \
        f"pruning-only path leaked merging-criteria keys: {keys & MERGING_CRITERIA_KEYS}"
    assert PRUNING_KEYS.issubset(keys), \
        f"pruning-only path missing consumed keys: {PRUNING_KEYS - keys}"


def test_pruning_only_matches_full_path_on_consumed_metrics():
    """The consumed pruning metrics must be identical whether or not the
    merging-criteria trackers are allocated (they must not perturb the
    reductions)."""
    full = _observe(record_pruning_metrics_only=False)
    only = _observe(record_pruning_metrics_only=True)
    for k in PRUNING_KEYS - {"ean_mean", "reap"}:  # OnlineStatsTracker -> .mean
        a, b = full[0][k], only[0][k]
        if isinstance(a, torch.Tensor):
            assert torch.allclose(a, b, atol=1e-5), f"{k} differs: {a} vs {b}"
```

## 6. Acceptance

- Audit grep confirms `prune.py` reads none of the four merging-criteria keys.
- `record_pruning_metrics_only` default is `True`.
- `tests/test_pruning_metrics_only_contract.py` passes (2 new tests).
- The merge E2E test (`tests/test_merge_pipeline.py`) still passes (it forces
  `False` explicitly).

## 7. What this unlocks

- **FREA / F2 correctness**: the kernels compute routed-only reductions; the
  contract test guarantees the consumer set stays routed-only.
- **Memory on the prune path**: drops 4 `OnlineStatsTracker` allocations per
  layer (the `(E,E)` + `(E,H)` trackers) — small (~1 MB) but removes their
  Welford update FLOPs and the `ttm_online` / `ca_dist_online` distance passes
  in `observer.py`'s `_hook_factory`.

## 8. Risk / rollback

- **Risk**: a user running `python -m reap.merge_pipeline` who *relies* on the
  default rather than passing the flag. Mitigated: `merge_pipeline.main()`
  forces `False` regardless.
- **Rollback**: revert the one-line default in `args.py`; the contract test
  will then assert the `True`-path still works, so re-flipping is safe.