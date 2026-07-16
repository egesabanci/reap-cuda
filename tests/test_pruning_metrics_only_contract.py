"""Contract: prune path produces only routed metrics (F3 / Phase 0)."""
from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig
from reap.pruning_metrics import MERGING_CRITERIA_KEYS, PRUNING_STATE_KEYS


def _make_model():
    cfg = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=1,
        norm_topk_prob=False,
    )
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


def _observe(model, record_pruning_metrics_only: bool, backend: str = "bmm"):
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long),
    }
    adapter = infer_model_adapter(model, model.config)
    hc = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=adapter.get_layer_config(
            adapter.layers(model)[0], model.config
        ).fused_experts,
        record_pruning_metrics_only=record_pruning_metrics_only,
        observe_backend=backend,
    )
    obs = MoETransformerObserver(model, hook_config=hc, adapter=adapter)
    with obs.set_attention_mask(batch["attention_mask"]):
        _ = model(**batch)
    state = obs.report_state()
    obs.close_hooks()
    return state


def test_pruning_only_excludes_merging_criteria():
    state = _observe(_make_model(), record_pruning_metrics_only=True)
    keys = set(state[0].keys())
    assert MERGING_CRITERIA_KEYS.isdisjoint(keys), (
        f"pruning-only path leaked merging-criteria keys: "
        f"{keys & MERGING_CRITERIA_KEYS}"
    )
    assert PRUNING_STATE_KEYS.issubset(keys), (
        f"pruning-only path missing consumed keys: {PRUNING_STATE_KEYS - keys}"
    )


def test_pruning_only_matches_full_path_on_consumed_metrics():
    # Same weights for both runs (seeded construction).
    full = _observe(_make_model(), record_pruning_metrics_only=False, backend="loop")
    only = _observe(_make_model(), record_pruning_metrics_only=True, backend="loop")
    for k in PRUNING_STATE_KEYS - {"ean_mean", "reap"}:
        a, b = full[0][k], only[0][k]
        if isinstance(a, torch.Tensor):
            assert torch.allclose(
                a.float().cpu(), b.float().cpu(), atol=1e-5
            ), f"{k} differs: {a} vs {b}"
