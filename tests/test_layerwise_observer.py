"""Layerwise (block-wise) observer regression tests.

Validates that ``LayerwiseMoEObserver`` (block-by-block forward with replay
cache) produces state that bit-for-bit matches the standard
``MoETransformerObserver`` (whole-model forward with hooks) on a real (tiny)
HuggingFace ``Qwen3MoeForCausalLM``. This is the gold-standard check that the
layerwise mechanism's port onto the adapter system preserves REAP semantics.

Runs on CPU with the project venv (pinned transformers 4.55, where Qwen3-MoE
experts are a non-fused ``ModuleList``).
"""
from __future__ import annotations

import copy

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.layerwise_observer import LayerwiseMoEObserver
from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig


def _make_qwen3_moe_model(num_hidden_layers: int = 1):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
    )
    model = Qwen3MoeForCausalLM(config)
    model.eval()
    return model


def _make_config(adapter) -> MoETransformerObserverConfig:
    return MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=False,
        record_pruning_metrics_only=True,
    )


def _assert_states_match(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    metrics_to_compare = [
        "total_tokens",
        "expert_frequency",
        "pairwise_expert_frequency",
        "weighted_expert_frequency_sum",
        "ean_sum",
        "weighted_ean_sum",
        "ean_mean",
        "reap",
    ]
    for layer_idx in a:
        for metric in metrics_to_compare:
            av, bv = a[layer_idx][metric], b[layer_idx][metric]
            if av.is_floating_point():
                assert torch.allclose(av, bv, rtol=1e-5, atol=1e-6), (
                    f"mismatch layer {layer_idx} {metric}: {av} vs {bv}"
                )
            else:
                assert torch.equal(av, bv), (
                    f"mismatch layer {layer_idx} {metric}: {av} vs {bv}"
                )


def test_layerwise_observer_matches_standard_observer():
    torch.manual_seed(0)
    model = _make_qwen3_moe_model()
    layerwise_model = copy.deepcopy(model)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
    }

    # standard observer (adapter-driven)
    sa = infer_model_adapter(model, model.config)
    observer = MoETransformerObserver(model, hook_config=_make_config(sa), adapter=sa)
    with observer.set_attention_mask(batch["attention_mask"]):
        _ = model(**batch)
    standard_state = observer.report_state()
    observer.close_hooks()

    # layerwise observer (adapter-driven)
    la = infer_model_adapter(layerwise_model, layerwise_model.config)
    layerwise_observer = LayerwiseMoEObserver(
        layerwise_model, hook_config=_make_config(la), adapter=la
    )
    layerwise_state = layerwise_observer.record_all_blocks([batch])
    layerwise_observer.close_hooks()

    expected_tokens = batch["attention_mask"].sum()
    assert layerwise_state[0]["total_tokens"] == expected_tokens
    assert standard_state[0]["total_tokens"] == expected_tokens

    assert torch.equal(
        layerwise_state[0]["expert_frequency"], standard_state[0]["expert_frequency"]
    )
    assert torch.equal(
        layerwise_state[0]["pairwise_expert_frequency"],
        standard_state[0]["pairwise_expert_frequency"],
    )
    for k in (
        "weighted_expert_frequency_sum",
        "ean_sum",
        "weighted_ean_sum",
        "ean_mean",
        "reap",
    ):
        assert torch.allclose(
            layerwise_state[0][k], standard_state[0][k], rtol=1e-5, atol=1e-6
        ), f"mismatch in {k}"


def test_layerwise_observer_grouped_batches_match_single_pass():
    torch.manual_seed(0)
    model = _make_qwen3_moe_model(num_hidden_layers=2)
    grouped_model = copy.deepcopy(model)

    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long
            ),
        },
        {
            "input_ids": torch.tensor([[6, 7, 8, 9], [10, 11, 12, 0]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long
            ),
        },
        {
            "input_ids": torch.tensor([[13, 14, 0, 0], [15, 16, 17, 18]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long
            ),
        },
    ]

    a = infer_model_adapter(model, model.config)
    single_pass = LayerwiseMoEObserver(model, hook_config=_make_config(a), adapter=a)
    single_pass_state = single_pass.record_all_blocks(batches)
    single_pass.close_hooks()

    ga = infer_model_adapter(grouped_model, grouped_model.config)
    grouped = LayerwiseMoEObserver(grouped_model, hook_config=_make_config(ga), adapter=ga)
    grouped_state = grouped.record_all_blocks(batches, batch_group_size=1)
    grouped.close_hooks()

    _assert_states_match(grouped_state, single_pass_state)