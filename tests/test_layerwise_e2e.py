"""End-to-end layerwise calibration + prune pipeline test.

Exercises the full wiring path restored in the layerwise port:
``record_activations_layerwise`` (adapter config construction + block-by-block
forward + state save) -> ``prune`` (adapter.slice_experts + save_pretrained).

Runs on CPU with the project venv (transformers 4.55) on a tiny
``Qwen3MoeForCausalLM`` (4 experts, 2 layers). No weights are downloaded.
"""
from __future__ import annotations

import pathlib
import tempfile

import torch
from dataclasses import replace
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.args import DatasetArgs, ObserverArgs, PruneArgs, LayerwiseArgs
from reap.layerwise_prune import record_activations_layerwise
from reap.prune import prune as prune_model


def _make_model(num_experts: int = 4, num_hidden_layers: int = 2):
    cfg = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=num_experts,
        num_experts_per_tok=1,
        norm_topk_prob=False,
    )
    return Qwen3MoeForCausalLM(cfg).eval()


def _n_experts(model) -> int:
    return len(model.model.layers[0].mlp.experts)


def test_layerwise_calibrate_then_prune_end_to_end():
    torch.manual_seed(0)
    model = _make_model(num_experts=4, num_hidden_layers=2)
    n_before = _n_experts(model)
    assert n_before == 4

    data_batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[6, 7, 8, 9], [1, 2, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long),
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = pathlib.Path(tmp)
        ds_args = replace(DatasetArgs(), dataset_name="test")
        obs_args = replace(ObserverArgs(), record_pruning_metrics_only=True)
        observer_data = record_activations_layerwise(
            model, None, data_batches, ds_args, obs_args, LayerwiseArgs(), results_dir
        )

        assert set(observer_data.keys()) == {0, 1}
        assert observer_data[0]["expert_frequency"].numel() == 4

        pruned_dir = results_dir / "pruned"
        prune_model(observer_data, model, PruneArgs(), 2, pruned_dir)

        assert _n_experts(model) == 2, "prune did not reduce experts 4 -> 2"
        saved = list(pruned_dir.glob("*.safetensors")) + list(pruned_dir.glob("*.bin"))
        assert saved, "pruned model was not saved"