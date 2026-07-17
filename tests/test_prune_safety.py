"""Regression coverage for protected pruning, atomic publication, and layerwise plans."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import reap.prune as prune_module
from reap.args import (
    ClusterArgs,
    DatasetArgs,
    EvalArgs,
    LayerwiseArgs,
    ModelArgs,
    ObserverArgs,
    PruneArgs,
    ReapArgs,
)
from reap.layerwise_prune import run as run_layerwise
from reap.residency import LoadPlan


def _tiny_qwen() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
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
    return Qwen3MoeForCausalLM(config).eval()


def _observer_data():
    return {
        layer: {
            "expert_frequency": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "total_tokens": torch.tensor(10.0),
        }
        for layer in range(2)
    }


def test_protected_experts_are_never_pruned_and_count_is_uniform(monkeypatch):
    model = _tiny_qwen()
    observer_data = _observer_data()
    adapter = prune_module.infer_model_adapter(model, model.config)
    kept: list[list[int]] = []
    original_slice = adapter.slice_experts

    def record_slice(moe, indices):
        kept.append(list(indices))
        return original_slice(moe, indices)

    monkeypatch.setattr(prune_module, "infer_model_adapter", lambda *_args: adapter)
    monkeypatch.setattr(adapter, "slice_experts", record_slice)
    monkeypatch.setattr(
        prune_module,
        "get_super_expert_indices",
        lambda *_args, **_kwargs: torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )

    # Requesting three removals is impossible with two protected experts in
    # each four-expert layer. The documented best-effort policy uniformly
    # reduces this to one unprotected removal per layer.
    prune_module.apply_pruning(
        observer_data, model, PruneArgs(preserve_super_experts=True), 3
    )

    assert len(kept) == 2
    assert all({0, 1}.issubset(indices) for indices in kept)
    assert {len(indices) for indices in kept} == {3}
    assert model.config.num_experts == 3
    assert model.config.num_experts_per_tok <= model.config.num_experts


class _Tokenizer:
    def save_pretrained(self, path):
        Path(path, "tokenizer.json").write_text("{}")


def test_smoke_failure_never_publishes_or_replaces_existing_output(tmp_path: Path, monkeypatch):
    destination = tmp_path / "pruned"
    destination.mkdir()
    (destination / "existing.txt").write_text("keep")
    monkeypatch.setattr(
        prune_module,
        "stream_save_pretrained",
        lambda *_args, **_kwargs: pytest.fail("save must not run after smoke failure"),
    )

    with pytest.raises(RuntimeError, match="forced smoke failure"):
        prune_module.publish_pruned_model(
            object(),
            _Tokenizer(),
            destination,
            smoke_test_fn=lambda: (_ for _ in ()).throw(RuntimeError("forced smoke failure")),
        )

    assert (destination / "existing.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".pruned.tmp-*"))


def test_successful_publish_replaces_destination_after_staging(tmp_path: Path, monkeypatch):
    destination = tmp_path / "pruned"
    destination.mkdir()
    (destination / "old.txt").write_text("old")

    def save_model(_model, output_dir):
        Path(output_dir, "model.safetensors").write_text("weights")

    monkeypatch.setattr(prune_module, "stream_save_pretrained", save_model)
    output = prune_module.publish_pruned_model(object(), _Tokenizer(), destination)

    assert output == destination
    assert (destination / "model.safetensors").exists()
    assert (destination / "tokenizer.json").exists()
    assert not (destination / "old.txt").exists()
    assert not list(tmp_path.glob(".pruned.tmp-*"))
    assert not list(tmp_path.glob(".pruned.old-*"))


def test_layerwise_prune_never_uses_gpu_full_plan_for_mutation(tmp_path: Path, monkeypatch):
    plans: list[str] = []
    fake_model = nn.Linear(1, 1)
    fake_model.config = type("Config", (), {})()
    fake_tokenizer = MagicMock()
    observer_data = {0: {"expert_frequency": torch.ones(4), "total_tokens": torch.tensor(1)}}

    def fake_plan(mode, **_kwargs):
        plans.append(mode)
        return LoadPlan(
            resolved=mode,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder=None,
            stream_save_from_gpu=True,
            avoid_cpu_materialize=True,
            reason="test",
        )

    monkeypatch.setattr("reap.residency.plan_load", fake_plan)
    monkeypatch.setattr("reap.residency.resolve_residency", lambda *_args, **_kwargs: ("layerwise", "test"))
    monkeypatch.setattr("reap.residency.estimate_model_bytes_from_config", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr("reap.residency.load_causal_lm", lambda *_args, **_kwargs: fake_model)
    monkeypatch.setattr("reap.layerwise_prune.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: fake_tokenizer)
    monkeypatch.setattr("reap.layerwise_prune.prepare_calibration_batches", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("reap.layerwise_prune.record_activations_layerwise", lambda *_args, **_kwargs: observer_data)
    monkeypatch.setattr("reap.layerwise_prune.apply_pruning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("reap.prune.publish_pruned_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("reap.residency.log_model_residency", lambda *_args, **_kwargs: None)

    run_layerwise(
        ReapArgs(
            profile=False,
            smoke_test=False,
            artifacts_dir=str(tmp_path),
            residency="layerwise",
        ),
        DatasetArgs(dataset_name="fixture"),
        ObserverArgs(output_file_name="observations.pt"),
        ModelArgs(model_name="fixture-model"),
        EvalArgs(run_lm_eval=False),
        PruneArgs(n_experts_to_prune=1),
        ClusterArgs(),
        LayerwiseArgs(),
    )

    assert plans and set(plans) == {"layerwise"}
