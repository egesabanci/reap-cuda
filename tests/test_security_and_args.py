"""Security-boundary and public argument wiring regressions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from typer.testing import CliRunner

from reap.args import DatasetArgs, ModelArgs
from reap.cli.app import app
from reap.cli.options import build_eval_args
from reap.pipeline import load_observer_artifact
from reap.residency import LoadPlan, load_causal_lm


def test_model_and_observation_trust_defaults_are_safe():
    assert ModelArgs().trust_remote_code is False
    assert not hasattr(ModelArgs(), "num_experts_per_tok_override")
    assert DatasetArgs().shuffle is True


def test_model_loader_propagates_explicit_trust_and_offline_controls(monkeypatch):
    captured: dict[str, object] = {}

    class FakeModel:
        def eval(self):
            return self

    def fake_load(_name, **kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_load)
    plan = LoadPlan("gpu_full", "auto", True, None, True, True, "test")
    load_causal_lm(
        "owner/model",
        plan,
        trust_remote_code=False,
        revision="deadbeef",
        local_files_only=True,
    )

    assert captured["trust_remote_code"] is False
    assert captured["revision"] == "deadbeef"
    assert captured["local_files_only"] is True


def test_legacy_observation_artifact_requires_explicit_trust(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {0: {"expert_frequency": torch.ones(2), "total_tokens": torch.tensor(2)}},
        path,
    )
    expected = {"schema_version": 3}
    with pytest.raises(ValueError, match="no manifest"):
        load_observer_artifact(path, expected_metadata=expected)
    loaded = load_observer_artifact(
        path, expected_metadata=expected, trust_legacy=True
    )
    assert torch.equal(loaded[0]["expert_frequency"], torch.ones(2))


def test_eval_task_csv_builder_is_deterministic_and_validates_values():
    args = build_eval_args(
        do_eval=True,
        lm_eval_tasks="hellaswag, arc_challenge",
        eval_batch_size=2,
        eval_limit=3,
    )
    assert args.lm_eval_tasks == ["hellaswag", "arc_challenge"]
    assert args.eval_batch_size == 2
    with pytest.raises(Exception, match="eval-batch-size"):
        build_eval_args(do_eval=True, eval_batch_size=0)


def test_cli_wires_shuffle_and_security_options(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **_kwargs):
        captured["dataset"] = args[1]
        captured["observer"] = args[2]
        captured["model"] = args[3]
        captured["eval"] = args[4]

    monkeypatch.setattr("reap.prune.run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "prune",
            "full",
            "--no-shuffle",
            "--trust-observation-artifact",
            "--model-revision",
            "deadbeef",
            "--local-files-only",
            "--observe-only",
            "--eval",
            "--eval-tasks",
            "hellaswag,arc_challenge",
            "--eval-num-fewshot",
            "2",
            "--eval-batch-size",
            "3",
            "--eval-limit",
            "4",
            "--eval-baseline",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["dataset"].shuffle is False
    assert captured["observer"].trust_observation_artifact is True
    assert captured["model"].model_revision == "deadbeef"
    assert captured["model"].local_files_only is True
    assert captured["eval"].lm_eval_tasks == ["hellaswag", "arc_challenge"]
    assert captured["eval"].eval_num_fewshot == 2
    assert captured["eval"].eval_batch_size == 3
    assert captured["eval"].eval_limit == 4
    assert captured["eval"].eval_baseline is True
