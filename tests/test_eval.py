"""Hermetic regression tests for configurable lm-eval integration."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from reap.args import EvalArgs, ModelArgs
from reap.eval import run_evaluate


def _install_fake_lm_eval(monkeypatch, calls: list[dict]) -> None:
    lm_eval = types.ModuleType("lm_eval")

    def simple_evaluate(**kwargs):
        calls.append(kwargs)
        model_name = getattr(kwargs["model"], "model_name", "candidate")
        score = 0.75 if model_name == "candidate" else 0.5
        return {"results": {"tiny_task": {"acc,none": score}}}

    lm_eval.simple_evaluate = simple_evaluate
    models = types.ModuleType("lm_eval.models")
    huggingface = types.ModuleType("lm_eval.models.huggingface")

    class HFLM:
        def __init__(self, *, pretrained, tokenizer, batch_size):
            self.model_name = pretrained.model_name
            self.tokenizer = tokenizer
            self.batch_size = batch_size

    huggingface.HFLM = HFLM
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", huggingface)


def test_run_evaluate_writes_results_and_baseline_diff(tmp_path: Path, monkeypatch):
    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls)

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

    monkeypatch.setattr(
        "reap.eval.AutoModelForCausalLM.from_pretrained",
        lambda model_name, **_kwargs: FakeModel(model_name),
    )
    monkeypatch.setattr(
        "reap.eval.AutoTokenizer.from_pretrained",
        lambda model_name, **_kwargs: f"tokenizer:{model_name}",
    )

    args = EvalArgs(
        lm_eval_tasks=["tiny_task"],
        eval_num_fewshot=2,
        eval_batch_size=4,
        eval_limit=2,
        eval_baseline=True,
    )
    result = run_evaluate(
        ModelArgs(model_name="candidate"),
        tmp_path,
        args,
        seed=7,
        baseline_model_name="baseline",
        baseline_model_args=ModelArgs(model_name="baseline"),
    )

    assert result["results"]["tiny_task"]["acc,none"] == 0.75
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "baseline_results.json").exists()
    diff = json.loads((tmp_path / "diff.json").read_text())
    assert diff["tasks"]["tiny_task"]["acc,none"]["delta"] == 0.25
    assert all(call["limit"] == 2 for call in calls)
    assert all(call["num_fewshot"] == 2 for call in calls)
    assert all(call["batch_size"] == 4 for call in calls)


def test_vllm_backend_errors_clearly_when_not_installed(tmp_path: Path):
    args = EvalArgs(lm_eval_tasks=["tiny_task"], eval_backend="vllm")
    with pytest.raises(RuntimeError, match="requires vllm"):
        run_evaluate(ModelArgs(model_name="candidate"), tmp_path, args, seed=1)


def test_requested_unimplemented_evaluator_warns(tmp_path: Path, caplog):
    args = EvalArgs(run_lm_eval=False, run_evalplus=True)
    run_evaluate(ModelArgs(model_name="candidate"), tmp_path, args, seed=1)
    assert "evalplus was requested but is not implemented" in caplog.text
