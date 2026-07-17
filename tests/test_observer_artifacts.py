"""Regression coverage for authoritative observer artifacts and manifests."""
from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import torch

from reap.args import DatasetArgs, ModelArgs, ObserverArgs, ReapArgs
from reap.pipeline import (
    _compute_artifact_metadata,
    create_results_directory,
    load_observer_artifact,
    record_activations,
    write_observer_metadata,
)


class _Observer:
    def __init__(self):
        self.frequency = torch.zeros(2)
        self.total_tokens = torch.tensor(0)

    def set_attention_mask(self, _mask):
        return contextlib.nullcontext()

    def report_state(self):
        return {
            0: {
                "expert_frequency": self.frequency.clone(),
                "total_tokens": self.total_tokens.clone(),
            }
        }

    def save_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.report_state(), path)

    def close_hooks(self):
        pass


class _Model:
    def __init__(self):
        self.observer: _Observer | None = None

    def __call__(self, *, input_ids, **_kwargs):
        assert self.observer is not None
        self.observer.frequency += torch.tensor([1.0, 2.0])
        self.observer.total_tokens += input_ids.numel()


def _args():
    return (
        ReapArgs(profile=False),
        ModelArgs(model_name="owner-a/model"),
        DatasetArgs(dataset_name="fixture", shuffle=False),
        ObserverArgs(output_file_name="observations.pt"),
    )


def test_full_observer_aggregate_contains_all_categories_and_reuses_manifest(
    tmp_path: Path, monkeypatch
):
    reap_args, model_args, ds_args, obs_args = _args()
    model = _Model()
    observer = _Observer()
    model.observer = observer
    batches = {
        "first": [{"input_ids": torch.tensor([[1, 2]])}],
        "second": [{"input_ids": torch.tensor([[3, 4, 5]])}],
    }
    monkeypatch.setattr("reap.pipeline._setup_observer", lambda *_args: observer)
    monkeypatch.setattr("reap.pipeline._primary_device", lambda _model: torch.device("cpu"))
    monkeypatch.setattr("reap.pipeline.load_category_batches", lambda **_kwargs: batches)

    result = record_activations(
        model, None, reap_args, model_args, ds_args, obs_args, tmp_path
    )
    assert torch.equal(result[0]["expert_frequency"], torch.tensor([2.0, 4.0]))
    assert result[0]["total_tokens"].item() == 5
    aggregate = tmp_path / "all" / "observations.pt"
    assert aggregate.exists()
    assert aggregate.with_suffix(".pt.meta.json").exists()

    monkeypatch.setattr(
        "reap.pipeline.load_category_batches",
        lambda **_kwargs: pytest.fail("valid aggregate cache should be reused"),
    )
    cached = record_activations(
        model, None, reap_args, model_args, ds_args, obs_args, tmp_path
    )
    assert torch.equal(cached[0]["expert_frequency"], torch.tensor([2.0, 4.0]))


def test_artifact_paths_include_model_namespace_and_hash(tmp_path: Path):
    a = create_results_directory("owner-a/model", "dataset", base=tmp_path)
    b = create_results_directory("owner-b/model", "dataset", base=tmp_path)
    assert a != b


def test_manifest_mismatch_and_invalid_schema_are_rejected(tmp_path: Path):
    reap_args, model_args, ds_args, obs_args = _args()
    metadata = _compute_artifact_metadata(reap_args, model_args, ds_args, obs_args)
    path = tmp_path / "observations.pt"
    valid = {0: {"expert_frequency": torch.ones(2), "total_tokens": torch.tensor(2)}}
    torch.save(valid, path)
    write_observer_metadata(path, metadata)

    loaded = load_observer_artifact(path, expected_metadata=metadata)
    assert torch.equal(loaded[0]["expert_frequency"], valid[0]["expert_frequency"])
    with pytest.raises(ValueError, match="incompatible manifest"):
        load_observer_artifact(path, expected_metadata={**metadata, "schema_version": 999})

    invalid = tmp_path / "invalid.pt"
    torch.save({0: {"total_tokens": torch.tensor(2)}}, invalid)
    with pytest.raises(ValueError, match="expert_frequency"):
        load_observer_artifact(invalid)
