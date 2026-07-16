"""Hermetic tests for offline / composite dataset loading (#33–#39)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset

from reap.data import (
    _load_local_dataset,
    _load_raw_dataset,
    load_category_batches,
    load_composite_category_batches,
    parse_composite_dataset_spec,
    resolve_component_dataset_path,
)


class _FakeTok:
    model_max_length = 32
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, *a, **k):
        return {
            "input_ids": torch.ones(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }

    def apply_chat_template(self, *a, **k):
        return "x"

    def encode(self, *a, **k):
        return [1, 2, 3]


def _alpaca_disk(tmp_path: Path, n: int = 4) -> Path:
    ds = Dataset.from_dict(
        {
            "instruction": [f"q{i}" for i in range(n)],
            "output": [f"a{i}" for i in range(n)],
        }
    )
    out = tmp_path / "alpaca"
    ds.save_to_disk(str(out))
    return out


# ---- #36 composite parse: batches not samples ----


def test_composite_spec_parses_batch_count_and_at_path():
    comps = parse_composite_dataset_spec(
        "theblackcat102/evol-codealpaca-v1:64@/data/local,"
        "open-r1/Mixture-of-Thoughts[code]:32"
    )
    assert comps is not None
    assert len(comps) == 2
    assert comps[0].num_batches == 64
    assert comps[0].local_path == "/data/local"
    assert comps[1].subset == "code"
    assert comps[1].num_batches == 32
    assert comps[1].local_path is None


# ---- #33 composite honors dataset_path ----


def test_composite_load_uses_per_component_path(tmp_path: Path):
    disk = _alpaca_disk(tmp_path)
    comps = parse_composite_dataset_spec(
        f"theblackcat102/evol-codealpaca-v1:1@{disk}"
    )
    out = load_composite_category_batches(
        comps,
        tokenizer=_FakeTok(),
        model_max_length=32,
        batch_size=1,
        return_vllm_tokens_prompt=False,
        truncate=True,
        global_dataset_path=None,
    )
    assert "all" in out
    assert len(out["all"]) >= 1


def test_composite_global_path_single_component(tmp_path: Path):
    disk = _alpaca_disk(tmp_path)
    comps = parse_composite_dataset_spec("theblackcat102/evol-codealpaca-v1:1")
    out = load_composite_category_batches(
        comps,
        tokenizer=_FakeTok(),
        model_max_length=32,
        batch_size=1,
        return_vllm_tokens_prompt=False,
        truncate=True,
        global_dataset_path=str(disk),
    )
    assert len(out["all"]) >= 1


def test_composite_multi_file_global_path_errors(tmp_path: Path):
    arrow = tmp_path / "one.arrow"
    # build via save_to_disk then we need a file — use disk for multi-comp with file
    disk = _alpaca_disk(tmp_path / "d")
    # fake file path
    f = tmp_path / "f.arrow"
    f.write_bytes(b"not real")  # will fail later if used; we error before load
    comps = parse_composite_dataset_spec(
        "theblackcat102/evol-codealpaca-v1:1,open-r1/Mixture-of-Thoughts:1"
    )
    # Point global path at a single file while multi-component → ValueError
    # Use a real file that exists
    real_file = tmp_path / "data.jsonl"
    real_file.write_text('{"instruction":"a","output":"b"}\n')
    with pytest.raises(ValueError, match="single file"):
        load_composite_category_batches(
            comps,
            tokenizer=_FakeTok(),
            model_max_length=32,
            batch_size=1,
            return_vllm_tokens_prompt=False,
            truncate=True,
            global_dataset_path=str(real_file),
        )


def test_record_activations_threads_dataset_path(tmp_path: Path):
    """#39: record_activations passes DatasetArgs.dataset_path into loaders."""
    from reap.args import DatasetArgs, ModelArgs, ObserverArgs, ReapArgs
    from reap.pipeline import record_activations

    disk = _alpaca_disk(tmp_path)
    art = tmp_path / "art"
    art.mkdir()

    with patch("reap.pipeline.load_category_batches") as mock_lcb, patch(
        "reap.pipeline._setup_observer"
    ) as mock_obs:
        mock_lcb.return_value = {
            "all": [
                {
                    "input_ids": torch.ones(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }
            ]
        }
        mock_observer = MagicMock()
        mock_obs.return_value = mock_observer
        mock_observer.set_attention_mask.return_value.__enter__ = lambda s: None
        mock_observer.set_attention_mask.return_value.__exit__ = lambda *a: False
        mock_observer.close_hooks = MagicMock()
        mock_observer.save_state = MagicMock()
        mock_observer.reset = MagicMock()

        model = MagicMock()
        type(model).__call__ = lambda self, **k: None

        with patch(
            "reap.pipeline._primary_device", return_value=torch.device("cpu")
        ):
            try:
                record_activations(
                    model,
                    _FakeTok(),
                    ReapArgs(profile=False, run_observer_only=True),
                    ModelArgs(model_name="dummy"),
                    DatasetArgs(
                        dataset_name="theblackcat102/evol-codealpaca-v1",
                        dataset_path=str(disk),
                    ),
                    ObserverArgs(
                        batches_per_category=1,
                        batch_size=1,
                        overwrite_observations=True,
                        model_max_length=32,
                    ),
                    art,
                )
            except Exception:
                pass  # only care that load received dataset_path
        assert mock_lcb.called
        assert mock_lcb.call_args.kwargs["dataset_path"] == str(disk)


# ---- #34 field validation ----


def test_local_wrong_fields_raises_clear_error(tmp_path: Path):
    ds = Dataset.from_dict({"prompt": ["hi"], "completion": ["yo"]})
    disk = tmp_path / "wrong"
    ds.save_to_disk(str(disk))
    with pytest.raises(ValueError, match="instruction"):
        load_category_batches(
            dataset_name="theblackcat102/evol-codealpaca-v1",
            split="train",
            subset=None,
            tokenizer=_FakeTok(),
            model_max_length=32,
            batch_size=1,
            split_by_category=False,
            return_vllm_tokens_prompt=False,
            truncate=True,
            batches_per_category=1,
            dataset_path=str(disk),
        )


def test_local_unknown_dataset_name_raises(tmp_path: Path):
    disk = _alpaca_disk(tmp_path)
    with pytest.raises(ValueError, match="No DatasetProcessor"):
        load_category_batches(
            dataset_name="my-unregistered-local",
            split="train",
            subset=None,
            tokenizer=_FakeTok(),
            model_max_length=32,
            batch_size=1,
            split_by_category=False,
            return_vllm_tokens_prompt=False,
            truncate=True,
            batches_per_category=1,
            dataset_path=str(disk),
        )


# ---- #35 arrow split warning ----


def test_arrow_file_loads_and_warns_on_non_train_split(tmp_path: Path, caplog):
    import logging

    # Build a real arrow via save_to_disk is a dir; for single .arrow use from_file write
    # datasets doesn't easily write single arrow without arrow writer — use save_to_disk
    # and also test warn path via monkeypatch
    disk = _alpaca_disk(tmp_path)
    # save_to_disk is a Dataset dir; loading with split=validation on single Dataset warns
    with caplog.at_level(logging.WARNING):
        ds = _load_local_dataset(str(disk), split="validation")
    assert len(ds) == 4
    assert any("ignoring" in r.message.lower() or "split" in r.message.lower() for r in caplog.records)


# ---- #37 c4 offline ----


def test_c4_offline_raises_clear_error(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="dataset-path|offline|HF_HUB"):
        _load_raw_dataset("allenai/c4", "train")


def test_hub_offline_blocks_generic_hub(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="dataset-path|OFFLINE"):
        load_category_batches(
            dataset_name="theblackcat102/evol-codealpaca-v1",
            split="train",
            subset=None,
            tokenizer=_FakeTok(),
            model_max_length=32,
            batch_size=1,
            split_by_category=False,
            return_vllm_tokens_prompt=False,
            truncate=True,
            batches_per_category=1,
            dataset_path=None,
        )


# ---- #38 hub error suggests dataset-path ----


def test_hub_failure_mentions_dataset_path(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)

    def _boom(*a, **k):
        raise OSError("Network unreachable")

    monkeypatch.setattr("reap.data.load_dataset", _boom)
    with pytest.raises(RuntimeError, match="dataset-path") as ei:
        _load_raw_dataset("some/hub-id", "train")
    assert "dataset-path" in str(ei.value).lower() or "local" in str(ei.value).lower()


# ---- resolve helper ----


def test_resolve_component_path_prefers_at_path(tmp_path: Path):
    from reap.data import CompositeDatasetComponent

    c = CompositeDatasetComponent(
        name="theblackcat102/evol-codealpaca-v1",
        split="train",
        subset=None,
        num_batches=1,
        local_path=str(tmp_path / "x"),
    )
    assert resolve_component_dataset_path(c, "/global") == str(tmp_path / "x")
