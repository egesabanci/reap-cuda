"""Unit tests for weight residency policy (no model downloads)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from reap.residency import (
    RESIDENCY_MODES,
    LoadPlan,
    MemorySnapshot,
    estimate_model_bytes_from_module,
    plan_load,
    preflight_or_warn,
    resolve_residency,
    stream_save_pretrained,
    validate_residency,
)


def _mem(
    *,
    host_total: int,
    host_available: int | None = None,
    gpu_total: int | None,
    gpu_available: int | None = None,
) -> MemorySnapshot:
    return MemorySnapshot(
        host_total=host_total,
        host_available=host_available if host_available is not None else host_total // 2,
        gpu_total=gpu_total,
        gpu_available=gpu_available if gpu_available is not None else gpu_total,
    )


class TestValidate:
    def test_valid_modes(self):
        for m in RESIDENCY_MODES:
            assert validate_residency(m) == m

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown residency"):
            validate_residency("disk_only")


class TestResolveAuto:
    def test_explicit_passthrough(self):
        mode, reason = resolve_residency(
            "gpu_full",
            model_bytes=16 * 1024**3,
            mem=_mem(host_total=16 * 1024**3, gpu_total=24 * 1024**3),
        )
        assert mode == "gpu_full"
        assert "explicit" in reason

    def test_g6_xlarge_like_picks_gpu_full(self):
        """~16GiB model, 16GiB host, 24GiB GPU → gpu_full (LFM2-8B case)."""
        mode, reason = resolve_residency(
            "auto",
            model_bytes=int(15.5 * 1024**3),
            mem=_mem(host_total=16 * 1024**3, gpu_total=24 * 1024**3),
            cli_prefers_layerwise=False,
        )
        assert mode == "gpu_full"
        assert "fits GPU" in reason or "gpu" in reason.lower()

    def test_layerwise_cli_with_huge_model(self):
        mode, reason = resolve_residency(
            "auto",
            model_bytes=60 * 1024**3,
            mem=_mem(host_total=46 * 1024**3, gpu_total=46 * 1024**3),
            cli_prefers_layerwise=True,
        )
        assert mode == "layerwise"

    def test_no_gpu_safe_host(self):
        mode, _ = resolve_residency(
            "auto",
            model_bytes=2 * 1024**3,
            mem=_mem(host_total=64 * 1024**3, gpu_total=None),
        )
        assert mode == "cpu_full"

    def test_no_estimate_with_cuda(self):
        mode, _ = resolve_residency(
            "auto",
            model_bytes=None,
            mem=_mem(host_total=16 * 1024**3, gpu_total=24 * 1024**3),
            cli_prefers_layerwise=False,
        )
        assert mode == "gpu_full"

    def test_layerwise_cli_no_estimate_limited_ram(self):
        mode, _ = resolve_residency(
            "auto",
            model_bytes=None,
            mem=_mem(host_total=16 * 1024**3, host_available=8 * 1024**3, gpu_total=24 * 1024**3),
            cli_prefers_layerwise=True,
        )
        assert mode == "layerwise"


class TestPlanLoad:
    def test_gpu_full_plan(self):
        plan = plan_load("gpu_full")
        assert plan.device_map == "auto"
        assert plan.stream_save_from_gpu is True
        assert plan.avoid_cpu_materialize is True
        assert plan.offload_folder is None

    def test_cpu_full_plan(self):
        plan = plan_load("cpu_full")
        assert plan.device_map == "cpu"
        assert plan.stream_save_from_gpu is False

    def test_layerwise_plan_uses_offload(self, tmp_path: Path):
        plan = plan_load("layerwise", offload_root=tmp_path / "off")
        assert plan.resolved == "layerwise"
        assert plan.offload_folder is not None
        assert Path(plan.offload_folder).exists()
        assert plan.device_map == "auto"
        assert plan.avoid_cpu_materialize is True

    def test_plan_rejects_auto(self):
        with pytest.raises(ValueError, match="resolved"):
            plan_load("auto")


class TestPreflight:
    def test_warns_cpu_full_too_big(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            preflight_or_warn(
                "cpu_full",
                int(14 * 1024**3),
                _mem(host_total=16 * 1024**3, gpu_total=24 * 1024**3),
            )
        assert any("cpu_full" in r.message for r in caplog.records)

    def test_strict_raises(self):
        with pytest.raises(RuntimeError, match="cpu_full"):
            preflight_or_warn(
                "cpu_full",
                int(14 * 1024**3),
                _mem(host_total=16 * 1024**3, gpu_total=24 * 1024**3),
                strict=True,
            )


class TestStreamSave:
    def test_stream_save_calls_save_pretrained(self, tmp_path: Path):
        model = MagicMock()
        model.save_pretrained = MagicMock()
        stream_save_pretrained(model, tmp_path / "out")
        model.save_pretrained.assert_called_once()
        assert (tmp_path / "out").is_dir()


class TestEstimateFromModule:
    def test_counts_params(self):
        m = nn.Linear(16, 32, bias=False)
        # 16*32 * 4 bytes float32
        assert estimate_model_bytes_from_module(m) == 16 * 32 * 4


class TestCliResidencyWiring:
    def test_help_shows_residency(self):
        from typer.testing import CliRunner

        from reap.cli.app import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["prune", "full", "--help"],
            color=False,
            env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"},
        )
        assert result.exit_code == 0
        assert "residency" in result.stdout.lower()

    @patch("reap.prune.run")
    def test_prune_full_passes_residency(self, mock_run: MagicMock):
        from typer.testing import CliRunner

        from reap.cli.app import app

        mock_run.return_value = None
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "prune",
                "full",
                "--observe-only",
                "--residency",
                "gpu_full",
            ],
            color=False,
            env={"NO_COLOR": "1", "TERM": "dumb"},
        )
        assert result.exit_code == 0, result.stdout
        mock_run.assert_called_once()
        reap_args = mock_run.call_args.args[0]
        assert reap_args.residency == "gpu_full"

    @patch("reap.layerwise_prune.run")
    def test_prune_layerwise_passes_residency(self, mock_run: MagicMock):
        from typer.testing import CliRunner

        from reap.cli.app import app

        mock_run.return_value = None
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["prune", "layerwise", "--observe-only", "--residency", "auto"],
            color=False,
            env={"NO_COLOR": "1", "TERM": "dumb"},
        )
        assert result.exit_code == 0, result.stdout
        assert mock_run.call_args.args[0].residency == "auto"


class TestDelegation:
    @patch("reap.layerwise_prune.run")
    def test_full_run_delegates_when_resolved_layerwise(self, mock_lw: MagicMock):
        from reap.args import (
            ClusterArgs,
            DatasetArgs,
            EvalArgs,
            ModelArgs,
            ObserverArgs,
            PruneArgs,
            ReapArgs,
        )
        from reap.prune import run as prune_run

        mock_lw.return_value = Path("/tmp/x")
        with patch(
            "reap.prune.resolve_residency",
            return_value=("layerwise", "test"),
        ), patch("reap.prune.estimate_model_bytes_from_config", return_value=None):
            out = prune_run(
                ReapArgs(residency="auto", run_observer_only=True),
                DatasetArgs(),
                ObserverArgs(),
                ModelArgs(model_name="dummy/model"),
                EvalArgs(),
                PruneArgs(),
                ClusterArgs(),
            )
        mock_lw.assert_called_once()
        assert out == Path("/tmp/x")
        assert mock_lw.call_args.kwargs.get("_residency_resolved") == "layerwise"

    @patch("reap.prune.run")
    def test_layerwise_run_delegates_when_gpu_full(self, mock_full: MagicMock):
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
        from reap.layerwise_prune import run as lw_run

        mock_full.return_value = Path("/tmp/y")
        with patch(
            "reap.residency.resolve_residency",
            return_value=("gpu_full", "test"),
        ), patch(
            "reap.residency.estimate_model_bytes_from_config", return_value=None
        ):
            out = lw_run(
                ReapArgs(residency="auto", run_observer_only=True),
                DatasetArgs(),
                ObserverArgs(),
                ModelArgs(model_name="dummy/model"),
                EvalArgs(),
                PruneArgs(),
                ClusterArgs(),
                LayerwiseArgs(),
            )
        mock_full.assert_called_once()
        assert out == Path("/tmp/y")
        assert mock_full.call_args.kwargs.get("_residency_resolved") == "gpu_full"
