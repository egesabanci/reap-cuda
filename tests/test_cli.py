"""Unit tests for the Typer-based REAP CLI.

Heavy pipeline work (model load, observe, prune, merge) is mocked so these
tests only validate CLI structure, option parsing, and dispatch into the
``run()`` entrypoints.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from reap.args import (
    ClusterArgs,
    DatasetArgs,
    EvalArgs,
    LayerwiseArgs,
    MergeArgs,
    ModelArgs,
    ObserverArgs,
    PruneArgs,
    ReapArgs,
)
from reap.cli.app import app
from reap.cli import options as opt

runner = CliRunner()

_ENV = {"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"}


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _invoke(*args: str, expect_exit: int = 0):
    result = runner.invoke(app, list(args), color=False, env=_ENV)
    assert result.exit_code == expect_exit, (
        f"exit={result.exit_code}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\nexc={result.exception}"
    )
    return result


def _help(*args: str) -> str:
    return _strip_ansi(_invoke(*args, "--help").stdout)


# ---------------------------------------------------------------------------
# Help / version / structure
# ---------------------------------------------------------------------------


class TestCliStructure:
    def test_root_help_lists_groups(self):
        out = _help()
        assert "prune" in out
        assert "merge" in out
        assert "version" in out

    def test_prune_help_lists_subcommands(self):
        out = _help("prune")
        assert "full" in out
        assert "layerwise" in out

    def test_merge_help_lists_subcommands(self):
        out = _help("merge")
        assert "full" in out
        assert "layerwise" in out

    def test_prune_full_help_options(self):
        out = _help("prune", "full")
        # Rich may ellipsize long flag names; match stable prefixes.
        for token in (
            "--model",
            "--dataset",
            "--compression-ratio",
            "--observe-backend",
            "--prune-method",
            "--observe-only",
            "--eval",
            "--smoke-test",
            "preserve-super",
        ):
            assert token in out, f"missing {token!r}"

    def test_prune_layerwise_help_options(self):
        out = _help("prune", "layerwise")
        for flag in (
            "--model",
            "--batch-group-size",
            "--low-cpu-mem",
            "--save-intermediate",
            "--observe-backend",
            "--compression-ratio",
        ):
            assert flag in out, f"missing {flag}"

    def test_merge_full_help_options(self):
        out = _help("merge", "full")
        for flag in (
            "--expert-sim",
            "--merge-method",
            "--cluster-method",
            "--linkage",
            "--skip-first",
            "--skip-last",
            "--permute",
            "--distance",
        ):
            assert flag in out, f"missing {flag}"

    def test_merge_layerwise_help_options(self):
        out = _help("merge", "layerwise")
        for flag in ("--batch-group-size", "--expert-sim", "--merge-method"):
            assert flag in out, f"missing {flag}"

    def test_version_prints_semver_like(self):
        out = _strip_ansi(_invoke("version").stdout).strip()
        assert re.match(r"^\d+\.\d+\.\d+", out)

    def test_unknown_command_fails(self):
        result = runner.invoke(app, ["not-a-command"], color=False, env=_ENV)
        assert result.exit_code != 0

    def test_prune_without_subcommand_shows_help(self):
        # no_args_is_help=True on prune group
        result = runner.invoke(app, ["prune"], color=False, env=_ENV)
        assert result.exit_code in (0, 2)
        out = _strip_ansi(result.stdout + (result.stderr or ""))
        assert "full" in out or "Usage" in out or "layerwise" in out


# ---------------------------------------------------------------------------
# Option builders (pure, no Typer)
# ---------------------------------------------------------------------------


class TestOptionBuilders:
    def test_build_prune_args_maps_preserve_fields(self):
        args = opt.build_prune_args(
            preserve_super_experts=True,
            preserve_outliers=False,
            prune_method="reap",
            n_experts_to_prune=16,
            overwrite_pruned_model=True,
        )
        assert isinstance(args, PruneArgs)
        assert args.perserve_super_experts is True
        assert args.perserve_outliers is False
        assert args.prune_method == "reap"
        assert args.n_experts_to_prune == 16
        assert args.overwrite_pruned_model is True

    def test_build_observer_args(self):
        args = opt.build_observer_args(
            observe_backend="bmm",
            batch_size=2,
            batches_per_category=8,
            record_pruning_metrics_only=False,
            overwrite_observations=True,
        )
        assert isinstance(args, ObserverArgs)
        assert args.observe_backend == "bmm"
        assert args.batch_size == 2
        assert args.batches_per_category == 8
        assert args.record_pruning_metrics_only is False
        assert args.overwrite_observations is True

    def test_build_reap_args(self):
        args = opt.build_reap_args(
            seed=7,
            run_observer_only=True,
            do_eval=True,
            smoke_test=False,
            profile=False,
        )
        assert isinstance(args, ReapArgs)
        assert args.seed == 7
        assert args.run_observer_only is True
        assert args.do_eval is True
        assert args.smoke_test is False
        assert args.profile is False

    def test_build_cluster_and_merge_args(self):
        c = opt.build_cluster_args(
            compression_ratio=0.25,
            expert_sim="characteristic_activation",
            cluster_method="agglomerative",
            linkage_method="ward",
        )
        m = opt.build_merge_args(
            merge_method="ties",
            skip_first=True,
            skip_last=True,
            permute="direct",
        )
        assert isinstance(c, ClusterArgs)
        assert c.compression_ratio == 0.25
        assert c.expert_sim == "characteristic_activation"
        assert c.linkage_method == "ward"
        assert isinstance(m, MergeArgs)
        assert m.merge_method == "ties"
        assert m.skip_first and m.skip_last
        assert m.permute == "direct"

    def test_build_layerwise_args(self):
        args = opt.build_layerwise_args(
            batch_group_size=4,
            save_intermediate=True,
            low_cpu_mem_usage=False,
        )
        assert isinstance(args, LayerwiseArgs)
        assert args.batch_group_size == 4
        assert args.save_intermediate is True
        assert args.low_cpu_mem_usage is False

    def test_build_eval_args_disables_stubs_and_gates_lm_eval(self):
        off = opt.build_eval_args(do_eval=False)
        assert off.run_lm_eval is False
        assert off.run_evalplus is False
        assert off.run_livecodebench is False

        on = opt.build_eval_args(do_eval=True, lm_eval_tasks=["arc_easy"])
        assert on.run_lm_eval is True
        assert on.lm_eval_tasks == ["arc_easy"]

    def test_build_model_and_dataset_args(self):
        m = opt.build_model_args(model_name="org/model")
        d = opt.build_dataset_args(
            dataset_name="ds/name",
            dataset_config_name="cfg",
            split="validation",
        )
        assert isinstance(m, ModelArgs) and m.model_name == "org/model"
        assert isinstance(d, DatasetArgs)
        assert d.dataset_name == "ds/name"
        assert d.dataset_config_name == "cfg"
        assert d.split == "validation"


# ---------------------------------------------------------------------------
# Dispatch: prune full / layerwise (mocked run)
# ---------------------------------------------------------------------------


class TestPruneFullDispatch:
    @patch("reap.prune.run")
    def test_dispatches_with_mapped_options(self, mock_run: MagicMock):
        mock_run.return_value = Path("/tmp/pruned")
        result = _invoke(
            "prune",
            "full",
            "--model",
            "org/Tiny-MoE",
            "--dataset",
            "my/data",
            "--prune-method",
            "reap",
            "--compression-ratio",
            "0.3",
            "--n-experts-to-prune",
            "10",
            "--observe-backend",
            "bmm",
            "--batch-size",
            "2",
            "--batches-per-category",
            "16",
            "--model-max-length",
            "512",
            "--seed",
            "99",
            "--observe-only",
            "--overwrite-observations",
            "--overwrite-pruned",
            "--all-metrics",
            "--no-renorm-router",
            "--preserve-super-experts",
            "--no-smoke-test",
            "--eval",
            "--no-profile",
            "--dataset-config",
            "subset_a",
            "--split",
            "train",
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        (
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
        ) = mock_run.call_args.args

        assert isinstance(reap_args, ReapArgs)
        assert reap_args.seed == 99
        assert reap_args.run_observer_only is True
        assert reap_args.do_eval is True
        assert reap_args.smoke_test is False
        assert reap_args.profile is False

        assert model_args.model_name == "org/Tiny-MoE"
        assert ds_args.dataset_name == "my/data"
        assert ds_args.dataset_config_name == "subset_a"
        assert ds_args.split == "train"

        assert obs_args.observe_backend == "bmm"
        assert obs_args.batch_size == 2
        assert obs_args.batches_per_category == 16
        assert obs_args.model_max_length == 512
        assert obs_args.overwrite_observations is True
        assert obs_args.record_pruning_metrics_only is False  # --all-metrics
        assert obs_args.renormalize_router_weights is False

        assert prune_args.prune_method == "reap"
        assert prune_args.n_experts_to_prune == 10
        assert prune_args.overwrite_pruned_model is True
        assert prune_args.perserve_super_experts is True

        assert cluster_args.compression_ratio == pytest.approx(0.3)
        assert eval_args.run_lm_eval is True

    @patch("reap.prune.run")
    def test_short_flags_model_and_dataset(self, mock_run: MagicMock):
        mock_run.return_value = None
        _invoke("prune", "full", "-m", "M", "-d", "D", "--observe-only")
        _, ds_args, _, model_args, *_ = mock_run.call_args.args
        assert model_args.model_name == "M"
        assert ds_args.dataset_name == "D"

    @patch("reap.prune.run")
    def test_defaults_for_full_prune(self, mock_run: MagicMock):
        mock_run.return_value = None
        _invoke("prune", "full", "--observe-only")
        (
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
        ) = mock_run.call_args.args
        assert model_args.model_name == "Qwen/Qwen3-30B-A3B"
        assert ds_args.dataset_name == "theblackcat102/evol-codealpaca-v1"
        assert prune_args.prune_method == "reap"
        assert cluster_args.compression_ratio == pytest.approx(0.5)
        assert obs_args.observe_backend == "auto"
        assert obs_args.record_pruning_metrics_only is True
        assert reap_args.do_eval is False
        assert eval_args.run_evalplus is False

    @patch("reap.prune.run", side_effect=RuntimeError("boom"))
    def test_pipeline_exception_surfaces(self, mock_run: MagicMock):
        result = runner.invoke(
            app,
            ["prune", "full", "--observe-only"],
            color=False,
            env=_ENV,
        )
        assert result.exit_code != 0
        assert result.exception is not None
        assert "boom" in str(result.exception)


class TestPruneLayerwiseDispatch:
    @patch("reap.layerwise_prune.run")
    def test_dispatches_layerwise_options(self, mock_run: MagicMock):
        mock_run.return_value = Path("/tmp/lw")
        _invoke(
            "prune",
            "layerwise",
            "-m",
            "org/MoE",
            "-d",
            "ds/x",
            "--prune-method",
            "frequency",
            "--compression-ratio",
            "0.4",
            "--observe-backend",
            "f2",
            "--batch-size",
            "1",
            "--batch-group-size",
            "3",
            "--save-intermediate",
            "--no-low-cpu-mem",
            "--seed",
            "1",
            "--observe-only",
            "--preserve-outliers",
        )
        mock_run.assert_called_once()
        (
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
            layerwise_args,
        ) = mock_run.call_args.args

        assert model_args.model_name == "org/MoE"
        assert ds_args.dataset_name == "ds/x"
        assert reap_args.run_observer_only is True
        assert reap_args.profile is False
        assert reap_args.smoke_test is False
        assert obs_args.observe_backend == "f2"
        assert obs_args.batch_size == 1
        assert prune_args.prune_method == "frequency"
        assert prune_args.perserve_outliers is True
        assert cluster_args.compression_ratio == pytest.approx(0.4)
        assert isinstance(layerwise_args, LayerwiseArgs)
        assert layerwise_args.batch_group_size == 3
        assert layerwise_args.save_intermediate is True
        assert layerwise_args.low_cpu_mem_usage is False


# ---------------------------------------------------------------------------
# Dispatch: merge full / layerwise (mocked run)
# ---------------------------------------------------------------------------


class TestMergeFullDispatch:
    @patch("reap.merge_pipeline.run")
    def test_dispatches_with_cluster_and_merge_options(self, mock_run: MagicMock):
        mock_run.return_value = Path("/tmp/merged")
        _invoke(
            "merge",
            "full",
            "-m",
            "org/MoE",
            "-d",
            "ds/y",
            "--compression-ratio",
            "0.5",
            "--expert-sim",
            "characteristic_activation",
            "--cluster-method",
            "agglomerative",
            "--linkage",
            "complete",
            "--merge-method",
            "ties",
            "--distance",
            "cosine",
            "--observe-backend",
            "loop",
            "--skip-first",
            "--skip-last",
            "--no-frequency-penalty",
            "--permute",
            "wm",
            "--overwrite-merged",
            "--overwrite-observations",
            "--observe-only",
            "--seed",
            "5",
            "--no-profile",
        )
        mock_run.assert_called_once()
        (
            reap_args,
            model_args,
            ds_args,
            obs_args,
            cluster_args,
            merge_args,
            eval_args,
        ) = mock_run.call_args.args

        # Argument order for merge_pipeline.run: reap, model, ds, obs, cluster, merge, eval
        assert model_args.model_name == "org/MoE"
        assert ds_args.dataset_name == "ds/y"
        assert reap_args.seed == 5
        assert reap_args.run_observer_only is True
        assert reap_args.profile is False

        assert obs_args.record_pruning_metrics_only is False  # merge always needs all
        assert obs_args.distance_measure == "cosine"
        assert obs_args.observe_backend == "loop"
        assert obs_args.overwrite_observations is True

        assert cluster_args.expert_sim == "characteristic_activation"
        assert cluster_args.cluster_method == "agglomerative"
        assert cluster_args.linkage_method == "complete"
        assert cluster_args.frequency_penalty is False
        assert cluster_args.compression_ratio == pytest.approx(0.5)

        assert merge_args.merge_method == "ties"
        assert merge_args.skip_first is True
        assert merge_args.skip_last is True
        assert merge_args.permute == "wm"
        assert merge_args.overwrite_merged_model is True

    @patch("reap.merge_pipeline.run")
    def test_merge_defaults_force_all_metrics(self, mock_run: MagicMock):
        mock_run.return_value = None
        _invoke("merge", "full", "--observe-only")
        # order: reap, model, ds, obs, cluster, merge, eval
        args = mock_run.call_args.args
        assert args[3].record_pruning_metrics_only is False
        assert args[4].expert_sim == "characteristic_activation"
        assert args[5].merge_method == "frequency_weighted_average"


class TestMergeLayerwiseDispatch:
    @patch("reap.layerwise_merge.run")
    def test_dispatches_layerwise_merge(self, mock_run: MagicMock):
        mock_run.return_value = Path("/tmp/lm")
        _invoke(
            "merge",
            "layerwise",
            "-m",
            "org/MoE",
            "--expert-sim",
            "ttm",
            "--merge-method",
            "average",
            "--batch-group-size",
            "2",
            "--save-intermediate",
            "--observe-backend",
            "bmm",
            "--observe-only",
        )
        mock_run.assert_called_once()
        (
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            cluster_args,
            merge_args,
            layerwise_args,
        ) = mock_run.call_args.args

        assert model_args.model_name == "org/MoE"
        assert reap_args.run_observer_only is True
        assert obs_args.record_pruning_metrics_only is False
        assert obs_args.observe_backend == "bmm"
        assert cluster_args.expert_sim == "ttm"
        assert merge_args.merge_method == "average"
        assert layerwise_args.batch_group_size == 2
        assert layerwise_args.save_intermediate is True


# ---------------------------------------------------------------------------
# Root callback / verbose
# ---------------------------------------------------------------------------


class TestRootCallback:
    @patch("reap.prune.run", return_value=None)
    def test_verbose_flag_accepted(self, mock_run: MagicMock):
        result = _invoke("-v", "prune", "full", "--observe-only")
        assert result.exit_code == 0
        mock_run.assert_called_once()
