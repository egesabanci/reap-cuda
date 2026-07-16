"""CLI structure tests (no model downloads / no GPU work)."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from reap.cli.app import app

runner = CliRunner()


def _invoke(*args: str) -> str:
    result = runner.invoke(
        app,
        list(args),
        color=False,
        env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    # Strip residual ANSI if any.
    return re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)


def test_root_help():
    out = _invoke("--help")
    assert "prune" in out
    assert "merge" in out


def test_prune_help():
    out = _invoke("prune", "--help")
    assert "full" in out
    assert "layerwise" in out


def test_prune_full_help_options():
    out = _invoke("prune", "full", "--help")
    assert "--model" in out
    assert "--compression-ratio" in out
    assert "--observe-backend" in out
    assert "--prune-method" in out


def test_prune_layerwise_help():
    out = _invoke("prune", "layerwise", "--help")
    assert "--batch-group-size" in out


def test_merge_help():
    out = _invoke("merge", "--help")
    assert "full" in out
    assert "layerwise" in out


def test_merge_full_help_options():
    out = _invoke("merge", "full", "--help")
    assert "--expert-sim" in out
    assert "--merge-method" in out
    assert "--cluster-method" in out


def test_version():
    out = _invoke("version")
    assert out.strip()


def test_builders_preserve_super_maps_to_dataclass():
    from reap.cli.options import build_prune_args

    args = build_prune_args(preserve_super_experts=True, prune_method="reap")
    assert args.perserve_super_experts is True
    assert args.prune_method == "reap"
