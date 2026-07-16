"""Root Typer application for REAP.

Usage examples::

    reap --help
    reap prune --help
    reap prune layerwise --model Qwen/Qwen3-30B-A3B --compression-ratio 0.5
    reap merge full --expert-sim characteristic_activation
"""

from __future__ import annotations

import logging

import typer

from reap.cli.merge_cmd import app as merge_app
from reap.cli.prune_cmd import app as prune_app

app = typer.Typer(
    name="reap",
    help=(
        "REAP — Router-weighted Expert Activation Pruning for MoE compression.\n\n"
        "Use [bold]prune[/bold] to remove experts, [bold]merge[/bold] to cluster "
        "and fuse them. Prefer [bold]layerwise[/bold] subcommands on a single GPU."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(prune_app, name="prune")
app.add_typer(merge_app, name="merge")


@app.callback()
def _root(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging.",
    ),
) -> None:
    """REAP end-to-end CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


@app.command("version")
def version() -> None:
    """Print the installed package version."""
    try:
        from importlib.metadata import version as pkg_version

        typer.echo(pkg_version("reap"))
    except Exception:
        typer.echo("0.1.0")


@app.command("kernels")
def kernels_status() -> None:
    """Show observation-backend / Triton readiness (no model load)."""
    import torch

    from reap.kernels.backend import select_observe_backend, triton_status

    status = triton_status()
    typer.echo(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"cuda device: {torch.cuda.get_device_name(0)}")
    typer.echo(f"triton package: {status['package']}")
    typer.echo(f"triton runtime: {status['runtime']}")
    if status["import_error"]:
        typer.echo(f"triton import error: {status['import_error']}")
    if status["disabled_env"]:
        typer.echo(f"REAP_DISABLE_TRITON: {status['disabled_env']}")
    typer.echo(f"auto backend: {select_observe_backend('auto')}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
