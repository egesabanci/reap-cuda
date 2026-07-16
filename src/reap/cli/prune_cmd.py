"""``reap prune`` command group."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from reap.cli import options as opt

app = typer.Typer(
    name="prune",
    help="Prune MoE experts using REAP / frequency / EAN saliency.",
    no_args_is_help=True,
)


@app.command("full")
def prune_full(
    model: opt.ModelName = "Qwen/Qwen3-30B-A3B",
    dataset: opt.DatasetName = "theblackcat102/evol-codealpaca-v1",
    compression_ratio: opt.CompressionRatio = 0.5,
    prune_method: Annotated[
        str,
        typer.Option(
            "--prune-method",
            help=(
                "Saliency: reap | frequency | ean_sum | ean_mean | "
                "weighted_frequency_sum | weighted_ean_sum | max_activations | ean_ca"
            ),
            rich_help_panel="Compression",
        ),
    ] = "reap",
    n_experts_to_prune: Annotated[
        Optional[int],
        typer.Option(
            "--n-experts-to-prune",
            help="Absolute experts to remove (overrides --compression-ratio).",
            rich_help_panel="Compression",
        ),
    ] = None,
    observe_backend: opt.ObserveBackend = "auto",
    batch_size: opt.BatchSize = 8,
    batches_per_category: opt.BatchesPerCategory = 1024,
    model_max_length: opt.ModelMaxLength = 2048,
    seed: opt.Seed = 42,
    residency: opt.Residency = "auto",
    dataset_path: opt.DatasetPath = None,
    artifacts_dir: opt.ArtifactsDir = None,
    observe_only: Annotated[
        bool,
        typer.Option(
            "--observe-only/--no-observe-only",
            help="Only run calibration; skip prune/save.",
            rich_help_panel="Run",
        ),
    ] = False,
    overwrite_observations: Annotated[
        bool,
        typer.Option(
            "--overwrite-observations/--keep-observations",
            help="Re-run observer even if a cache exists.",
            rich_help_panel="Observer",
        ),
    ] = False,
    overwrite_pruned_model: Annotated[
        bool,
        typer.Option(
            "--overwrite-pruned/--keep-pruned",
            help="Overwrite an existing pruned checkpoint.",
            rich_help_panel="Compression",
        ),
    ] = False,
    record_pruning_metrics_only: Annotated[
        bool,
        typer.Option(
            "--pruning-metrics-only/--all-metrics",
            help="Record only prune-path metrics (default). Use --all-metrics for merge criteria.",
            rich_help_panel="Observer",
        ),
    ] = True,
    renormalize_router_weights: Annotated[
        bool,
        typer.Option(
            "--renorm-router/--no-renorm-router",
            help="Renormalize top-k router weights when the model uses norm_topk_prob.",
            rich_help_panel="Observer",
        ),
    ] = True,
    preserve_super_experts: Annotated[
        bool,
        typer.Option(
            "--preserve-super-experts/--no-preserve-super-experts",
            help="Never prune identified super-experts (first 75%% of layers).",
            rich_help_panel="Compression",
        ),
    ] = False,
    preserve_outliers: Annotated[
        bool,
        typer.Option(
            "--preserve-outliers/--no-preserve-outliers",
            help="Never prune outlier experts (all layers).",
            rich_help_panel="Compression",
        ),
    ] = False,
    smoke_test: Annotated[
        bool,
        typer.Option(
            "--smoke-test/--no-smoke-test",
            help="Generate a short sample after pruning.",
            rich_help_panel="Run",
        ),
    ] = True,
    do_eval: Annotated[
        bool,
        typer.Option(
            "--eval/--no-eval",
            help="Run lm-eval after pruning (requires [eval] extra).",
            rich_help_panel="Run",
        ),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(
            "--profile/--no-profile",
            help="Profile a max-length forward before calibration.",
            rich_help_panel="Run",
        ),
    ] = True,
    dataset_config: Annotated[
        Optional[str],
        typer.Option("--dataset-config", help="HF dataset config/subset.", rich_help_panel="Data"),
    ] = None,
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split.", rich_help_panel="Data"),
    ] = "train",
) -> None:
    """Prune with the **full model on GPU** (multi-GPU / large VRAM)."""
    from reap.prune import run as run_prune

    run_prune(
        opt.build_reap_args(
            seed=seed,
            profile=profile,
            run_observer_only=observe_only,
            do_eval=do_eval,
            smoke_test=smoke_test,
            residency=residency,
            artifacts_dir=artifacts_dir,
        ),
        opt.build_dataset_args(
            dataset_name=dataset,
            dataset_config_name=dataset_config,
            dataset_path=dataset_path,
            split=split,
        ),
        opt.build_observer_args(
            batches_per_category=batches_per_category,
            batch_size=batch_size,
            model_max_length=model_max_length,
            overwrite_observations=overwrite_observations,
            record_pruning_metrics_only=record_pruning_metrics_only,
            renormalize_router_weights=renormalize_router_weights,
            observe_backend=observe_backend,
        ),
        opt.build_model_args(model_name=model),
        opt.build_eval_args(do_eval=do_eval),
        opt.build_prune_args(
            prune_method=prune_method,
            n_experts_to_prune=n_experts_to_prune,
            overwrite_pruned_model=overwrite_pruned_model,
            preserve_super_experts=preserve_super_experts,
            preserve_outliers=preserve_outliers,
        ),
        opt.build_cluster_args(compression_ratio=compression_ratio),
    )


@app.command("layerwise")
def prune_layerwise(
    model: opt.ModelName = "Qwen/Qwen3-30B-A3B",
    dataset: opt.DatasetName = "theblackcat102/evol-codealpaca-v1",
    compression_ratio: opt.CompressionRatio = 0.5,
    prune_method: Annotated[
        str,
        typer.Option(
            "--prune-method",
            help="Saliency method (see ``reap prune full --help``).",
            rich_help_panel="Compression",
        ),
    ] = "reap",
    n_experts_to_prune: Annotated[
        Optional[int],
        typer.Option(
            "--n-experts-to-prune",
            help="Absolute experts to remove (overrides --compression-ratio).",
            rich_help_panel="Compression",
        ),
    ] = None,
    observe_backend: opt.ObserveBackend = "auto",
    batch_size: opt.BatchSize = 4,
    batches_per_category: opt.BatchesPerCategory = 1024,
    model_max_length: opt.ModelMaxLength = 2048,
    batch_group_size: Annotated[
        Optional[int],
        typer.Option(
            "--batch-group-size",
            help="Process this many calib batches through all blocks at a time (CPU RAM).",
            rich_help_panel="Layerwise",
        ),
    ] = None,
    save_intermediate: Annotated[
        bool,
        typer.Option(
            "--save-intermediate/--no-save-intermediate",
            help="Save per-block metrics during layerwise calibration.",
            rich_help_panel="Layerwise",
        ),
    ] = False,
    low_cpu_mem_usage: Annotated[
        bool,
        typer.Option(
            "--low-cpu-mem/--no-low-cpu-mem",
            help="Memory-efficient model load on CPU.",
            rich_help_panel="Layerwise",
        ),
    ] = True,
    seed: opt.Seed = 42,
    residency: opt.Residency = "auto",
    dataset_path: opt.DatasetPath = None,
    artifacts_dir: opt.ArtifactsDir = None,
    observe_only: Annotated[
        bool,
        typer.Option(
            "--observe-only/--no-observe-only",
            help="Only run calibration; skip prune/save.",
            rich_help_panel="Run",
        ),
    ] = False,
    overwrite_observations: Annotated[
        bool,
        typer.Option(
            "--overwrite-observations/--keep-observations",
            rich_help_panel="Observer",
        ),
    ] = False,
    overwrite_pruned_model: Annotated[
        bool,
        typer.Option(
            "--overwrite-pruned/--keep-pruned",
            rich_help_panel="Compression",
        ),
    ] = False,
    record_pruning_metrics_only: Annotated[
        bool,
        typer.Option(
            "--pruning-metrics-only/--all-metrics",
            rich_help_panel="Observer",
        ),
    ] = True,
    renormalize_router_weights: Annotated[
        bool,
        typer.Option(
            "--renorm-router/--no-renorm-router",
            rich_help_panel="Observer",
        ),
    ] = True,
    preserve_super_experts: Annotated[
        bool,
        typer.Option(
            "--preserve-super-experts/--no-preserve-super-experts",
            rich_help_panel="Compression",
        ),
    ] = False,
    preserve_outliers: Annotated[
        bool,
        typer.Option(
            "--preserve-outliers/--no-preserve-outliers",
            rich_help_panel="Compression",
        ),
    ] = False,
    do_eval: Annotated[
        bool,
        typer.Option("--eval/--no-eval", rich_help_panel="Run"),
    ] = False,
    dataset_config: Annotated[
        Optional[str],
        typer.Option("--dataset-config", rich_help_panel="Data"),
    ] = None,
    split: Annotated[str, typer.Option("--split", rich_help_panel="Data")] = "train",
) -> None:
    """Prune with **one decoder block on GPU** (30B+ on a single L40S).

    With ``--residency auto|gpu_full``, small models that fit VRAM may run the
    full GPU path instead of pinning weights in host RAM.
    """
    from reap.layerwise_prune import run as run_layerwise_prune

    run_layerwise_prune(
        opt.build_reap_args(
            seed=seed,
            run_observer_only=observe_only,
            do_eval=do_eval,
            smoke_test=False,
            profile=False,
            residency=residency,
            artifacts_dir=artifacts_dir,
        ),
        opt.build_dataset_args(
            dataset_name=dataset,
            dataset_config_name=dataset_config,
            dataset_path=dataset_path,
            split=split,
        ),
        opt.build_observer_args(
            batches_per_category=batches_per_category,
            batch_size=batch_size,
            model_max_length=model_max_length,
            overwrite_observations=overwrite_observations,
            record_pruning_metrics_only=record_pruning_metrics_only,
            renormalize_router_weights=renormalize_router_weights,
            observe_backend=observe_backend,
        ),
        opt.build_model_args(model_name=model),
        opt.build_eval_args(do_eval=do_eval),
        opt.build_prune_args(
            prune_method=prune_method,
            n_experts_to_prune=n_experts_to_prune,
            overwrite_pruned_model=overwrite_pruned_model,
            preserve_super_experts=preserve_super_experts,
            preserve_outliers=preserve_outliers,
        ),
        opt.build_cluster_args(compression_ratio=compression_ratio),
        opt.build_layerwise_args(
            batch_group_size=batch_group_size,
            save_intermediate=save_intermediate,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ),
    )
