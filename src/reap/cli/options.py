"""Shared Typer option types and dataclass builders for the REAP CLI."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

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

# ---------------------------------------------------------------------------
# Reusable Annotated option aliases
# ---------------------------------------------------------------------------

ModelName = Annotated[
    str,
    typer.Option(
        "--model",
        "-m",
        help="HuggingFace model id or local path.",
        rich_help_panel="Model",
    ),
]
DatasetName = Annotated[
    str,
    typer.Option(
        "--dataset",
        "-d",
        help=(
            "Calibration dataset id, composite spec "
            "(name:N,...), or 'combined' for cached observations."
        ),
        rich_help_panel="Data",
    ),
]
Seed = Annotated[
    int,
    typer.Option("--seed", help="RNG seed.", rich_help_panel="Run"),
]
CompressionRatio = Annotated[
    Optional[float],
    typer.Option(
        "--compression-ratio",
        help="Fraction of experts to remove (or merge down by). Default 0.5.",
        rich_help_panel="Compression",
    ),
]
ObserveBackend = Annotated[
    str,
    typer.Option(
        "--observe-backend",
        help="Observation backend: auto | loop | bmm | frea | f2.",
        rich_help_panel="Observer",
    ),
]
BatchSize = Annotated[
    int,
    typer.Option("--batch-size", help="Calibration batch size.", rich_help_panel="Data"),
]
BatchesPerCategory = Annotated[
    int,
    typer.Option(
        "--batches-per-category",
        help="Number of calibration batches (per category if split).",
        rich_help_panel="Data",
    ),
]
ModelMaxLength = Annotated[
    Optional[int],
    typer.Option(
        "--model-max-length",
        help="Max sequence length for calibration tokens.",
        rich_help_panel="Data",
    ),
]


def build_reap_args(
    *,
    seed: int = 42,
    debug: bool = False,
    profile: bool = True,
    run_observer_only: bool = False,
    do_eval: bool = False,
    smoke_test: bool = True,
) -> ReapArgs:
    return ReapArgs(
        seed=seed,
        debug=debug,
        profile=profile,
        run_observer_only=run_observer_only,
        do_eval=do_eval,
        smoke_test=smoke_test,
    )


def build_model_args(
    *,
    model_name: str = "Qwen/Qwen3-30B-A3B",
    num_experts_per_tok_override: int | None = None,
) -> ModelArgs:
    return ModelArgs(
        model_name=model_name,
        num_experts_per_tok_override=num_experts_per_tok_override,
    )


def build_dataset_args(
    *,
    dataset_name: str = "theblackcat102/evol-codealpaca-v1",
    dataset_config_name: str | None = None,
    split: str = "train",
    shuffle: bool = True,
) -> DatasetArgs:
    return DatasetArgs(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        split=split,
        shuffle=shuffle,
    )


def build_observer_args(
    *,
    batches_per_category: int = 1024,
    split_by_category: bool = False,
    batch_size: int = 8,
    model_max_length: int | None = 2048,
    truncate: bool = False,
    overwrite_observations: bool = False,
    distance_measure: str = "angular",
    output_file_name: str = "observations_1024_cosine.pt",
    record_pruning_metrics_only: bool = True,
    renormalize_router_weights: bool = True,
    observe_backend: str = "auto",
) -> ObserverArgs:
    return ObserverArgs(
        batches_per_category=batches_per_category,
        split_by_category=split_by_category,
        batch_size=batch_size,
        model_max_length=model_max_length,
        truncate=truncate,
        overwrite_observations=overwrite_observations,
        distance_measure=distance_measure,
        output_file_name=output_file_name,
        record_pruning_metrics_only=record_pruning_metrics_only,
        renormalize_router_weights=renormalize_router_weights,
        observe_backend=observe_backend,
    )


def build_prune_args(
    *,
    prune_method: str = "reap",
    n_experts_to_prune: int | None = None,
    overwrite_pruned_model: bool = False,
    preserve_super_experts: bool = False,
    preserve_outliers: bool = False,
) -> PruneArgs:
    return PruneArgs(
        prune_method=prune_method,
        n_experts_to_prune=n_experts_to_prune,
        overwrite_pruned_model=overwrite_pruned_model,
        # Keep legacy field names on the dataclass.
        perserve_super_experts=preserve_super_experts,
        perserve_outliers=preserve_outliers,
    )


def build_cluster_args(
    *,
    compression_ratio: float | None = 0.5,
    num_clusters: int | None = None,
    expert_sim: str = "ttm",
    cluster_method: str = "agglomerative",
    linkage_method: str = "average",
    frequency_penalty: bool = True,
    softmax_temperature: float | None = None,
    multi_layer: int | None = None,
    max_cluster_size: int | None = None,
    singleton_super_experts: bool = False,
    singleton_outlier_experts: bool = False,
    cluster_description: str | None = None,
) -> ClusterArgs:
    return ClusterArgs(
        compression_ratio=compression_ratio,
        num_clusters=num_clusters,
        expert_sim=expert_sim,
        cluster_method=cluster_method,
        linkage_method=linkage_method,
        frequency_penalty=frequency_penalty,
        softmax_temperature=softmax_temperature,
        multi_layer=multi_layer,
        max_cluster_size=max_cluster_size,
        singleton_super_experts=singleton_super_experts,
        singleton_outlier_experts=singleton_outlier_experts,
        cluster_description=cluster_description,
    )


def build_merge_args(
    *,
    merge_method: str = "frequency_weighted_average",
    overwrite_merged_model: bool = False,
    merged_model_dir_name: str | None = None,
    skip_first: bool = False,
    skip_last: bool = False,
    dom_as_base: bool = False,
    select_top_k: float = 0.1,
    permute: str | None = None,
    save_as_tied_params: bool = False,
) -> MergeArgs:
    return MergeArgs(
        merge_method=merge_method,
        overwrite_merged_model=overwrite_merged_model,
        merged_model_dir_name=merged_model_dir_name,
        skip_first=skip_first,
        skip_last=skip_last,
        dom_as_base=dom_as_base,
        select_top_k=select_top_k,
        permute=permute,
        save_as_tied_params=save_as_tied_params,
    )


def build_layerwise_args(
    *,
    batch_group_size: int | None = None,
    save_intermediate: bool = False,
    low_cpu_mem_usage: bool = True,
) -> LayerwiseArgs:
    return LayerwiseArgs(
        batch_group_size=batch_group_size,
        save_intermediate=save_intermediate,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def build_eval_args(
    *,
    do_eval: bool = False,
    greedy: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    run_lm_eval: bool = True,
    lm_eval_tasks: list[str] | None = None,
) -> EvalArgs:
    args = EvalArgs(
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        run_lm_eval=run_lm_eval and do_eval,
        # Disable unimplemented backends by default in the Typer CLI.
        run_evalplus=False,
        run_livecodebench=False,
        run_wildbench=False,
        run_math=False,
    )
    if lm_eval_tasks is not None:
        args.lm_eval_tasks = lm_eval_tasks
    return args
