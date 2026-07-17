"""``reap merge`` command group."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from reap.cli import options as opt

app = typer.Typer(
    name="merge",
    help="Cluster and merge MoE experts into super-experts.",
    no_args_is_help=True,
)


@app.command("full")
def merge_full(
    model: opt.ModelName = "Qwen/Qwen3-30B-A3B",
    dataset: opt.DatasetName = "theblackcat102/evol-codealpaca-v1",
    compression_ratio: opt.CompressionRatio = 0.5,
    num_clusters: Annotated[
        int | None,
        typer.Option(
            "--num-clusters",
            help=(
                "Explicit number of clusters per layer. Overrides "
                "--compression-ratio when set."
            ),
            rich_help_panel="Cluster",
        ),
    ] = None,
    expert_sim: Annotated[
        str,
        typer.Option(
            "--expert-sim",
            help=(
                "Similarity for clustering: ttm | characteristic_activation | "
                "routed_characteristic_activation | router_logits | "
                "online_characteristic_activation_dist | dynamic_ttm"
            ),
            rich_help_panel="Cluster",
        ),
    ] = "characteristic_activation",
    cluster_method: Annotated[
        str,
        typer.Option(
            "--cluster-method",
            help="agglomerative | kmeans | mc_smoe",
            rich_help_panel="Cluster",
        ),
    ] = "agglomerative",
    linkage_method: Annotated[
        str,
        typer.Option(
            "--linkage",
            help="Linkage for agglomerative: average | ward | complete | single",
            rich_help_panel="Cluster",
        ),
    ] = "average",
    merge_method: Annotated[
        str,
        typer.Option(
            "--merge-method",
            help=(
                "frequency_weighted_average | average | ties | multislerp | "
                "sce | karcher | submoe"
            ),
            rich_help_panel="Merge",
        ),
    ] = "frequency_weighted_average",
    distance_measure: Annotated[
        str,
        typer.Option(
            "--distance",
            help="angular | euclidean | jsd | cka | cosine",
            rich_help_panel="Cluster",
        ),
    ] = "angular",
    observe_backend: opt.ObserveBackend = "auto",
    frea_backend: opt.FreaBackend = "auto",
    batch_size: opt.BatchSize = 8,
    batches_per_category: opt.BatchesPerCategory = 1024,
    model_max_length: opt.ModelMaxLength = 2048,
    seed: opt.Seed = 42,
    residency: opt.Residency = "auto",
    dataset_path: opt.DatasetPath = None,
    artifacts_dir: opt.ArtifactsDir = None,
    trust_remote_code: opt.TrustRemoteCode = False,
    skip_first: Annotated[
        bool,
        typer.Option("--skip-first/--no-skip-first", rich_help_panel="Merge"),
    ] = False,
    skip_last: Annotated[
        bool,
        typer.Option("--skip-last/--no-skip-last", rich_help_panel="Merge"),
    ] = False,
    frequency_penalty: Annotated[
        bool,
        typer.Option(
            "--frequency-penalty/--no-frequency-penalty",
            rich_help_panel="Cluster",
        ),
    ] = True,
    permute: Annotated[
        Optional[str],
        typer.Option(
            "--permute",
            help="Optional weight permutation before merge: direct | wm",
            rich_help_panel="Merge",
        ),
    ] = None,
    observe_only: Annotated[
        bool,
        typer.Option("--observe-only/--no-observe-only", rich_help_panel="Run"),
    ] = False,
    overwrite_observations: Annotated[
        bool,
        typer.Option(
            "--overwrite-observations/--keep-observations",
            rich_help_panel="Observer",
        ),
    ] = False,
    overwrite_merged_model: Annotated[
        bool,
        typer.Option("--overwrite-merged/--keep-merged", rich_help_panel="Merge"),
    ] = False,
    do_eval: Annotated[
        bool,
        typer.Option("--eval/--no-eval", rich_help_panel="Run"),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option("--profile/--no-profile", rich_help_panel="Run"),
    ] = True,
    dataset_config: Annotated[
        Optional[str],
        typer.Option("--dataset-config", rich_help_panel="Data"),
    ] = None,
    split: Annotated[str, typer.Option("--split", rich_help_panel="Data")] = "train",
) -> None:
    """Merge with the **full model on GPU**."""
    from reap.merge_pipeline import run as run_merge_full

    run_merge_full(
        opt.build_reap_args(
            seed=seed,
            profile=profile,
            run_observer_only=observe_only,
            do_eval=do_eval,
            smoke_test=False,
            residency=residency,
            artifacts_dir=artifacts_dir,
        ),
        opt.build_model_args(model_name=model, trust_remote_code=trust_remote_code),
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
            distance_measure=distance_measure,
            record_pruning_metrics_only=False,
            observe_backend=observe_backend,
            frea_backend=frea_backend,
        ),
        opt.build_cluster_args(
            compression_ratio=compression_ratio,
            num_clusters=num_clusters,
            expert_sim=expert_sim,
            cluster_method=cluster_method,
            linkage_method=linkage_method,
            frequency_penalty=frequency_penalty,
        ),
        opt.build_merge_args(
            merge_method=merge_method,
            overwrite_merged_model=overwrite_merged_model,
            skip_first=skip_first,
            skip_last=skip_last,
            permute=permute,
        ),
        opt.build_eval_args(do_eval=do_eval),
    )


@app.command("layerwise")
def merge_layerwise(
    model: opt.ModelName = "Qwen/Qwen3-30B-A3B",
    dataset: opt.DatasetName = "theblackcat102/evol-codealpaca-v1",
    compression_ratio: opt.CompressionRatio = 0.5,
    num_clusters: Annotated[
        int | None,
        typer.Option(
            "--num-clusters",
            help=(
                "Explicit number of clusters per layer. Overrides "
                "--compression-ratio when set."
            ),
            rich_help_panel="Cluster",
        ),
    ] = None,
    expert_sim: Annotated[
        str,
        typer.Option("--expert-sim", rich_help_panel="Cluster"),
    ] = "characteristic_activation",
    cluster_method: Annotated[
        str,
        typer.Option("--cluster-method", rich_help_panel="Cluster"),
    ] = "agglomerative",
    linkage_method: Annotated[
        str,
        typer.Option("--linkage", rich_help_panel="Cluster"),
    ] = "average",
    merge_method: Annotated[
        str,
        typer.Option("--merge-method", rich_help_panel="Merge"),
    ] = "frequency_weighted_average",
    distance_measure: Annotated[
        str,
        typer.Option("--distance", rich_help_panel="Cluster"),
    ] = "angular",
    observe_backend: opt.ObserveBackend = "auto",
    frea_backend: opt.FreaBackend = "auto",
    batch_size: opt.BatchSize = 4,
    batches_per_category: opt.BatchesPerCategory = 1024,
    model_max_length: opt.ModelMaxLength = 2048,
    batch_group_size: Annotated[
        Optional[int],
        typer.Option("--batch-group-size", rich_help_panel="Layerwise"),
    ] = None,
    save_intermediate: Annotated[
        bool,
        typer.Option(
            "--save-intermediate/--no-save-intermediate",
            rich_help_panel="Layerwise",
        ),
    ] = False,
    low_cpu_mem_usage: Annotated[
        bool,
        typer.Option("--low-cpu-mem/--no-low-cpu-mem", rich_help_panel="Layerwise"),
    ] = True,
    seed: opt.Seed = 42,
    residency: opt.Residency = "auto",
    dataset_path: opt.DatasetPath = None,
    artifacts_dir: opt.ArtifactsDir = None,
    trust_remote_code: opt.TrustRemoteCode = False,
    skip_first: Annotated[
        bool,
        typer.Option("--skip-first/--no-skip-first", rich_help_panel="Merge"),
    ] = False,
    skip_last: Annotated[
        bool,
        typer.Option("--skip-last/--no-skip-last", rich_help_panel="Merge"),
    ] = False,
    frequency_penalty: Annotated[
        bool,
        typer.Option(
            "--frequency-penalty/--no-frequency-penalty",
            rich_help_panel="Cluster",
        ),
    ] = True,
    permute: Annotated[
        Optional[str],
        typer.Option("--permute", rich_help_panel="Merge"),
    ] = None,
    observe_only: Annotated[
        bool,
        typer.Option("--observe-only/--no-observe-only", rich_help_panel="Run"),
    ] = False,
    overwrite_observations: Annotated[
        bool,
        typer.Option(
            "--overwrite-observations/--keep-observations",
            rich_help_panel="Observer",
        ),
    ] = False,
    overwrite_merged_model: Annotated[
        bool,
        typer.Option("--overwrite-merged/--keep-merged", rich_help_panel="Merge"),
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
    """Merge with **layerwise calibration** (one block on GPU)."""
    from reap.layerwise_merge import run as run_layerwise_merge

    run_layerwise_merge(
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
            distance_measure=distance_measure,
            record_pruning_metrics_only=False,
            observe_backend=observe_backend,
            frea_backend=frea_backend,
        ),
        opt.build_model_args(model_name=model, trust_remote_code=trust_remote_code),
        opt.build_eval_args(do_eval=do_eval),
        opt.build_cluster_args(
            compression_ratio=compression_ratio,
            num_clusters=num_clusters,
            expert_sim=expert_sim,
            cluster_method=cluster_method,
            linkage_method=linkage_method,
            frequency_penalty=frequency_penalty,
        ),
        opt.build_merge_args(
            merge_method=merge_method,
            overwrite_merged_model=overwrite_merged_model,
            skip_first=skip_first,
            skip_last=skip_last,
            permute=permute,
        ),
        opt.build_layerwise_args(
            batch_group_size=batch_group_size,
            save_intermediate=save_intermediate,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ),
    )
