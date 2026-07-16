"""Expert-merging pipeline: cluster experts from observer state, then merge
each cluster's experts into a single super-expert in-place, and save the
merged model.

Restored from the original Cerebras REAP codebase (deleted in fe39201) and
ported off the removed ``model_util.MODEL_ATTRS`` / ``get_moe`` helpers onto
the layout-based adapter system. Clustering is model-agnostic (operates on
observer state tensors); merging reads per-expert weight attribute names via
``adapter.expert_weight_attrs()``.

Entry point: ``python -m reap.merge_pipeline`` (mirrors the original
``experiments/merging-cli.sh``). The observer step reuses the standard
``record_activations`` path with ``record_pruning_metrics_only=False`` so the
merging-criteria metrics (ttm, characteristic_activation, ca_dist,
router_logit_similarity) are produced for clustering.
"""
from __future__ import annotations

import logging
import pathlib
import pickle
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed

from reap.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    ClusterArgs,
    MergeArgs,
    EvalArgs,
)
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
    multi_layer_hierarchical_clustering,
    mc_smoe_clustering,
    multi_layer_kmeans_clustering_on_ca,
    restricted_hierarchical_clustering,
    kmeans_clustering,
)
from reap.pipeline import record_activations, dump_args_to_yaml, create_results_directory
from reap.merge import MergeMethod, MoEExpertMerger
from reap.metrics import get_distance_fn
from reap.model_adapters import infer_model_adapter
from reap.eval import run_evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (ported off model_util.MODEL_ATTRS onto the adapter system)
# ---------------------------------------------------------------------------


def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    """Identify high-activation "super experts" to exclude from merging."""
    logger.info("Identifying super experts to preserve...")
    quantile = 99.5
    times = 10
    all_max_activations = [layer["max_activations"] for layer in observer_data.values()]
    num_layers = len(all_max_activations)
    all_max_activations = torch.cat(all_max_activations).flatten()
    percentile_threshold = torch.quantile(all_max_activations, quantile / 100.0).item()
    abs_threshold = all_max_activations.max().item() / times
    final_threshold = max(percentile_threshold, abs_threshold)
    all_max_activations = all_max_activations.reshape(num_layers, -1)
    super_experts_mask = all_max_activations > final_threshold
    if not include_last_layers:
        logger.info(
            "Only considering first 75% of layers for super expert "
            "identification since singleton_outlier_experts is False"
        )
        num_layers = int(num_layers * 0.75)
        super_experts_mask[num_layers:, :] = False
    super_expert_idx = torch.argwhere(super_experts_mask)
    logger.info(
        f"Identified {super_experts_mask.sum().item()} super experts "
        f"with threshold: {final_threshold:.4f}"
    )
    return super_expert_idx


def assert_merge(model: nn.Module, merged_moe: nn.Module, cluster_label, model_attrs):
    """Verify that experts within each cluster were merged (weights tied/equal)."""
    assert hasattr(merged_moe, "experts"), (
        "The merged module must have an 'experts' attribute."
    )
    gate_proj = model_attrs["gate_proj"]
    down_proj = model_attrs["down_proj"]
    up_proj = model_attrs["up_proj"]

    if model_attrs["fused"]:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert torch.allclose(
                    getattr(merged_moe.experts, gate_proj)[dom_expert],
                    getattr(merged_moe.experts, gate_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
                assert torch.allclose(
                    getattr(merged_moe.experts, down_proj)[dom_expert],
                    getattr(merged_moe.experts, down_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
    else:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert (
                    getattr(merged_moe.experts[dom_expert], up_proj).weight
                    == getattr(merged_moe.experts[expert], up_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], down_proj).weight
                    == getattr(merged_moe.experts[expert], down_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], gate_proj).weight
                    == getattr(merged_moe.experts[expert], gate_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster(
    data: dict[int, dict[str, Any]],
    num_clusters: int,
    cluster_args: ClusterArgs,
    merge_args: MergeArgs,
    distance_measure: str,
    results_dir: pathlib.Path,
) -> dict[int, torch.Tensor]:
    """Cluster the model's experts based on the specified clustering method."""
    logger.info(f"Clustering experts using settings:\n{cluster_args}\n")

    cluster_labels: dict[int, torch.Tensor] = {}
    distances: dict[int, torch.Tensor] = {}
    all_layer_expert_proba: dict[int, torch.Tensor] = {}
    if cluster_args.singleton_super_experts or cluster_args.singleton_outlier_experts:
        super_expert_idx = get_super_expert_indices(
            data, include_last_layers=cluster_args.singleton_outlier_experts
        )
    for layer in tqdm(data, "Clustering experts..."):
        # Honour skip_first/skip_last: skipped layers get identity clusters
        if merge_args.skip_first and layer == min(data.keys()):
            num_experts = len(data[layer]["expert_frequency"])
            cluster_labels[layer] = torch.arange(num_experts)
            logger.info(f"Skipping clustering for layer {layer} as per 'skip_first'.")
            continue
        if merge_args.skip_last and layer == max(data.keys()):
            num_experts = len(data[layer]["expert_frequency"])
            cluster_labels[layer] = torch.arange(num_experts)
            logger.info(f"Skipping clustering for layer {layer} as per 'skip_last'.")
            continue
        expert_prob = data[layer]["expert_frequency"] / data[layer]["total_tokens"]
        ttm_sim_matrix = None
        try:
            ttm_sim_matrix = data[layer]["ttm_similarity_matrix"]
        except KeyError:
            pass
        online_characteristic_activation_dist = None
        try:
            online_characteristic_activation_dist = data[layer][
                "online_characteristic_activation_dist"
            ]
        except KeyError:
            pass
        ca = data[layer]["characteristic_activation"]
        routed_ca = None
        try:
            routed_ca = data[layer]["routed_characteristic_activation"]
        except KeyError:
            pass
        router_logits = data[layer]["router_logit_similiarity"]

        expert_similarity_scores = {
            "ttm": ttm_sim_matrix,
            "dynamic_ttm": ttm_sim_matrix,
            "characteristic_activation": ca,
            "routed_characteristic_activation": routed_ca,
            "router_logits": router_logits,
            "online_characteristic_activation_dist": online_characteristic_activation_dist,
        }
        distance = expert_similarity_scores[cluster_args.expert_sim]

        if (
            cluster_args.expert_sim
            in [
                "characteristic_activation",
                "routed_characteristic_activation",
                "router_logits",
            ]
            and cluster_args.cluster_method != "kmeans"
        ):
            distance_fn = get_distance_fn(distance_measure)
            distance = distance_fn(distance.unsqueeze(0), distance.unsqueeze(1))

        if cluster_args.singleton_super_experts:
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][
                :, 1
            ]
            if len(super_experts_in_layer) > 0:
                max_value = torch.finfo(distance.dtype).max
                distance[:, super_experts_in_layer] = max_value
                distance[super_experts_in_layer, :] = max_value

        distances[layer] = distance
        all_layer_expert_proba[layer] = expert_prob
        if cluster_args.multi_layer or cluster_args.cluster_method == "mc_smoe":
            continue
        if cluster_args.frequency_penalty and cluster_args.expert_sim != "dynamic_ttm":
            penalty = get_penalty_vector(
                expert_prob,
                cluster_args.softmax_temperature,
            )
            penalty_matrix = penalty.unsqueeze(0) + penalty.unsqueeze(1)
            penalized_distance = distance * penalty_matrix
            penalized_distance[penalized_distance.isnan()] = float("inf")
            distance = penalized_distance

        if cluster_args.expert_sim == "dynamic_ttm":
            cluster_label = dynamic_frequency_penalized_clustering(
                distance,
                expert_prob,
                num_clusters,
                cluster_args.softmax_temperature,
            )
        elif cluster_args.cluster_method == "agglomerative":
            if (
                hasattr(cluster_args, "max_cluster_size")
                and cluster_args.max_cluster_size is None
            ):
                cluster_label = hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                )
            else:
                cluster_label = restricted_hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                    max_cluster_size=cluster_args.max_cluster_size,
                )
            if isinstance(cluster_label, np.ndarray):
                cluster_label = torch.tensor(cluster_label)
        elif cluster_args.cluster_method == "kmeans":
            cluster_label = kmeans_clustering(distance, num_clusters)
        else:
            raise NotImplementedError(
                f"Clustering method '{cluster_args.cluster_method}' is not implemented."
            )
        cluster_labels[layer] = cluster_label

    if cluster_args.multi_layer:
        logger.info(f"Multi layer clustering with multi_layer={cluster_args.multi_layer}")
        if cluster_args.cluster_method == "agglomerative":
            cluster_labels = multi_layer_hierarchical_clustering(
                distances,
                cluster_args.multi_layer,
                cluster_args.linkage_method,
                num_clusters,
            )
        elif cluster_args.cluster_method == "kmeans":
            if cluster_args.expert_sim != "characteristic_activation":
                raise ValueError(
                    "multi_layer kmeans clustering on ca only implemented for "
                    "characteristic_activation expert sim"
                )
            cluster_labels = multi_layer_kmeans_clustering_on_ca(
                distances,
                num_layers=cluster_args.multi_layer,
                n_clusters=num_clusters,
            )

    if cluster_args.cluster_method == "mc_smoe":
        logger.info("Performing MC-SMoE adaptive layer-wise merging...")
        cluster_labels = mc_smoe_clustering(
            distances,
            all_layer_expert_proba,
            total_clusters=len(distances) * num_clusters,
        )
    return cluster_labels


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge(
    model: nn.Module,
    cluster_labels: dict[int, torch.Tensor],
    observer_data: dict[int, dict[str, Any]],
    merge_args: MergeArgs,
) -> None:
    """Merge experts based on the clustering results (in-place)."""
    logger.info(f"Merging experts using method '{merge_args.merge_method}'")
    adapter = infer_model_adapter(model, model.config)
    if adapter is None:
        raise ValueError(
            f"No model adapter for {model.__class__.__name__}. REAP currently "
            "supports Qwen3-MoE, Llama4-MoE, and Mixtral-style architectures."
        )
    _moe_indices = adapter.identify_moe_layers(model)
    first_moe = (
        adapter.get_moe(adapter.layers(model)[_moe_indices[0]])
        if _moe_indices
        else None
    )
    model_attrs = dict(adapter.expert_weight_attrs(first_moe))
    try:
        merge_method = MergeMethod(merge_args.merge_method)
    except ValueError:
        raise NotImplementedError(
            f"Merge method '{merge_args.merge_method}' is not implemented. "
            f"Supported methods: {[method.value for method in MergeMethod]}"
        )

    layers = adapter.layers(model)
    for layer_idx, layer in enumerate(tqdm(cluster_labels, "Merging layers...")):
        if merge_args.skip_first and layer_idx == 0:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_first'."
            )
            continue
        if merge_args.skip_last and layer_idx == len(cluster_labels) - 1:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_last'."
            )
            continue

        expert_proba = (
            observer_data[layer]["expert_frequency"]
            / observer_data[layer]["total_tokens"]
        )
        cluster_label = cluster_labels[layer]
        moe = adapter.get_moe(layers[layer])
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_label,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=merge_method,
            dom_as_base=merge_args.dom_as_base,
            permute=merge_args.permute,
            tie_tensors=merge_args.save_as_tied_params,
            select_top_k=merge_args.select_top_k,
        )
        merger.merge_experts()
        assert_merge(model, moe, cluster_label, model_attrs)


def save_merged_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    merged_model_dir: pathlib.Path,
    safe_serialization: bool,
) -> pathlib.Path:
    logger.info("Saving merged model...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model.save_pretrained(merged_model_dir, safe_serialization=safe_serialization)
    if tokenizer is not None:
        tokenizer.save_pretrained(merged_model_dir)
    logger.info(
        f"Merged model saved to {merged_model_dir} in {time.time() - start:.2f} seconds"
    )
    return merged_model_dir


def get_merged_model_dir(
    results_dir: pathlib.Path,
    num_clusters: int,
    cluster_args: ClusterArgs,
    obs_args: ObserverArgs,
    merge_args: MergeArgs,
) -> pathlib.Path:
    cluster_desc = cluster_args.cluster_description or (
        f"{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters}_"
        f"{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}"
        f"_softmax-{cluster_args.softmax_temperature}_multi_layer-{cluster_args.multi_layer}"
    )
    sub = merge_args.merged_model_dir_name or (
        f"{merge_args.merge_method}-permute_{merge_args.permute}-"
        f"skip_first_{merge_args.skip_first}-skip_last_{merge_args.skip_last}-"
        f"multilayer_{cluster_args.multi_layer}"
    )
    return results_dir / "merged_models" / sub / cluster_desc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_merge(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    observer_data: dict,
    reap_args: ReapArgs,
    model_args: ModelArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    cluster_args: ClusterArgs,
    merge_args: MergeArgs,
    eval_args: EvalArgs,
    results_dir: pathlib.Path,
) -> pathlib.Path:
    """Cluster -> merge -> save, given pre-collected observer data."""
    experts_per_layer = len(
        observer_data[next(iter(observer_data))]["expert_frequency"]
    )
    num_layers = len(observer_data)

    # Guard: skip_first/skip_last must not exclude every layer.
    merged_layers = num_layers - int(merge_args.skip_first) - int(merge_args.skip_last)
    if merged_layers <= 0:
        raise ValueError(
            "skip_first/skip_last exclude all layers; nothing to merge."
        )

    # Each merged layer is compressed by the configured ratio; skipped layers
    # keep all experts (identity clusters), handled in cluster(). Using the
    # per-layer formula (rather than redistributing a global cluster budget
    # across merged layers) guarantees num_clusters <= experts_per_layer, so
    # the underlying agglomerative clustering never receives an impossible
    # cluster count (n_clusters > n_samples).
    num_clusters = int(experts_per_layer * (1 - cluster_args.compression_ratio))
    if num_clusters < 1:
        raise ValueError(
            f"compression_ratio {cluster_args.compression_ratio} yields 0 "
            f"clusters (experts_per_layer={experts_per_layer})."
        )

    logger.info(
        f"Calculated num_clusters: {num_clusters} (compression_ratio "
        f"{cluster_args.compression_ratio}, {experts_per_layer} experts/layer, "
        f"{num_layers} layers, merged_layers={merged_layers}, "
        f"skip_first={merge_args.skip_first}, skip_last={merge_args.skip_last})"
    )

    cluster_labels = cluster(
        observer_data, num_clusters, cluster_args, merge_args, obs_args.distance_measure, results_dir
    )
    logger.info("Clustering completed.")

    merged_model_dir = get_merged_model_dir(
        results_dir, num_clusters, cluster_args, obs_args, merge_args
    )
    if (
        merged_model_dir.exists()
        and list(merged_model_dir.glob("*.safetensors"))
        and not merge_args.overwrite_merged_model
    ):
        logger.info(f"Merged model already exists at {merged_model_dir}. Skipping.")
        return merged_model_dir

    merge(model, cluster_labels, observer_data, merge_args)
    merged_model_dir = save_merged_model(
        model,
        tokenizer,
        merged_model_dir,
        safe_serialization=True if not merge_args.save_as_tied_params else False,
    )

    cluster_analysis_dir = merged_model_dir / "clusters"
    cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(cluster_analysis_dir / "clusters.pkl", "wb") as f:
        pickle.dump(cluster_labels, f)

    dump_args_to_yaml(
        merged_model_dir,
        reap_args=reap_args,
        ds_args=ds_args,
        obs_args=obs_args,
        model_args=model_args,
        eval_args=eval_args,
        cluster_args=cluster_args,
        merge_args=merge_args,
    )
    logger.info(f"Merged model saved to {merged_model_dir}.")

    if reap_args.do_eval:
        logger.info("Starting evaluation...")
        model_args_eval = ModelArgs(model_name=str(merged_model_dir))
        run_evaluate(model_args_eval, merged_model_dir / "eval", eval_args, reap_args.seed)

    return merged_model_dir


def run(
    reap_args: ReapArgs,
    model_args: ModelArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    cluster_args: ClusterArgs,
    merge_args: MergeArgs,
    eval_args: EvalArgs,
):
    """Full-model observe (merge metrics) → cluster → merge → save."""
    if cluster_args.singleton_super_experts and cluster_args.singleton_outlier_experts:
        raise ValueError(
            "Only one of singleton_super_experts or singleton_outlier_experts can be True."
        )

    # Merge needs the merging-criteria metrics -> must NOT record pruning-only.
    if obs_args.record_pruning_metrics_only:
        logger.info(
            "Merging requires merging-criteria metrics; forcing "
            "record_pruning_metrics_only=False."
        )
        obs_args.record_pruning_metrics_only = False

    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    model_name = model_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info(
        f"Running observer to collect activation+merging data for "
        f"{model_args.model_name} on {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
    )
    if reap_args.run_observer_only:
        logger.info("Observer run completed. Exiting (run_observer_only=True).")
        return None

    return run_merge(
        model,
        tokenizer,
        observer_data,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        merge_args,
        eval_args,
        results_dir,
    )


def main():
    """CLI entry (HfArgumentParser). Prefer ``reap merge full`` (Typer)."""
    parser = HfArgumentParser(
        (ReapArgs, ModelArgs, DatasetArgs, ObserverArgs, ClusterArgs, MergeArgs, EvalArgs)
    )
    (
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        merge_args,
        eval_args,
    ) = parser.parse_args_into_dataclasses()
    run(
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        merge_args,
        eval_args,
    )


if __name__ == "__main__":
    main()