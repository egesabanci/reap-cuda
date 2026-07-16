"""
Layerwise Expert Merging for MoE Models.

This module provides a memory-efficient entry point for expert merging
that processes the model one layer at a time, computing merging-criteria
metrics (characteristic activation, ttm similarity, router logit similarity)
alongside the standard pruning metrics, enabling cluster-then-merge on
large MoE models (30B+) on a single GPU.

Key differences from standard merge_pipeline.py:
1. Model is loaded on CPU with device_map="cpu"
2. Only one transformer block is on GPU at a time
3. Hidden states are cached between blocks (ReplayCache)
4. Merging-criteria metrics (OnlineStatsTracker) live on CPU
5. Significantly reduced GPU memory requirements

Usage:
    python -m reap.layerwise_merge \
        --model_name "Qwen/Qwen3-30B-A3B" \
        --dataset_name "theblackcat102/evol-codealpaca-v1" \
        --cluster_method "agglomerative" \
        --expert_sim "characteristic_activation" \
        --compression_ratio 0.5 \
        --batch_size 4
"""

from __future__ import annotations
import logging
import pathlib
from typing import Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed

from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
    MergeArgs,
    LayerwiseArgs,
)
from reap.data import load_category_batches, parse_composite_dataset_spec
from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserverConfig
from reap.layerwise_observer import LayerwiseMoEObserver
from reap.merge_pipeline import run_merge
from reap.pipeline import dump_args_to_yaml, create_results_directory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_observer_output_path(
    results_dir: pathlib.Path,
    dataset_name: str,
    output_file_name: str,
) -> pathlib.Path:
    if (
        dataset_name == "combined"
        or parse_composite_dataset_spec(dataset_name) is not None
    ):
        return results_dir / "all" / output_file_name
    return results_dir / "layerwise" / output_file_name


def prepare_calibration_batches(
    tokenizer,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
) -> List[torch.Tensor]:
    """Prepare calibration samples for layerwise processing.

    Returns a list of tokenized input tensors.
    """
    logger.info(f"Loading dataset {ds_args.dataset_name}...")

    composite_components = parse_composite_dataset_spec(
        ds_args.dataset_name, default_split=ds_args.split
    )

    if composite_components is not None:
        all_batches = []
        total_samples = sum(component.num_batches for component in composite_components)
        logger.info(
            f"Composite dataset specified, using {len(composite_components)} "
            f"components, {total_samples} total samples."
        )
        for component in composite_components:
            batches = load_category_batches(
                tokenizer,
                component.dataset_name,
                obs_args,
                ds_args,
                component.num_batches,
            )
            all_batches.extend(batches)
        return all_batches

    batches = load_category_batches(
        tokenizer,
        ds_args.dataset_name,
        obs_args,
        ds_args,
        obs_args.batches_per_category,
    )
    return batches


def record_activations_layerwise_merge(
    model,
    tokenizer,
    data_batches: List[torch.Tensor],
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    layerwise_args: LayerwiseArgs,
    results_dir: pathlib.Path,
):
    """Block-wise calibration collecting merging-criteria metrics.

    Mirrors ``record_activations_layerwise`` in ``layerwise_prune.py`` but
    forces ``record_pruning_metrics_only=False`` so the layerwise observer
    also computes characteristic_activation, ttm_similarity, and
    router_logit_similiarity — all required by clustering.
    """
    logger.info("Starting layerwise activation recording (merge mode)...")

    adapter = infer_model_adapter(model, model.config)
    if adapter is None:
        raise ValueError(
            f"No model adapter for {model.__class__.__name__}. REAP currently "
            "supports Qwen3-MoE, Llama4-MoE, LFM2-MoE, and Mixtral-style architectures."
        )

    moe_indices = adapter.identify_moe_layers(model)
    if not moe_indices:
        raise ValueError("Model has no MoE layers to observe.")
    first_moe_layer = adapter.layers(model)[moe_indices[0]]
    layer_cfg = adapter.get_layer_config(first_moe_layer, model.config)

    hook_config = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=layer_cfg.fused_experts,
        renormalize_router_weights=obs_args.renormalize_router_weights,
        record_pruning_metrics_only=False,  # merge needs ALL metrics
    )

    observer = LayerwiseMoEObserver(
        model=model,
        hook_config=hook_config,
        adapter=adapter,
    )

    save_path = (
        _get_observer_output_path(
            results_dir,
            ds_args.dataset_name,
            obs_args.output_file_name,
        ).parent
        / "layerwise_merge_intermediate"
        if layerwise_args.save_intermediate
        else None
    )

    observer_data = observer.record_all_blocks(
        data_batches=data_batches,
        save_path=save_path,
        batch_group_size=layerwise_args.batch_group_size,
    )

    output_file = _get_observer_output_path(
        results_dir,
        ds_args.dataset_name,
        obs_args.output_file_name,
    )
    observer.save_state(output_file)
    observer.close_hooks()

    logger.info(f"Layerwise merge calibration complete. Saved to {output_file}")
    return observer_data


def main():
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            ClusterArgs,
            MergeArgs,
            LayerwiseArgs,
        )
    )
    (
        reap_args,
        ds_args,
        obs_args,
        model_args,
        eval_args,
        cluster_args,
        merge_args,
        layerwise_args,
    ) = parser.parse_args_into_dataclasses()

    if cluster_args.singleton_super_experts and cluster_args.singleton_outlier_experts:
        raise ValueError(
            "Only one of singleton_super_experts or singleton_outlier_experts can be True."
        )

    # Merge needs merging-criteria metrics -> must NOT record pruning-only.
    if obs_args.record_pruning_metrics_only:
        logger.info(
            "Merging requires merging-criteria metrics; forcing "
            "record_pruning_metrics_only=False."
        )
        obs_args.record_pruning_metrics_only = False

    if layerwise_args.batch_group_size is not None and layerwise_args.batch_group_size < 1:
        raise ValueError("layerwise batch_group_size must be at least 1 when provided.")

    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    model_name = model_args.model_name

    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # The merge pipeline always needs the model weights on hand to merge into,
    # regardless of whether observer data is freshly recorded or loaded from
    # cache. Load it once here (CPU, block-wise) so run_merge() can mutate it.
    logger.info(f"Loading model {model_name} on CPU for layerwise processing...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=layerwise_args.low_cpu_mem_usage,
    )
    model.eval()
    logger.info(f"Model loaded: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params / 1e9:.2f}B")

    cached_data_path = _get_observer_output_path(
        results_dir, ds_args.dataset_name, obs_args.output_file_name
    )

    if cached_data_path.exists() and not obs_args.overwrite_observations:
        logger.info(f"Loading cached observer data from {cached_data_path}")
        observer_data = torch.load(cached_data_path, weights_only=False)
    else:
        if ds_args.dataset_name == "combined":
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data at "
                f"{cached_data_path}"
            )
        logger.info("Preparing calibration samples...")
        data_batches = prepare_calibration_batches(tokenizer, ds_args, obs_args)

        logger.info("Recording activations with layerwise merge observer...")
        observer_data = record_activations_layerwise_merge(
            model, tokenizer, data_batches, ds_args, obs_args, layerwise_args,
            results_dir,
        )

    if reap_args.run_observer_only:
        logger.info("Observer run completed. Exiting (run_observer_only=True)")
        return

    logger.info("Starting merge pipeline from layerwise-calibrated observer data...")
    merged_dir = run_merge(
        model, tokenizer, observer_data,
        reap_args, model_args, ds_args, obs_args,
        cluster_args, merge_args, eval_args, results_dir,
    )
    logger.info(f"Layerwise merge complete. Model saved to {merged_dir}")


if __name__ == "__main__":
    main()
