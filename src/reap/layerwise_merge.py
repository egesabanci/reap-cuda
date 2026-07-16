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
from reap.data import (
    load_category_batches,
    load_composite_category_batches,
    parse_composite_dataset_spec,
)
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

    Mirrors ``layerwise_prune.prepare_calibration_batches``.
    """
    logger.info(f"Loading dataset {ds_args.dataset_name}...")

    composite_components = parse_composite_dataset_spec(
        ds_args.dataset_name, default_split=ds_args.split
    )

    global_path = getattr(ds_args, "dataset_path", None)
    if composite_components is not None:
        total_batches = sum(c.num_batches for c in composite_components)
        logger.info(
            f"Composite dataset specified, using {len(composite_components)} "
            f"components, {total_batches} total **batches**."
        )
        category_data_batches = load_composite_category_batches(
            composite_components,
            tokenizer=tokenizer,
            model_max_length=obs_args.model_max_length,
            batch_size=obs_args.batch_size,
            return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
            truncate=obs_args.truncate,
            global_dataset_path=global_path,
        )
        all_batches = []
        for _category, batches in category_data_batches.items():
            all_batches.extend(batches)
        return all_batches

    category_data_batches = load_category_batches(
        dataset_name=ds_args.dataset_name,
        split=ds_args.split,
        subset=ds_args.dataset_config_name,
        tokenizer=tokenizer,
        model_max_length=obs_args.model_max_length,
        split_by_category=obs_args.split_by_category,
        return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
        truncate=obs_args.truncate,
        batches_per_category=obs_args.batches_per_category,
        batch_size=obs_args.batch_size,
        dataset_path=global_path,
    )
    all_batches: List[torch.Tensor] = []
    for _category, samples in category_data_batches.items():
        all_batches.extend(samples)
    return all_batches


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

    try:
        from reap.kernels.triton_frea import set_frea_backend

        set_frea_backend(getattr(obs_args, "frea_backend", "auto"))
    except Exception:
        pass

    hook_config = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=layer_cfg.fused_experts,
        renormalize_router_weights=obs_args.renormalize_router_weights,
        record_pruning_metrics_only=False,  # merge needs ALL metrics
        observe_backend=getattr(obs_args, "observe_backend", "auto"),
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

    from reap.kernels.triton_utils import log_triton_usage_summary, reset_triton_usage

    reset_triton_usage()
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
    log_triton_usage_summary()

    logger.info(f"Layerwise merge calibration complete. Saved to {output_file}")
    return observer_data


def run(
    reap_args: ReapArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    model_args: ModelArgs,
    eval_args: EvalArgs,
    cluster_args: ClusterArgs,
    merge_args: MergeArgs,
    layerwise_args: LayerwiseArgs,
    *,
    _residency_resolved: str | None = None,
):
    """Block-wise observe (merge metrics) → cluster → merge → save."""
    from reap.residency import (
        estimate_model_bytes_from_config,
        load_causal_lm,
        plan_load,
        preflight_or_warn,
        resolve_residency,
        validate_residency,
    )

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

    if _residency_resolved is None:
        residency = validate_residency(getattr(reap_args, "residency", "auto"))
        model_bytes = estimate_model_bytes_from_config(model_args.model_name)
        resolved, reason = resolve_residency(
            residency,
            model_bytes=model_bytes,
            cli_prefers_layerwise=True,
        )
        logger.info("Residency resolved: %s (%s)", resolved, reason)
        preflight_or_warn(resolved, model_bytes)
    else:
        resolved = validate_residency(_residency_resolved)
        model_bytes = estimate_model_bytes_from_config(model_args.model_name)
        logger.info("Residency (pre-resolved): %s", resolved)
        preflight_or_warn(resolved, model_bytes)

    if resolved in ("gpu_full", "cpu_full"):
        from reap.merge_pipeline import run as run_full_merge

        logger.info(
            "Delegating to full merge path (residency=%s) — avoids full-CPU pin",
            resolved,
        )
        return run_full_merge(
            reap_args,
            model_args,
            ds_args,
            obs_args,
            cluster_args,
            merge_args,
            eval_args,
            _residency_resolved=resolved,
        )

    set_seed(reap_args.seed)
    results_dir = create_results_directory(
        model_args.model_name,
        ds_args.dataset_name,
        base=getattr(reap_args, "artifacts_dir", None),
    )

    model_name = model_args.model_name

    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Prefer auto+disk offload over pinning the entire model in host RAM.
    offload_root = results_dir / ".offload"
    plan = plan_load(
        "layerwise",
        offload_root=offload_root,
        low_cpu_mem_usage=layerwise_args.low_cpu_mem_usage,
    )
    logger.info("Loading model for layerwise merge (%s)...", plan.reason)
    model = load_causal_lm(model_name, plan)
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
        return None

    logger.info("Starting merge pipeline from layerwise-calibrated observer data...")
    merged_dir = run_merge(
        model, tokenizer, observer_data,
        reap_args, model_args, ds_args, obs_args,
        cluster_args, merge_args, eval_args, results_dir,
    )
    logger.info(f"Layerwise merge complete. Model saved to {merged_dir}")
    return merged_dir


def main():
    """CLI entry (HfArgumentParser). Prefer ``reap merge layerwise`` (Typer)."""
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
    run(
        reap_args,
        ds_args,
        obs_args,
        model_args,
        eval_args,
        cluster_args,
        merge_args,
        layerwise_args,
    )


if __name__ == "__main__":
    main()
