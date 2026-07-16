"""
Layerwise Expert Pruning for MoE Models.

This module provides a memory-efficient entry point for expert pruning
that processes the model one layer at a time, enabling calibration of
large MoE models on a single GPU.

Key differences from standard prune.py:
1. Model is loaded on CPU with device_map="cpu"
2. Only one transformer block is on GPU at a time
3. Hidden states are cached between blocks
4. Significantly reduced GPU memory requirements

Usage:
    python -m reap.layerwise_prune \
        --model_name "Qwen/Qwen3-30B-A3B" \
        --dataset_name "theblackcat102/evol-codealpaca-v1" \
        --prune_method "reap" \
        --compression_ratio 0.5 \
        --batch_size 4
"""

from __future__ import annotations
import logging
import dataclasses
import pathlib
import hashlib
from typing import Any, Dict, List
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed

from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
    LayerwiseArgs,
)
from reap.data import load_category_batches, parse_composite_dataset_spec
from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserverConfig
from reap.layerwise_observer import LayerwiseMoEObserver
from reap.layerwise_model_utils import cleanup_memory
from reap.eval import run_evaluate
from reap.prune import prune as prune_model
from reap.prune import get_pruned_model_dir
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
    """
    Prepare calibration samples for layerwise processing.

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
            f"Composite dataset specified, overwriting given batches_per_category={obs_args.batches_per_category} "
            f"with values in composite dataset spec."
        )
        logger.info(
            f"Preparing composite calibration data with {len(composite_components)} "
            f"components, {total_samples} total samples."
        )

        for component in composite_components:
            comp_label = f"{component.name}[{component.split}]"
            logger.info(
                f"Loading composite component {comp_label} ({component.num_batches} batches)"
            )
            category_data_batches = load_category_batches(
                dataset_name=component.name,
                split=component.split,
                subset=component.subset,
                tokenizer=tokenizer,
                model_max_length=obs_args.model_max_length,
                split_by_category=False,
                return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
                truncate=obs_args.truncate,
                batches_per_category=component.num_batches,
                batch_size=obs_args.batch_size,
                dataset_path=None,
            )
            for category, batches in category_data_batches.items():
                all_batches.extend(batches)
                logger.info(f"Added {len(batches)} batches from category: {category}")

        logger.info(f"Total calibration batches: {len(all_batches)}")
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
        dataset_path=getattr(ds_args, "dataset_path", None),
    )

    # Flatten all batches into a single list
    all_batches = []
    for category, samples in category_data_batches.items():
        all_batches.extend(samples)
        logger.info(f"Added {len(samples)} samples from category: {category}")

    logger.info(f"Total calibration samples: {len(all_batches)}")
    return all_batches


def record_activations_layerwise(
    model,
    tokenizer,
    data_batches: List[torch.Tensor],
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    layerwise_args: LayerwiseArgs,
    results_dir: pathlib.Path,
) -> Dict[int, Dict[str, Any]]:
    """
    Record MoE activations using layerwise processing.

    This function processes the model one block at a time to minimize
    GPU memory usage.
    """
    logger.info("Starting layerwise activation recording...")

    # Build the adapter + observer config from the model's layout (replaces the
    # old OBSERVER_CONFIG_REGISTRY[model_class_name] lookup).
    adapter = infer_model_adapter(model, model.config)
    if adapter is None:
        raise ValueError(
            f"No model adapter for {model.__class__.__name__}. REAP currently "
            "supports Qwen3-MoE, Llama4-MoE, and Mixtral-style architectures."
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
        record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
        observe_backend=getattr(obs_args, "observe_backend", "auto"),
    )

    # Create layerwise observer
    observer = LayerwiseMoEObserver(
        model=model,
        hook_config=hook_config,
        adapter=adapter,
    )

    # Process all blocks
    save_path = (
        _get_observer_output_path(
            results_dir,
            ds_args.dataset_name,
            obs_args.output_file_name,
        ).parent
        / "layerwise_intermediate"
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

    # Save complete state
    output_file = _get_observer_output_path(
        results_dir,
        ds_args.dataset_name,
        obs_args.output_file_name,
    )
    observer.save_state(output_file)
    log_triton_usage_summary()

    logger.info(f"Layerwise activation recording complete. Saved to {output_file}")

    return observer_data


def run(
    reap_args: ReapArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    model_args: ModelArgs,
    eval_args: EvalArgs,
    prune_args: PruneArgs,
    cluster_args: ClusterArgs,
    layerwise_args: LayerwiseArgs,
    *,
    _residency_resolved: str | None = None,
):
    """Block-wise observe → prune (one decoder block on GPU at a time).

    Honors ``reap_args.residency``. If resolved to ``gpu_full`` / ``cpu_full``,
    delegates to :func:`reap.prune.run` instead of pinning the full model on CPU.
    """
    from reap.residency import (
        estimate_model_bytes_from_config,
        load_causal_lm,
        plan_load,
        preflight_or_warn,
        resolve_residency,
        validate_residency,
    )

    # Validation
    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError(
            "Only one of perserve_super_experts or perserve_outliers can be True."
        )
    if (
        layerwise_args.batch_group_size is not None
        and layerwise_args.batch_group_size < 1
    ):
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
        from reap.prune import run as run_full

        logger.info(
            "Delegating to full prune path (residency=%s) — avoids full-CPU pin",
            resolved,
        )
        return run_full(
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
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
    model = None

    cached_data_path = _get_observer_output_path(
        results_dir,
        ds_args.dataset_name,
        obs_args.output_file_name,
    )

    if ds_args.dataset_name == "combined":
        if cached_data_path.exists():
            logger.info(f"Loading cached observer data from {cached_data_path}")
            observer_data = torch.load(cached_data_path, weights_only=False)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {cached_data_path}"
            )
    else:
        logger.info("Preparing calibration samples...")
        data_batches = prepare_calibration_batches(tokenizer, ds_args, obs_args)

        # Layerwise: auto+disk offload instead of pinning entire model in host RAM.
        offload_root = results_dir / ".offload"
        plan = plan_load(
            "layerwise",
            offload_root=offload_root,
            low_cpu_mem_usage=layerwise_args.low_cpu_mem_usage,
        )
        logger.info(
            "Loading model for layerwise processing (%s)...", plan.reason
        )
        model = load_causal_lm(model_name, plan)
        logger.info(f"Model loaded: {model.__class__.__name__}")
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {num_params / 1e9:.2f}B")

        if cached_data_path.exists() and not obs_args.overwrite_observations:
            logger.info(f"Loading cached observer data from {cached_data_path}")
            observer_data = torch.load(cached_data_path, weights_only=False)
        else:
            logger.info("Recording activations using layerwise processing...")
            observer_data = record_activations_layerwise(
                model,
                tokenizer,
                data_batches,
                ds_args,
                obs_args,
                layerwise_args,
                results_dir,
            )

    if reap_args.run_observer_only:
        logger.info("Observer run completed. Exiting (run_observer_only=True)")
        return None

    # Calculate number of experts to prune
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either n_experts_to_prune or compression_ratio must be set."
            )
        total_experts = len(
            observer_data[next(iter(observer_data))]["expert_frequency"]
        )
        n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
        logger.info(
            f"Calculated n_experts_to_prune: {n_experts_to_prune} "
            f"(compression_ratio: {cluster_args.compression_ratio})"
        )
    else:
        total_experts = len(
            observer_data[next(iter(observer_data))]["expert_frequency"]
        )

    # Get output directory
    pruned_model_dir = get_pruned_model_dir(
        results_dir,
        n_experts_to_prune,
        total_experts,
        prune_args,
        reap_args.seed,
        obs_args.renormalize_router_weights,
        name_prefix="layerwise_",
    )

    # Check if already pruned
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model already exists at {pruned_model_dir}. Skipping pruning."
        )
    else:
        # Reload for mutate/save with GPU-first residency (never pin full model
        # on CPU when host RAM is the bottleneck).
        logger.info("Reloading model for pruning (gpu_full plan)...")
        if model is not None:
            del model
        cleanup_memory()

        from reap.residency import load_causal_lm, plan_load

        plan = plan_load("gpu_full")
        try:
            model = load_causal_lm(
                model_name, plan, local_files_only=True
            )
        except Exception:
            model = load_causal_lm(model_name, plan, local_files_only=False)

        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune_model(
            observer_data,
            model,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )

        # Save tokenizer
        tokenizer.save_pretrained(pruned_model_dir)

        # Save args
        dump_args_to_yaml(
            pruned_model_dir,
            reap_args=reap_args,
            ds_args=ds_args,
            obs_args=obs_args,
            model_args=model_args,
            eval_args=eval_args,
            prune_args=prune_args,
            cluster_args=cluster_args,
            layerwise_args=layerwise_args,
        )

        logger.info("Pruning completed successfully!")

    # Evaluation
    if reap_args.do_eval:
        logger.info("Starting evaluation...")
        if model is not None:
            del model
        del observer_data
        cleanup_memory()

        model_args.model_name = pruned_model_dir
        run_evaluate(
            model_args,
            pruned_model_dir / "eval",
            eval_args,
            reap_args.seed,
        )

    return pruned_model_dir


def main():
    """CLI entry (HfArgumentParser). Prefer ``reap prune layerwise`` (Typer)."""
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            PruneArgs,
            ClusterArgs,
            LayerwiseArgs,
        )
    )
    (
        reap_args,
        ds_args,
        obs_args,
        model_args,
        eval_args,
        prune_args,
        cluster_args,
        layerwise_args,
    ) = parser.parse_args_into_dataclasses()
    run(
        reap_args,
        ds_args,
        obs_args,
        model_args,
        eval_args,
        prune_args,
        cluster_args,
        layerwise_args,
    )


if __name__ == "__main__":
    main()
