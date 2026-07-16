from __future__ import annotations
import gc
import logging
import pathlib
import time
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.pipeline import record_activations, smoke_test, create_results_directory
from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
)
from reap.model_adapters import infer_model_adapter
from reap.pruning_metrics import PRUNE_METHOD_KEY_MAP
from reap.eval import run_evaluate
from reap.merge_pipeline import get_super_expert_indices

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _resolve_saliency(
    layer_data: dict[str, Any],
    prune_method: str,
    *,
    model_device: torch.device | str,
) -> torch.Tensor:
    """Map CLI prune_method to a per-expert saliency vector (lower = prune first)."""
    if prune_method == "ean_ca":
        rca = layer_data.get("routed_characteristic_activation")
        if rca is None:
            raise ValueError(
                "ean_ca requires routed_characteristic_activation; re-run the "
                "observer with --record_pruning_metrics_only False "
                "(or use a merge-mode calibration)."
            )
        num_experts = rca.shape[0]
        ean = torch.zeros(num_experts, device=model_device, dtype=torch.float32)
        for i in range(num_experts):
            ean[i] = torch.linalg.norm(rca[i].float(), dim=-1).sum()
        return ean

    key = PRUNE_METHOD_KEY_MAP.get(prune_method, prune_method)
    saliency_data = layer_data.get(key)
    if saliency_data is None:
        raise ValueError(
            f"Prune method {prune_method!r} (key {key!r}) not found in observer "
            f"data. Available keys: {list(layer_data.keys())}"
        )
    if not isinstance(saliency_data, torch.Tensor):
        saliency_data = torch.as_tensor(saliency_data)
    return saliency_data.to(device=model_device)


def prune(
    observer_data,
    model,
    prune_args,
    n_experts_to_prune,
    pruned_model_dir,
):
    """Prune the model based on the observer data."""
    adapter = infer_model_adapter(model, model.config)
    if adapter is None:
        raise ValueError(
            f"Cannot detect a supported MoE adapter for "
            f"{model.__class__.__name__}. REAP currently supports Qwen3-MoE, "
            "Llama4-MoE, LFM2-MoE, and Mixtral-style architectures."
        )
    layers = adapter.layers(model)
    model_device = next(model.parameters()).device

    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    # Optional super/outlier expert preservation (mirrors merge path).
    protected: dict[int, set[int]] = {}
    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_idx = get_super_expert_indices(
            observer_data,
            include_last_layers=bool(prune_args.perserve_outliers),
        )
        for row in super_idx:
            layer_i, expert_i = int(row[0].item()), int(row[1].item())
            protected.setdefault(layer_i, set()).add(expert_i)
        logger.info(
            "Preserving %s super/outlier experts across layers during prune",
            int(super_idx.shape[0]),
        )

    retained_expert_indicies = None
    for layer in tqdm(observer_data, "Pruning layers..."):
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        saliency = _resolve_saliency(
            observer_data[layer], prune_args.prune_method, model_device=model_device
        )
        # Mask protected experts so they are never among the lowest-k.
        if layer in protected:
            saliency = saliency.clone()
            for e in protected[layer]:
                if 0 <= e < num_experts:
                    saliency[e] = torch.finfo(saliency.dtype).max

        n_prune = min(n_experts_to_prune, num_experts - 1)
        if n_prune < 1:
            retained_expert_indicies = list(range(num_experts))
            continue

        _, experts_to_prune = torch.topk(saliency, n_prune, largest=False)
        prune_set = set(experts_to_prune.tolist())
        retained_expert_indicies = [
            i for i in range(num_experts) if i not in prune_set
        ]
        moe = adapter.get_moe(layers[layer])
        adapter.slice_experts(moe, retained_expert_indicies)

    logger.info("Saving pruned model...")
    retained_experts = len(retained_expert_indicies)
    moe_layer_indices = adapter.identify_moe_layers(model)
    first_moe_layer = layers[moe_layer_indices[0]]
    layer_cfg = adapter.get_layer_config(first_moe_layer, model.config)
    adapter.update_config(model.config, retained_experts, layer_cfg.top_k)

    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # Strip accelerate dispatch hooks so save streams GPU tensors via safetensors
    # without materializing a full CPU state dict.
    remove_hook_from_module(model, recurse=True)
    model.save_pretrained(pruned_model_dir)
    end = time.time()
    logger.info(
        f"Pruned model saved to {pruned_model_dir} in {end - start:.2f} seconds"
    )
    return pruned_model_dir


def get_pruned_model_dir(
    results_dir: pathlib.Path,
    n_experts_to_prune: int,
    total_experts: int,
    prune_args: PruneArgs,
    seed: int,
    renorm: bool,
    name_prefix: str = None,
) -> pathlib.Path:
    """Generate output directory path for pruned model."""
    compression_ratio_str = f"{(n_experts_to_prune / total_experts):.2f}"
    name_prefix = "" if name_prefix is None else name_prefix
    pruned_model_name = f"{name_prefix}{prune_args.prune_method}"

    if prune_args.perserve_super_experts:
        pruned_model_name += "-perserve_super"
    elif prune_args.perserve_outliers:
        pruned_model_name += "-perserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"

    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")

    return pruned_model_dir


def main():
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            PruneArgs,
            ClusterArgs,
        )
    )
    reap_args, ds_args, obs_args, model_args, eval_args, prune_args, cluster_args = (
        parser.parse_args_into_dataclasses()
    )
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
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    logger.info("Start of pruning")
    total_experts = len(
        observer_data[next(iter(observer_data))]["expert_frequency"]
    )
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either n_experts_to_prune or compression_ratio must be set for pruning."
            )
        n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
        logger.info(
            f"Calculated n_experts to prune: {n_experts_to_prune} from compression_ratio: {cluster_args.compression_ratio}"
        )

    pruned_model_dir = get_pruned_model_dir(
        results_dir,
        n_experts_to_prune,
        total_experts,
        prune_args,
        reap_args.seed,
        obs_args.renormalize_router_weights,
    )
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model directory {pruned_model_dir} already exists and contains pruned model files. "
            "Skipping pruning step."
        )
    else:
        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune(
            observer_data,
            model,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )
        logger.info("pruning completed.")

        if reap_args.smoke_test:
            logger.info("Running smoke test on the pruned model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")

        tokenizer.save_pretrained(pruned_model_dir)
        logger.info("Pruning completed.")

    if reap_args.do_eval and pruned_model_dir.exists():
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = str(pruned_model_dir)
        run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
