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
from reap.residency import (
    estimate_model_bytes_from_config,
    estimate_model_bytes_from_module,
    load_causal_lm,
    plan_load,
    preflight_or_warn,
    resolve_residency,
    stream_save_pretrained,
    validate_residency,
)

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


def apply_pruning(
    observer_data,
    model,
    prune_args,
    n_experts_to_prune,
) -> None:
    """Mutate the model in-memory by slicing / pruning low-saliency experts.

    Does **not** write any files. After this call the model is pruned but
    still in its current device placement.  Call ``publish_pruned_model``
    (after a smoke test) to persist weights.
    """
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
    if prune_args.preserve_super_experts or prune_args.preserve_outliers:
        super_idx = get_super_expert_indices(
            observer_data,
            include_last_layers=bool(prune_args.preserve_outliers),
        )
        for row in super_idx:
            layer_i, expert_i = int(row[0].item()), int(row[1].item())
            protected.setdefault(layer_i, set()).add(expert_i)
        logger.info(
            "Preserving %s super/outlier experts across layers during prune",
            int(super_idx.shape[0]),
        )

    # Compute per-layer capacity to determine a single uniform prune count.
    capacity: dict[int, int] = {}
    for layer in observer_data:
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        if layer in protected:
            protected_count = len({e for e in protected[layer] if 0 <= e < num_experts})
        else:
            protected_count = 0
        max_possible = num_experts - protected_count - 1  # always leave >= 1
        capacity[layer] = max(0, max_possible)

    effective_n_prune = min(n_experts_to_prune, *capacity.values())
    if effective_n_prune < n_experts_to_prune:
        constrained = [l for l, c in capacity.items() if c < n_experts_to_prune]
        logger.warning(
            "Requested %d pruned but layers %s can only accommodate %d; "
            "using %d so all layers keep the same number of experts.",
            n_experts_to_prune,
            constrained,
            min(capacity.values()),
            effective_n_prune,
        )

    retained_expert_indicies = None
    for layer in tqdm(observer_data, "Pruning layers..."):
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        saliency = _resolve_saliency(
            observer_data[layer], prune_args.prune_method, model_device=model_device
        )

        if layer in protected:
            protected_set = {
                e for e in protected[layer] if 0 <= e < num_experts
            }
            unprotected = [
                i for i in range(num_experts) if i not in protected_set
            ]
        else:
            protected_set = set()
            unprotected = list(range(num_experts))

        if effective_n_prune < 1:
            retained_expert_indicies = list(range(num_experts))
            continue

        if unprotected:
            unprotected_saliency = saliency[unprotected]
            _, unprotected_prune_rel = torch.topk(
                unprotected_saliency, effective_n_prune, largest=False
            )
            prune_set = {unprotected[i] for i in unprotected_prune_rel.tolist()}
        else:
            prune_set = set()

        if protected_set:
            actually_pruned = prune_set & protected_set
            if actually_pruned:
                logger.warning(
                    "Layer %d: %d protected experts were selected for pruning "
                    "\u2014 this indicates a bug in the protection logic; skipping prune",
                    layer,
                    len(actually_pruned),
                )
                retained_expert_indicies = list(range(num_experts))
                continue

        retained_expert_indicies = [
            i for i in range(num_experts) if i not in prune_set
        ]
        moe = adapter.get_moe(layers[layer])
        adapter.slice_experts(moe, retained_expert_indicies)

    retained_experts = len(retained_expert_indicies)
    moe_layer_indices = adapter.identify_moe_layers(model)
    first_moe_layer = layers[moe_layer_indices[0]]
    layer_cfg = adapter.get_layer_config(first_moe_layer, model.config)
    adapter.update_config(model.config, retained_experts, layer_cfg.top_k)
    logger.info(
        "Pruning applied: %d MoE layers each \u2192 %d experts",
        len(observer_data), retained_experts,
    )


def publish_pruned_model(
    model,
    tokenizer,
    pruned_model_dir: pathlib.Path,
    *,
    smoke_test_fn=None,
    extra_yaml_args: dict[str, object] | None = None,
) -> pathlib.Path:
    """Write pruned model weights, tokenizer, and metadata to disk.

    Uses a staging directory so that partial output is never written to
    the final path.  If ``smoke_test_fn`` is provided, it is called
    *before* writing files; when it raises, no output is left behind.
    """
    # Run smoke test before writing any files.
    if smoke_test_fn is not None:
        logger.info("Running smoke test before publishing...")
        smoke_test_fn()

    # Write to a staging directory next to the final path.
    import uuid
    pruned_model_dir = pathlib.Path(pruned_model_dir)
    stage = pruned_model_dir.parent / f".{pruned_model_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        stage.mkdir(parents=True, exist_ok=True)
        start = time.time()
        stream_save_pretrained(model, stage)
        tokenizer.save_pretrained(stage)
        if extra_yaml_args:
            from reap.pipeline import dump_args_to_yaml
            dump_args_to_yaml(stage, **extra_yaml_args)
        end = time.time()
        logger.info(
            "Pruned model saved to staging %s in %.2f seconds",
            stage, end - start,
        )
        # Atomically promote.
        if pruned_model_dir.exists():
            import shutil
            shutil.rmtree(pruned_model_dir)
        stage.rename(pruned_model_dir)
        logger.info("Published pruned model to %s", pruned_model_dir)
    except BaseException:
        if stage.exists():
            import shutil
            shutil.rmtree(stage, ignore_errors=True)
            logger.warning("Staging directory %s cleaned up after failure", stage)
        raise

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

    if prune_args.preserve_super_experts:
        pruned_model_name += "-preserve_super"
    elif prune_args.preserve_outliers:
        pruned_model_name += "-preserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"

    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")

    return pruned_model_dir


def run(
    reap_args: ReapArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    model_args: ModelArgs,
    eval_args: EvalArgs,
    prune_args: PruneArgs,
    cluster_args: ClusterArgs,
    *,
    _residency_resolved: str | None = None,
) -> pathlib.Path | None:
    """Observe → prune → optional eval.

    Honors ``reap_args.residency``. If resolved mode is ``layerwise``, delegates
    to :func:`reap.layerwise_prune.run`.
    """
    tcr = bool(getattr(model_args, "trust_remote_code", False))
    if _residency_resolved is None:
        residency = validate_residency(getattr(reap_args, "residency", "auto"))
        model_bytes = estimate_model_bytes_from_config(model_args.model_name, trust_remote_code=tcr)
        resolved, reason = resolve_residency(
            residency,
            model_bytes=model_bytes,
            cli_prefers_layerwise=False,
        )
        logger.info("Residency resolved: %s (%s)", resolved, reason)
        preflight_or_warn(resolved, model_bytes)
    else:
        resolved = validate_residency(_residency_resolved)
        model_bytes = estimate_model_bytes_from_config(model_args.model_name, trust_remote_code=tcr)
        logger.info("Residency (pre-resolved): %s", resolved)
        preflight_or_warn(resolved, model_bytes)

    if resolved == "layerwise":
        from reap.args import LayerwiseArgs
        from reap.layerwise_prune import run as run_layerwise

        logger.info("Delegating to layerwise prune (residency=%s)", resolved)
        return run_layerwise(
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
            LayerwiseArgs(),
            _residency_resolved=resolved,
        )

    set_seed(reap_args.seed)
    results_dir = create_results_directory(
        model_args.model_name,
        ds_args.dataset_name,
        base=getattr(reap_args, "artifacts_dir", None),
    )

    model_name = model_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=tcr)
    plan = plan_load("cpu_full" if resolved == "cpu_full" else "gpu_full")

    model = load_causal_lm(model_name, plan, trust_remote_code=tcr)
    try:
        live_bytes = estimate_model_bytes_from_module(model)
        logger.info("Loaded model weight footprint ~%.2f GiB", live_bytes / 1024**3)
    except Exception:
        pass

    logger.info(
        "Running observer to collect activation data for model %s on dataset %s.",
        model_args.model_name,
        ds_args.dataset_name,
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
        return None

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
        apply_pruning(
            observer_data,
            model,
            prune_args,
            n_experts_to_prune,
        )
        logger.info("pruning completed.")

        # Collect extra YAML args for the published artifact.
        yaml_kwargs = dict(
            reap_args=reap_args,
            ds_args=ds_args,
            obs_args=obs_args,
            model_args=model_args,
            eval_args=eval_args,
            prune_args=prune_args,
            cluster_args=cluster_args,
        )

        publish_pruned_model(
            model,
            tokenizer,
            pruned_model_dir,
            smoke_test_fn=(
                lambda: smoke_test(model, tokenizer)
                if reap_args.smoke_test else None
            ),
            extra_yaml_args=yaml_kwargs,
        )

        if reap_args.smoke_test:
            logger.info("Smoke test passed; model published.")

    if reap_args.do_eval and pruned_model_dir.exists():
        remove_hook_from_module(model, recurse=True)
        # Avoid model.to("cpu") on low-RAM hosts — drop references instead.
        del model
        del observer_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = str(pruned_model_dir)
        run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)

    return pruned_model_dir


def main():
    """CLI entry (HfArgumentParser). Prefer ``reap prune full`` (Typer)."""
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
    run(reap_args, ds_args, obs_args, model_args, eval_args, prune_args, cluster_args)


if __name__ == "__main__":
    main()
