from __future__ import annotations
import dataclasses
import logging
import pathlib
import re

import json
import yaml
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from accelerate.utils import set_seed


from reap.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    EvalArgs,
)
from reap.data import (
    load_category_batches,
    load_composite_category_batches,
    parse_composite_dataset_spec,
)
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig
from reap.model_adapters import infer_model_adapter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def str_to_directory_name(s: str) -> str:
    """Convert a string to a valid directory name by replacing special characters."""
    return re.sub(r"[^\w\-_.]", "_", s)


def resolve_artifacts_root(base: str | pathlib.Path | None = None) -> pathlib.Path:
    """Resolve artifacts root: explicit arg → ``REAP_ARTIFACTS_DIR`` → ``./artifacts``."""
    import os

    if base is not None and str(base).strip():
        return pathlib.Path(base).expanduser().resolve()
    env = os.environ.get("REAP_ARTIFACTS_DIR") or os.environ.get("REAP_OUTPUT_DIR")
    if env and env.strip():
        return pathlib.Path(env).expanduser().resolve()
    return pathlib.Path("./artifacts").resolve()


def create_results_directory(
    model_name: str,
    dataset_name: str,
    *,
    base: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Create a clean directory name from model and dataset names.

    For composite dataset specs (comma-separated), uses a short hash of the
    full spec as the directory name: ``composite_<md5[:8]>``.

    *base* overrides the root (also via ``REAP_ARTIFACTS_DIR`` / ``REAP_OUTPUT_DIR``).
    """
    import hashlib

    model_clean = model_name.split("/")[-1]
    model_clean = str_to_directory_name(model_clean)

    if "," in dataset_name:
        spec_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
        dataset_clean = f"composite_{spec_hash}"
        logger.info(
            f"Composite dataset spec detected. Using directory name: {dataset_clean}"
        )
    else:
        dataset_clean = dataset_name.split("/")[-1]
        dataset_clean = str_to_directory_name(dataset_clean)

    root = resolve_artifacts_root(base)
    results_dir = root / model_clean / dataset_clean

    if results_dir.exists():
        logger.warning(f"Directory '{results_dir}' already exists")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created artifacts directory: {results_dir}")

    return results_dir


def _setup_observer(model, obs_args):
    """Create and return an MoETransformerObserver for the given model."""
    adapter = infer_model_adapter(model, model.config)
    if adapter is None:
        raise ValueError(
            f"Cannot detect a supported MoE adapter for "
            f"{model.__class__.__name__}. REAP currently supports Qwen3-MoE, "
            "Llama4-MoE, LFM2-MoE, and Mixtral-style architectures."
        )

    moe_layer_indices = adapter.identify_moe_layers(model)
    if not moe_layer_indices:
        raise ValueError(
            f"No MoE layers found in {model.__class__.__name__}; "
            "cannot configure observer."
        )
    first_moe_layer = adapter.layers(model)[moe_layer_indices[0]]

    renormalize_router_weights = (
        getattr(model.config, "norm_topk_prob", False)
        and obs_args.renormalize_router_weights
    )
    if renormalize_router_weights:
        logger.info("Renormalizing topk router weights to sum to 1.")

    # Honor ObserverArgs.frea_backend for kernel dispatch.
    try:
        from reap.kernels.triton_frea import set_frea_backend

        set_frea_backend(getattr(obs_args, "frea_backend", "auto"))
    except Exception:
        pass

    observer_config = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=adapter.get_layer_config(
            first_moe_layer, model.config
        ).fused_experts,
        distance_measure="cosine",
        renormalize_router_weights=renormalize_router_weights,
        record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
        observe_backend=getattr(obs_args, "observe_backend", "auto"),
    )
    return MoETransformerObserver(
        model=model,
        hook_config=observer_config,
        adapter=adapter,
    )


def _primary_device(model: torch.nn.Module) -> torch.device:
    """Best-effort device for accelerate / device_map models."""
    try:
        dev = getattr(model, "device", None)
        if isinstance(dev, torch.device):
            return dev
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _profile_model(model, tokenizer, model_args, obs_args, observer):
    """Run a profiling forward pass to avoid OOM at inference time."""
    with torch.no_grad():
        try:
            model_max_length = obs_args.model_max_length
            if model_max_length is None:
                model_max_length = tokenizer.model_max_length
            logger.info(f"Profiling at model max length: {model_max_length}.")
            s = "hello " * model_max_length
            tokenized = tokenizer(
                [s],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )
            tokenized = {
                k: v.to(_primary_device(model)) for k, v in tokenized.items()
            }
            for _ in range(2):
                _ = model(**tokenized)
        except Exception as e:
            raise RuntimeError(
                f"Failed to run model with max input length {model_max_length}: {e}"
            )
    logger.info(
        f"Model {model_args.model_name} successfully loaded and profiled at max length {model_max_length}."
    )
    observer.reset()


def _compute_artifact_metadata(
    reap_args, model_args, ds_args, obs_args,
) -> dict[str, object]:
    """Compute a stable metadata dict for cache invalidation."""
    return {
        "version": 2,
        "model_name": model_args.model_name,
        "dataset_name": ds_args.dataset_name,
        "dataset_config_name": getattr(ds_args, "dataset_config_name", None),
        "dataset_path": getattr(ds_args, "dataset_path", None),
        "split": ds_args.split,
        "seed": reap_args.seed,
        "shuffle": getattr(ds_args, "shuffle", True),
        "model_max_length": obs_args.model_max_length,
        "batches_per_category": obs_args.batches_per_category,
        "batch_size": obs_args.batch_size,
        "truncate": obs_args.truncate,
        "split_by_category": obs_args.split_by_category,
        "distance_measure": obs_args.distance_measure,
        "record_pruning_metrics_only": obs_args.record_pruning_metrics_only,
        "renormalize_router_weights": obs_args.renormalize_router_weights,
        "observe_backend": getattr(obs_args, "observe_backend", "auto"),
        "frea_backend": getattr(obs_args, "frea_backend", "auto"),
    }


def load_observer_artifact(path: pathlib.Path) -> dict[str, object]:
    """Load an observer state dict with safe tensor-only deserialization."""
    value = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Observer artifact {path} is not a dict")
    for layer_key, layer_data in value.items():
        if isinstance(layer_data, dict):
            expected_fields = {"expert_frequency", "total_tokens"}
            if not expected_fields.issubset(layer_data):
                logger.warning(
                    "Layer %s in %s lacks expected fields %s",
                    layer_key, path, expected_fields - set(layer_data),
                )
    return value


def record_activations(
    model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
):
    from reap.kernels.triton_utils import log_triton_usage_summary, reset_triton_usage

    reset_triton_usage()
    dataset_path = getattr(ds_args, "dataset_path", None)

    # Compute aggregate path early (before any model/observer setup).
    all_dir = results_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = all_dir / obs_args.output_file_name
    meta_path = aggregate_path.with_suffix(aggregate_path.suffix + ".meta.json")

    # Early return if valid aggregate cache exists.
    if aggregate_path.exists() and not obs_args.overwrite_observations:
        try:
            current_meta = _compute_artifact_metadata(reap_args, model_args, ds_args, obs_args)
            if meta_path.exists():
                with open(meta_path) as f:
                    stored_meta = json.load(f)
                if stored_meta == current_meta:
                    logger.info(
                        "Aggregate cache hit @ %s; returning cached data.",
                        aggregate_path,
                    )
                    return load_observer_artifact(aggregate_path)
                else:
                    logger.info(
                        "Aggregate cache @ %s has mismatched metadata; recomputing.",
                        aggregate_path,
                    )
            else:
                logger.info(
                    "Aggregate cache @ %s has no metadata; recomputing.",
                    aggregate_path,
                )
        except Exception as exc:
            logger.warning("Aggregate cache validation failed (%s); recomputing.", exc)

    if ds_args.dataset_name == "combined":
        if aggregate_path.exists():
            return load_observer_artifact(aggregate_path)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {aggregate_path}"
            )

    # check for composite dataset specification
    composite_components = parse_composite_dataset_spec(
        ds_args.dataset_name,
        default_split=ds_args.split,
    )
    if composite_components is not None:
        total_batches = sum(c.num_batches for c in composite_components)
        logger.info(
            f"Composite dataset specified, overwriting given batches_per_category={obs_args.batches_per_category} "
            f"with values in composite dataset spec "
            f"({len(composite_components)} components, {total_batches} total **batches**)."
        )
        if dataset_path:
            logger.info(
                "Composite + --dataset-path=%s (per-component @path overrides; "
                "or {path}/<short_name> subdirs)",
                dataset_path,
            )
        category_data_batches = load_composite_category_batches(
            composite_components,
            tokenizer=tokenizer,
            model_max_length=obs_args.model_max_length,
            batch_size=obs_args.batch_size,
            return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
            truncate=obs_args.truncate,
            global_dataset_path=dataset_path,
            shuffle=getattr(ds_args, "shuffle", True),
            seed=reap_args.seed,
        )
    else:
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
            dataset_path=dataset_path,
            shuffle=getattr(ds_args, "shuffle", True),
            seed=reap_args.seed,
        )

    logger.info(
        "Loaded and processed data for categories: %s",
        str(list(category_data_batches.keys())),
    )

    # load observer and hook model
    observer = _setup_observer(model, obs_args)
    primary_device = _primary_device(model)

    if reap_args.profile:
        _profile_model(model, tokenizer, model_args, obs_args, observer)

    # Process all categories into a single accumulated observer state.
    # Per-category artifacts are saved as diagnostics; the aggregate is the
    # authoritative cache.  Every category always runs — a per-category skip
    # would leave the aggregate with empty state.
    with torch.no_grad():
        for category, cat_data in category_data_batches.items():
            logger.info(f"Processing category: {category}...")
            cat_dir = results_dir / str_to_directory_name(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            for sample in tqdm(cat_data, desc=f"Processing {category} samples"):
                attention_mask = sample.get("attention_mask", None)
                sample = {
                    k: v.to(primary_device) if torch.is_tensor(v) else v
                    for k, v in sample.items()
                }
                with observer.set_attention_mask(attention_mask):
                    model(**sample)
            # Save per-category diagnostic snapshot.
            cat_path = cat_dir / obs_args.output_file_name
            observer.save_state(cat_path)
            logger.info(f"Category '{category}' data saved to {cat_path}")

    # Write aggregate (all categories combined in one observer).
    observer.save_state(aggregate_path)
    # Write sidecar metadata for future cache validation.
    meta = _compute_artifact_metadata(reap_args, model_args, ds_args, obs_args)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(
        "Aggregate cache saved @ %s (version=%d)", aggregate_path, meta["version"]
    )

    # Capture the in-memory state BEFORE close_hooks() resets it.
    observer_data = observer.report_state()
    observer.close_hooks()
    log_triton_usage_summary()
    return observer_data


@torch.no_grad()
def smoke_test(model: torch.nn.Module, tokenizer: AutoTokenizer):
    """Run a short generate smoke test (transformers 5.x BatchEncoding-safe)."""
    device = _primary_device(model)
    prompt = "What is your name?"
    test_input = [{"role": "user", "content": prompt}]
    try:
        inputs = tokenizer.apply_chat_template(
            test_input,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        if hasattr(inputs, "items"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif torch.is_tensor(inputs):
            inputs = {"input_ids": inputs.to(device)}
        else:
            raise TypeError(f"Unexpected apply_chat_template return type: {type(inputs)}")
    except Exception as exc:
        logger.warning(
            "apply_chat_template failed (%s); falling back to plain tokenize", exc
        )
        enc = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in enc.items()}

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        pad_token_id=pad_id,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    logger.info("Smoke test response: %s", response[0])





def dump_args_to_yaml(
    pruned_model_dir: pathlib.Path,
    **all_args,
):
    """Dump all arguments to a YAML file."""

    def convert_paths_to_str(data):
        if isinstance(data, dict):
            return {k: convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_paths_to_str(i) for i in data]
        elif isinstance(data, pathlib.Path):
            return str(data)
        else:
            return data

    serializable_args = {}
    for name, arg in all_args.items():
        if dataclasses.is_dataclass(arg):
            serializable_args[name] = convert_paths_to_str(dataclasses.asdict(arg))
        else:
            serializable_args[name] = convert_paths_to_str(arg)

    output_path = pruned_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"Arguments saved to {output_path}")

