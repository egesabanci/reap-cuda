from __future__ import annotations
import dataclasses
import hashlib
import logging
import os
import pathlib
import re
from importlib.metadata import PackageNotFoundError, version

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


def _normalized_identifier(value: str) -> str:
    """Normalize a local path or Hub identifier without losing its namespace."""
    path = pathlib.Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value.strip()


def _artifact_path_component(value: str, *, prefix: str) -> str:
    """Return a readable, collision-resistant component for an artifact path."""
    normalized = _normalized_identifier(value)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    readable = str_to_directory_name(normalized.replace("/", "--"))[-72:]
    return f"{prefix}_{readable}-{digest}"


def create_results_directory(
    model_name: str,
    dataset_name: str,
    *,
    base: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Create collision-resistant artifact directories for a model/dataset pair.

    The full normalized model and dataset identities are retained in readable
    form and suffixed with stable hashes. This prevents ``owner-a/model`` and
    ``owner-b/model`` (or distinct local paths with a shared basename) from
    reusing each other's observations/checkpoints.
    """
    model_clean = _artifact_path_component(model_name, prefix="model")
    dataset_clean = _artifact_path_component(dataset_name, prefix="dataset")

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


_OBSERVATION_SCHEMA_VERSION = 3


def observer_metadata_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _fingerprint_path(value: str | None) -> str | None:
    """Hash local path identity and bounded content samples for cache safety."""
    if not value:
        return None
    root = pathlib.Path(value).expanduser()
    if not root.exists():
        return None
    digest = hashlib.sha256(str(root.resolve()).encode("utf-8"))
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for file_path in files[:10_000]:
        stat = file_path.stat()
        digest.update(str(file_path.relative_to(root) if root.is_dir() else file_path.name).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        # Detect content changes that preserve timestamp/size without reading a
        # potentially multi-gigabyte model or dataset in full.
        with file_path.open("rb") as handle:
            digest.update(handle.read(64 * 1024))
            if stat.st_size > 64 * 1024:
                handle.seek(max(0, stat.st_size - 64 * 1024))
                digest.update(handle.read(64 * 1024))
    digest.update(str(len(files)).encode())
    return digest.hexdigest()


def _reap_version() -> str:
    try:
        return version("reap")
    except PackageNotFoundError:
        return "local-source"


def _compute_artifact_metadata(
    reap_args, model_args, ds_args, obs_args,
) -> dict[str, object]:
    """Compute a versioned, provenance-rich observation cache manifest."""
    return {
        "schema_version": _OBSERVATION_SCHEMA_VERSION,
        "reap_version": _reap_version(),
        "observer_schema_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
        "model": {
            "name": model_args.model_name,
            "normalized_id": _normalized_identifier(model_args.model_name),
            "revision": getattr(model_args, "model_revision", None),
            "local_fingerprint": _fingerprint_path(model_args.model_name),
            "trust_remote_code": bool(getattr(model_args, "trust_remote_code", False)),
        },
        "tokenizer": {
            "revision": getattr(model_args, "model_revision", None),
            "local_fingerprint": _fingerprint_path(model_args.model_name),
        },
        "dataset": {
            "name": ds_args.dataset_name,
            "config": getattr(ds_args, "dataset_config_name", None),
            "path": getattr(ds_args, "dataset_path", None),
            "path_fingerprint": _fingerprint_path(getattr(ds_args, "dataset_path", None)),
            "split": ds_args.split,
            "shuffle": getattr(ds_args, "shuffle", True),
            "seed": reap_args.seed,
        },
        "observer": {
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
        },
    }


def write_observer_metadata(path: pathlib.Path, metadata: dict[str, object]) -> None:
    metadata_path = observer_metadata_path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)


def _validate_observer_schema(value: object, path: pathlib.Path) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Observer artifact {path} must be a non-empty dict.")
    for layer_key, layer_data in value.items():
        if not isinstance(layer_key, (int, str)) or not isinstance(layer_data, dict):
            raise ValueError(f"Observer artifact {path} has invalid layer entry {layer_key!r}.")
        frequency = layer_data.get("expert_frequency")
        total_tokens = layer_data.get("total_tokens")
        if not isinstance(frequency, torch.Tensor) or frequency.ndim != 1:
            raise ValueError(
                f"Observer artifact {path} layer {layer_key!r} requires a 1-D "
                "tensor 'expert_frequency'."
            )
        if not isinstance(total_tokens, (torch.Tensor, int, float)):
            raise ValueError(
                f"Observer artifact {path} layer {layer_key!r} requires numeric 'total_tokens'."
            )
    return value


def load_observer_artifact(
    path: pathlib.Path,
    *,
    expected_metadata: dict[str, object] | None = None,
    trust_legacy: bool = False,
) -> dict[str, object]:
    """Safely load and schema-check an observation artifact and its manifest."""
    path = pathlib.Path(path)
    value = _validate_observer_schema(
        torch.load(path, weights_only=True, map_location="cpu"), path
    )
    metadata_file = observer_metadata_path(path)
    if expected_metadata is not None:
        if not metadata_file.exists():
            if not trust_legacy:
                raise ValueError(
                    f"Observation artifact {path} has no manifest. Recompute it or pass "
                    "--trust-observation-artifact for a legacy artifact you trust."
                )
            logger.warning("Loading trusted legacy observation artifact without manifest: %s", path)
        else:
            with metadata_file.open() as handle:
                stored_metadata = json.load(handle)
            if stored_metadata != expected_metadata:
                raise ValueError(f"Observation artifact {path} has an incompatible manifest.")
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
    current_meta = _compute_artifact_metadata(reap_args, model_args, ds_args, obs_args)

    # The aggregate is the sole authoritative cache. Its manifest and schema
    # must match before it can bypass model/data loading.
    if aggregate_path.exists() and not obs_args.overwrite_observations:
        try:
            data = load_observer_artifact(
                aggregate_path,
                expected_metadata=current_meta,
                trust_legacy=bool(getattr(obs_args, "trust_observation_artifact", False)),
            )
            logger.info("Aggregate cache hit @ %s; returning cached data.", aggregate_path)
            return data
        except Exception as exc:
            logger.warning("Aggregate cache is not reusable (%s); recomputing.", exc)

    if ds_args.dataset_name == "combined":
        if aggregate_path.exists():
            return load_observer_artifact(
                aggregate_path,
                expected_metadata=current_meta,
                trust_legacy=bool(getattr(obs_args, "trust_observation_artifact", False)),
            )
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
            write_observer_metadata(cat_path, {**current_meta, "category": str(category)})
            logger.info(f"Category '{category}' data saved to {cat_path}")

    # Write aggregate (all categories combined in one observer).
    observer.save_state(aggregate_path)
    write_observer_metadata(aggregate_path, current_meta)
    logger.info(
        "Aggregate cache saved @ %s (schema_version=%d)",
        aggregate_path,
        _OBSERVATION_SCHEMA_VERSION,
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

