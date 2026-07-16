from __future__ import annotations
import dataclasses
import logging
import pathlib
import re

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
from reap.data import load_category_batches, parse_composite_dataset_spec
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig
from reap.model_adapters import infer_model_adapter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def str_to_directory_name(s: str) -> str:
    """Convert a string to a valid directory name by replacing special characters."""
    return re.sub(r"[^\w\-_.]", "_", s)


def create_results_directory(model_name: str, dataset_name: str) -> pathlib.Path:
    """Create a clean directory name from model and dataset names.

    For composite dataset specs (comma-separated), uses a short hash of the
    full spec as the directory name: ``composite_<md5[:8]>``.
    """
    import hashlib

    model_clean = model_name.split("/")[-1]
    model_clean = str_to_directory_name(model_clean)

    if "," in dataset_name:
        # Composite dataset spec — use hash-based directory name
        spec_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
        dataset_clean = f"composite_{spec_hash}"
        logger.info(
            f"Composite dataset spec detected. Using directory name: {dataset_clean}"
        )
    else:
        dataset_clean = dataset_name.split("/")[-1]
        dataset_clean = str_to_directory_name(dataset_clean)

    results_dir = pathlib.Path("./artifacts") / model_clean / dataset_clean

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

    observer_config = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=adapter.get_layer_config(
            first_moe_layer, model.config
        ).fused_experts,
        distance_measure="cosine",
        renormalize_router_weights=renormalize_router_weights,
        record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
    )
    return MoETransformerObserver(
        model=model,
        hook_config=observer_config,
        adapter=adapter,
    )


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
            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
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


def record_activations(
    model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
):
    if ds_args.dataset_name == "combined":
        # just return the combined data
        cat_dir = results_dir / "all"
        f_name = cat_dir / obs_args.output_file_name
        if f_name.exists():
            return torch.load(f_name, weights_only=False)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {f_name}"
            )

    # check for composite dataset specification
    composite_components = parse_composite_dataset_spec(
        ds_args.dataset_name,
        default_split=ds_args.split,
    )
    if composite_components is not None:
        combined_batches = []
        total_batches = sum(c.num_batches for c in composite_components)
        logger.info(
            f"Composite dataset specified, overwriting given batches_per_category={obs_args.batches_per_category} "
            f"with values in composite dataset spec."
        )
        logger.info(
            f"Loading composite dataset with {len(composite_components)} "
            f"components, {total_batches} total data batches."
        )

        for comp_idx, component in enumerate(composite_components):
            comp_label = (
                f"{component.name}"
                f"{f'[{component.subset}]' if component.subset is not None else ''}"
                f"[{component.split}]"
            )
            logger.info(
                f"[{comp_idx + 1}/{len(composite_components)}] Loading component: "
                f"{comp_label} ({component.num_batches} batches)"
            )
            component_batches = load_category_batches(
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
            )
            combined_batches.extend(component_batches["all"])

        category_data_batches = {"all": combined_batches}
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
        )

    logger.info(
        "Loaded and processed data for categories: %s",
        str(list(category_data_batches.keys())),
    )
    
    # load observer and hook model
    observer = _setup_observer(model, obs_args)

    if reap_args.profile:
        _profile_model(model, tokenizer, model_args, obs_args, observer)

    # run samples over model and save observer state
    with torch.no_grad():
        for category, cat_data in category_data_batches.items():
            logger.info(f"Processing category: {category}...")
            cat_dir = results_dir / str_to_directory_name(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            f_name = cat_dir / obs_args.output_file_name
            if f_name.exists() and not obs_args.overwrite_observations:
                logger.info(
                    f"Category '{category}' previously processed. Skipping to next category..."
                )
                continue
            try:
                logger.info("No previous data found @ %s", f_name)
                for sample in tqdm(cat_data, desc=f"Processing {category} samples"):
                    attention_mask = sample.get("attention_mask", None)
                    sample = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in sample.items()
                    }
                    with observer.set_attention_mask(attention_mask):
                        model(**sample)
            except Exception as e:
                logger.error(f"Error processing category '{category}'")
                logger.info(
                    f"Saving partial results for category '{category}' and exiting"
                )
                observer.save_state(cat_dir / "partial.pkl")
                logger.info(
                    f"{category} data processed and saved to "
                    f"{cat_dir / obs_args.output_file_name}"
                )
                raise e
            observer.save_state(cat_dir / obs_args.output_file_name)
            observer.reset()
            logger.info(
                f"{category} data processed and saved to "
                f"{cat_dir / obs_args.output_file_name}"
            )
    observer.close_hooks()
    with open(f"{cat_dir / obs_args.output_file_name}", "rb") as f:
        observer_data = torch.load(f, weights_only=False)
    return observer_data





@torch.no_grad()
def smoke_test(model: torch.nn.Module, tokenizer: AutoTokenizer):
    """Run a smoke test to ensure the model is functioning correctly."""
    prompt = "What is your name?"
    test_input = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        test_input,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        # enable_thinking=False,
    ).to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=True,
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

