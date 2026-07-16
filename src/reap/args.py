from dataclasses import dataclass, field


@dataclass
class ReapArgs:
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    debug: bool = field(
        default=False, metadata={"help": "Enable debug mode for more verbose output."}
    )
    profile: bool = field(
        default=True, metadata={"help": "Enable profiling prior to run to avoid OOM."}
    )
    run_observer_only: bool = field(
        default=False,
        metadata={"help": "Whether to only run the observer to collect activation data."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation after pruning."},
    )
    smoke_test: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to run a smoke test on the model to ensure it works "
                "as expected prior to saving"
            )
        },
    )


@dataclass
class ModelArgs:
    model_name: str = field(
        default="Qwen/Qwen3-30B-A3B",
        metadata={
            "help": "Name of the model to use.",
        },
    )
    num_experts_per_tok_override: int | None = field(
        default=None,
        metadata={
            "help": (
                "Override the number of experts per token. If None, uses the model's "
                "default number of experts per token."
            )
        },
    )


@dataclass
class DatasetArgs:
    dataset_name: str = field(
        default="theblackcat102/evol-codealpaca-v1",
        metadata={
            "help": (
                "Name of the dataset to use. Can be a single HuggingFace dataset name "
                "(e.g., 'theblackcat102/evol-codealpaca-v1') or a composite specification "
                "with comma-separated entries of <dataset>[<subset>](<split>):<num_samples>. "
                "Example: 'theblackcat102/evol-codealpaca-v1:4096,"
                "open-r1/Mixture-of-Thoughts[code]:4096,"
                "SWE-bench/SWE-smith-trajectories(tool):4096'. "
                "Use 'combined' to load pre-recorded combined observation data."
            ),            
        },
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "Configuration name of the dataset."}
    )
    split: str = field(default="train", metadata={"help": "Dataset split to use."})
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the dataset."}
    )
    # for SFT only
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})


@dataclass
class ObserverArgs:
    batches_per_category: int = 1024
    split_by_category: bool = False
    select_only_categories: list[str] | str | None = field(
        default=None,
        metadata={
            "help": (
                "List of categories to select for observation. If None, all categories "
                "are selected."
            )
        },
    )
    batch_size: int = 8
    model_max_length: int | None = 2048
    return_vllm_tokens_prompt: bool = False
    truncate: bool = False
    overwrite_observations: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite existing observer data files."},
    )
    distance_measure: str = field(
        default="angular",
        metadata={
            "help": "Distance function to use for clustering.",
            "choices": ["angular", "euclidean", "jsd", "cka", "cosine"],
        },
    )
    output_file_name: str = field(
        default=f"observations_1024_cosine.pt",
        metadata={"help": "Name of the output file for observer data."},
    )
    record_pruning_metrics_only: bool = field(
        default=True,
        metadata={
            "help": (
                "Only record prune-path metrics (routed-token saliency). Default True "
                "because prune never reads merge-criteria tensors. Merge entrypoints "
                "force this to False."
            )
        },
    )
    renormalize_router_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to renormalize topk router weights to sum to 1 if the model.config.norm_topk_prob is True."
            )
        }, 
    )
    observe_backend: str = field(
        default="auto",
        metadata={
            "help": (
                "Observation backend: auto|loop|bmm|frea|f2. "
                "auto picks f2 on CUDA+Triton else bmm (routed-only GPU). "
                "loop is the legacy full (E,T,H) parity oracle."
            ),
            "choices": ["auto", "loop", "bmm", "frea", "f2"],
        },
    )

@dataclass
class ClusterArgs:
    cluster_description: str | None = field(
        default=None,
        metadata={
            "help": (
                "Description of the clustering run, used for naming output dir. "
                "If None, use: "
                "f'{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters}_"
                "{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}_"
                "softmax-{cluster_args.softmax_temperature}'"
            )
        },
    )
    expert_sim: str = field(
        default="ttm",
        metadata={
            "help": "Expert similiarty method.",
            "choices": [
                "ttm",
                "dynamic_ttm",
                "characteristic_activation",
                "routed_characteristic_activation",
                "router_logits",
                "online_characteristic_activation_dist"
            ],
        },
    )
    compression_ratio: float | None = field(
        default=0.5,
        metadata={
            "help": (
                "Compression ratio for clustering experts. If None, num_clusters must "
                "be set."
            )
        },
    )
    num_clusters: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of clusters to place experts into per layer. If None, "
                "num_clusters is calculated as int(num_experts * compression_ratio)."
            )
        },
    )
    cluster_method: str = field(
        default="agglomerative",
        metadata={
            "help": "Clustering method to use.",
            "choices": ["agglomerative", "kmeans", "spectral", "mc_smoe"],
        },
    )
    linkage_method: str = field(
        default="average",
        metadata={
            "help": "Linkage method for agglomerative clustering.",
            "choices": ["ward", "complete", "average", "single"],
        },
    )
    frequency_penalty: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to apply frequency penalty to expert similarity scores. "
                "If True, the frequency of each expert is used to scale the similarity"
            )
        },
    )
    softmax_temperature: float | None = field(
        default=None,
        metadata={
            "help": (
                "Temperature for softmax scaling of expert probabilities to calculate "
                "distance penalty vector. If 0 or None, expert probabilites are max "
                "normalized."
            )
        },
    )
    multi_layer: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of layers to merge at once. If None, merges all layers "
                "separately."
            )
        },
    )
    max_cluster_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "If not None, maximum number of experts per cluster. Only agglomerative"
                " cluster method supported"
            )
        }
    )
    singleton_super_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep super experts in a singleton when clustering"
            )
        }
    )
    singleton_outlier_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep outlier experts in a singleton when clustering"
            )
        }
    )



@dataclass
class MergeArgs:
    overwrite_merged_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite existing merged model files."
        },
    )
    merged_model_dir_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name of the merged model. If None, uses a concatenation of releveant hyperparameters:"
                "'merge_args.merge_method-merge_args.dom_as_base-merge_args.select_top_k-permute_merge_args.permute'"
            )
        },
    )
    merge_method: str = field(
        default="frequency_weighted_average",
        metadata={
            "help": "Method to use for merging experts.",
            "choices": [
                "frequency_weighted_average",
                "average",  # alias of frequency_weighted_average with uniform weights
                "ties",
                "multislerp",
                "sce",
                "karcher",
                "submoe"
            ],
        },
    )
    skip_first: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to skip the first layer when merging experts. "
            )
        }
    )
    skip_last: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to skip the last layer when merging experts. "
            )
        }
    )
    dom_as_base: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the most frequent expert as the base model for "
                "multislerp."
            )
        },
    )
    select_top_k: float = field(
        default=0.1,
        metadata={
            "help": (
                "Top-k percentage of weights to keep in non-dom experts for TIES."
            )
        }
    )
    permute: str | None = field(
        default=None,
        metadata={
            "help": (
                "Permutation to apply prior to merge"
            ),
            "choices": [
                None,
                "direct",
                "wm",
            ]
        },
    )
    save_as_tied_params: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the merged model as tied parameters. "
                "If False, saves merged experts as copies."
            )
        },
    )

@dataclass
class EvalArgs:
    greedy: bool = field(
        default=True,
        metadata={
            "help": "Whether to use greedy decoding for evaluation. If False, uses sampling."
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={
            "help": "Temperature for sampling during evaluation. Ignored if greedy=True."
        },
    )
    top_p: float = field(
        default=0.8,
        metadata={"help": "Top-p value for nucleus sampling during evaluation. Ignored if greedy=True."},
    )
    top_k: int = field(
        default=20,
        metadata={
            "help": "Top-k value for top-k sampling during evaluation. Ignored if greedy=True."
        },
    )
    min_p: float = field(
        default=0.00,
        metadata={
            "help": "Minimum probability for sampling during evaluation. Ignored if greedy=True."
        },
    )
    results_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Directory to save evaluation results. If None, results are saved "
                "in artifacts/model_name directory."
            )
        },
    )
    run_lm_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation on the merged model."},
    )
    run_evalplus: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation using evalplus."},
    )
    run_livecodebench: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation using livecodebench."},
    )
    run_wildbench: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation using wildbench."},
    )
    run_math: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation using math tasks."},
    )

    lm_eval_tasks: list[str] = field(
        default_factory=lambda: [
            "winogrande",
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "mmlu",
            "openbookqa",
            "rte",
        ],
        metadata={
            "help": "List of tasks to evaluate on using lm-eval.",
        },
    )
    evalplus_tasks: list[str] = field(
        default_factory=lambda: [
            "mbpp",
            "humaneval",
        ],
        metadata={
            "help": "List of tasks to evaluate on using evalplus.",
        },
    )
    parallel_tasks: int = field(
        default=32,
        metadata={
            "help": "Number of parallel tasks to run during evalplus evaluation."
        },
    )

@dataclass
class PruneArgs:
    overwrite_pruned_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite existing pruned model files."
        },
    )
    prune_method: str = field(
        default="frequency",
        metadata={
            "help": "Method to use for pruning experts.",
            "choices": [
                "frequency",
                "ean_ca",
                "ean_sum",
                'ean_mean',
                "weighted_frequency_sum",
                "weighted_ean_sum",
                "weighted_ean_sum_l2",
                "reap",
                "reap_l2",
                "max_activations"
            ]
        },
    )
    n_experts_to_prune: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of experts to keep after pruning. If None, use "
                "--compression-ratio."
            )
        },
    )
    perserve_super_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to preserve super experts when pruning. Excludes last 25%% of layers"
            )
        }
    )
    perserve_outliers: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to preserve outlier experts when pruning, includes all layers"
            )
        }
    )


@dataclass
class LayerwiseArgs:
    """Arguments for layerwise (memory-efficient) calibration."""

    batch_group_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of pre-tokenized calibration batches to process at a time. "
                "If not set, process all the batches generated from the dataset. "
                "If set, the layerwise observer processes one group through all blocks "
                "before moving to the next group, which reduces CPU RAM usage from "
                "cached first-layer inputs."
            )
        },
    )
    save_intermediate: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save intermediate results after each block during layerwise "
                "calibration. Useful for debugging and recovery."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": ("Use memory-efficient model loading. Recommended for large models."),
        },
    )



