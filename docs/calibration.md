# Calibration

Calibration supplies token sequences for MoE observation. REAP does not train;
it only runs forward passes and accumulates routing statistics.

Implementation: `src/reap/data.py` (`load_category_batches`,
`load_composite_category_batches`, `_load_local_dataset`), wired through
`pipeline.record_activations` and layerwise prepare helpers.

## Inputs

| Source | CLI / args | Behavior |
| --- | --- | --- |
| Single HF dataset | `--dataset name` | Hub load; tokenize via **registered** processor |
| Local offline | `--dataset name --dataset-path PATH` | Disk load; `--dataset` is the **processor id** (must match columns) |
| Composite | `--dataset "a:64,b[sub]:32"` | Trailing `N` is a **batch count** (not sample count) |
| Composite offline | `@path` and/or `--dataset-path` | Per-component path or `{root}/<short_name>` |
| Cached observations | `--dataset combined` | Skip data load; require existing `.pt` |

Also:

| Flag / env | Role |
| --- | --- |
| `--split` | Dataset split (default `train`). Single `.arrow` files have **no** split metadata — non-`train` warns and loads all rows |
| `--dataset-config` | HF config/subset name |
| `--batch-size` | Sequences per batch |
| `--batches-per-category` | Batch count for single-dataset mode (overridden by composite `N`) |
| `--model-max-length` | Max tokens per sequence |
| `--truncate` | Hard truncation in processors |
| `--artifacts-dir` / `REAP_ARTIFACTS_DIR` | Output root (not calibration load) |
| `HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` | Block hub fetches; use `--dataset-path` |

## Offline / local load (`--dataset-path`)

```bash
reap prune full \
  -d theblackcat102/evol-codealpaca-v1 \
  --dataset-path /data/datasets/evol-codealpaca-calib-200 \
  --artifacts-dir /data/reap-artifacts
```

Rules:

1. **`--dataset` selects the field-mapping processor** (must be in
   `DATASET_REGISTRY`). It is not guessed from the path.
2. **Supported local layouts:** `.arrow` file, `.parquet`, `.json`/`.jsonl`,
   directory of shards, or HuggingFace `save_to_disk` (Dataset or DatasetDict).
3. **Column validation:** known processors check required columns before
   tokenization (e.g. evol-codealpaca needs `instruction` + `output`). Mismatch
   → clear `ValueError` naming present vs required columns.
4. **No silent fallback** to evol-codealpaca for unknown names.
5. Hub failures (and offline env without a path) **suggest `--dataset-path`**.

### `allenai/c4`

Hub load uses a **single remote train shard** (not full C4) and warns. Under
`HF_*_OFFLINE`, it errors with a path hint instead of hanging on the network.

## Composite dataset spec

Format (comma-separated components):

```txt
<dataset_name>[subset](split):<num_batches>[@local_path]
```

| Piece | Meaning |
| --- | --- |
| `dataset_name` | Registry / hub id (processor + default hub source) |
| `[subset]` | Optional HF config name |
| `(split)` | Optional split (default from `--split`) |
| `:N` | **`N` calibration batches** (not raw examples) |
| `@path` | Optional offline path for this component only |

Examples:

```bash
# Two sources, 64 batches each (batch count × batch_size ≈ sequences, not "64 samples")
reap prune layerwise \
  --dataset "theblackcat102/evol-codealpaca-v1:64,open-r1/Mixture-of-Thoughts[code]:64"

# Per-component offline paths
reap prune full \
  --dataset "theblackcat102/evol-codealpaca-v1:64@/data/evol,open-r1/Mixture-of-Thoughts[code]:32@/data/mot"

# Shared root with subdirs {root}/evol-codealpaca-v1 and {root}/Mixture-of-Thoughts
reap prune full \
  --dataset "theblackcat102/evol-codealpaca-v1:64,open-r1/Mixture-of-Thoughts:32" \
  --dataset-path /data/datasets
```

Resolution order for offline composite (`load_composite_category_batches`):

1. Per-component `@path`
2. `{--dataset-path}/<short_name>` if that exists
3. `--dataset-path` itself (single-component or loadable dir/file)

**Errors clearly** when multi-component + global path is a single **file** without
`@` paths. Full and layerwise pipelines both use this helper (no silent
`dataset_path=None`).

## Processors

Registered in `DATASET_REGISTRY` (`src/reap/data.py`). Each processor:

1. Maps raw rows to chat/LM fields
2. Tokenizes with the model tokenizer
3. Returns batches with at least `input_ids` (usually `attention_mask`)

Supported registry keys (processor id = usual hub id):

| Registry key | Role (summary) |
| --- | --- |
| `theblackcat102/evol-codealpaca-v1` | Instruction / output code chat (common EC2 default) |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | Evol-instruct chat |
| `m-a-p/CodeFeedback-Filtered-Instruction` | Code feedback chat |
| `open-r1/Mixture-of-Thoughts` | Mixture-of-Thoughts (often with `[code]` subset) |
| `allenai/tulu-3-sft-mixture` | Tulu SFT chat mixture |
| `allenai/tulu-3-sft-personas-math` | Personas math |
| `allenai/c4` | LM text (hub uses **one** remote train shard; prefer local) |
| `cais/mmlu` | MMLU-style chat |
| `euclaise/WritingPrompts_curated` | Writing prompts |
| `Salesforce/xlam-function-calling-60k` | Function-calling |
| `SWE-bench/SWE-smith-trajectories` | SWE-smith trajectories |

Unsupported names raise `ValueError` listing supported keys. To add a dataset:

1. Implement a processor subclass
2. Register it in `DATASET_REGISTRY`
3. Optionally add required columns to `_PROCESSOR_REQUIRED_COLUMNS` for offline checks
4. Document here if user-facing

## Batching and padding

- Batches may include padding; observers use `attention_mask` so **padding does
  not contribute** to saliency.
- Prefer realistic `--model-max-length` for meaningful routing.

## How many batches?

| Goal | Suggestion |
| --- | --- |
| CLI smoke / wiring | 4–16 batches, batch size 1–2 |
| Paper-like prune | hundreds–1024 batches (default `batches_per_category=1024`) |
| Debug metrics | small N with `--overwrite-observations` |

Composite `:N` **overrides** `--batches-per-category` for that component.

Rough token count ≈ `N × batch_size × average_seq_len` (not `N` samples).

## Caching

- Full path: per-category under `artifacts/.../<category>/<output_file_name>`
  (or `--artifacts-dir` root)
- Layerwise: `.../layerwise/<output_file_name>`
- Skip recompute unless `--overwrite-observations`
- `combined` requires a pre-built cache at the expected path

## Tests

```bash
uv run pytest tests/test_dataset_loading.py -q
```

Covers composite `@path`, column validation, arrow split warnings, offline C4/hub
guards, hub error hints, and pipeline `dataset_path` threading.

## Related

- [cli.md](cli.md)
- [pipeline.md](pipeline.md)
- [observation-and-metrics.md](observation-and-metrics.md)
- [development.md](development.md)
