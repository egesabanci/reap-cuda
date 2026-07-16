# CLI Reference

The preferred interface is the **Typer** app exposed as `reap`.

```bash
uv run reap --help
uv run python -m reap.cli --help
```

Implementation: `src/reap/cli/` (`app.py`, `prune_cmd.py`, `merge_cmd.py`,
`options.py`).

## Command tree

```txt
reap
├── prune
│   ├── full
│   └── layerwise
├── merge
│   ├── full
│   └── layerwise
└── version
```

Global:

| Flag | Description |
| --- | --- |
| `-v` / `--verbose` | DEBUG logging |
| `-h` / `--help` | Help |

## `reap prune full`

Whole-model GPU observe → prune → save.

```bash
reap prune full \
  --model Qwen/Qwen3-30B-A3B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend auto
```

### Important options

| Option | Default | Panel |
| --- | --- | --- |
| `--model` / `-m` | `Qwen/Qwen3-30B-A3B` | Model |
| `--dataset` / `-d` | evol-codealpaca | Data |
| `--dataset-config` | unset | Data |
| `--split` | `train` | Data |
| `--batch-size` | `8` | Data |
| `--batches-per-category` | `1024` | Data |
| `--model-max-length` | `2048` | Data |
| `--prune-method` | `reap` | Compression |
| `--compression-ratio` | `0.5` | Compression |
| `--n-experts-to-prune` | unset | Compression |
| `--overwrite-pruned` / `--keep-pruned` | keep | Compression |
| `--preserve-super-experts` | off | Compression |
| `--preserve-outliers` | off | Compression |
| `--observe-backend` | `auto` | Observer |
| `--frea-backend` | `auto` | Observer |
| `--pruning-metrics-only` / `--all-metrics` | pruning-only | Observer |
| `--renorm-router` / `--no-renorm-router` | renorm on | Observer |
| `--overwrite-observations` / `--keep-observations` | keep | Observer |
| `--dataset-path` | unset | Data |
| `--artifacts-dir` | `./artifacts` or env | Run |
| `--observe-only` | off | Run |
| `--smoke-test` / `--no-smoke-test` | smoke on | Run |
| `--eval` / `--no-eval` | no eval | Run |
| `--profile` / `--no-profile` | profile on | Run |
| `--seed` | `42` | Run |
| `--residency` | `auto` | Residency |

### `--frea-backend` (all prune/merge subcommands)

Controls the **FREA expert-MLP** implementation when the observe backend uses
FREA (`auto`→`f2`, `frea`, or `f2`). Orthogonal to `--observe-backend`.

| Value | Behavior |
| --- | --- |
| `auto` | Time Triton vs cuBLAS once per shape; keep the winner (default) |
| `triton` | Force Triton when tiles fit shared mem |
| `pytorch` | Force grouped `F.linear` (often fastest on L4/T4) |

```bash
reap prune full --frea-backend pytorch   # throughput on small-SM GPUs
reap prune full --frea-backend triton    # force kernel / memory-lean path
```

Full story: [frea-throughput.md](frea-throughput.md).

### `--dataset-path` / `--artifacts-dir`

| Flag | Meaning |
| --- | --- |
| `--dataset-path PATH` | Local arrow/json/jsonl/dir; offline load. `--dataset` still selects the field-mapping processor. |
| `--artifacts-dir PATH` | Root for pruned/merged models and observations (else `REAP_ARTIFACTS_DIR` / `./artifacts`). |

```bash
reap prune full \
  -d theblackcat102/evol-codealpaca-v1 \
  --dataset-path /data/datasets/evol-codealpaca-calib-200 \
  --artifacts-dir /data/reap-artifacts
```

### `--residency` (all prune/merge subcommands)

Controls **where model weights live** during load and save (host RAM vs GPU vs
disk offload). Orthogonal to full vs layerwise observe schedule.

| Value | Behavior |
| --- | --- |
| `auto` | Pick from host/GPU memory + model-size estimate |
| `gpu_full` | `device_map="auto"`; stream-save; no full CPU pin |
| `layerwise` | Block observe; auto + disk offload; mutate/save via `gpu_full` plan |
| `cpu_full` | Pin full model on CPU (needs ample host RAM) |

```bash
# Small-RAM GPU instance (model fits VRAM)
reap prune full --residency gpu_full -m LiquidAI/LFM2-8B-A1B ...

# Explicit layerwise weights policy on layerwise CLI
reap prune layerwise --residency auto -m Qwen/Qwen3-30B-A3B ...
```

Full policy, heuristics, and full↔layerwise **delegation**: [residency.md](residency.md).

## `reap prune layerwise`

Same goals; block-wise calibration. Defaults lean smaller (`batch-size=4`, no
smoke/profile).

Extra flags:

| Option | Default |
| --- | --- |
| `--batch-group-size` | unset (all batches) |
| `--save-intermediate` | off |
| `--low-cpu-mem` / `--no-low-cpu-mem` | low mem on |
| `--residency` | `auto` |
| `--frea-backend` | `auto` |
| `--dataset-path` | unset |
| `--artifacts-dir` | env / `./artifacts` |

With `--residency auto`, large models stay on the layerwise path; if the model
fits VRAM and host RAM is tight, residency may resolve to `gpu_full` and
**delegate** to the full prune pipeline (avoids full-CPU pin). See
[residency.md](residency.md).

```bash
reap prune layerwise \
  -m Qwen/Qwen3-30B-A3B \
  -d theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend auto \
  --frea-backend auto \
  --residency auto \
  --batches-per-category 64 \
  --batch-size 1
```

## `reap merge full` / `reap merge layerwise`

Force merge-criteria observation. Core options:

| Option | Default |
| --- | --- |
| `--expert-sim` | `characteristic_activation` |
| `--cluster-method` | `agglomerative` |
| `--linkage` | `average` |
| `--merge-method` | `frequency_weighted_average` |
| `--distance` | `angular` |
| `--compression-ratio` | `0.5` |
| `--skip-first` / `--skip-last` | off |
| `--frequency-penalty` | on |
| `--permute` | unset (`direct` \| `wm`) |
| `--overwrite-merged` | keep |
| `--residency` | `auto` |
| `--frea-backend` | `auto` |
| `--dataset-path` | unset |
| `--artifacts-dir` | env / `./artifacts` |

Layerwise merge adds the same layerwise flags as layerwise prune.

```bash
reap merge layerwise \
  -m Qwen/Qwen3-30B-A3B \
  -d theblackcat102/evol-codealpaca-v1 \
  --expert-sim characteristic_activation \
  --compression-ratio 0.5
```

## Observe-only

```bash
reap prune layerwise --observe-only --overwrite-observations ...
```

Writes observer `.pt` without pruning. Useful for metric inspection and reusing
calibration across prune settings.

## Legacy console scripts

Still installed for compatibility (HfArgumentParser, underscore-style flags):

| Script | Maps roughly to |
| --- | --- |
| `reap-prune` | `reap prune full` |
| `reap-layerwise` | `reap prune layerwise` |
| `reap-merge` | `reap merge full` |
| `reap-layerwise-merge` | `reap merge layerwise` |

Prefer Typer for new docs and scripts.

## Exit behavior

- Typer validation errors → non-zero exit, usage printed
- Pipeline `RuntimeError` / OOM → non-zero; partial observer `partial.pkl` may
  exist on full-path observe failures
- Unit tests mock `run()`; see `tests/test_cli.py`

## Related

- [residency.md](residency.md)
- [frea-throughput.md](frea-throughput.md)
- [gpu-and-backends.md](gpu-and-backends.md)
- [pipeline.md](pipeline.md)
- [pruning.md](pruning.md)
- [merging.md](merging.md)
- [development.md](development.md)
