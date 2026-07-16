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
| `--pruning-metrics-only` / `--all-metrics` | pruning-only | Observer |
| `--renorm-router` / `--no-renorm-router` | renorm on | Observer |
| `--overwrite-observations` / `--keep-observations` | keep | Observer |
| `--observe-only` | off | Run |
| `--smoke-test` / `--no-smoke-test` | smoke on | Run |
| `--eval` / `--no-eval` | no eval | Run |
| `--profile` / `--no-profile` | profile on | Run |
| `--seed` | `42` | Run |

## `reap prune layerwise`

Same goals; block-wise calibration. Defaults lean smaller (`batch-size=4`, no
smoke/profile).

Extra flags:

| Option | Default |
| --- | --- |
| `--batch-group-size` | unset (all batches) |
| `--save-intermediate` | off |
| `--low-cpu-mem` / `--no-low-cpu-mem` | low mem on |

```bash
reap prune layerwise \
  -m Qwen/Qwen3-30B-A3B \
  -d theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend bmm \
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

- [pipeline.md](pipeline.md)
- [pruning.md](pruning.md)
- [merging.md](merging.md)
- [development.md](development.md)
