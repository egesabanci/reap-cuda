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
├── kernels   # CUDA / Triton / auto-backend status (no model load)
└── version
```

Global:

| Flag | Description |
| --- | --- |
| `-v` / `--verbose` | DEBUG logging |
| `-h` / `--help` | Help |

## `reap version` / `reap kernels`

```bash
reap version    # package version (e.g. 0.1.0)
reap kernels    # torch.cuda, Triton package/runtime, resolved auto backend
```

`kernels` loads no model — useful for EC2 bring-up before a long prune.

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
| `--model-revision` / `--local-files-only` | unset / off | Model |
| `--trust-remote-code` | off | Model |
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
| `--shuffle` / `--no-shuffle` | shuffle on | Data |
| `--trust-observation-artifact` | off | Security |
| `--artifacts-dir` | `./artifacts` or env | Run |
| `--observe-only` | off | Run |
| `--smoke-test` / `--no-smoke-test` | smoke on | Run |
| `--eval` / `--no-eval` | no eval | Run |
| `--eval-tasks`, `--eval-backend`, `--eval-num-fewshot` | default tasks / `hf` / `0` | Evaluation |
| `--eval-batch-size`, `--eval-limit`, `--eval-baseline`, `--eval-data-path` | `1` / unset / off / unset | Evaluation |
| `--profile` / `--no-profile` | profile on | Run |
| `--seed` | `42` | Run |
| `--residency` | `auto` | Residency |

### `--frea-backend` (all prune/merge subcommands)

Controls the **FREA expert-MLP** implementation when the observe backend uses
FREA (`auto`→`f2`, `frea`, or `f2`). Orthogonal to `--observe-backend`.

| Value | Behavior |
| --- | --- |
| `auto` | Time Triton vs cuBLAS once per shape; keep the winner (default; picks **pytorch** on L4 for large MoE shapes) |
| `triton` | Force Triton when tiles fit (L4: often 128×64 with SM opt-in, not 128×128) |
| `pytorch` | Force grouped `F.linear` (usually fastest on L4/T4) |

```bash
reap prune full --frea-backend auto      # recommended default
reap prune full --frea-backend pytorch   # explicit L4 throughput
reap prune full --frea-backend triton    # experiment / big-SM GPUs
```

Full story + L4 SM erratum: [frea-throughput.md](frea-throughput.md).

### `--dataset` / `--dataset-path` / `--artifacts-dir`

| Flag | Meaning |
| --- | --- |
| `--dataset` / `-d` | Hub id, composite `name:N_batches,...` (optional `@path`), or `combined` |
| `--dataset-path PATH` | Offline arrow/json/dir. **Processor is still `--dataset`** (must match columns). Composite: `@path` per component or `{path}/<short_name>` |
| `--artifacts-dir PATH` | Root for pruned/merged models and observations (else `REAP_ARTIFACTS_DIR` / `./artifacts`) |

Composite trailing `N` is a **batch count**, not sample count.

```bash
reap prune full \
  -d theblackcat102/evol-codealpaca-v1 \
  --dataset-path /data/datasets/evol-codealpaca-calib-200 \
  --artifacts-dir /data/reap-artifacts

# Composite offline
reap prune full \
  -d "theblackcat102/evol-codealpaca-v1:64@/data/evol,open-r1/Mixture-of-Thoughts[code]:32@/data/mot"
```

Full data rules: [calibration.md](calibration.md).

### `--residency` (all prune/merge subcommands)

Controls **where model weights live** during load and save (host RAM vs GPU vs
disk offload). Orthogonal to full vs layerwise observe schedule.

| Value | Behavior |
| --- | --- |
| `auto` | Pick from host/GPU memory + model-size estimate |
| `gpu_full` | `device_map="auto"`; stream-save; no full CPU pin |
| `layerwise` | Block observe; auto + disk offload; mutate/save reuses the offloaded model (no `gpu_full` reload) |
| `cpu_full` | Pin full model on CPU (needs ample host RAM) |

```bash
# Small-RAM GPU instance (model fits VRAM)
reap prune full --residency gpu_full -m LiquidAI/LFM2.5-8B-A1B ...

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
