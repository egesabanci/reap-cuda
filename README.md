<div align="center">

# REAP CUDA

**Router-weighted Expert Activation Pruning for MoE models on CUDA / PyTorch.**

[Quick Start](#quick-start) ·
[Workflow](#workflow) ·
[Supported Models](#supported-models) ·
[CLI](#cli-reference) ·
[Metrics](#pruning-methods) ·
[Docs](#technical-docs) ·
[Development](#development) ·
[License](#license)

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Runtime](https://img.shields.io/badge/Runtime-PyTorch%20%7C%20CUDA-black)
![Package](https://img.shields.io/badge/Package-reap-green)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

</div>

REAP CUDA compresses HuggingFace Mixture-of-Experts LLMs by **pruning** or
**merging** routed experts. A short calibration pass records router-weighted
activation statistics; low-saliency experts are removed (or clustered and fused)
and the mutated model is saved as a standard transformers checkpoint.

Use it for one-shot MoE compression on NVIDIA GPUs — including **single-GPU
layerwise** calibration for 30B-class models on ~46 GB cards, and
**GPU-resident weights** on small-RAM hosts (e.g. g6.xlarge 16 GiB RAM + L4).

## Highlights

- **Typer CLI** — `reap prune|merge` × `full|layerwise` with rich help panels
- **Adapter-based MoE support** — Qwen3 / Qwen3.5–3.6 / Llama4 / Mixtral / LFM2
- **Weight residency** — `--residency auto|gpu_full|layerwise|cpu_full` avoids
  full-CPU pins when VRAM fits but host RAM is tight; stream-save from GPU
- **FREA profitability** — `--frea-backend auto` probes Triton vs cuBLAS per
  host/shape (L4-safe throughput); force `triton` or `pytorch` when needed
- **GPU-first observation** — saliency stays on the compute device; routed-only
  backends avoid `(E, T, H)` activation materialization
- **Layerwise mode** — one decoder block on GPU at a time for large MoEs
- **Prune + merge** — REAP/EAN/frequency saliency; agglomerative / TIES / …
- **Layout-normalized kernels package** — F4 weight cache, F5 / native router,
  grouped bmm / FREA / F2 (Triton when profitable; always safe fallbacks)
- **Hermetic tests** — tiny in-memory models, mocked CLI dispatch (no Hub)

## Quick Start

**Prerequisites:** Python 3.12+, `uv`, NVIDIA GPU for real runs (CPU/MPS for
control-flow and unit tests).

```bash
git clone https://github.com/egesabanci/reap-cuda.git
cd reap-cuda
uv venv .venv --seed --python 3.12
uv pip install --editable .
uv pip install pytest

# Optional on CUDA hosts
uv pip install -e '.[cuda]'    # triton
uv pip install -e '.[eval]'    # lm-eval
```

Check the CLI and tests:

```bash
uv run reap --help
uv run reap version
uv run pytest tests/ -q
```

### Minimal prune (layerwise — single GPU friendly)

```bash
uv run reap prune layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend bmm \
  --residency auto \
  --batches-per-category 64 \
  --batch-size 1
```

### Full-model prune (multi-GPU / large VRAM)

```bash
uv run reap prune full \
  --model Qwen/Qwen3-30B-A3B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --residency auto
```

### Small-RAM GPU host (e.g. g6.xlarge 16 GiB RAM + L4)

When the model fits VRAM but is large vs host RAM, prefer **GPU-resident**
weights — do **not** pin the full model on CPU:

```bash
uv run reap prune full \
  --model LiquidAI/LFM2-8B-A1B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --residency gpu_full \
  --observe-backend bmm \
  --batches-per-category 8 \
  --batch-size 1
```

`--residency auto` usually picks `gpu_full` in this situation.

### Merge (cluster experts → fuse weights)

```bash
uv run reap merge layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --expert-sim characteristic_activation \
  --cluster-method agglomerative \
  --compression-ratio 0.5
```

Artifacts land under `artifacts/<model>/<dataset>/` (pruned or merged
checkpoints, observer `.pt` files, optional eval outputs).

## Workflow

| Step | What happens | Evidence |
| --- | --- | --- |
| 0. Residency | Resolve `--residency` → load/save plan (GPU / offload / CPU) | log: `Residency resolved: …` |
| 1. Load | HF model + tokenizer per plan (`device_map` auto / offload / cpu) | model class / config |
| 2. Calibrate | Tokenize calibration batches (single or composite dataset) | batch count |
| 3. Observe | Routed expert stats via hooks or block replay | observer `.pt` |
| 4. Decide | Rank experts (prune) or cluster (merge) | saliency / labels |
| 5. Mutate | `slice_experts` or in-place merge | live module shapes |
| 6. Save | Stream `save_pretrained` (hooks stripped; no full CPU dump) | safetensors |
| 7. Validate | Optional smoke generate / lm-eval | logs / `eval/` |

```txt
residency → load → calibrate → observe → prune|merge → stream-save → smoke|eval
```

## Supported Models

| Adapter | Family | Experts layout | Notes |
| --- | --- | --- | --- |
| `Qwen3MoeModelAdapter` | Qwen3-MoE | Fused `gate_up` / `down` (TF ≥5) | Default fused path |
| `Qwen3_5MoeModelAdapter` | Qwen3.5 / 3.6 MoE | Fused + **shared expert** | Shared expert kept |
| `Llama4MoeModelAdapter` | Llama4 Text MoE | Fused **bmm** layout | Router attr `.router` |
| `MixtralMoeModelAdapter` | Mixtral / PhiMoE | Non-fused ModuleList | `num_local_experts` |
| `Lfm2MoeModelAdapter` | LFM2.5 MoE | Fused linear | Slices `expert_bias` |

Requires **`transformers>=5.5.0`** for current fused Qwen stacks. Layout
detection is runtime-based (`infer_model_adapter`). See
[docs/model-adapters.md](docs/model-adapters.md).

## Memory Modes

Two orthogonal controls:

### 1. Observe schedule (`full` vs `layerwise` subcommand)

| Command | Peak VRAM during observe (order of magnitude) | Use when |
| --- | --- | --- |
| `reap prune full` / `reap merge full` | Whole model (~60 GB bf16 for 30B-class) | Multi-GPU / A100-80 / H100 |
| `reap prune layerwise` / `reap merge layerwise` | One block (~1–2 GB + routed transients) | Single L40S 46 GB, 30B+ |

Layerwise still **reloads the full model** for the final prune mutate/save step
(via `gpu_full` plan). Plan that VRAM separately from calibration.

### 2. Weight residency (`--residency`)

Where **parameters** live during load / save — critical on **low host RAM**:

| Mode | Load | Save | Typical host |
| --- | --- | --- | --- |
| `auto` | Heuristic from host/GPU + model size | per resolved mode | Default |
| `gpu_full` | `device_map="auto"` on GPU | Stream from device (no full CPU pin) | g6.xlarge-class 16 GiB RAM + GPU |
| `layerwise` | `auto` + disk offload (not full CPU pin) | Reload `gpu_full` then stream | Large MoE, mid GPU |
| `cpu_full` | `device_map="cpu"` | Normal | Ample host RAM / debug |

```bash
# Prefer GPU weights when model fits VRAM but is large vs RAM
reap prune full --residency gpu_full ...

# Or let auto pick (g6-like: → gpu_full)
reap prune full --residency auto ...
```

Full policy, heuristics, delegation (full↔layerwise), and env knobs:
**[docs/residency.md](docs/residency.md)**.

## Pruning Methods

`--prune-method` ranks experts; **higher scores are kept**.

| Method | Meaning |
| --- | --- |
| `reap` | Router-weighted activation-norm mean (default in Typer CLI) |
| `frequency` | Top-k assignment counts |
| `ean_sum` / `ean_mean` | Sum / mean of routed L2 norms |
| `weighted_ean_sum` | Sum of `norm × router_weight` |
| `weighted_frequency_sum` | Sum of router weights |
| `max_activations` | Max activation element over routed outputs |
| `ean_ca` | Norm of routed characteristic activation (needs full metrics) |

`--compression-ratio` in `[0, 1)` removes `int(E × ratio)` experts per layer
(always keeps ≥1). Or set `--n-experts-to-prune`.

## Observation Backends

| Backend | Role |
| --- | --- |
| `auto` | `f2` if CUDA+Triton available, else `bmm` |
| `bmm` | Grouped routed-only matmuls (recommended first EC2 path) |
| `frea` / `f2` | FREA expert MLP + (for `f2`) F2 scatter reduce |
| `loop` | Legacy / parity oracle |

**FREA sub-policy** (when using `auto`/`frea`/`f2`):

| `--frea-backend` | Role |
| --- | --- |
| `auto` | Probe Triton vs cuBLAS once per shape; keep winner (default; L4 → pytorch) |
| `triton` | Force Triton when tiles fit (L4 max often 128×64, not 128×128) |
| `pytorch` | Force cuBLAS grouped path (usually best throughput on L4/T4) |

```bash
reap prune layerwise --observe-backend bmm ...
reap prune full --observe-backend auto --frea-backend auto ...
reap prune full --frea-backend pytorch   # prefer throughput on small-SM GPUs
```

Saliency tensors stay on GPU until save. Design / ops:
[docs/gpu-and-backends.md](docs/gpu-and-backends.md),
[docs/frea-throughput.md](docs/frea-throughput.md),
[docs/kernels/](docs/kernels/README.md).

## CLI Reference

```bash
uv run reap --help
uv run reap prune --help
uv run reap merge full --help
```

| Command | Memory | Purpose |
| --- | --- | --- |
| `reap prune full` | Full GPU | Observe → prune → save |
| `reap prune layerwise` | Block GPU | Same, layerwise calib |
| `reap merge full` | Full GPU | Observe → cluster → merge |
| `reap merge layerwise` | Block GPU | Same, layerwise calib |
| `reap version` | — | Package version |

Common flags: `-m/--model`, `-d/--dataset`, `--compression-ratio`,
`--observe-backend`, `--frea-backend`, `--residency`, `--dataset-path`,
`--artifacts-dir`, `--observe-only`, `--eval`, `--seed`.

Full flag tables: [docs/cli.md](docs/cli.md).

Legacy scripts (`reap-prune`, `reap-layerwise`, `reap-merge`,
`reap-layerwise-merge`) remain for HfArgumentParser workflows.

## Data

| Mode | Example |
| --- | --- |
| Single (hub) | `--dataset theblackcat102/evol-codealpaca-v1` |
| Offline local | `--dataset theblackcat102/evol-codealpaca-v1 --dataset-path /data/…` |
| Composite | `--dataset "ds_a:64,ds_b[code]:64"` (`:N` = **batch** count) |
| Composite offline | `name:N@/local/path` and/or shared `--dataset-path` root |
| Cached observations | `--dataset combined` (requires prior `.pt`) |

`--dataset` always selects the **field-mapping processor** (columns must match);
`--dataset-path` only chooses the files. Offline env vars
`HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` need a local path or they fail with a
hint. Full rules: [docs/calibration.md](docs/calibration.md).

## Technical Docs

Maintainer documentation (SoC, one concern per file):

| Start here | |
| --- | --- |
| **[docs/setup.md](docs/setup.md)** | Install, CUDA/Triton, first run, how kernels attach |
| **[docs/index.md](docs/index.md)** | Full documentation map |
| **[docs/architecture.md](docs/architecture.md)** | Codebase structure and data flow |

| Doc | Topic |
| --- | --- |
| [Setup](docs/setup.md) | Install + kernels usage |
| [Architecture](docs/architecture.md) | Modules, data flow, invariants |
| [Pipeline](docs/pipeline.md) | Phase-by-phase prune/merge |
| [Model adapters](docs/model-adapters.md) | Families, slice contract |
| [Observation & metrics](docs/observation-and-metrics.md) | Saliency state |
| [GPU & backends](docs/gpu-and-backends.md) | Device policy, F4/F5/FREA/Triton |
| [**FREA throughput**](docs/frea-throughput.md) | `--frea-backend`, probe, tiles, L4 tradeoff |
| [**Weight residency**](docs/residency.md) | `--residency`, auto heuristics, stream save, g6.xlarge |
| [Pruning](docs/pruning.md) | Ranking and save |
| [Merging](docs/merging.md) | Cluster + fuse |
| [Layerwise](docs/layerwise.md) | Block replay memory mode |
| [CLI](docs/cli.md) | Command reference |
| [Evaluation](docs/evaluation.md) | Smoke + lm-eval |
| [Development](docs/development.md) | Tests and extension |
| [Kernels design](docs/kernels/README.md) | Kernel phase design docs |

## Repository Layout

```txt
reap-cuda/
  README.md
  LICENSE
  pyproject.toml
  docs/
    index.md
    *.md
    kernels/           # kernel design (SoC phases)
  src/reap/
    cli/               # Typer app
    kernels/           # observe backends
    residency.py       # weight load/save policy
    model_adapters.py
    observer.py
    layerwise_*.py
    prune.py
    merge*.py
    ...
  tests/
  scripts/
  data/                # small fixtures (e.g. smoke jsonl)
```

## Development

```bash
uv pip install --editable . pytest
uv run pytest tests/ -q
uv run reap --help
git diff --check
```

Conventional Commits are used (`feat:`, `fix:`, `test:`, `docs:`, …).

## Related

- [reap-mlx](https://github.com/egesabanci/reap-mlx) — Apple Silicon / MLX port
- Paper: [REAP the Experts (arXiv 2510.13999)](https://arxiv.org/abs/2510.13999)
- Upstream inspiration: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)

## License

Apache License 2.0. See [LICENSE](LICENSE).
