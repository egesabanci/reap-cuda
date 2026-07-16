# Setup and first run

End-to-end guide: install REAP CUDA, verify the environment, enable Triton
kernels (optional), and run prune/merge. For design depth, follow the links at
the end.

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python **3.12+** | `requires-python = ">=3.12"` |
| `uv` (recommended) | Or plain `pip` |
| **Dev (Mac / CPU)** | Unit tests + CLI help; no NVIDIA kernels |
| **EC2 / NVIDIA GPU** | Real MoE runs; Triton optional via `[cuda]` extra |
| Disk / network | HuggingFace model + calibration dataset on first run |

Versions pinned in `pyproject.toml`: `torch>=2.10`, `transformers>=5.5`.

## 2. Install (any machine)

```bash
git clone https://github.com/egesabanci/reap-cuda.git
cd reap-cuda
uv venv .venv --seed --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate

uv pip install --editable .
uv pip install pytest
```

Check the package and CLI:

```bash
uv run python -c "import reap; print('ok')"
uv run reap --help
uv run reap version
uv run pytest tests/ -q
```

## 3. Install for CUDA + Triton kernels (EC2 / NVIDIA)

Triton is an **optional extra**. Without it, REAP still runs using pure-PyTorch
backends (`bmm`). With it, `auto` prefers the `f2` path that can launch custom
kernels.

```bash
# On a machine with NVIDIA drivers + CUDA-capable PyTorch already matching the GPU
uv pip install -e '.[cuda]'     # pulls triton>=2.3
uv pip install -e '.[eval]'     # optional: lm-eval
```

Verify:

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
uv run python -c "import triton; print('triton', triton.__version__)"
uv run reap kernels
```

Expected on a healthy EC2 GPU box:

```text
torch.cuda.is_available: True
triton package: True
triton runtime: True
auto backend: f2
```

On a Mac without CUDA you will see:

```text
torch.cuda.is_available: False
triton package: False   # or True if installed but still no runtime
triton runtime: False
auto backend: bmm
```

That is **normal**: custom `@triton.jit` kernels only run when **both** the
`triton` package and a CUDA device are available.

### Force pure PyTorch (disable Triton)

```bash
export REAP_DISABLE_TRITON=1
# or always:
reap prune layerwise --observe-backend bmm ...
```

## 4. Mental model of the codebase

```text
CLI (Typer)  ──►  run() pipelines  ──►  observe  ──►  prune | merge  ──►  save
                      │                    │
                      │                    ├─ adapters (layout)
                      │                    └─ kernels (bmm / Triton F5·FREA·F2)
                      └─ data / metrics / cluster
```

| Layer | Package / docs |
| --- | --- |
| Commands | `src/reap/cli/` · [cli.md](cli.md) |
| Orchestration | `prune.py`, `layerwise_*.py`, `merge_*.py` · [pipeline.md](pipeline.md) |
| Weight residency | `residency.py` · [residency.md](residency.md) |
| Model layouts | `model_adapters.py` · [model-adapters.md](model-adapters.md) |
| Observation | `observer.py`, `kernels/observe.py` · [observation-and-metrics.md](observation-and-metrics.md) |
| GPU / kernels | `kernels/` · [gpu-and-backends.md](gpu-and-backends.md) · [kernels/README.md](kernels/README.md) |
| Full architecture | [architecture.md](architecture.md) |

## 5. How kernels plug in (usage)

You do **not** call Triton APIs yourself. Choose an **observe backend**; the
library dispatches:

| `--observe-backend` | Behavior |
| --- | --- |
| `auto` | `f2` if Triton runtime OK, else `bmm` |
| `bmm` | Pure PyTorch grouped routed matmuls (parity / safest bring-up) |
| `frea` | Try Triton FREA SwiGLU; fallback to bmm |
| `f2` | Try Triton FREA + F2 reduce; fallback to PyTorch |
| `loop` | Legacy path (parity oracle) |

```bash
# Recommended first real run on a single L40S-class GPU
uv run reap prune layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend bmm \
  --batches-per-category 8 \
  --batch-size 1 \
  --observe-only

# After kernels verified
uv run reap prune layerwise \
  --model Qwen/Qwen3-30B-A3B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend auto
```

### Where the Triton source lives

| File | Role |
| --- | --- |
| `kernels/triton_utils.py` | Detect package + CUDA; `REAP_DISABLE_TRITON` |
| `kernels/triton_softmax.py` | F5 row-softmax (`@triton.jit`) |
| `kernels/triton_frea.py` | FREA SwiGLU (`@triton.jit`) |
| `kernels/triton_reduce.py` | F2 scatter reduce (`@triton.jit`) |
| `kernels/bmm.py` | PyTorch fallback (always available) |

Imports of `triton` are **lazy** (inside functions / try-except) so Mac/CPU
installs do not crash. See [gpu-and-backends.md](gpu-and-backends.md).

### Gates before a Triton path runs

1. `triton` import succeeds  
2. `torch.cuda.is_available()`  
3. Tensors on CUDA, dtype in `{fp16, bf16, fp32}`  
4. FREA: SiLU activation and `H, I ≥ 16` (tiny test models stay on PyTorch)  
5. Any launch error → automatic PyTorch fallback  

## 6. Common workflows

### Weight residency (read this on small-RAM hosts)

| Flag | When |
| --- | --- |
| `--residency auto` | Default; picks GPU vs offload vs CPU from memory + model size |
| `--residency gpu_full` | Model fits VRAM; **host RAM is tight** (e.g. g6.xlarge 16 GiB + L4) |
| `--residency layerwise` | Force block-wise observe + disk offload |
| `--residency cpu_full` | Explicit full CPU pin (needs ample RAM — often OOMs at 16 GiB) |

```bash
# g6.xlarge-class: keep weights on GPU, stream-save (do not pin CPU)
uv run reap prune full \
  -m LiquidAI/LFM2-8B-A1B \
  -d theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --residency gpu_full \
  --observe-backend bmm \
  --batches-per-category 8 \
  --batch-size 1
```

Full policy: [residency.md](residency.md).

### Prune (layerwise — single GPU)

```bash
uv run reap prune layerwise \
  -m Qwen/Qwen3-30B-A3B \
  -d theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --observe-backend auto \
  --residency auto
```

### Prune (full model — multi-GPU / large VRAM)

```bash
uv run reap prune full \
  -m Qwen/Qwen3-30B-A3B \
  -d theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --residency auto
```

### Merge

```bash
uv run reap merge layerwise \
  -m Qwen/Qwen3-30B-A3B \
  -d theblackcat102/evol-codealpaca-v1 \
  --expert-sim characteristic_activation \
  --compression-ratio 0.5 \
  --residency auto
```

Artifacts: `artifacts/<model>/<dataset>/…`  
Details: [pipeline.md](pipeline.md), [cli.md](cli.md), [residency.md](residency.md).

## 7. Tests related to kernels and residency

```bash
# Always (CPU): fallbacks + dispatch
uv run pytest tests/test_triton_kernels.py tests/test_kernel_parity_bmm.py -q

# On CUDA host: also runs @requires_triton cases
uv run pytest tests/test_triton_kernels.py -q

# Weight residency heuristics + CLI wiring (hermetic)
uv run pytest tests/test_residency.py tests/test_cli.py -q
```

## 8. Doc map (everything else)

| Want… | Read |
| --- | --- |
| Module map & invariants | [architecture.md](architecture.md) |
| Phase-by-phase prune/merge | [pipeline.md](pipeline.md) |
| **Weight residency / low-RAM hosts** | **[residency.md](residency.md)** |
| Backends + activation device policy | [gpu-and-backends.md](gpu-and-backends.md) |
| Kernel **design** (SoC phases) | [kernels/README.md](kernels/README.md) |
| Metrics / saliency keys | [observation-and-metrics.md](observation-and-metrics.md) |
| Adapters / new models | [model-adapters.md](model-adapters.md) |
| Layerwise memory mode | [layerwise.md](layerwise.md) |
| Full CLI flags | [cli.md](cli.md) |
| Extending the code | [development.md](development.md) |
| Hub index | [index.md](index.md) |
