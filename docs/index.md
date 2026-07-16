# REAP CUDA — Technical Documentation

Maintainer-focused reference for **reap-cuda**: Router-weighted Expert Activation
Pruning (and expert merging) for HuggingFace MoE models on **CUDA / PyTorch**.

This tree follows **Separation of Concerns (SoC)**: one topic per document.
Cross-links replace duplication. Design detail for observation kernels lives
under [`kernels/`](kernels/README.md).

## Runtime shape

```txt
CLI (Typer)
  -> load model + tokenizer
  -> calibrate (tokenized batches)
  -> observe (per-layer routed expert stats, GPU-resident)
  -> prune | merge
  -> save checkpoint
  -> optional smoke / lm-eval
```

Two **observe schedules** share the same metrics and adapters:

| Mode | Command family | VRAM footprint (observe) |
| --- | --- | --- |
| **Full** | `reap prune full`, `reap merge full` | Whole model on GPU |
| **Layerwise** | `reap prune layerwise`, `reap merge layerwise` | One decoder block on GPU |

Separately, **weight residency** (`--residency auto|gpu_full|layerwise|cpu_full`)
controls whether parameters are GPU-mapped, disk-offloaded, or CPU-pinned —
critical on low host-RAM instances (e.g. g6.xlarge). See [residency.md](residency.md).

## Start here

| Goal | Document |
| --- | --- |
| **Install, first run, enable Triton** | **[Setup](setup.md)** |
| Architecture overview | [Architecture](architecture.md) |
| How to run prune/merge | [CLI](cli.md) · [Pipeline](pipeline.md) |
| Low-RAM / GPU weight placement | **[Weight residency](residency.md)** |
| FREA speed vs memory on L4/T4 | **[FREA throughput](frea-throughput.md)** |
| Kernels / backends | [GPU and Backends](gpu-and-backends.md) · [Kernels design](kernels/README.md) |

## Documentation map

| Document | Concern |
| --- | --- |
| [Setup](setup.md) | Install, CUDA/Triton, first prune, verification |
| [Architecture](architecture.md) | Module boundaries, data flow, invariants |
| [Pipeline](pipeline.md) | End-to-end prune and merge execution phases |
| [Model Adapters](model-adapters.md) | Supported families, layout detection, slicing contract |
| [Calibration](calibration.md) | Datasets, composite specs, local `--dataset-path` |
| [Observation and Metrics](observation-and-metrics.md) | Observers, saliency state, prune vs merge metrics |
| [GPU and Backends](gpu-and-backends.md) | Activation/device policy, observe backends, F4/F5/FREA/F2 |
| [**FREA Throughput**](frea-throughput.md) | `--frea-backend`, probe, L4 SM erratum (48/99 KiB), run 3–4 |
| [**Weight Residency**](residency.md) | `--residency`, auto heuristics, stream save, delegation |
| [Pruning](pruning.md) | Saliency methods, ranking, `slice_experts`, config patch |
| [Merging](merging.md) | Clustering, merge methods, skip layers, super-experts |
| [Layerwise Mode](layerwise.md) | Block replay, CPU activation cache, memory trade-offs |
| [CLI](cli.md) | Typer commands, flags, legacy scripts |
| [Evaluation](evaluation.md) | lm-eval, stubs, smoke tests |
| [Development](development.md) | Install, tests, extension checklist |
| [Kernels design](kernels/README.md) | Phased kernel design reference (SoC under `kernels/`) |

## Core guarantees

- **Adapter isolation**: architecture differences live in `model_adapters.py`
  (and weight-layout convention for kernels). Pipelines do not hardcode class
  names for matmul math.
- **Routed-only prune metrics**: default observation records only metrics
  consumed by pruning (`record_pruning_metrics_only=True`). Merge forces full
  merge-criteria trackers.
- **GPU-first saliency**: hot-path accumulators stay on the compute device;
  host transfer is deferred to save/report.
- **Weight residency**: default `auto` prefers GPU-mapped weights + stream save
  when the model fits VRAM but is large vs host RAM; layerwise uses disk
  offload instead of pinning the full model in host RAM ([residency.md](residency.md)).
- **FREA profitability**: default `--frea-backend auto` probes Triton vs cuBLAS
  per host/shape so shared-mem-bound GPUs do not silently pick a slower path
  ([frea-throughput.md](frea-throughput.md)).
- **Routed-only expert work** (backends `bmm` / `frea` / `f2`): no full
  `(E, T, H)` activation materialization on the prune path.
- **Post-slice runnable modules**: fused `slice_experts` updates live
  `num_experts` / `top_k` so in-memory smoke and save stay consistent.
- **Shared experts preserved**: Qwen3.5/3.6 and Llama4 shared experts are not
  routed and are not sliced by prune.

## Supported model families (summary)

| Adapter | Family | Experts layout |
| --- | --- | --- |
| `Qwen3MoeModelAdapter` | Qwen3-MoE | Fused `(E, 2I, H)` / ModuleList legacy |
| `Qwen3_5MoeModelAdapter` | Qwen3.5 / 3.6 MoE | Fused + shared expert |
| `Llama4MoeModelAdapter` | Llama4 Text MoE | Fused bmm `(E, H, 2I)` |
| `MixtralMoeModelAdapter` | Mixtral / PhiMoE | Non-fused ModuleList |
| `Lfm2MoeModelAdapter` | LFM2.5 MoE | Fused linear + optional `expert_bias` |

Requires `transformers>=5.5.0` for current fused Qwen defaults. Details:
[model-adapters.md](model-adapters.md).

## Quick maintainer commands

```bash
uv venv .venv --seed --python 3.12
uv pip install --editable . pytest
uv pip install -e '.[cuda]'   # optional triton on CUDA hosts
uv run pytest tests/ -q
uv run reap --help
uv run reap kernels           # Triton / backend readiness
uv run reap version
```

Full walkthrough: [setup.md](setup.md).

## Related projects

- [reap-mlx](https://github.com/egesabanci/reap-mlx) — Apple Silicon / MLX port
- Paper: [REAP the Experts (arXiv 2510.13999)](https://arxiv.org/abs/2510.13999)
- Upstream inspiration: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
