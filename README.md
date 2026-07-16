# REAP — Router-weighted Expert Activation Pruning (CUDA)

Router-weighted Expert Activation Pruning for Mixture-of-Experts LLM
compression on CUDA. Based on the paper
[REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression](https://arxiv.org/abs/2510.13999)
(arXiv 2510.13999).

REAP decides **which experts to remove** (prune) or **merge together** in
MoE transformer layers using activation statistics collected during a
calibration forward pass over a small dataset.

## Quickstart

```bash
git clone https://github.com/egesabanci/reap-cuda && cd reap-cuda
uv venv .venv --seed --python 3.12
uv pip install --editable .

# Typer CLI (preferred)
reap --help
reap prune --help
reap merge --help

# Full-model prune (whole model on GPU — needs ~60 GB for 30B)
reap prune full \
  --model "Qwen/Qwen3-30B-A3B" \
  --dataset "theblackcat102/evol-codealpaca-v1" \
  --prune-method reap --compression-ratio 0.5

# Layerwise prune (one block on GPU — works on 46 GB L40S)
reap prune layerwise \
  --model "Qwen/Qwen3-30B-A3B" \
  --dataset "theblackcat102/evol-codealpaca-v1" \
  --prune-method reap --compression-ratio 0.5 \
  --observe-backend bmm

# Full-model merge (cluster → merge → save)
reap merge full \
  --model "Qwen/Qwen3-30B-A3B" \
  --dataset "theblackcat102/evol-codealpaca-v1" \
  --expert-sim characteristic_activation \
  --cluster-method agglomerative \
  --compression-ratio 0.5

# Layerwise merge (30B+ on a single L40S)
reap merge layerwise \
  --model "Qwen/Qwen3-30B-A3B" \
  --dataset "theblackcat102/evol-codealpaca-v1" \
  --expert-sim characteristic_activation \
  --compression-ratio 0.5
```

### CLI map

| Command | Memory mode | What it does |
|---|---|---|
| `reap prune full` | Whole model on GPU | Observe → prune → save |
| `reap prune layerwise` | One block on GPU | Same, block-wise calib |
| `reap merge full` | Whole model on GPU | Observe merge metrics → cluster → merge |
| `reap merge layerwise` | One block on GPU | Same, block-wise calib |
| `reap version` | — | Package version |

Common flags: `--model` / `-m`, `--dataset` / `-d`, `--compression-ratio`,
`--observe-backend {auto,loop,bmm,frea,f2}`, `--observe-only`, `--eval`.

Legacy console scripts (`reap-prune`, `reap-layerwise`, `reap-merge`,
`reap-layerwise-merge`) still work via HfArgumentParser.

## Supported Models

| Architecture | Adapter | Experts layout | `transformers` |
|---|---|---|---|
| Qwen3-MoE | `Qwen3MoeModelAdapter` | Fused stacked `gate_up_proj`/`down_proj` | `>=5.5.0` |
| Qwen3.5/3.6-MoE | `Qwen3_5MoeModelAdapter` | Fused stacked params + shared expert | `>=5.5.0` |
| Llama4-MoE | `Llama4MoeModelAdapter` | Fused `gate_up_proj`/`down_proj` | `>=4.50` |
| Mixtral / PhiMoE | `MixtralMoeModelAdapter` | Non-fused + `num_local_experts` | `>=4.50` |
| LFM2.5 MoE | `Lfm2MoeModelAdapter` | Fused (requires ≥5.2) | `>=5.2` |

## Memory Modes

| Command | Peak VRAM | Use when |
|---|---|---|
| `reap-prune` | Whole model (~60 GB) | Multi-GPU / A100-80GB / H100 |
| `reap-layerwise` | One block (~1.2 GB + transient) | Single L40S 46 GB, 30B+ |
| `reap-merge` | Whole model (~60 GB) | Multi-GPU / large instance |
| `reap-layerwise-merge` | One block (~1.2 GB + transient) | Single L40S 46 GB, 30B+ |

## Architecture

```
cli/                 — Typer CLI (reap prune|merge full|layerwise)
pipeline.py          — Helpers: record_activations, _setup_observer, smoke_test
model_adapters.py    — Layout-based adapters (weight convention + fused detection)
observer.py          — Standard forward-hook observer (whole model on GPU)
layerwise_observer.py — Block-wise observer (one block on GPU at a time)
pruning_metrics.py   — GPU-resident REAP/EAN/frequency saliency
kernels/             — observe backends: loop | bmm | frea | f2 (F4 weight cache, F5 router)
metrics.py           — Distance functions + OnlineStatsTracker (Welford/Kahan)

prune.py / layerwise_prune.py     — Prune run() APIs (+ legacy scripts)
merge_pipeline.py / layerwise_merge.py — Merge run() APIs (+ legacy scripts)
cluster.py           — Hierarchical, k-means, MC-SMoE, restricted clustering
merge.py             — Merge methods (frequency-weighted, average, TIES, MultiSLERP, ...)
permute.py           — Weight permutation / matching for merge alignment
eval.py              — lm-eval harness (HF backend)
```

## Observation backends (GPU-first)

```bash
# Default: auto → f2 on CUDA+Triton, else routed-only bmm (no (E,T,H) materialization)
reap-prune ... --observe_backend auto

# Explicit backends
reap-layerwise ... --observe_backend bmm   # pure PyTorch grouped routed matmul
reap-layerwise ... --observe_backend frea  # FREA (compiled/grouped on CUDA)
reap-layerwise ... --observe_backend loop  # legacy full-expert loop (parity oracle)
```

Saliency accumulators stay on the compute device; tensors are moved to CPU only
when saving observer state. Prune path defaults to
`--record_pruning_metrics_only True` (merge entrypoints force `False`).

## Environments

- **Dev (macOS / Linux):** CPU/MPS — pure-PyTorch `bmm`/`frea` fallbacks.
  Run tests with `uv run pytest tests/ -q`.
- **EC2:** g6e.2xlarge (L40S, sm_89) or larger. Optional Triton:
  `uv pip install -e '.[cuda]'`.

## Data

- **Single dataset:** `--dataset_name "theblackcat102/evol-codealpaca-v1"`
- **Composite (multi-dataset):** comma-separated spec:
  `"theblackcat102/evol-codealpaca-v1:4096,open-r1/Mixture-of-Thoughts[code]:4096"`
- **Pre-recorded combined:** `--dataset_name combined` loads cached
  observer data (requires prior calibration run).

## Why `transformers>=5.5.0` is required

`transformers>=5.x` uses a fused `Qwen3MoeExperts` module for **both**
Qwen3-MoE and Qwen3.5/3.6-MoE: per-expert weights are stacked as a single
`nn.Parameter` on dim 0 (`gate_up_proj` / `down_proj`) instead of per-expert
`nn.Linear` in a `ModuleList`, and the router is a `Qwen3_5MoeTopKRouter`
that returns a `(logits, scores, indices)` tuple (Qwen3.5/3.6 MoE blocks add
a shared expert). The adapter system detects the fused layout at runtime
(`_is_fused_experts`) and dispatches to the fused observer / slice / save
paths; `Qwen3_5MoeModelAdapter` keys on `Qwen3_5MoeSparseMoeBlock` and leaves
the shared expert untouched by pruning. The legacy non-fused `ModuleList`
path (transformers 4.55) is still supported by the loop observer branch but
is no longer the default target.

## Testing

```bash
uv pip install pytest
uv run pytest tests/ -q        # CPU-only suite (parity, adapters, slice, contracts)
```

All tests construct tiny in-memory `Qwen3MoeForCausalLM` models (4 experts,
2 layers, hidden_size=8) — no weights are downloaded from HuggingFace Hub.

## Related

- [reap-mlx](https://github.com/egesabanci/reap-mlx) — MLX/Aarch64 port
  for Apple Silicon
- Original paper: [arXiv 2510.13999](https://arxiv.org/abs/2510.13999)
