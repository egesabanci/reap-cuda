# REAP Observation Kernels — Current Status & Design

> Parent docs: [../index.md](../index.md) · [../setup.md](../setup.md) ·
> [../gpu-and-backends.md](../gpu-and-backends.md)  
> Implementation: `src/reap/kernels/`

This directory documents the **observation acceleration stack** (routed expert
activation + saliency). Design history lives in the numbered phase files; this
README is the **authoritative current status** of the code on `main`.

## Current status (code, not aspiration)

| Component | Code | Status |
| --- | --- | --- |
| **F3** prune-only metrics default | `args.ObserverArgs.record_pruning_metrics_only=True` | **Shipped** + `tests/test_pruning_metrics_only_contract.py` |
| **F4** weight cache | `kernels/weight_cache.py` | **Shipped** (layout-normalized linear stacks) |
| **F5** router + pairs | `kernels/router.py` + `triton_softmax.py` | **Shipped** (PyTorch topk/CSR; **Triton** softmax when eligible) |
| **bmm** grouped routed MLP | `kernels/bmm.py` | **Shipped** — parity oracle / Mac default |
| **FREA** SwiGLU | `kernels/frea.py` + `triton_frea.py` | **Shipped** (**Triton** `@triton.jit` + PyTorch fallback) |
| **F2** saliency reduce | `pruning_metrics.update_pruning_state_routed` + `triton_reduce.py` | **Shipped** (**Triton** atomics + PyTorch `index_add_` / Welford) |
| Unified observe entry | `kernels/observe.py` → `observe_moe_batch` | **Shipped** (both full + layerwise observers) |
| Backend select | `kernels/backend.py` · CLI `--observe-backend` | **Shipped** (`auto\|loop\|bmm\|frea\|f2`) |
| Status CLI | `reap kernels` | **Shipped** |

**Triton is real** (`@triton.jit` in `triton_softmax.py`, `triton_frea.py`,
`triton_reduce.py`). Imports are **lazy** so Mac/CPU installs work without the
`triton` package. On this machine without CUDA, runtime uses **PyTorch
fallbacks** (`bmm` path). On EC2 with `uv pip install -e '.[cuda]'`,
`--observe-backend auto` prefers `f2` and can launch Triton.

There is **no** dependency on `torch.compile` for correctness.

## Package map

```txt
src/reap/kernels/
  observe.py           # observe_moe_batch (single entry for both observers)
  backend.py           # select_observe_backend, triton_status
  weight_cache.py      # F4
  router.py            # F5 API (pairs + CSR)
  bmm.py               # grouped PyTorch FREA math
  frea.py              # dispatch → Triton or bmm
  f2.py                # dispatch → update_pruning_state_routed
  triton_utils.py      # capability detection, REAP_DISABLE_TRITON
  triton_softmax.py    # F5 Triton softmax
  triton_frea.py       # FREA Triton SwiGLU
  triton_reduce.py     # F2 Triton scatter
```

## How a routed observe step runs

```txt
flat_input (T, H)
  → extract_router_logits
  → F5: softmax (Triton|PyTorch) + topk + sort pairs by expert (CSR)
  → F4: W_gate/up (E,I,H), W_down (E,H,I)  [linear convention]
  → FREA: SwiGLU on pairs only (Triton|grouped linear) → (n_pairs, H)
  → F2: scatter norms/weights (Triton|index_add) + Welford means (always PyTorch)
```

Default CLI backends:

| `--observe-backend` | Behavior |
| --- | --- |
| `auto` | `f2` if Triton **runtime** OK, else `bmm` |
| `bmm` | Always PyTorch grouped path |
| `frea` | Try Triton FREA; fallback bmm |
| `f2` | Try Triton FREA + reduce; fallback PyTorch |
| `loop` | Legacy dense/loop path (parity) |

```bash
reap kernels                          # print Triton readiness
export REAP_DISABLE_TRITON=1          # force PyTorch
reap prune layerwise --observe-backend bmm ...
```

### Triton eligibility (gates)

| Kernel | Needs |
| --- | --- |
| Softmax | CUDA + triton package; `E ≤ 1024` for single-tile path |
| FREA | CUDA + triton; SiLU; `H ≥ 16` and `I ≥ 16`; weights on CUDA |
| F2 reduce | CUDA + triton; typically `H ≥ 16` for Triton path |
| Always | On failure → PyTorch (debug log `Triton … fallback`) |

Tiny unit-test models (H=8) **intentionally** stay on PyTorch.

## Expected impact (projections vs loop baseline)

Reference: Qwen3-30B-A3B-class (E=128, top_k=8, H=2048, I=768, T=8192).
Wall-clock is **projected** until measured on EC2 (`08-expected-improvements.md`).

| Stage | Expert FLOPs vs loop | Peak act mem vs ~8.6 GB `(E,T,H)` | Observe wall-clock (proj.) |
| --- | --- | --- | --- |
| F4 | ~same | +~1.2 GB weight cache (temporary) | ~1× alone |
| F5 | n/a (router) | MB-scale pairs | small absolute; enables CSR |
| **bmm** | **~16× less** | **GB → ~MB** | **~10–15×** |
| **FREA Triton** | same as bmm | same as bmm | **~15–25×** vs loop; modest over bmm |
| **F2** | n/a | no big act re-read | helps total **~20–30×** vs loop |

Does **not** speed up: original HF forward (attention + MoE still run), prune
topk/save, clustering. Observer still **recomputes** experts for metrics
(double expert work vs invasive fusion).

## Document index (phase design + history)

Numbered docs are design notes. Prefer this README + code for “what runs today.”
Stale line numbers in old snippets may not match `main`; trust `src/reap/kernels/`.

| Doc | Phase | Concern | Status vs code |
| --- | --- | --- | --- |
| [00-cost-model.md](00-cost-model.md) | — | Loop baseline costs | **Historical baseline** (loop path still available via `--observe-backend loop`) |
| [01-f3-dead-metric-audit.md](01-f3-dead-metric-audit.md) | 0 | Prune-only metrics contract | **Landed** (default `True` + contract tests) |
| [02-bmm-baseline.md](02-bmm-baseline.md) | 1 | Grouped routed PyTorch | **Landed** as `bmm.py` (grouped form, not naive weight-gather) |
| [03-f5-router-fusion.md](03-f5-router-fusion.md) | 2 | Softmax + topk + pairs | **Landed** (`router.py` + Triton softmax) |
| [04-frea-kernel.md](04-frea-kernel.md) | 3 | Routed SwiGLU | **Landed** (`triton_frea.py` + bmm fallback) |
| [05-f2-saliency-accumulator.md](05-f2-saliency-accumulator.md) | 4 | Saliency reduce | **Landed** (pair scatter + Welford; not full in-kernel Welford) |
| [06-f4-weight-stacking.md](06-f4-weight-stacking.md) | 5 | Weight cache | **Landed** (`weight_cache.py`, linear + bmm layouts) |
| [07-validation-strategy.md](07-validation-strategy.md) | — | Tests | **Partial** — see tests list below |
| [08-expected-improvements.md](08-expected-improvements.md) | — | Perf projections | Reference (unmeasured wall-clock) |

## Tests (actual files)

| Test | What |
| --- | --- |
| `tests/test_pruning_metrics_only_contract.py` | F3 contract |
| `tests/test_kernel_parity_bmm.py` | loop vs bmm/frea metrics on tiny Qwen3 |
| `tests/test_f4_weight_cache.py` | F4 shapes / Llama transpose |
| `tests/test_triton_kernels.py` | softmax/F5/FREA/reduce; CUDA cases `@requires_triton` |
| `tests/test_cli.py` | includes `reap kernels` |

```bash
uv run pytest tests/test_triton_kernels.py tests/test_kernel_parity_bmm.py -q
uv run reap kernels
```

## Fallback discipline

Never call a Triton kernel without a PyTorch path. Dispatch helpers:

- `triton_utils.triton_runtime_available()` / `prefer_triton_for(tensor)`
- `select_observe_backend(...)`
- Per-kernel try/except → `log_triton_fallback(...)`

## Environment

| | Dev (Apple Silicon) | EC2 |
|---|---|---|
| GPU | MPS / CPU — **no Triton launch** | NVIDIA CUDA + optional Triton |
| What runs | `bmm` / loop, parity tests | `auto`→`f2`, Triton when eligible |
| Install | `uv pip install -e .` | `uv pip install -e '.[cuda]'` |
| transformers | `>=5.5` (fused Qwen default) | same |

## Related

- User setup: [../setup.md](../setup.md)
- Runtime backends: [../gpu-and-backends.md](../gpu-and-backends.md)
- Metrics keys: [../observation-and-metrics.md](../observation-and-metrics.md)
