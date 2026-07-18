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
| **FREA** SwiGLU | `kernels/frea.py` + `triton_frea.py` | **Shipped** (Triton + PyTorch; **`--frea-backend` probe**) |
| **F2** saliency reduce | `pruning_metrics.update_pruning_state_routed` + `triton_reduce.py` | **Shipped** (**fp64** Triton atomics + PyTorch `index_add_` / Welford) |
| Unified observe entry | `kernels/observe.py` → `observe_moe_batch` | **Shipped** (both full + layerwise observers) |
| Backend select | `kernels/backend.py` · CLI `--observe-backend` | **Shipped** (`auto\|loop\|bmm\|frea\|f2`) |
| FREA policy | `triton_frea.set_frea_backend` · CLI `--frea-backend` | **Shipped** (`auto\|triton\|pytorch`) |
| Status CLI | `reap kernels` | **Shipped** (package/runtime; run summary is separate) |

**Triton is real** (`@triton.jit` in `triton_softmax.py`, `triton_frea.py`,
`triton_reduce.py`). Imports are **lazy** so Mac/CPU installs work without the
`triton` package. On hosts without CUDA, runtime uses **PyTorch fallbacks**. On
EC2 with `uv pip install -e '.[cuda]'`, `--observe-backend auto` prefers `f2`;
**`--frea-backend auto`** then probes whether FREA-Triton or cuBLAS is faster
on that GPU (see [frea-throughput.md](../frea-throughput.md)).

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
  → router: F5 softmax+topk  OR  native module router (bias/sigmoid families)
  → F4: W_gate/up (E,I,H), W_down (E,H,I)  [linear convention; ≤1 cache entry]
  → FREA: SwiGLU on pairs (Triton|grouped linear per --frea-backend) → (n_pairs, H)
  → F2: scatter norms/weights fp64 (Triton|index_add) + Welford (always PyTorch)
```

Default CLI backends:

| `--observe-backend` | Behavior |
| --- | --- |
| `auto` | `f2` if Triton **runtime** OK, else `bmm` |
| `bmm` | Always PyTorch grouped path |
| `frea` | FREA stage (+ `--frea-backend` policy) |
| `f2` | FREA + F2 reduce |
| `loop` | Legacy dense/loop path (parity) |

```bash
reap kernels                          # print Triton package/runtime readiness
export REAP_DISABLE_TRITON=1          # force all kernels to PyTorch
reap prune layerwise --observe-backend bmm ...
reap prune full --frea-backend auto   # probe Triton vs cuBLAS for FREA
```

### Triton eligibility (gates)

| Kernel | Needs |
| --- | --- |
| Softmax | CUDA + triton package; `E ≤ 1024` for single-tile path |
| FREA | CUDA + triton; SiLU; `H,I ≥ 16`; tiles fit shared mem; then **`--frea-backend`** |
| F2 reduce | CUDA + triton; typically `H ≥ 16`; **fp64** accumulators |
| Always | On failure → PyTorch (WARN once, then DEBUG; end-of-run INFO summary) |

Tiny unit-test models (H=8) **intentionally** stay on PyTorch.

Ops detail for FREA tiles / probe / L4 SM erratum (48 KiB default / 99 KiB
opt-in, not 164 KiB): **[../frea-throughput.md](../frea-throughput.md)**.

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

## Deferred redesigns (not shipped)

The following high-risk kernel redesigns are **intentionally deferred** and
not implemented in the current codebase:

- **Single-launch CSR FREA**: The per-expert Python loop in `triton_frea.py`
  remains. A single CSR launch requires a new grid mapping from program IDs to
  variable-length CSR segments, per-program expert-weight addressing, tuning
  across routing imbalance, and a performance study.
- **Multi-tile F5 softmax**: `triton_softmax.py` falls back to `F.softmax` for
  `E > 1024`. A multi-tile online softmax needs numerically stable multi-pass
  reduction, workspace management, and wide-MoE parity/stress tests.
- **In-kernel Welford fusion**: `OnlineStatsTracker` updates stay in PyTorch;
  fusing them into F2 changes cross-batch reduction semantics.
- **F2 grid redesign**: The one-program-per-pair grid is simple but can suffer
  from atomic contention on skewed routing.
- **Cache thread-safety**: `_STACK_CACHE` and FREA global state are not
  thread-safe for `DataParallel`/DDP.

See [07-validation-strategy.md](07-validation-strategy.md) for the test matrix
and deferred-design prerequisites.

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
