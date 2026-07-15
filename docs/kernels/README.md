# REAP Custom Kernels — Design & Implementation Guide

This directory is the **authoritative design reference** for the custom Triton
kernels that accelerate REAP's MoE pruning/merge calibration. Each document is
a self-contained module following **Separation of Concerns (SoC)**: one phase,
one kernel, or one cross-cutting topic per file. Files cross-reference each
other rather than duplicating content.

## Why kernels exist

REAP's pruning saliency is computed from **activations of routed experts**
during a calibration forward pass over a dataset. On a 128- or 256-expert MoE,
the stock HuggingFace MoE block runs a **Python `for` loop over every expert**
(`for idx, expert in enumerate(module.experts)`), launching hundreds of small
matmuls per layer and materializing a full `(E, T, H)` activation tensor — while
REAP's saliency only depends on the **~6% of `(expert, token)` pairs that are
routed** (`top_k / E`). The kernels eliminate that loop and that materialization.

## Target hardware & environment

| | Dev | EC2 |
|---|---|---|
| Machine | Apple Silicon (M-series) | `g6e.2xlarge` |
| GPU | MPS (no CUDA, no Triton) | NVIDIA L40S, sm_89, 46 GB |
| What runs | pure-PyTorch fallbacks, parity tests | Triton kernels, benchmarks |
| Repo venv | `.venv` (python 3.12, torch 2.7.1, transformers 4.55) | same |

Every Triton kernel ships with a **pure-PyTorch fallback** (selected when
`torch.cuda.is_available()` is false or `triton` is unavailable) so the whole
pipeline stays runnable on the Mac for control-flow and parity, and only the
fast path is gated to EC2.

## Reference model

Concrete numbers throughout this guide use **Qwen3-30B-A3B** (the realistic
single-L40S target):

| Config | Value |
|---|---|
| `num_hidden_layers` | 48 |
| `num_experts` (E) | 128 |
| `num_experts_per_tok` (top_k) | 8 |
| `hidden_size` (H) | 2048 |
| `moe_intermediate_size` (I) | 768 |
| routed fraction (top_k / E) | 6.25 % |

256-expert variants (Qwen3.5/3.6 large) are called out where the factor changes.

## Document index

| Doc | Phase | Concern | Status |
|---|---|---|---|
| [`00-cost-model.md`](00-cost-model.md) | — | The current bottleneck: quantified | reference |
| [`01-f3-dead-metric-audit.md`](01-f3-dead-metric-audit.md) | 0 | Prerequisite contract: prune consumes routed-only metrics | planned |
| [`02-bmm-baseline.md`](02-bmm-baseline.md) | 1 | Pure-PyTorch bmm baseline (parity oracle, MPS-runnable) | planned |
| [`03-f5-router-fusion.md`](03-f5-router-fusion.md) | 2 | Fused router softmax + topk + gather-index builder | planned |
| [`04-frea-kernel.md`](04-frea-kernel.md) | 3 | FREA: fused routed expert activation (headline kernel) | planned |
| [`05-f2-saliency-accumulator.md`](05-f2-saliency-accumulator.md) | 4 | F2: fused online saliency accumulator (all consumed metrics) | planned |
| [`06-f4-weight-stacking.md`](06-f4-weight-stacking.md) | 5 | F4: expert weight pre-stacking cache | planned |
| [`07-validation-strategy.md`](07-validation-strategy.md) | — | Parity + benchmark harness | planned |
| [`08-expected-improvements.md`](08-expected-improvements.md) | — | Performance & memory projections | reference |

## Phase dependency graph

```
            ┌─────────────────────────────────────────────┐
            │  Phase 0 — F3 dead-metric audit + default     │
            │  (makes "routed-only" load-bearing)           │
            └──────────────────────┬──────────────────────┘
                                   │ unlocks (proves correctness contract)
                                   ▼
            ┌─────────────────────────────────────────────┐
            │  Phase 1 — bmm baseline (parity oracle)       │
            │  pure PyTorch, runs on MPS                    │
            └──────────────────────┬──────────────────────┘
                                   │ reference for every Triton kernel
            ┌──────────────────────┴──────────────────────┐
            ▼                                               ▼
 ┌────────────────────┐                         ┌────────────────────────┐
 │ Phase 5 — F4       │                         │ Phase 2 — F5 router    │
 │ weight stacking    │                         │ softmax+topk+gather    │
 │ (feeds FREA tiles) │                         └───────────┬────────────┘
 └─────────┬──────────┘                                       │
           │                                                  ▼
           │                                    ┌────────────────────────┐
           └─────────────►─────────────────────►│ Phase 3 — FREA          │
                                                │ fused routed activation │
                                                └───────────┬────────────┘
                                                            │
                                                            ▼
                                                ┌────────────────────────┐
                                                │ Phase 4 — F2           │
                                                │ fused saliency accum.  │
                                                └────────────────────────┘
```

Phases 2–4 are **CUDA/Triton-gated** (no Mac execution); Phase 1 is the
Mac-runnable oracle they must match.

## Tracking

Implementation is tracked in GitHub issue
[`#13`](https://github.com/egesabanci/reap-cuda/issues/13) (epic). Prerequisite
issues: [`#4`](https://github.com/egesabanci/reap-cuda/issues/4) (fused
Qwen3.5/3.6 support), [`#3`](https://github.com/egesabanci/reap-cuda/issues/3)
(layerwise merge path).

## Conventions

- **Line references** use the format `src/reap/<file>.py:<line>`. They are
  accurate against `main` at the time of writing (commit `5ba965e`); re-resolve
  with `git grep` before editing.
- **Parity contract**: a kernel is correct iff its per-layer output state
  matches the bmm baseline (Phase 1) **bit-for-bit** (within fp32 accumulation
  tolerance) on a tiny `Qwen3MoeForCausalLM`. See `07-validation-strategy.md`.
- **Fallback discipline**: every kernel site calls a `_select_backend()` helper
  that returns `"triton"`, `"bmm"`, or `"loop"` based on capability. Never call
  a Triton kernel directly without the fallback.