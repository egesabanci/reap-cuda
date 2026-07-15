# 07 — Validation Strategy

> **Concern:** how we prove the kernels are correct and measure the speedup.
> Two tracks: **parity** (correctness, the gate that blocks every merge) and
> **performance** (the speedup claim). Parity runs on Mac (MPS) where possible;
**performance** runs on EC2 (L40S) only.

## 1. The parity ladder

Correctness is established in a **chain**: each level is validated against the
level below it, and the bottom is the existing loop observer (which is
already E2E-tested on real tiny models, commit `1d5dfde` / `5ba965e`).

```
Level 4:  FREA-Triton  ──must match──►  FREA-PyTorch (= bmm grouped, Level 3)
Level 3:  F2-Triton    ──must match──►  F2-PyTorch (= bmm + update_pruning_state_routed)
Level 2:  bmm baseline ──must match──►  loop observer (Level 1)           [Mac, MPS]
Level 1:  loop observer ──already E2E tested (tests/test_layerwise_e2e.py,
                                               tests/test_merge_pipeline.py)
Level 0:  F3 contract  ──prune path consumes only routed metrics           [Mac]
```

**Rule**: a Triton kernel (Level 3/4) is **never** validated directly against
the loop on Mac (Triton doesn't run on MPS). It is validated against its
PyTorch fallback, which **is** validated against the loop on Mac. So the
Triton-vs-loop equivalence is transitive.

## 2. The parity oracle contract (consumed metrics)

Every parity test compares **exactly the keys `prune.py` reads** (the
"consumed" set), no more, no less:

```python
CONSUMED_KEYS = [
    "total_tokens",
    "expert_frequency",
    "pairwise_expert_frequency",
    "ean_sum",
    "ean_mean",        # OnlineStatsTracker -> .mean
    "reap",            # OnlineStatsTracker -> .mean
    "weighted_ean_sum",
    "weighted_expert_frequency_sum",
    "max_activations",
    "routed_characteristic_activation",  # only when prune_method == "ean_ca"
]
```

These are the keys enumerated in `00-cost-model.md` §6 and read in
`src/reap/prune.py:60–73`. Comparing the merging-criteria keys is
**out of scope** for the kernel parity tests (those are merge-path metrics;
F2 does not produce them — see `05-f2-saliency-accumulator.md` §11).

### Tolerance

- **Per-batch, tiny model, fp32 weights**: `atol=1e-5, rtol=1e-5` (bit-for-bit
  in practice for the grouped bmm; the only divergence is accumulation order).
- **Multi-batch (Welford path for `ean_mean`/`reap`)**: `atol=1e-5` after
  ≥3 batches of **different sizes** (exercises the cross-batch Welford update —
  the trickiest fidelity, see `05-f2-saliency-accumulator.md` §3).
- **Real model (Qwen3-30B-A3B, bf16 weights, fp32 accum)**: `atol=1e-3,
  rtol=1e-3` — bf16 matmuls make bit-for-bit impossible; compare per-layer
  consumed metrics on a small calibration subset (e.g. 4 batches).

## 3. Test files (SoC — one file per phase/concern)

| File | Runs on | Compares | Gate for |
|---|---|---|---|
| `tests/test_pruning_metrics_only_contract.py` | Mac | F3 default + key sets | Phase 0 |
| `tests/test_kernel_parity_bmm.py` | Mac (MPS) | bmm vs loop, consumed keys | Phase 1 (and the oracle for FREA) |
| `tests/test_kernel_parity_f5.py` | EC2 (Triton) | F5-Triton vs F5-PyTorch (router outputs) | Phase 2 |
| `tests/test_kernel_parity_frea.py` | EC2 (Triton) | FREA-Triton vs FREA-PyTorch (= bmm grouped) | Phase 3 |
| `tests/test_kernel_parity_f2.py` | EC2 (Triton) | F2-Triton vs F2-PyTorch, multi-batch Welford | Phase 4 |
| `tests/test_f4_weight_cache.py` | Mac | shapes + fused-view-sharing-storage | Phase 5 |

Mac tests use `pytest.mark.skipif(not torch.cuda.is_available() and _HAS_TRITON,
reason="Triton kernel requires CUDA")` so the suite stays green on Mac (the
Triton tests skip on MPS, the fallback tests run).

### The Mac parity test (the most important one)

`tests/test_kernel_parity_bmm.py` is the **only** parity gate that runs on Mac
and therefore the **only** one that runs in CI (`08-…`? no — see the CI
issue #12). It proves the bmm fallback (which is what runs on Mac and what
FREA-Triton is validated against) matches the loop. Keep this test
fast (tiny model, < 5 s) and hermetic (no HF Hub download — use
`Qwen3MoeConfig`/`Qwen3MoeForCausalLM` constructed in-memory, as the existing
tests do).

## 4. The tiny-model harness (shared)

All parity tests use the same in-memory tiny Qwen3-MoE construction pattern
(established by `tests/test_layerwise_observer.py` and
`tests/test_merge_pipeline.py`):

```python
def make_tiny_qwen3_moe(num_experts=8, num_layers=2, top_k=2, hidden=16, inter=16):
    cfg = Qwen3MoeConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=inter,
        moe_intermediate_size=inter, num_hidden_layers=num_layers,
        num_attention_heads=2, num_key_value_heads=2,
        num_experts=num_experts, num_experts_per_tok=top_k, norm_topk_prob=False)
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()
```

Keep it tiny so the suite runs on free CI runners without GPU. For the Llama4
fused-layout parity, an equivalent tiny `Llama4ForCausalLM` config is
constructed (when transformers 4.55 exposes it; otherwise the fused path is
tested via a mock fused module as in `tests/test_model_adapters.py`).

## 5. Performance benchmark harness

`scripts/bench_observer.py` (new, EC2-only):

```bash
# Observer-only wall-clock + peak VRAM on Qwen3-30B-A3B, three backends:
uv run python scripts/bench_observer.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --dataset_name <calibration> \
    --observe-backend {loop,bmm,frea} \
    --run_observer_only \
    --batch_size 4 --batches_per_category 64
```

Outputs (per backend): total wall-clock, per-layer mean wall-clock, peak
`torch.cuda.max_memory_allocated()`, kernel-launch count (via
`torch.profiler`).

### Targets (vs loop baseline, Qwen3-30B-A3B, single L40S)

| Backend | Observer wall-clock | Peak VRAM (layerwise, one block) |
|---|---|---|
| loop (current) | 1× (baseline) | block weights + 8.6 GB activation transient |
| bmm (Phase 1) | ~10–15× faster | block weights + ~50 MB (grouped) |
| frea (Phase 3) | ~15–25× faster | block weights + ~1 MB stats |
| f2 (Phase 4) | ~20–30× faster | block weights + ~1 MB stats |

For E=256 (Qwen3.5/3.6 large): the FLOP cut is 32× → top end ~30–40×.

### Profiling

Use `torch.profiler` with `CUDA` + `Kernel` activities to confirm:
- launch count drops (384/layer → 1/layer for FREA),
- no `(E, T, H)` allocation (search the memory snapshot for the 8.6 GB tensor —
  must be absent on the FREA path),
- the kernel is not memory-bound on the expert weights (F4 coalescing).

## 6. End-to-end regression (EC2)

After the kernels pass parity, run the full prune E2E on Qwen3-30B-A3B with
each backend and assert the **pruned model** is identical (same experts
retained, same `config.num_experts`):

```bash
uv run python -m reap.layerwise_prune \
    --model_name Qwen/Qwen3-30B-A3B --prune_method reap \
    --compression_ratio 0.5 --observe-backend frea ...
# then with --observe-backend loop on a small subset; assert identical retained experts
```

This catches any kernel bug that passes the per-layer parity test but
diverges in the `prune.py` topk selection (e.g. an fp32-vs-fp64 accumulation
order that crosses a topk boundary).

## 7. Acceptance summary

- ✅ `test_pruning_metrics_only_contract.py` (Phase 0)
- ✅ `test_kernel_parity_bmm.py` bmm vs loop on Mac (Phase 1) — **the CI gate**
- ✅ F4 shape/view test on Mac (Phase 5)
- ✅ F5 / FREA / F2 Triton parity on EC2 (Phases 2–4)
- ✅ Observer-only bench on Qwen3-30B-A3B: FREA ≥ 10× faster than loop, peak
  VRAM < block weights + 100 MB
- ✅ E2E prune on Qwen3-30B-A3B: same retained experts as loop on a subset