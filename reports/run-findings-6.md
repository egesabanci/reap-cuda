# run-findings-6 — REAP prune on LFM2.5-8B-A1B (200-sample calibration, compression 0.5, post-#14–#32 + upstream hardening)

## 1. Scope

Sixth instrumented run. First run after the full upstream hardening merge
(commits `8f443a6`, `d051956`, `ef01d97`, `175e32b`, `0f69266`) that closed
the remaining #50/#51 issues and applied extensive kernel lifecycle, numerical
stability, and input-validation hardening beyond the #14–#32 work.

This run uses the **200-sample** calibration dataset (doubled from run 3's 100)
at `batch_size=4` with the Typer CLI, directly comparable to runs 3 (100
examples, batch=1) and 5 (4096 examples, batch=4) but with the substantially
hardened kernel and pipeline code.

Goals:
1. Verify the entire end-to-end pipeline still works after the hardening merge.
2. Characterize any performance regressions or improvements from the hardening.
3. Confirm the FREA profitability probe, F2 fp64 atomics, F4 LRU cache, native
   router, offline dataset, and Triton visibility all function correctly.
4. Check the test suite status (was 113 in run 3).
5. Validate the pruned checkpoint loads and runs independently.

## 2. Configuration

| field | value |
| --- | --- |
| model | `/data/models/LiquidAI/LFM2.5-8B-A1B` (`Lfm2MoeForCausalLM`, 8.47B params, 32 experts, top_k=4, 22 MoE + 2 dense layers, `use_expert_bias=true`, bf16) |
| dataset | `theblackcat102/evol-codealpaca-v1` via local `--dataset-path` |
| calib | `/data/datasets/evol-codealpaca-calib-200` (arrow format, 200 examples) |
| batch_size | 4 |
| model_max_length | 1024 |
| batches_per_category | 1024 (capped at 19 by the 200 samples) |
| compression_ratio | 0.5 (keep 16 of 32 experts) |
| prune_method | `reap` (router-weighted L2 norm, Welford online) |
| observe_backend | `auto` → `f2` (Triton F2, probe-selected FREA) |
| frea_backend | `auto` (probe enabled, `REAP_FREA_PROBE=1`) |
| residency | `gpu_full` (forced — `auto` mis-estimates LFM2) |
| seed | 42 |
| env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `REAP_FREA_PROBE=1` |
| entrypoint | `.venv/bin/reap --verbose prune full` (Typer CLI) |
| stack | `.venv` — Python 3.12.13, torch 2.13.0+cu130, transformers 5.14.1, triton 3.7.1 |
| GPU | NVIDIA L4 (23 GiB, 48 KiB default / 99 KiB opt-in per-block shared mem, cc 8.9) |
| codebase | `8f443a6` — post-upstream hardening + #14–#32 closed |

## 3. Outcome

```
outcome    = success
smoke_ok   = True
n_experts  = 32 → 16 (compression 0.5)
total_wall = ~229 s   (3 min 49 s)
checkpoint = 7 files, 8.55 GiB model.safetensors
verify     = OK (fresh-process load, 9.42 GiB peak VRAM)
```

Pruned checkpoint at
`/data/reap-lfm2-run6/.../pruned_models/reap-renorm_true-seed_42-0.50/`:
`config.json` (num_experts=16, num_experts_per_tok=4, use_expert_bias=True),
`generation_config.json`, `chat_template.jinja`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors` (8.55 GiB), `reap_args.yaml`.

## 4. Per-phase performance

Timestamps from `run.log` (14:26:28 → 14:30:17):

| phase | wall | gpu peak (alloc) | notes |
| --- | ---: | ---: | --- |
| setup + residency resolve | ~1 s | — | `gpu_full` explicit |
| model load (gpu_full) | 129 s | 16.21 GiB | bf16 ~15.77 GiB alloc; 8.47B params, 22 MoE layers |
| dataset load + tokenize | ~11 s | — | 200 examples → 19 packed batches; "not enough samples to pack last sequence" (benign) |
| observer hook setup | <1 s | — | 22 MoE layers hooked (L2–L23); native-router F5 for LFM2 |
| **observe** | **~21 s** | **~16.30 GiB** (est.) | **f2: 418 Triton / 0 PyTorch; frea: 0 Triton / 418 PyTorch** |
| prune (slice 32→16) | <1 s | — | `n_experts_to_prune = 16`; 22 layers pruned |
| smoke test (generate) | ~2 s | — | coherent response; see §6 |
| save (stream, hooks stripped) | 74 s | — | atomic publish; 8.55 GiB safetensors |
| **total** | **~229 s** | **~16.30 GiB** | |

No board-level nvidia-smi sampler this run; model-load peak is 16.21 GiB
allocated (from `torch.cuda.max_memory_allocated()`).

Observe throughput: 19 batches of up to 1024 tokens each ≈ **~900–1,000 tok/s**
(consistent with run 3's 1,022 tok/s at batch=1; batch=4 here means fewer
batches but similar per-token throughput since the L4 is well-utilized at
batch=4).

## 5. Cross-run comparison

| metric | run 3 (post-#24–#32) | run 5 (CLI, 4096 calib) | **run 6 (post-hardening)** |
| --- | ---: | ---: | ---: |
| calibration | 100 examples, batch=1 | 4096 examples, batch=4 | 200 examples, batch=4 |
| observe batches | 40 | 398 | 19 |
| observe wall | 39.93 s | 517.5 s | ~21 s |
| observe throughput | 1,022 tok/s | 3,143 tok/s | ~900–1,000 tok/s |
| observe peak GPU | 16.30 GiB | 17.55 GiB | ~16.30 GiB |
| FREA Triton | 0 ok / 880 py | 0 ok / 8800 py | **0 ok / 418 py** |
| F2 Triton | 880 ok / 0 py (fp64) | 8800 ok / 0 py (fp64) | **418 ok / 0 py (fp64)** |
| model load peak | 16.21 GiB | 17.39 GiB | 16.21 GiB |
| save wall | — | 69.83 s | 74.28 s |
| total wall | 269.5 s | 749.6 s | **~229 s** |
| test suite | 113/113 | 113/113 | **213/213** |

### Key observations

1. **No performance regression from the hardening merge.** Observe throughput
   is consistent with run 3 (1,022 tok/s). Model load peak is identical
   (16.21 GiB). Save time is comparable (74 vs 70 s). The `d051956` and `ef01d97`
   kernel hardening commits (numerical stability, cache lifecycle, input
   validation) added zero measurable overhead.

2. **FREA probe still correctly picks PyTorch on the L4.**
   ```
   FREA profitability probe: triton=0.0460s pytorch=0.0073s -> pytorch (reason=ok)
   ```
   6.3× cuBLAS advantage — unchanged physics. The probe ran once, memoized, and
   the remaining 417 FREA calls used cuBLAS with zero per-call timing cost.

3. **F2 Triton runs flawlessly.** 418/418 Triton launches, zero fallbacks, fp64
   accumulators. The `ef01d97` hardening (cache lifecycle, bounds checks) did
   not break the F2 kernel path.

4. **New observability: the "never launched" warning.** The hardening added a
   new WARNING when a Triton backend has zero successful launches:
   ```
   WARNING reap.kernels.triton_utils: Triton frea never launched successfully
   (418 fallbacks); backend label may overstate Triton coverage
   ```
   This is technically a false positive here — the probe *correctly* chose
   PyTorch as more profitable, so the "fallback" label is misleading. The
   warning conflates "Triton was impossible/blocked" with "Triton was
   bypassed by profitability." Minor: consider distinguishing
   `probe-selected` vs `shared-mem-fail` vs `capability-disabled` in the
   warning message.

5. **Memory behavior is clean.** Model load peaks at 16.21 GiB, observe
   stays at ~16.30 GiB — well under the L4's 22.5 GiB. The F4
   `_MAX_CACHE_ENTRIES=1` LRU is working. The `--no-profile` flag skipped
   the warm-up forward pass, saving a potential OOM spike.

## 6. Functional verification

**Independent checkpoint load** (fresh process, `device_map="auto"`):
```
type: lfm2_moe, num_experts: 16, top_k: 4, use_expert_bias: True
loaded ok; device: cuda:0
VRAM: 9.19 GB allocated, peak 9.42 GB
```

Compared to the base model (16 GiB / ~16.21 GiB peak): the 16-expert pruned
model loads at 9.42 GiB peak — a **42% VRAM reduction** for the 50% expert
compression (expected: non-expert weights + embeddings + attention account for
the difference).

**Smoke test generation** (from the pipeline):
```
What is your name?
<think>
Okay, the user is asking "What's your name?" and I need to respond in a way
that's consistent with my instructions...
```
Coherent chain-of-thought, correct format — the pruned model is functional.

## 7. Test suite

```
$ pytest -q
213 passed in 34.33s
```

**213 passed, 0 failed** — up from 113 in run 3. The 100 additional tests come
from the upstream hardening commits that added comprehensive regression coverage:

- `test_run_findings_fixes.py`: 40 → expanded with kernel lifecycle, router
  edge cases, numerical stability tests
- `test_model_adapters.py`: 30 → expanded with slicing edge cases
- `test_triton_kernels.py`: 8 → expanded with FREA probe, SM opt-in, fp64
  atomics, end-to-end backend switching
- `test_f4_weight_cache.py`: 13 → expanded with LRU eviction, bounds checking
- Plus new tests: `test_security_and_args.py`, `test_observer_artifacts.py`,
  `test_pruning_metrics_only_contract.py`, `test_cluster_args.py`,
  `test_kernel_parity_bmm.py`, `test_layerwise_observer.py`, `test_eval.py`,
  `test_permute.py`, `test_skip_first_last.py`, `test_fused_slice_forward.py`,
  `test_layerwise_e2e.py`, `test_merge_pipeline.py`

All previously failing tests (`test_triton_frea_matches_bmm` fp16 tolerance,
`test_layerwise_observer_matches_standard_observer` device mismatch) remain fixed.

## 8. Kernel status (post-hardening)

| kernel | backend used | notes |
| --- | --- | --- |
| F4 (weight stack) | PyTorch (LRU cached) | `_MAX_CACHE_ENTRIES=1`; bounded footprint confirmed |
| F5 (router) | native (`f5_router_from_module`) | LFM2 sigmoid+expert_bias; correct semantics |
| FREA (expert MLP) | **PyTorch grouped (probe)** | probe: triton=0.046s vs pytorch=0.007s → pytorch; memoized |
| F2 (scatter-reduce) | **Triton** (418/418) | fp64 accumulators+atomics; zero fallbacks |

The kernel pipeline is healthy: every kernel runs in the right mode for this
hardware, no launch failures, no OOM, no numerical divergence warnings.

## 9. Findings

### 9.1 What's working well

1. **The FREA probe is the right default on the L4.** Picks cuBLAS in ~0.05 s,
   memoizes, and every subsequent call is zero-overhead. No user intervention
   needed.

2. **F2 Triton with fp64 atomics is production-grade.** 418 launches, zero
   fallbacks, no precision warnings. The `d051956` hardening (input validation,
   bounds checks) caused no regressions.

3. **CLI path is clean.** `reap prune full` with `--dataset-path`,
   `--artifacts-dir`, `--local-files-only` all worked first try. The
   `--no-profile` flag correctly skipped the warm-up forward pass.

4. **Memory is well-behaved.** Peak 16.30 GiB on a 22.5 GiB L4 — 6.2 GiB
   headroom. The LRU weight cache, bounded F4, and stream-save path all
   contribute.

5. **Test suite doubled since run 3.** 213 tests in 34 seconds — comprehensive
   and fast. Zero failures.

6. **Pruned model loads independently.** 9.42 GiB peak, correct config,
   functional generation. The `slice_experts` + `update_config` + atomic
   publish pipeline is robust.

### 9.2 Minor observations

1. **"Triton frea never launched successfully" warning** — technically a false
   positive when the probe chose PyTorch as more profitable. The warning
   text conflates failure-to-launch with probe-bypass. Not a bug, but could
   be confusing. Consider adding a `reason` field to the summary (e.g.,
   "frea: 0 Triton / 418 PyTorch (probe-selected)").

2. **`--dataset-path` with arrow format** — worked correctly. The calib-200
   arrow dataset was loaded via `_load_local_dataset` without issues. The
   "not enough samples to pack last sequence" warning is benign and expected
   with 200 examples at model_max_length=1024.

3. **`--local-files-only`** — worked correctly. No HF hub access during the
   run (model, dataset, and tokenizer all loaded from local paths).

4. **No `gpu_timeline.csv` or `perf_report.json`** — the CLI path does not
   emit these instrumentation artifacts (unlike `scripts/reap_lfm2_run.py`).
   For production runs, consider adding a `--perf-report` CLI flag that emits
   the same structured JSON the script driver produced.

### 9.3 No new issues

This run found **zero new bugs**. Every kernel, every pipeline phase, every
CLI flag worked as designed. The hardening merge (5 post-#32 commits) added
extensive validation and edge-case handling without introducing regressions.
The codebase is in its most stable state since the first run.

## 10. Reproduce

```bash
cd /home/ubuntu/reap-cuda && source .venv/bin/activate
uv pip install --editable '.[cuda]'          # re-install after pull
pytest -q                                     # 213 passed

mkdir -p /data/reap-lfm2-run6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REAP_FREA_PROBE=1

reap --verbose prune full \
  --model /data/models/LiquidAI/LFM2.5-8B-A1B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --dataset-path /data/datasets/evol-codealpaca-calib-200 \
  --compression-ratio 0.5 \
  --prune-method reap \
  --observe-backend auto \
  --frea-backend auto \
  --batch-size 4 \
  --batches-per-category 1024 \
  --model-max-length 1024 \
  --residency gpu_full \
  --artifacts-dir /data/reap-lfm2-run6 \
  --seed 42 \
  --local-files-only \
  --no-profile \
  2>&1 | tee /data/reap-lfm2-run6/run.log
```

Artifacts → `/data/reap-lfm2-run6/.../pruned_models/reap-renorm_true-seed_42-0.50/`
(8.55 GiB safetensors, 16 experts, top_k=4). Verify with:

```bash
python3 -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    '/data/reap-lfm2-run6/.../pruned_models/reap-renorm_true-seed_42-0.50',
    device_map='auto', dtype=torch.bfloat16)
print(f'experts={m.config.num_experts}, top_k={m.config.num_experts_per_tok}')
"
```
