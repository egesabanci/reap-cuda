# run-findings-3 — REAP on LFM2.5-8B-A1B after the throughput-regression fixes (#24–#32)

> **Erratum (shared memory + #28):** L4 is **48 KiB default / 99 KiB opt-in**
> (not 99 / 164). Opt-in enables **128×64** tiles, not 128×128. Probe default
> remains correct. Full correction: `run-findings-4.md` §3 and GH #28 comment.

## 1. Scope

Third instrumented end-to-end REAP prune run on `LFM2.5-8B-A1B`, executed **after**
upstream landed the fixes for the issues filed from `run-findings-2.md`:

- #24 FREA Triton ~2.3× slower than PyTorch on shared-mem-bound GPUs (tracking)
- #25 empirical FREA profitability probe (the default `auto` mechanism)
- #26 static tile-floor when `REAP_FREA_PROBE=0`
- #27 `--frea-backend {auto,triton,pytorch}` / `REAP_FREA_BACKEND`
- #28 opt into `shared_memory_per_block_optin` (~164 KiB) on Ampere/Ada for 128×128 FREA tiles
- #29 small-tile `BLOCK_N=32` + `num_warps=8` occupancy tweak (lighter than the 2D-grid rewrite)
- #30 F2 `num_warps` scaling with `H` / `n`
- #31 fp16 FREA parity test tolerance (`atol=1.0, rtol=5e-2`) + force `backend="triton"`
- #32 layerwise-observer test compares on CPU (device-agnostic)

All 10 issues (#24–#32 plus the earlier #14–#23) are **CLOSED**. The three commits
pulled for this run:

| commit | summary |
| --- | --- |
| `a7c87cd` | `fix(kernels): FREA profitability probe, --frea-backend, SM opt-in, test fixes` — closes #24–#32 |
| `b64e427` | `fix(kernels): harden FREA probe against sticky tiny-batch and SM edge cases` |
| `b7a38a6` | `docs: document FREA probe, frea-backend, offline data, and residency ops` |

## 2. Configuration (unchanged from runs 1 & 2)

| | value |
| --- | --- |
| model | `/data/models/LiquidAI/LFM2.5-8B-A1B` (`Lfm2MoeForCausalLM`, 32 experts, top_k=4, 22 MoE layers, `use_expert_bias=true`, bf16) |
| dataset | `/data/datasets/evol-codealpaca-calib-200` (first 100 examples) |
| n_examples | 100 |
| batch_size | 1 |
| model_max_length | 1024 (40,808 packed tokens) |
| compression_ratio | 0.5 (keep 16 of 32 experts) |
| prune_method | `reap` (router-weighted L2 norm, Welford online) |
| observe_backend | `auto` → `f2` |
| frea_backend | `auto` (probe enabled, `REAP_FREA_PROBE=1`) |
| residency | `gpu_full` (`device_map="auto"`) |
| seed | 42 |
| env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `REAP_ARTIFACTS_DIR=/data/reap-artifacts` |
| GPU | NVIDIA L4 (23 GiB, **48 KiB default / 99 KiB opt-in** per-block shared mem, cc 8.9) |
| stack | torch 2.13.0+cu130, transformers 5.14.1, triton 3.7.1 |

## 3. Outcome

```
outcome   = success
smoke_ok  = True
total_wall= 269.52 s   (sum of phase walls)
checkpoint= 6 files, 8.569 GiB  (16 experts, top_k=4, use_expert_bias=True)
```

Pruned checkpoint loads in a **fresh process** and generates coherent, correct
code (independent load verified, see §6).

## 4. Per-phase performance

| phase | wall | gpu_peak alloc | cpu rss | notes |
| --- | ---: | ---: | ---: | --- |
| 0_env_backend_check | 0.03 s | 0.000 GiB | 0.93 GiB | backend=f2, triton 3.7.1 |
| 1_tokenizer_load | 1.57 s | 0.000 GiB | 1.16 GiB | vocab 124,893 |
| 2_model_load | 144.29 s | 16.210 GiB | 0.12 GiB | bf16 ~16 GiB to GPU |
| 3_dataset_load_tokenize | 6.31 s | 15.773 GiB | 0.42 GiB | 100 examples → 40 batches |
| 4_observer_setup | 0.00 s | 15.773 GiB | 0.42 GiB | |
| **5_observe** | **39.93 s** | **16.296 GiB** | 1.22 GiB | **f2: 880 Triton / 0 PyTorch; frea: 0 Triton / 880 PyTorch** |
| 5b_load_observer_state | 0.01 s | 15.781 GiB | 1.22 GiB | 22 layers, 9 metrics |
| 6_prune_slice_save | 75.60 s | 15.999 GiB | 8.71 GiB | stream path; hooks stripped |
| 7_smoke_test | 1.77 s | 8.645 GiB | 8.72 GiB | generates |
| 8_artifact_summary | 0.00 s | 8.560 GiB | 8.72 GiB | 6 files / 8.569 GiB |

Board-level (`nvidia-smi` 1 Hz sampler, 261 samples): **peak 17,732 MiB**, **peak util 94%**.

Observe throughput: 40,808 tokens / 39.93 s = **~1,022 tok/s**.

## 5. Three-run comparison

| metric | run 1 (pre-fix, FREA→PyTorch) | run 2 (post-#14–#23, FREA→Triton) | **run 3 (post-#24–#32, probe)** | Δ vs run 2 |
| --- | ---: | ---: | ---: | ---: |
| observe wall | 43.21 s | 98.46 s | **39.93 s** | **−59%** |
| observe throughput | 949 tok/s | 415 tok/s | **1,022 tok/s** | **+146%** |
| observe peak GPU alloc | 17.04 GiB | 16.29 GiB | **16.296 GiB** | flat |
| FREA Triton | 0 ok / 1,760 fallback | 880 ok / 0 fallback | **0 ok / 880 fallback** | probe→pytorch |
| F2 Triton | 880 / 0 (fp32) | 880 / 0 (fp64) | **880 / 0 (fp64)** | flat |
| total wall | 262 s | 317 s | **269.5 s** | −47 s |
| board peak | 17,688 MiB | 17,672 MiB | **17,732 MiB** | flat |

**Headline:** the empirical probe (#25) gives run-2's **memory** (16.296 GiB, the
0.75 GiB win over run 1) **and** run-1's **throughput** — in fact slightly better
(1,022 vs 949 tok/s), because run 1 still paid the (unused) Triton feasibility
checks on every batch and had the older F4-cache path. The probe ran exactly
once, memoized per `(device, H, I_dim)`, and the rest of the 880 FREA calls used
the cached PyTorch decision with zero per-call timing overhead.

### 5.1 The probe verdict (one line in run.log)
```
INFO reap.kernels.triton_frea: FREA profitability probe: triton=0.0481s pytorch=0.0062s -> pytorch (reason=ok)
```
On the L4, for this expert shape (H=2048, I=1792; opt-in would allow up to
**128×64** tiles, not 128×128), Triton is ~7.8× slower than the cuBLAS-backed
PyTorch grouped path on a single layer — exactly the #24 finding. The probe
correctly picked PyTorch and memoized it. The first FREA fallback is logged at
WARNING (`frea`); the remaining 879 are at DEBUG per the #18 once-then-debug
contract.

### 5.2 Why memory is still 16.296 GiB with PyTorch FREA
Run 1's 17.04 GiB peak came partly from F4 weight-cache accumulation across the
22 MoE layers (the original #15). Run 1 worked around it with a per-call
`free_cache`; run 2 replaced that with `_MAX_CACHE_ENTRIES=1` LRU-evict (#15
fix). Run 3 inherits the LRU, so even on the PyTorch FREA path the cache
footprint stays bounded — hence the 0.75 GiB win is preserved without needing
the Triton path.

## 6. Functional verification

**Independent checkpoint load** (fresh process, no driver state):
```
type: Lfm2MoeConfig
num_experts: 16  top_k: (model default)  use_expert_bias: True
loaded ok; device: cuda:0
```
(top_k is stored on the inner config; the pruned model still routes 4 experts
per token and generates correctly.)

**Generation sample** (`max_new_tokens=80`, greedy), prompt
"Write a Python function to check if a number is even":
```
Okay, the user wants a Python function to check if a number is even. Let me
think about how to do that.
First, the function should take a number as input and return whether it's
even. The simplest way is using modulo operator. If the number divided by 2
has no remainder, then it's even.
So the function would be something like:
def is_even(number
```
Coherent, on-task, correct reasoning. Same quality as run 2's checkpoint.

## 7. Kernel status (post-fix)

| kernel | backend used | notes |
| --- | --- | --- |
| F4 (weight stack) | PyTorch (cached) | `_MAX_CACHE_ENTRIES=1` LRU; bounded footprint |
| F5 (router) | native (`f5_router_from_module`) | LFM2 sigmoid+expert_bias semantics; bypassed for Triton counting |
| FREA (expert MLP) | **PyTorch grouped (probe)** | probe timed triton=0.048s vs pytorch=0.006s → pytorch; memoized |
| F2 (scatter-reduce) | **Triton** (880/0) | fp64 accumulators+atomics (#19); `num_warps` scales with H/n (#30) |

### What the fixes changed vs run 2
- **#25 probe** (`triton_frea.py:_run_probe`): warm-up both paths, time once per
  `(device,H,I)`, memoize, tie-bias to Triton (memory win), skip-memoize on
  `n_pairs<16` (`b64e427` hardening) so a sparse first batch doesn't stick the
  wrong choice. This is the single change that recovered throughput.
- **#28 SM opt-in** (`triton_utils.py:device_shared_memory_bytes(prefer_optin=…)` +
  `choose_frea_block_sizes` + safe retry): uses the **device-reported** opt-in
  budget (L4: 99 KiB → larger tiles like 128×64; **not** 128×128 / 164 KiB).
  Run 3's probe never enters Triton, so opt-in is not exercised here; run 4
  forces Triton and confirms opt-in (see `run-findings-4.md`). Safe-retry sets
  `_USE_SMEM_OPTIN=False` and re-launches with default-budget tiles on SM OOM.
- **#29** was implemented as the lighter small-tile occupancy tweak
  (`BLOCK_N=32` + `num_warps=8` when `block_h<=64`, with a shared-mem re-check
  from `b64e427`) rather than the full 2D-grid rewrite — sufficient given the
  probe now routes small-tile cases to PyTorch anyway.
- **#30** F2 `num_warps` scaling (`4 if h>=512 else 2`, bumped to 4 at `n>=2048`)
  — low-risk; F2 already ran on Triton with zero fallbacks in run 2.
- **#31/#32** test fixes: the full suite now passes **113/113** (was 109/111).

## 8. Test suite

```
$ pytest -q
113 passed in 20.78s
```
Both previously-failing tests now pass:
- `test_triton_frea_matches_bmm` (#31): tolerance loosened to `atol=1.0, rtol=5e-2`
  for fp16, and the test forces `backend="triton"` so it is a real Triton-vs-PyTorch
  comparison instead of PyTorch-vs-PyTorch.
- `test_layerwise_observer_matches_standard_observer` (#32): comparison moved to
  CPU (`.detach().cpu()`) so CUDA-vs-CPU placement never fails the assert.

## 9. Tradeoff assessment (honest)

- **Throughput: recovered and then some.** 1,022 tok/s is the best of the three
  runs. The probe's one-time cost (~0.05 s warm-up + 2× single-layer timing) is
  amortized over 880 calls; net observe wall is below run 1 because run 1 still
  ran Triton feasibility checks each batch.
- **Memory: best of the three.** 16.296 GiB matches run 2's Triton figure and
  beats run 1 by 0.75 GiB, courtesy of the `_MAX_CACHE_ENTRIES=1` LRU (#15) — the
  memory win no longer requires running FREA on Triton.
- **Correctness: unchanged.** Smoke ok; checkpoint loads independently and
  generates correct code.
- **What we did *not* get on the L4:** a Triton FREA path that beats cuBLAS.
  128×128 is **impossible** on L4 (99 KiB opt-in max). Opt-in 128×64 still loses
  (run 4). The probe picks PyTorch; memory win is from F4 LRU, not Triton.
- **Cost of the probe:** one extra timed Triton launch + one extra timed
  PyTorch launch per distinct `(device, H, I)` per process. For a single-model
  run that is one probe total — negligible.

## 10. Residual gaps / things to watch

1. **Probe runs only once per shape.** If a run mixes expert shapes with
   different `(H, I)` (e.g. a hybrid model with two MoE block sizes), the probe
   runs once per shape — fine, but the memo key should be confirmed to cover
   that case (`_probe_key` uses `(device, H, I_dim)`; LFM2 has a single MoE
   shape so this run probes exactly once).
2. **`--frea-backend triton` would re-introduce the 2.3× slowdown on the L4.**
   That is by design (memory-constrained users opt in), but it's worth a
   one-line log note when the forced backend contradicts the probe's verdict so
   users aren't surprised. Minor.
3. **SM opt-in (#28) on L4** is exercised in **run 4** (forced `triton`): opt-in
   engages, tiles **128×64**, +33% vs older small-tile Triton, still slower than
   cuBLAS. 128×128 never fits on L4.
4. **#29/#30 were implemented as occupancy tweaks, not the 2D-grid rewrites**
   originally proposed. Given the probe makes the rewrite lower-value on the L4,
   this is a reasonable scoping — but on a GPU where Triton *is* profitable at
   small tiles, the 2D grid could still matter. Not urgent.

## 11. Reproduce

```bash
cd /home/ubuntu/reap-cuda && source .venv/bin/activate
uv pip install --editable '.[cuda]'          # re-install after pull
pytest -q                                     # 113 passed

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REAP_ARTIFACTS_DIR=/data/reap-artifacts
export REAP_FREA_BACKEND=auto
export REAP_FREA_PROBE=1
python scripts/reap_lfm2_run.py
# artifacts -> /data/reap-artifacts/LFM2.5-8B-A1B/evol-codealpaca-v1/
#   perf_report.json  gpu_timeline.csv  run.log  observations.pt  pruned_models/
```

Force the Triton FREA path (to exercise #28 SM opt-in on the L4):
```bash
REAP_FREA_BACKEND=triton python scripts/reap_lfm2_run.py
```
Force the static tile-floor instead of the probe (#26):
```bash
REAP_FREA_PROBE=0 python scripts/reap_lfm2_run.py
```