# Run Findings — REAP prune on LiquidAI/LFM2.5-8B-A1B

End-to-end instrumented prune run on the LFM2.5-8B-A1B MoE with the REAP CUDA
codebase, plus every issue, gap, and kernel problem surfaced during testing.

Artifacts for the run live under
`/data/reap-artifacts/LFM2.5-8B-A1B/evol-codealpaca-v1/`
(`perf_report.json`, `perf_report.csv`, `gpu_timeline.csv`, `run.log`,
`observations.pt`, and the pruned checkpoint).

## 1. Run configuration

| Setting | Value |
| --- | --- |
| Model | `/data/models/LiquidAI/LFM2.5-8B-A1B` (`Lfm2MoeForCausalLM`, `model_type=lfm2_moe`) |
| Adapter | `Lfm2MoeModelAdapter` (`src/reap/model_adapters.py`) |
| Dataset | local arrow `evol-codealpaca-calib-200`, first 100 examples |
| Calibration | 100 examples → 40 packed batches, 40,808 tokens, `model_max_length=1024`, `batch_size=1` |
| Residency | `gpu_full` (`device_map="auto"`, no CPU pin, stream-save) |
| Observe backend | `auto` → resolved to `f2` (Triton F5+FREA+F2, with PyTorch fallback) |
| Prune | method `reap`, `compression_ratio=0.5` → prune 16 of 32 experts, keep 16, `top_k=4` |
| Hardware | NVIDIA L4, 23 GiB VRAM, 99 KiB per-block shared memory |
| Software | torch 2.13.0+cu130, transformers 5.14.1, triton 3.7.1, Python 3.12.13 |
| Driver | `scripts/reap_lfm2_run.py` (instrumented, calls the codebase `run()`-level functions) |

Model facts (from `config.json`): 32 routed experts, `num_experts_per_tok=4`,
`use_expert_bias=true`, `norm_topk_prob=true`, `routed_scaling_factor=1.0`,
24 hybrid layers (conv / full_attention / MoE), `num_dense_layers=2` → 22 MoE
layers at indices 2–23.

## 2. End-to-end result: SUCCESS

Outcome: `success`. A complete, loadable HuggingFace checkpoint was produced:

```
pruned_models/reap-renorm_true-seed_42-0.50/
  chat_template.jinja
  config.json                       # num_experts=16, num_experts_per_tok=4
  generation_config.json
  model.safetensors                 # 8.55 GiB (was 16 GiB)
  tokenizer.json
  tokenizer_config.json
```

The checkpoint was verified to load and generate on a fresh process
(`AutoModelForCausalLM.from_pretrained(..., device_map="auto")` →
`model.generate(...)`), confirming `slice_experts` + `update_config` produced a
forward-correct 16-expert model.

### Per-phase performance (`perf_report.json`)

| Phase | Wall (s) | Peak GPU alloc (GiB) | CPU RSS end (GiB) | Notes |
| --- | ---: | ---: | ---: | --- |
| 0 env/backend check | 0.05 | 0.00 | 0.92 | `auto → f2`, triton 3.7.1 |
| 1 tokenizer load | 1.29 | 0.00 | 1.16 | |
| 2 model load (gpu_full) | 137.17 | 16.21 | 0.12 | 8.47 B params, 22 MoE layers |
| 3 dataset load + tokenize | 6.29 | 15.77 | 0.42 | 40 batches / 40,808 tokens |
| 4 observer setup | 0.00 | 15.77 | 0.42 | hook regex `Lfm2MoeSparseMoeBlock` |
| **5 observe (f2 backend)** | **43.21** | **17.04** | 1.21 | 0.93 batch/s, 949 tok/s |
| 5b load observer state | 0.01 | 15.78 | 1.21 | 22 layers, `reap` key present |
| 6 prune (slice + stream save) | 70.69 | 16.00 | 8.68 | 32 → 16 experts |
| 7 smoke test (generate) | 3.36 | 8.64 | 8.76 | `smoke_ok=true` |
| 8 artifact summary | 0.00 | 8.56 | 8.76 | 6 files, 8.569 GiB |
| **Total** | **262.1** | peak **17.04** | | |

Board-level GPU (from `gpu_timeline.csv`, 254 samples @ 1 Hz): peak
`memory.used = 17,688 MiB`, peak `utilization.gpu = 94%`. Observe peak GPU
**never exceeded 17.04 GiB** on the 23 GiB L4.

## 3. Codebase changes made during this run

Three real fixes were required to make the LFM2.5 prune work end-to-end. They
are committed separately (see §6) and are the basis of the GitHub issues filed.

### 3.1 LFM2 router semantics — `src/reap/kernels/router.py` + `observe.py`
LFM2 uses `sigmoid(logits) + expert_bias` → topk-on-scores → gather sigmoid
weights → renorm → `× scaling_factor`. REAP's F5 did softmax+topk and called
the router without `expert_bias` (crash). Added `f5_router_from_module`
(`router.py:141`) that calls the model's own router and rebuilds the CSR pair
tensors; wired in `observe_moe_batch` (`observe.py:137`) gated on
`adapter.adapter_name == "lfm2_moe"`.

### 3.2 F4 weight-cache leak — `src/reap/kernels/observe.py`
`_STACK_CACHE` (`weight_cache.py:22`) was never freed in the full-observer path,
so all 22 layers' stacked weights (~448 MiB each ≈ 9.6 GiB) accumulated on top of
the 15.8 GiB model → OOM at 21.34 GiB. Added `free_cache(moe)` at the end of
`observe_moe_batch` (`observe.py:196`). Peak dropped 21.34 → 17.04 GiB.

### 3.3 `smoke_test` transformers-5.14 compat — `src/reap/pipeline.py`
`apply_chat_template(..., return_tensors="pt").to(...)` raised an empty
`AttributeError` under transformers 5.14. Fixed to `return_dict=True` +
`**inputs` to `generate` (`pipeline.py:274-286`).

## 4. Issues, gaps, and bugs found during testing

1. **LFM2 router semantics unsupported in F5** — sigmoid + expert_bias +
   topk-on-scores; `extract_router_logits` crashes (`router.py:35`); the `loop`
   backend (`observe.py:_loop_activations`) is also wrong for LFM2 (topk on raw
   logits). Fixed only for routed backends (bmm/frea/f2).
2. **F4 weight-cache leak in full-observer path** — `_STACK_CACHE` never freed
   (`weight_cache.py:22`); OOM on tight-VRAM GPUs. Fixed via `free_cache` per
   `observe_moe_batch`, at the cost of re-stacking every batch×layer.
3. **`smoke_test` incompatible with transformers 5.14** —
   `apply_chat_template(...).to()` breaks (`pipeline.py:268`). Fixed.
4. **FREA Triton kernel falls back on the L4** — 1,760 fallbacks in this run
   (every batch×layer). See §5.
5. **No "Triton actually ran" summary / silent fallback** — `reap kernels`
   reports `auto backend: f2` but FREA (the heaviest kernel) ran zero Triton
   launches. Only DEBUG logs reveal it.
6. **F2 Triton reduce accumulates in fp32 vs documented fp64** —
   `triton_reduce.py:105-107,147-149`; precision divergence vs the PyTorch path
   (`_scatter_pytorch` uses fp64 `index_add_`).
7. **`prefer_triton_for` too permissive** — `triton_utils.py:55` checks only
   CUDA + numel>0 + dtype; no device-capability / shared-mem / profitability
   check, so it greenlights FREA on hardware where it can't launch.
8. **No offline/local dataset path in the CLI** — `--dataset` must match a
   `DATASET_REGISTRY` key (`data.py:251`) and is passed to `load_dataset`
   (`data.py:217`), requiring the HF hub or a pre-built cache snapshot. No
   `--dataset-path` for local arrow/jsonl.
9. **`create_results_directory` hardcodes `./artifacts`** (`pipeline.py:35`) —
   no configurable output root; caused a mid-save "No space left on device" on
   the 73 GiB root partition. Relocated to `/data`.
10. **Unbounded `transformers`/`torch` ceiling** — `pyproject.toml` floors
    `torch>=2.10, transformers>=5.5` with no upper pin; the smoke_test API
    break shows newer transformers changed `apply_chat_template` semantics.
11. **Misspelled state key** — `router_logit_similiarity` (sic) in
    `pruning_metrics.py`; docs say "do not rename without migration." Not hit
    in this prune-only run.

## 5. Custom Triton kernel problems (extreme detail)

Three Triton kernels ship: F5 softmax (`triton_softmax.py`), FREA SwiGLU
(`triton_frea.py`), F2 scatter-reduce (`triton_reduce.py`).

### 5.1 FREA SwiGLU (`triton_frea.py`) — the heaviest kernel; never ran on the L4

**Evidence**: 1,760 `Triton frea fallback → PyTorch` log lines in `run.log`,
all with the same reason:
`out of resource: shared memory, Required: 139264, Hardware limit: 101376.
Reducing block sizes or num_stages may help.`

**Shared-memory arithmetic.** In `_frea_triton_impl` (`triton_frea.py:110-112`):
```python
block_h = max(_MIN_DOT, min(next_power_of_2(h), 128))      # h=2048     → 128
block_i = max(_MIN_DOT, min(next_power_of_2(i_dim), 128))   # i_dim=1792 → 128
block_n = 16
```
The kernel does three `tl.dot` per inner loop (`x[16×128] @ trans(wg[128×128])`,
same for `wu`, then `act @ trans(wd)`). Each weight tile is 128×128 fp32 =
65,536 B = 64 KiB; `wg` and `wu` are both live in the `h0` loop → ~128 KiB +
`x` tile (8 KiB) + accumulators + pipeline staging = **139,264 B (136 KiB)**.
The L4 (AD104, cc 8.9) default per-block dynamic shared-memory limit is
**101,376 B (99 KiB)**. 136 KiB > 99 KiB → launch fails.

**Problem A1 — block sizes are hardcoded, not hardware-aware.** The cap is
`min(next_power_of_2(dim), 128)` with `_MIN_DOT=16` (`triton_frea.py:28`). There
is no query of `torch.cuda.get_device_properties(0).shared_memory_per_block` /
`shared_memory_per_block_optin`. So on any GPU with < 136 KiB per-block shared
mem (L4, T4, consumer Ampere/Ada) FREA cannot launch. The kernel was evidently
tuned only on a big-shared-mem datacenter GPU (L40S/A100).

**Problem A2 — no opt-in to extended shared memory.** Ampere/Ada can expose up
to ~164 KiB of dynamic shared memory per block via
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize, …)`, but Triton does not
enable this by default and the code never requests it. So the hardware *could*
hold 136 KiB with an opt-in, but the kernel runs under the 99 KiB default and
fails.

**Problem A3 — `num_stages` is not set.** The launch (`triton_frea.py:216-238`)
sets `num_warps = 4 if h <= 1024 else 8` (8 for h=2048) but never `num_stages`.
Triton's default `num_stages` for `tl.dot`-heavy loops (typically 3–4)
**multiplies** the shared-memory footprint (each pipeline stage duplicates
operand buffers). The error message itself says *"Reducing block sizes or
`num_stages` may help"* — the code does neither.

**Problem A4 — the eligibility gate lies.** `_triton_frea_supported`
(`triton_frea.py:35`) checks only: SiLU, `triton_runtime_available()`,
`prefer_triton_for(...)`, and `h,i_dim ≥ 16`. It does **not** pre-flight
shared-memory feasibility. So on the L4 it returns `(True, "")` every call,
then `_frea_triton_impl` is entered, the launch is attempted, and it throws.
The gate and the launch are inconsistent.

**Problem A5 — no failure memoization → 1,760 wasted launch attempts.** Because
the gate always returns True and the failure is caught fresh each call
(`frea_triton_activations`, `triton_frea.py:56-69`), every batch×layer re-enters
`_frea_triton_impl`, re-attempts the launch, re-fails, and falls back. Triton
caches the *compilation*, but the *launch attempt* (which fails on the
shared-mem allocation) is repeated 1,760 times. No "once bitten" flag.

**Problem A6 — the fallback is silent to the user.** `log_triton_fallback`
(`triton_utils.py:71`) emits DEBUG only. `reap kernels` reports
`auto backend: f2`. Nothing at INFO/WARN tells the user FREA ran zero Triton
launches. The performance contract ("f2 = Triton FREA + F2") is silently broken.

**Net effect**: for the FREA stage the run got **neither** the Triton speedup
nor a memory win — it ran the pure-PyTorch grouped path
(`routed_expert_activations_grouped` in `bmm.py`), which loops over experts in
Python and materializes per-expert intermediates.

### 5.2 F2 scatter-reduce (`triton_reduce.py`) — runs, but with a precision divergence

This one **did** run on Triton (zero `f2_reduce fallback` log lines). 1D grid
(`grid = (n,)`, n = n_pairs ≈ 4096), `BLOCK_H = min(next_power_of_2(h), 128) =
128`, `num_warps=2`. No `tl.dot` → tiny shared mem → fits the L4.

**Problem B1 — fp32 atomic accumulation vs. fp64 on the PyTorch path.** The
Triton kernel allocates the accumulators as fp32
(`triton_reduce.py:105-107`) and updates them with `tl.atomic_add` in fp32
(`triton_reduce.py:147-149`), then casts to fp64 at return
(`ean_sum_f.to(torch.float64)`, line 172). The PyTorch path
(`_scatter_pytorch`, `triton_reduce.py:56`) uses fp64 `index_add_`
(`triton_reduce.py:67-69`). The state schema
(`docs/observation-and-metrics.md`) says `ean_sum (E,) fp64` and
`weighted_ean_sum (E,) fp64`. So:
- The Triton path violates the documented fp64 contract: it accumulates 40
  batches × ~4096 pairs of L2-norm sums in fp32 with non-deterministic
  atomic-add ordering, then upcasts.
- Over many batches, fp32 rounding + non-associative atomic order can drift
  `ean_sum`/`weighted_ean_sum` vs the fp64 path. The `reap` metric (Welford
  mean of `mean(‖y‖·w)`) is derived from these sums, so the Triton and PyTorch
  backends can produce slightly different expert rankings — a
  backend-dependent saliency divergence (not a crash).
- `batch_max` uses `tl.atomic_max` in fp32 (fine — max is not order-sensitive).

**Problem B2 — permissive `prefer_triton_for`.** `scatter_pair_stats`
(`triton_reduce.py:20`) gates on `triton_runtime_available() and
prefer_triton_for(pair_out)`. `prefer_triton_for` (`triton_utils.py:55`) only
checks `is_cuda`, `numel()>0`, dtype. So F2 always tries Triton regardless of
profitability; for tiny batches the launch overhead could exceed PyTorch
`index_add_`.

**Problem B3 — grid design leaves parallelism on the table.** One program per
pair, each looping sequentially over H=2048 in 128-chunks (16 iterations). A
2D grid (pairs × H-blocks) with a cross-block reduction would expose more
parallelism. Not a bug; a throughput ceiling.

### 5.3 F5 row-softmax (`triton_softmax.py`) — not exercised for LFM2

`softmax_rows` (`triton_softmax.py:19`) tries `_softmax_triton` when
`prefer_triton_for(logits) and triton_runtime_available()`, with an
`E > BLOCK_E` → `F.softmax` fallback. Simple row-wise kernel; would fit the L4.

**Problem C1 — bypassed entirely for LFM2.** `f5_router_from_module` calls the
model's own router (correct sigmoid+expert_bias semantics), so
`f5_router`/`softmax_rows` is never invoked for LFM2. On this run the F5 Triton
softmax is dead code. For softmax-router models (Qwen3/Mixtral/Llama4) it would
run; LFM2's sigmoid routing means F5's softmax is the wrong tool — the codebase
had no LFM2 router path before this fix.

### 5.4 Cross-cutting kernel problems

- **`prefer_triton_for` is too permissive** (`triton_utils.py:55`): CUDA +
  numel>0 + dtype. No device-capability, shared-mem feasibility, or
  profitability check. It greenlights FREA on the L4 even though the kernel
  provably can't launch. Root cause of the silent-fallback problem.
- **No pre-flight shared-memory check.** The FREA gate could compute the
  required shared mem from `block_h/block_i/block_n + num_stages` and compare
  to the device limit, returning `(False, "shared mem …")` *before* attempting
  the launch. Instead the gate returns True and the launch fails — every call.
- **No "Triton actually ran" accounting.** No counter/summary emitted at end
  of run (e.g., "FREA: 0 Triton / 1760 PyTorch; F2: 880 Triton / 0 PyTorch;
  softmax: 0/0 bypassed"). The user must grep DEBUG logs to discover the
  heaviest kernel never ran.
- **Fallback is correct but the performance contract is silently broken.**
  Every fallback path produces numerically valid results (parity-tested), so
  correctness tests pass. But `select_observe_backend → "f2"` and reality
  (PyTorch-FREA + Triton-F2) diverge with no signal.

## 6. Suggested next steps

1. Make FREA actually launch on the L4: cap `block_h`/`block_i` to 64 (→ ~68
   KiB, fits), or query the device shared-mem limit and auto-scale, or set
   `num_stages=1`, or opt into 164 KiB dynamic shared mem. Add a pre-flight
   shared-mem check in `_triton_frea_supported`.
2. Memoize kernel failures so FREA doesn't re-attempt the failing launch 1,760
   times.
3. Emit a per-run "Triton ran / fell back" summary at INFO.
4. Fix F2 reduce to accumulate in fp64 (or document the fp32 divergence).
5. Add a `--dataset-path` option for local arrow/jsonl and a configurable
   artifacts root.
6. Fix the `loop` backend for LFM2 (currently still topk-on-raw-logits).

## 7. Reproducing

```bash
source /home/ubuntu/reap-cuda/.venv/bin/activate
cd /home/ubuntu/reap-cuda
python scripts/reap_lfm2_run.py
# reports: /data/reap-artifacts/LFM2.5-8B-A1B/evol-codealpaca-v1/perf_report.json
```