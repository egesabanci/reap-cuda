# Run Findings 2 — REAP prune on LiquidAI/LFM2.5-8B-A1B (post-fix re-run)

Second end-to-end instrumented prune run on the LFM2.5-8B-A1B MoE, this time
against the **fixed codebase** (commits `b01176d` "resolve EC2 run-findings
issues (#14–#23)" and `2150fdf` "tighten run-findings residual gaps"). The
purpose of this run was to verify that the ten filed-and-closed issues actually
hold end-to-end, that the custom Triton kernels run seamlessly on the L4, and to
characterize the new performance/memory tradeoff that emerged.

Artifacts for this run live under
`/data/reap-artifacts/LFM2.5-8B-A1B/evol-codealpaca-v1/`
(`perf_report.json`, `perf_report.csv`, `gpu_timeline.csv`, `run.log`,
`observations.pt`, and the pruned checkpoint). The previous run's artifacts were
backed up to `..._prevrun_1784225551/` for an apples-to-apples comparison.

## 1. Run configuration

Identical to the first run (same model, same 100 calibration examples, same
compression, same hardware) so the only variable is the codebase fixes:

| Setting | Value |
| --- | --- |
| Model | `/data/models/LiquidAI/LFM2.5-8B-A1B` (`Lfm2MoeForCausalLM`, `model_type=lfm2_moe`) |
| Adapter | `Lfm2MoeModelAdapter` (`src/reap/model_adapters.py`) |
| Dataset | local arrow `evol-codealpaca-calib-200`, first 100 examples |
| Calibration | 100 examples → 40 packed batches, 40,808 tokens, `model_max_length=1024`, `batch_size=1` |
| Residency | `gpu_full` (`device_map="auto"`, no CPU pin, stream-save) |
| Observe backend | `auto` → resolved to `f2` (Triton F5+FREA+F2, with PyTorch fallback) |
| Prune | method `reap`, `compression_ratio=0.5` → prune 16 of 32 experts, keep 16, `top_k=4` |
| Hardware | NVIDIA L4, 23 GiB VRAM, **99 KiB per-block shared memory** |
| Software | torch 2.13.0+cu130, transformers 5.14.1, triton 3.7.1, Python 3.12.13 |
| Driver | `scripts/reap_lfm2_run.py` (instrumented; now also captures the Triton usage summary) |
| Codebase | post-fix `b01176d` + `2150fdf` (issues #14–#23 closed) |

Model facts unchanged: 32 routed experts, `num_experts_per_tok=4`,
`use_expert_bias=true`, `norm_topk_prob=true`, 24 hybrid layers,
`num_dense_layers=2` → 22 MoE layers at indices 2–23.

## 2. Codebase state going into this run

All ten issues filed from the first run are now **CLOSED** by `b01176d`
("Closes #14 #15 #16 #17 #18 #19 #20 #21 #22 #23"), with a follow-up tightening
commit `2150fdf`. The fixes relevant to this run's behavior:

- **#14 LFM2 router** — `prefers_native_router` (model-agnostic: detects
  `expert_bias` buffer / `use_expert_bias` / adapter flag / `lfm2_moe` name)
  routes non-softmax MoEs through `f5_router_from_module` (calls the model's own
  router). Wired for **both** routed and `loop` backends.
- **#15 F4 cache leak** — `_STACK_CACHE` now hard-bounded to **1 entry**
  (`_MAX_CACHE_ENTRIES=1`, LRU-evict) instead of the per-call `free_cache` hack.
- **#16 smoke_test** — `return_dict=True` + `**inputs` + `pad_token_id`.
- **#17 FREA shared-mem** — `choose_frea_block_sizes` + `estimate_frea_shared_bytes`
  auto-tile to the device's `shared_memory_per_block`; `num_stages=2`;
  memoized permanent disable (`_DISABLED`) so a doomed launch is attempted once.
- **#18 silent fallback** — `record_triton_ok` / `log_triton_fallback`
  (WARN once → DEBUG) / `triton_usage_snapshot` / `format_triton_usage_summary`
  emitted at INFO.
- **#19 F2 fp64** — accumulators now `torch.float64`, atomics cast to `tl.float64`.
- **#20 prefer_triton_for** — `shared_mem_feasible` + `device_shared_memory_bytes`
  + min-numel profitability gate.
- **#21 offline dataset** — `_load_local_dataset` + `--dataset-path` CLI.
- **#22 artifacts root** — `resolve_artifacts_root` + `--artifacts-dir` /
  `REAP_ARTIFACTS_DIR`.
- **#23 version ceilings** — `torch>=2.10,<2.15`, `transformers>=5.5,<6`.

The `2150fdf` follow-up tightened: softmax `E>BLOCK_E` fallback no longer counted
as Triton-ok; native-router detection uses actual top-k width; profile path is
`device_map`-safe via `_primary_device`; layerwise-merge observe emits the Triton
summary; FREA memo disables only on a real shared-mem failure.

A hermetic regression suite, `tests/test_run_findings_fixes.py` (13 tests),
covers all ten fixes and passes.

## 3. End-to-end result: SUCCESS

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

The checkpoint was verified on a **fresh process**
(`AutoModelForCausalLM.from_pretrained(..., device_map="auto")` →
`model.generate(...)`): `num_experts=16`, `top_k=4`, `use_expert_bias=True`,
forward + generate work.

### Per-phase performance vs. the first run

| Phase | This run (fixed) | First run | Δ wall | Δ peak GPU |
| --- | --- | --- | ---: | ---: |
| 0 env/backend check | 0.03s | 0.05s | — | — |
| 1 tokenizer load | 1.30s | 1.29s | — | — |
| 2 model load (gpu_full) | 137.14s, 16.21 GiB | 137.17s, 16.21 GiB | — | — |
| 3 dataset load + tokenize | 6.02s, 40 batches / 40,808 tok | 6.29s | — | — |
| 4 observer setup | 0.00s | 0.00s | — | — |
| **5 observe (f2 backend)** | **98.46s, peak 16.29 GiB, 415 tok/s** | 43.21s, 17.04 GiB, 949 tok/s | **+55.3s (+128%)** | **−0.75 GiB** |
| 5b load observer state | 0.01s | 0.01s | — | — |
| 6 prune (slice + stream save) | 71.76s, 16.0 GiB | 70.69s | — | — |
| 7 smoke test (generate) | 1.97s, ok | 3.36s, ok | −1.4s | — |
| 8 artifact summary | 0.00s, 6 files, 8.569 GiB | 0.00s | — | — |
| **Total** | **316.69s** | 262.08s | +54.6s | — |
| Board peak (nvidia-smi) | 17,672 MiB, 94% util | 17,688 MiB | flat | — |

Board-level GPU (from `gpu_timeline.csv`, 307 samples @ 1 Hz): peak
`memory.used = 17,672 MiB`, peak `utilization.gpu = 94%` — essentially flat vs.
the first run and well under the 22.5 GiB L4 limit.

## 4. Triton kernel status — the headline improvement

The new INFO-level Triton usage summary (issue #18 fix), captured by the driver
at the end of observe:

```
f2_reduce: 880 Triton / 0 PyTorch
frea:      880 Triton / 0 PyTorch
```

- **FREA: 880 Triton launches, 0 PyTorch fallbacks.** This is the big one. In
  the first run FREA fell back to pure-PyTorch grouped `F.linear` on **every**
  batch×layer — 1,760 fallbacks — because the 128×128 tile (136 KiB) exceeded the
  L4's 99 KiB per-block shared-mem limit. The #17 fix
  (`choose_frea_block_sizes` auto-tiles to the device's shared-mem limit +
  `num_stages=2` + a preflight `shared_mem_feasible` check + memoized permanent
  disable) now picks tile sizes that fit, so FREA actually launches on the L4.
  880 = 40 batches × 22 MoE layers.
- **F2 scatter-reduce: 880 Triton, 0 fallbacks.** Same as the first run (F2's
  tiny shared mem always fit), now accumulating in **fp64** (issue #19 fix), so
  the saliency sums match the PyTorch path and the documented state schema.
- **F5 row-softmax: bypassed for LFM2.** Correct — LFM2's sigmoid+expert_bias
  routing goes through the native-router path (`f5_router_from_module`,
  issue #14), so F5 softmax is not the right tool and is not invoked.
- **Zero** `fallback → PyTorch` DEBUG lines in `run.log`; **zero**
  `out of resource / shared memory` errors. The only WARNINGs are benign
  dataset-packing notes ("Not enough samples to pack last sequence…").

So the kernel contract now holds: `reap kernels` reports `auto backend: f2`, and
**f2 actually means Triton FREA + Triton F2** on the L4, with no silent
fallback. Issue #18's observability fix is what made this verifiable.

## 5. The performance-drop / memory-gain tradeoff

This is the central finding of the re-run. Observe is now **2.28× slower**
(98.46s vs 43.21s; 415 vs 949 tok/s) but uses **0.75 GiB less peak GPU**
(16.29 vs 17.04 GiB).

### Why it happened

In the first run, FREA's Triton launch failed (136 KiB > 99 KiB shared mem) and
fell back to `routed_expert_activations_grouped` — a pure-PyTorch path that loops
over the 32 experts and calls **cuBLAS** `F.linear` for each. cuBLAS GEMMs on
2048×1792 expert weights are highly optimized on the L4, so the "fallback" was
actually fast (43.2s).

Now FREA runs on Triton, but the auto-tile logic had to shrink the tiles to fit
the L4's 99 KiB limit. The first-run failure case was `block_h=block_i=128`
(139,264 B required). `choose_frea_block_sizes` walks the tile sizes down until
`shared_mem_feasible` passes, landing on **64×64** tiles (~34 KiB for the two
live weight tiles + staging). Smaller tiles mean **4× more tile iterations** per
pair (the `h0`/`i0` loops run 2048/64 × 1792/64 ≈ 32×28 iterations instead of
16×14). On the L4, that extra iteration overhead outweighs the fusion benefit,
so Triton-FREA-on-small-tiles is slower than cuBLAS-grouped.

The memory win is the flip side: the Triton FREA kernel streams weight tiles and
does not materialize per-expert intermediate activations the way the grouped
PyTorch path does, so peak GPU allocated drops 17.04 → 16.29 GiB.

### Net characterization

- **Throughput**: regression on shared-mem-bound GPUs (L4/T4/consumer
  Ampere/Ada). On a big-shared-mem GPU (L40S/A100 with 164 KiB/block), the
  128×128 tiles would fit and FREA-Triton would likely beat the PyTorch
  fallback — the kernel was evidently tuned for that class.
- **Memory**: improvement on every GPU. 0.75 GiB is meaningful on a 22.5 GiB
  L4: it is the difference between fitting `model_max_length=2048` vs `1024`, or
  fitting a slightly larger model.
- **Correctness**: unchanged. FREA parity is numerically fine (the test failure
  is a too-tight fp16 tolerance, see §7); F2 now matches the fp64 schema.

So the #17 fix traded throughput for memory on the L4. This is not a bug — both
paths produce correct results — but it is a **silently broken performance
contract for throughput-focused users on shared-mem-bound GPUs**. The
`prefer_triton_for` gate (#20) now checks **feasibility** but not
**profitability**: it launches FREA on Triton because it *can*, even though
PyTorch is ~2.3× faster here.

## 6. Functional verification of the pruned model

### 6.1 Independent load + generate
Fresh process, `from_pretrained(..., device_map="auto")`:
`num_experts=16, num_experts_per_tok=4, use_expert_bias=True`. Generate on
"What is 2+2?" produced coherent (if low-quality) text — forward + sampling
path intact.

### 6.2 Coding question battery (5 prompts, executed, not eyeballed)

Asked the pruned model 5 coding questions with `max_new_tokens=700` (greedy) so
the chain-of-thought + final code completed fully, then **executed** each
generated function against test cases:

| # | Task | Verdict | Detail |
| --- | --- | --- | --- |
| Q1 | `is_prime(n)` | ❌ BROKEN | Crashes `TypeError: 'float' object cannot be interpreted as an integer` for every n ≥ 3. The model wrote `range(3, math.sqrt(n)+1, 2)` with a float bound and never cast to int, and omitted the even-number guard it described. Only `n<2` and `n==2` base cases work. |
| Q2 | `bubble_sort(arr)` | ✅ CORRECT | In-place, ascending, early-exit. Passes unsorted/duplicate/empty. |
| Q3 | `fib(n)` | ✅ CORRECT | Iterative; `fib(20)=6765` ✓. |
| Q4 | `count_words(text)` | ✅ CORRECT | `re.split(r"\W+", text.lower())`; passes case-insensitivity + repeats. |
| Q5 | `flatten(lst)` | ✅ CORRECT | Recursive, arbitrary depth; `flatten([1,[2,[3,4],5],6]) == [1,2,3,4,5,6]` ✓. |

**4 of 5 generated functions pass real execution.** The one failure (Q1) is the
classic reasoning-vs-code gap: the chain-of-thought correctly worked out the
algorithm ("check divisibility up to √n", "cast to int", "check even first"),
but the final emitted code dropped two details (the `int()` cast and the
even-number guard). The model *understood* the problem; it didn't faithfully
transcribe its own reasoning into code.

This is a strong coding result for a 50%-pruned model calibrated on a small
coding set: the harder tasks (arbitrary-depth flatten, regex word-count) passed,
and only small transcription fidelity was lost — consistent with the first run's
observation that pruning kept coding/reasoning structure (the hard part) and
only sporadically dropped detail.

### 6.3 General knowledge spot-check (3 prompts, from the prior turn)
- "Capital of France" → **failed** (rambled, never said Paris).
- "Water chemical symbol" → **correct** (H2O, hydrogen + oxygen).
- "Moon landing year + mission" → **partially wrong** (said 1961, correctly
  named Apollo 11).

Expected: a 0.5 compression ratio pruned on a coding-calibrated set retains
coding/reasoning and loses long-tail factual recall. Knowledge degradation is
the known cost of router-weighted L2 saliency pruning, not a kernel/run defect.

## 7. Test suite status

`pytest tests/ -q`: **109 passed, 2 failed**. Both failures are pre-existing
test bugs, not regressions from this run:

- `tests/test_triton_kernels.py::TestFreaParity::test_triton_frea_matches_bmm` —
  **now a real signal**: FREA actually launches on Triton on the L4 (post-#17),
  so the test compares Triton-vs-PyTorch for the first time. Outputs match to
  ~0.25 on magnitudes ~400–1200 in fp16, but the test uses `atol=2e-2` — far too
  tight for fp16. The tolerance should be loosened (e.g. `rtol=5e-2` or
  `atol=1.0`). Previously this test "passed" trivially because both sides were
  the PyTorch fallback.
- `tests/test_layerwise_observer.py::test_layerwise_observer_matches_standard_observer` —
  known CPU-vs-CUDA device-mismatch test bug.

The 13 hermetic tests in `tests/test_run_findings_fixes.py` all pass.

## 8. Driver enhancement in this run

`scripts/reap_lfm2_run.py` was extended to capture the new Triton usage summary
(issue #18 fix): imports `reset_triton_usage` / `triton_usage_snapshot` /
`format_triton_usage_summary` from `reap.kernels.triton_utils`; calls
`reset_triton_usage()` at the start of Phase 5; records the snapshot + summary
text into the `5_observe` phase row of `perf_report.json` and logs the summary at
INFO. This is what produced the `f2_reduce: 880 Triton / 0 PyTorch; frea: 880
Triton / 0 PyTorch` line cited in §4.

## 9. Proposals for the throughput/memory tradeoff

The kernels now run seamlessly and correctly on the L4. The remaining gap is
that, on shared-mem-bound GPUs, FREA-on-Triton is slower than the PyTorch
fallback it replaced. Options, in rough order of effort vs. payoff:

1. **Empirical profitability probe (recommended, lowest-risk).** On the first
   observe batch, time FREA-Triton vs FREA-PyTorch on a single layer; pick the
   winner for the rest of the run and memoize. This is hardware- and
   shape-agnostic, needs no heuristics, and directly optimizes the user's actual
   workload. ~30 lines in `observe_moe_batch` / a tiny harness.

2. **Tile-size profitability gate in `_triton_frea_supported`.** When
   `choose_frea_block_sizes` is forced below `block_h=128` (or below the shape's
   natural next-power-of-2), return `(False, "auto-tiled blocks below
   profitability threshold")` so the cuBLAS-backed PyTorch fallback is used on
   shared-mem-bound GPUs. Keep Triton as the default only when tiles can be
   ≥128 (big-shared-mem GPUs). This generalizes #20's feasibility check to a
   profitability check.

3. **Expose a `--frea-backend {auto,triton,pytorch}` CLI knob.** Lets
   throughput-focused users force the cuBLAS fallback and memory-constrained
   users force Triton. Pairs with `--observe-backend` (which already exists for
   the coarse backend choice). Low effort, high control.

4. **Opt into 164 KiB dynamic shared memory on Ampere/Ada.** The L4 (cc 8.9)
   can expose ~164 KiB per-block via `cudaFuncSetAttribute(MaxDynamicShared
   MemorySize)`. If Triton exposes this (or via a launch flag), 128×128 tiles
   would fit on the L4 → far fewer iterations → likely beats cuBLAS. This was
   the "or opt into 164 KiB" option noted in issue #17 that was not implemented.
   Highest payoff on Ada/Ampere but Triton-API-dependent.

5. **Restructure the FREA kernel for the small-tile regime.** Currently one
   program per pair, sequential `h0`/`i0` tile loops. A 2D grid
   (pairs × h-blocks) with a cross-block reduction would expose more
   parallelism and could recover throughput even at 64×64 tiles. Larger effort,
   kernel-rewrite scope.

6. **F2 is fine — leave it.** F2 runs on Triton, fits shared mem trivially (no
   `tl.dot`), and now accumulates in fp64. No action needed.

My recommendation: ship **(1) the empirical probe** as the default (it makes the
right choice per host with zero tuning), and add **(3) the `--frea-backend`
knob** as the escape hatch. Pursue **(4) the 164 KiB opt-in** as a follow-up if
Triton exposes it, since it would make the L4/T4 case genuinely fast on Triton
rather than just falling back. The memory gain (0.75 GiB) is real and worth
keeping for memory-constrained hosts, so the Triton path should remain
selectable even when the probe picks PyTorch for throughput.

## 10. Reproducing

```bash
source /home/ubuntu/reap-cuda/.venv/bin/activate
cd /home/ubuntu/reap-cuda
python scripts/reap_lfm2_run.py
# reports: /data/reap-artifacts/LFM2.5-8B-A1B/evol-codealpaca-v1/perf_report.json
# Triton usage summary is in perf_report.json["phases"][5]["triton_usage_summary"]
```