# run-findings-4 — Forced `REAP_FREA_BACKEND=triton` run on LFM2.5-8B-A1B (L4)

## 1. Scope

Fourth instrumented run. Same config as run 3 (`run-findings-3.md`) but with the
FREA profitability **probe bypassed** and FREA forced onto the Triton path:

```
REAP_FREA_BACKEND=triton   (probe skipped; frea_triton_activations mode=="triton")
REAP_FREA_PROBE=1          (irrelevant when backend is forced)
```

Goal (from the §10 follow-up in run-findings-3): actually exercise the #28 SM
opt-in on the L4 and check whether the larger Triton tiles beat cuBLAS — i.e.
whether the probe's "pick PyTorch" verdict would ever be wrong on this GPU.

## 2. Headline result

| metric | run 3 (`auto`, probe→pytorch) | **run 4 (`triton` forced)** | Δ |
| --- | ---: | ---: | ---: |
| observe wall | 39.93 s | **74.19 s** | **+86%** |
| observe throughput | 1,022 tok/s | **550 tok/s** | **−46%** |
| observe peak GPU | 16.296 GiB | 16.294 GiB | flat |
| FREA | 0 Triton / 880 PyTorch | **880 Triton / 0 PyTorch** | forced |
| F2 | 880 Triton / 0 PyTorch | 880 Triton / 0 PyTorch | flat |
| total wall | 269.5 s | 304.7 s | +35 s |
| board peak | 17,732 MiB | 17,333 MiB | −399 MiB |

```
outcome = success, smoke_ok = True
Triton usage: f2_reduce: 880 Triton / 0 PyTorch; frea: 880 Triton / 0 PyTorch
```

Zero FREA fallbacks, zero shared-mem errors — the Triton kernel ran end-to-end.
But it is **1.86× slower than the cuBLAS PyTorch path** on the L4 (74.19 s vs
39.93 s). The probe's verdict in run 3 ("pick PyTorch") is **correct for the L4**.

## 3. Erratum: the L4's shared-memory limits (corrects run-findings-2/3 and issue #28)

`torch.cuda.get_device_properties('cuda')` on the L4 (AD104, cc 8.9):
```
shared_memory_per_block        = 49,152 B   (48 KiB)   <- default
shared_memory_per_block_optin  = 101,376 B  (99 KiB)   <- opt-in maximum
```

So:
- The L4's **default** per-block shared mem is **48 KiB**, not 99 KiB.
- The L4's **opt-in maximum** is **99 KiB**, **not 164 KiB**. The 164 KiB opt-in
  belongs to A100/L40S-class (Ampere cc 8.0 / Ada cc 8.9 *datacenter* dies), not
  the AD104 consumer die.

Consequence: **128×128 FREA tiles cannot fit on the L4 at all.**
`estimate_frea_shared_bytes(16,128,128) = 143,360 B (140 KiB) > 99 KiB` opt-in.
The #28 issue's premise ("164 KiB opt-in would let 128×128 fit on L4/T4") is
**wrong for the L4** — correct only for A100/L40S-class GPUs. The probe is
therefore essential on the L4: no tile config makes Triton win.

Confirmed via `choose_frea_block_sizes`:
```
h=2048 i=1792:  default(48KiB)->(128, 32, 16)   optin(99KiB)->(128, 64, 16)
128x128 needs 143,360 B  -> does NOT fit even with opt-in on the L4
```

run-findings-2 §6 and the #17/#24/#28 issue bodies' "L4 = 99 KiB default /
164 KiB opt-in" wording should be read as "L4 = 48 KiB default / 99 KiB opt-in".
This does **not** change any run-2/run-3 conclusion (Triton still loses to
cuBLAS on the L4); it only corrects the *tile sizes* and the *reason*.

## 4. What the #28 opt-in actually did on the L4

With `_USE_SMEM_OPTIN` starting `None`, `choose_frea_block_sizes` tries the
opt-in budget first and returns **(128, 64, 16)** — i.e. the opt-in doubled the
**I-tile** from 32 → 64 (H-tile was already 128). The H/I tile-iteration product
halves: 2048/128 × 1792/64 = 16×28 = **448** iterations vs the default budget's
2048/128 × 1792/32 = 16×56 = **896**.

- Triton **does** opt into the 99 KiB dynamic shared mem automatically — the
  128×64 kernel needs 77,824 B shared mem, which exceeds the 48 KiB static
  default, and the launch succeeded with **zero fallbacks**, so Triton's launcher
  called `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, …)` for us. No
  explicit `cudaFuncSetAttribute` is needed in user code.
- The opt-in's payoff: run 4 (128×64, opt-in) observe = **74.19 s / 550 tok/s**
  vs run 2 (128×32, pre-#28, no opt-in) observe = **98.46 s / 415 tok/s** →
  **+33% throughput** from the opt-in + the #29 `BLOCK_N=32`/`num_warps=8`
  occupancy tweaks. The opt-in alone roughly halves the I-loop, but Python
  per-expert dispatch (`for expert_id in range(e)` in `_frea_triton_impl`) and
  the remaining overhead cap the gain at ~33%.
- But 128×64-on-99KiB is still **1.86× slower than cuBLAS** (74.19 s vs 39.93 s).
  On the L4 there is no tile config that closes the gap; the gap is structural
  (per-expert Python loop + Triton tile overhead vs a single cuBLAS
  `F.linear` per expert).

## 5. Isolated micro-benchmark (single layer, e=8, n=512, h=2048, i=1792, fp16)

```
tiles used (opt-in): (128, 64, 16)
triton  = 14.63 ms
pytorch =  1.32 ms
ratio   = 11.08x slower (triton)
parity  = max abs diff 0.00195  (perfect fp16 parity)
```

The isolated ratio (11×) is much worse than the full-run ratio (1.86×) because
at n=512/e=8 each expert gets only ~64 tokens — per-expert Python dispatch and
Triton launch overhead dominate; the real run's denser per-expert segments
amortize that. The point of the micro-bench is the **parity** (0.002 max abs
diff on fp16 magnitudes ~hundreds) and the **tile confirmation** (128×64).

## 6. Functional verification

Pruned checkpoint (16 experts, `use_expert_bias=True`) loads in a fresh
process and generates coherent, correct code (bubble-sort prompt):
```
Okay, the user is asking to write a Python bubble sort function. Let me recall
how bubble sort works. Bubble sort is a comparison-based sorting algorithm that
gets its name from the way it moves elements through the list. It's like
rearranging the array by moving the larg…
```
Identical quality to the run-3 checkpoint (same seed, same pruning decision —
the FREA *backend* does not change the observed statistics or the pruned
selection; FREA computes the same routed activations on both backends, only the
implementation differs).

## 7. Per-phase performance (run 4)

| phase | wall | gpu_peak | rss |
| --- | ---: | ---: | ---: |
| 0_env_backend_check | 0.04 s | 0.000 GiB | 0.93 GiB |
| 1_tokenizer_load | 1.32 s | 0.000 GiB | 1.16 GiB |
| 2_model_load | 148.95 s | 16.210 GiB | 0.11 GiB |
| 3_dataset_load_tokenize | 6.10 s | 15.773 GiB | 0.41 GiB |
| 4_observer_setup | 0.00 s | 15.773 GiB | 0.41 GiB |
| **5_observe** | **74.19 s** | **16.294 GiB** | 1.34 GiB |
| 5b_load_observer_state | 0.01 s | 15.781 GiB | 1.34 GiB |
| 6_prune_slice_save | 75.88 s | 15.999 GiB | 8.83 GiB |
| 7_smoke_test | 1.81 s | 8.645 GiB | 8.83 GiB |
| 8_artifact_summary | 0.00 s | 8.560 GiB | 8.83 GiB |

Board sampler (299 samples, 1 Hz): peak 17,333 MiB, peak util 94%.

## 8. Conclusion & recommendation

1. **The probe is correctly calibrated for the L4.** Forcing Triton (run 4)
   loses 46% throughput with no memory benefit (16.294 vs 16.296 GiB — within
   noise). `--frea-backend auto` (run 3) remains the right default.
2. **#28's SM opt-in works and helps where it can.** On the L4 it enables
   128×64 (vs 128×32) for a 33% Triton-side speedup — real, just not enough to
   beat cuBLAS. On A100/L40S-class GPUs (164 KiB opt-in) it would enable true
   128×128 and the probe should be re-evaluated there.
3. **128×128 is impossible on the L4** (hardware: 99 KiB opt-in max; 140 KiB
   needed). Issue #28's "L4/T4" framing should be corrected to
   "A100/L40S-class"; the L4 can never reach 128×128 regardless of opt-in.
4. **No new issue to file.** The forced-triton path ran cleanly (zero
   fallbacks, correct parity, correct checkpoint) — it is simply slower on
   this GPU, which is exactly what the probe exists to detect.
5. **Memory tradeoff note:** runs 3 and 4 both peak at ~16.29 GiB — the 0.75 GiB
   win over run 1 comes from the F4 LRU cache fix (#15), not from the FREA
   backend choice. So forcing Triton buys nothing on memory on the L4 either.

## 9. Reproduce

```bash
cd /home/ubuntu/reap-cuda && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REAP_ARTIFACTS_DIR=/data/reap-artifacts
export REAP_FREA_BACKEND=triton        # force FREA onto Triton (probe skipped)
python scripts/reap_lfm2_run.py

# inspect the opt-in tile decision on this GPU:
python -c "
import torch
from reap.kernels.triton_utils import device_shared_memory_bytes
from reap.kernels.triton_frea import choose_frea_block_sizes, estimate_frea_shared_bytes
d=torch.device('cuda')
print('default', device_shared_memory_bytes(d, prefer_optin=False))
print('optin  ', device_shared_memory_bytes(d, prefer_optin=True))
print('tiles optin', choose_frea_block_sizes(2048,1792,device=d))
print('128x128 needs', estimate_frea_shared_bytes(16,128,128))
"
```