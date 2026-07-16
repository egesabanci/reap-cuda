# FREA Throughput vs Memory

Operational guide for the **FREA** (fused routed expert activation) stage.
Complements [kernels/04-frea-kernel.md](kernels/04-frea-kernel.md) and
[gpu-and-backends.md](gpu-and-backends.md).

Grounded in four instrumented LFM2.5-8B-A1B runs on an **NVIDIA L4**
(`run-findings.md` … `run-findings-4.md`).

## Two orthogonal knobs

| Knob | Controls | Doc |
| --- | --- | --- |
| `--observe-backend` | Coarse path: `loop` / `bmm` / `frea` / `f2` / `auto` | [gpu-and-backends.md](gpu-and-backends.md) |
| **`--frea-backend`** | **Which FREA implementation** when the coarse path uses FREA | this page |

```bash
--frea-backend auto|triton|pytorch   # default: auto
export REAP_FREA_BACKEND=auto|triton|pytorch
```

## What each mode does

| Mode | Behavior |
| --- | --- |
| **`auto` (default)** | One-shot **profitability probe**: warm-up then time Triton vs cuBLAS grouped PyTorch on the first dense batch for `(device, H, I)`. Memoize the winner. Tiny batches (`n_pairs < 16`) use PyTorch **without** memoizing so a later dense batch can still probe. |
| **`triton`** | Always try Triton when tiles fit shared memory; fall back only on unsupported shapes / launch failure. |
| **`pytorch`** | Always `routed_expert_activations_grouped` (per-expert `F.linear` / cuBLAS). Often max throughput on L4/T4. |

### Static tile floor (no probe)

```bash
export REAP_FREA_PROBE=0   # with --frea-backend auto
```

If the largest tiles that fit have **both** `BLOCK_H` and `BLOCK_I` below 128,
treat Triton as unprofitable and use PyTorch. Prefer the empirical probe on
mixed hardware.

## Shared-memory limits (measured, not guessed)

Code queries `torch.cuda.get_device_properties` — never hardcodes SKUs. Typical
values from EC2 runs:

| GPU class | Default `shared_memory_per_block` | Opt-in `shared_memory_per_block_optin` | 128×128 FREA (~140 KiB)? |
| --- | ---: | ---: | --- |
| **L4 / T4-class (AD104)** | **48 KiB** (49 152 B) | **99 KiB** (101 376 B) | **No** — opt-in max is 99 KiB |
| **A100 / L40S-class** | larger default | often **~164 KiB** | **Yes**, via opt-in when needed |

> **Erratum:** Earlier notes (and closed issue #28 title) said “L4 ≈ 99 KiB
> default / 164 KiB opt-in.” That is **wrong for L4**. Correct: **48 KiB
> default / 99 KiB opt-in**. The 164 KiB opt-in is for A100/L40S-class dies, not
> consumer AD104. See `run-findings-4.md` §3 and the erratum comment on #28.

`estimate_frea_shared_bytes(16, 128, 128) ≈ 143 360 B (~140 KiB)` → cannot fit
on L4 even with full opt-in.

### What opt-in still does on L4

With opt-in, `choose_frea_block_sizes(h=2048, i=1792)` typically returns
**(128, 64, 16)** instead of **(128, 32, 16)** under the 48 KiB default — larger
I-tile, fewer I-loop iterations. Triton opts into dynamic SM automatically
(`cudaFuncSetAttribute` via the launcher); no explicit user code is required.

Measured (forced `--frea-backend triton` on L4, LFM2.5 shapes):

| Tiles | Observe | Throughput |
| --- | ---: | ---: |
| ~128×32 (pre-opt-in era) | 98 s | 415 tok/s |
| **128×64 (opt-in)** | **74 s** | **550 tok/s** (+33% Triton-side) |
| PyTorch (probe default) | **40 s** | **1 022 tok/s** |

So opt-in **helps Triton on L4** but **does not beat cuBLAS**. On A100/L40S,
128×128 can fit and the probe should be re-evaluated.

## Why the probe exists (four-run summary)

Same model/calib on L4 (LFM2.5-8B-A1B, 100 examples, 0.5 prune, `gpu_full`):

| Run | FREA path | Observe | Peak GPU | Note |
| --- | --- | ---: | ---: | --- |
| 1 | PyTorch (Triton failed to launch) | 43 s / 949 tok/s | 17.0 GiB | F4 cache leak era |
| 2 | Triton (small tiles) | 98 s / 415 tok/s | 16.3 GiB | Feasible but slow |
| **3** | **Probe → PyTorch** | **40 s / 1 022 tok/s** | **16.3 GiB** | **Default after #24–#32** |
| 4 | Force Triton + opt-in 128×64 | 74 s / 550 tok/s | 16.3 GiB | Proves probe correct |

Memory win (~0.75 GiB vs run 1) is from **F4 single-entry cache**, not from
running FREA on Triton.

Probe line (run 3):

```text
INFO FREA profitability probe: triton=0.0481s pytorch=0.0062s -> pytorch (reason=ok)
```

## Shared-memory tile selection (code)

Implementation: `choose_frea_block_sizes` in `triton_frea.py`.

1. Prefer **opt-in** budget when larger than default; else default.
2. Walk `(128, 64, 32, 16)` for `BLOCK_H` / `BLOCK_I` until estimate fits
   (+ safety margin).
3. Optional larger `BLOCK_N` when H/I tiles are small, re-checked against SM.
4. On launch SM OOM: set `_USE_SMEM_OPTIN=False`, retry with default budget;
   permanent disable memo for hard failures.

**“Tile fit”** = chosen block sizes fit this GPU’s per-block shared-memory
limit so the kernel can launch.

## Observability

```text
INFO Triton usage summary: frea: N Triton / M PyTorch; f2_reduce: …
INFO FREA profitability probe: triton=…s pytorch=…s -> pytorch|triton
```

First fallback per component: **WARNING**; further at DEBUG.  
`reap kernels` = package/runtime readiness only — use the run summary for
“did FREA actually run?”

## Recommended settings

| Goal | Flags |
| --- | --- |
| Default / mixed hosts | `--observe-backend auto --frea-backend auto` |
| Max observe throughput on L4/T4 | `--frea-backend pytorch` (or trust `auto`) |
| Force Triton (debug / A100 experiment) | `--frea-backend triton` |
| Bring-up without Triton at all | `--observe-backend bmm` |
| Offline + big disk | `--dataset-path … --artifacts-dir …` |

## Env vars

| Variable | Role |
| --- | --- |
| `REAP_FREA_BACKEND` | Same as `--frea-backend` |
| `REAP_FREA_PROBE` | `0` / `false` → static tile-floor under `auto` (no timing) |
| `REAP_DISABLE_TRITON` | Disable all Triton (softmax / FREA / F2) |
| `REAP_OBSERVE_BACKEND` | Coarse backend if CLI omits `--observe-backend` |

## Field reports (repo root)

| File | Content |
| --- | --- |
| `run-findings.md` | First run: router crash, F4 OOM, silent FREA fallback |
| `run-findings-2.md` | Triton launches; throughput regression; some SM wording outdated |
| `run-findings-3.md` | Probe recovers throughput; memory via F4 LRU |
| `run-findings-4.md` | Forced Triton; **L4 SM erratum**; opt-in 128×64 confirmed |

Prefer **run-findings-4** for shared-mem numbers.

## Related

- [gpu-and-backends.md](gpu-and-backends.md)
- [kernels/04-frea-kernel.md](kernels/04-frea-kernel.md)
- [cli.md](cli.md)
- [residency.md](residency.md)
