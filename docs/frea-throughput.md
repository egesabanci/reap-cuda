# FREA Throughput vs Memory

Operational guide for the **FREA** (fused routed expert activation) stage after
the LFM2.5 + L4 EC2 runs. Complements the design notes in
[kernels/04-frea-kernel.md](kernels/04-frea-kernel.md) and the device policy in
[gpu-and-backends.md](gpu-and-backends.md).

## Two orthogonal knobs

| Knob | Controls | Doc |
| --- | --- | --- |
| `--observe-backend` | Coarse path: `loop` / `bmm` / `frea` / `f2` / `auto` | [gpu-and-backends.md](gpu-and-backends.md) |
| **`--frea-backend`** | **Which FREA implementation** when the coarse path uses FREA | this page |

When `--observe-backend` is `bmm` or `loop`, FREA is not used (or only as part
of a non-Triton path). When it is `auto` / `frea` / `f2`, FREA runs under:

```bash
--frea-backend auto|triton|pytorch   # default: auto
# or
export REAP_FREA_BACKEND=auto|triton|pytorch
```

## What each mode does

| Mode | Behavior |
| --- | --- |
| **`auto` (default)** | One-shot **profitability probe**: warm-up then time Triton vs cuBLAS grouped PyTorch on the first dense batch for this `(device, H, I)`. Memoize the winner for the process. Tiny batches (`n_pairs < 16`) use PyTorch **without** memoizing so a later dense batch can still probe. |
| **`triton`** | Always try the Triton SwiGLU kernel when tiles fit shared memory; fall back to PyTorch only on unsupported shapes / launch failure. |
| **`pytorch`** | Always use `routed_expert_activations_grouped` (per-expert `F.linear` / cuBLAS). Max throughput on many L4/T4-class GPUs. |

### Static tile floor (no probe)

```bash
export REAP_FREA_PROBE=0   # with --frea-backend auto
```

If the largest tiles that fit are below `128` on **both** H and I, treat Triton
as unprofitable and use PyTorch. Prefer the empirical probe (default) over this
heuristic on mixed hardware.

## Why this exists (L4 lesson)

On NVIDIA L4 (~99 KiB default shared mem / block):

1. Natural **128×128** FREA tiles need ~136 KiB → launch failed → PyTorch fallback
   (fast cuBLAS) → observe ~**949 tok/s**.
2. After auto-tiling, Triton **launched** with smaller tiles → correct, less peak
   VRAM (~0.75 GiB), but **~2.3× slower** observe (~415 tok/s) because many more
   tile iterations beat fusion gains.
3. Feasibility alone is not profitability. **`auto` probe** picks the faster
   path per host/shape without hardcoding SKUs.

On large-SM GPUs (or with opt-in ~164 KiB), 128×128 can fit and Triton often
wins on both speed and memory.

## Shared-memory tiles (hardware-agnostic)

Implementation: `choose_frea_block_sizes` in `triton_frea.py`.

1. Query `shared_memory_per_block` and, when useful,
   `shared_memory_per_block_optin` (Ampere/Ada ~164 KiB).
2. Walk candidates `(128, 64, 32, 16)` for `BLOCK_H` / `BLOCK_I` until the
   estimate fits (plus a small safety margin).
3. On launch `out of resource: shared memory`, disable opt-in and retry once;
   permanent disable memo avoids re-attempting a doomed config thousands of times.

**“Tile fit”** = chosen block sizes fit this GPU’s per-block shared memory so the
kernel can launch. See also the FREA design doc.

## Observability

At end of observe (full and layerwise):

```text
INFO Triton usage summary: frea: N Triton / M PyTorch; f2_reduce: …
```

First fallback per component is **WARNING**; further at DEBUG.  
`reap kernels` only shows package/runtime readiness — use the run summary for
“did FREA actually run?”

Probe log line:

```text
INFO FREA profitability probe: triton=0.0123s pytorch=0.0056s -> pytorch (reason=ok)
```

## Recommended settings

| Goal | Suggested flags |
| --- | --- |
| Default / mixed hosts | `--observe-backend auto --frea-backend auto` |
| Max observe throughput on L4/T4 | `--frea-backend pytorch` |
| Force Triton / min intermediate memory | `--frea-backend triton` |
| Bring-up without Triton | `--observe-backend bmm` |
| Offline calib + big disk | `--dataset-path … --artifacts-dir …` ([cli.md](cli.md)) |

## Env vars

| Variable | Role |
| --- | --- |
| `REAP_FREA_BACKEND` | Same as `--frea-backend` (overrides process default if set) |
| `REAP_FREA_PROBE` | `0` / `false` → disable empirical probe; use tile-floor heuristic under `auto` |
| `REAP_DISABLE_TRITON` | Disable all Triton kernels (softmax / FREA / F2) |
| `REAP_OBSERVE_BACKEND` | Coarse backend if CLI omits `--observe-backend` |

## Field reports

- `run-findings.md` — first LFM2.5 run (FREA silent fallback, F4 OOM, router)
- `run-findings-2.md` — post-fix re-run (full Triton, throughput/memory tradeoff)

## Related

- [gpu-and-backends.md](gpu-and-backends.md)
- [kernels/04-frea-kernel.md](kernels/04-frea-kernel.md)
- [cli.md](cli.md)
- [residency.md](residency.md) — weight placement (orthogonal)
