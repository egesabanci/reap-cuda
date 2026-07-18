# 04 — Phase 3: FREA — Fused Routed Expert Activation

> **Status: LANDED**  
> Dispatch: `kernels/frea.py` → `triton_frea.frea_triton_activations` or  
> `bmm.routed_expert_activations_grouped`  
> Triton source: `kernels/triton_frea.py` (`@triton.jit` SwiGLU)  
> Ops / throughput: **[../frea-throughput.md](../frea-throughput.md)**  
> **Not** `torch.compile` — that path was removed.

> **Concern:** expert MLP only on routed pairs; no `(E,T,H)` buffer.

## Math (per pair `(t,e)`)

Weights are F4 **linear** convention:

```txt
g = silu-linear(x, W_gate[e])   # W_gate[e]: (I, H)
u = linear(x, W_up[e])
h = silu(g) * u
y = linear(h, W_down[e])        # W_down[e]: (H, I)
```

## Triton path

- One launch **per expert** with `n_e > 0` tokens (CSR segment from F5 / native router)
- Tiled SwiGLU in fp32 accum, write back model dtype
- **Tiles:** `choose_frea_block_sizes` walks 128→16 for H/I against live SM
  budgets (`shared_memory_per_block` and `_optin` when larger). Examples:
  L4 opt-in **99 KiB** → often **128×64** for large MoEs; **128×128 needs ~140 KiB**
  and only fits on GPUs with larger opt-in (e.g. A100/L40S ~164 KiB).
- **Gates:** CUDA + triton package; SiLU only; `H ≥ 16`, `I ≥ 16`; weights on CUDA
- On any failure → **automatic** grouped PyTorch fallback (`log_triton_fallback`,
  WARN once then DEBUG; permanent memo for hard SM failures)

Tiny models in unit tests (H=8) always use PyTorch.

## Backend policy (`--frea-backend`)

| Mode | Behavior |
| --- | --- |
| `auto` | Empirical probe: warm-up + time Triton vs cuBLAS; memoize winner per `(device_type, device_index, dtype, H, I)` |
| `triton` | Force Triton when supported |
| `pytorch` | Force grouped `F.linear` |

Env: `REAP_FREA_BACKEND`, `REAP_FREA_PROBE=0` (static tile-floor instead of probe).

The probe key is scoped by **device identity and input dtype** so a decision
for one GPU or precision does not leak to another. Shared-memory opt-in state
(`_USE_SMEM_OPTIN`) and permanent disable memos are likewise **per-device**.

When the support check fails (CPU input, non-SiLU, disabled, etc.), the probe
returns ``pytorch`` immediately — **no timing, no CUDA sync, no memoization**.

Probe timing uses **CUDA events** on the launch device (falling back to
``torch.cuda.synchronize(device)`` when events are unavailable) so unrelated
streams are not globally synchronized. Warm-up launches are stream-ordered
before the timed region, so no explicit synchronize is needed between
warm-up and timing when events are used. The wall-clock fallback path uses a
narrowly scoped ``torch.cuda.synchronize(device)`` only around the timed
interval.

CSR expert offsets are bulk-transferred to host once via `.tolist()`
before the per-expert Python loop, eliminating O(E) scalar `.item()`
synchronizations.

See [frea-throughput.md](../frea-throughput.md): on L4, Triton can launch with
opt-in tiles yet remain **~1.9× slower** than cuBLAS; the probe correctly
defaults to PyTorch.

## PyTorch path

`routed_expert_activations_grouped`: `index_select` inputs, then per-expert
`apply_swiglu` (`F.linear` × 3). Often the **throughput** winner on L4/T4.

## Integration

```txt
observe_moe_batch(..., backend in {bmm,frea,f2})
  use_triton = backend in {frea,f2} and triton_runtime_available()
  pair_out = frea_activations(..., use_triton=..., frea_backend=get_frea_backend())
```

## Expected impact vs loop

| | Loop | FREA (Triton or bmm) |
|---|---|---|
| Expert FLOPs | E×T×… | top_k×T×… (**~16× less**) |
| Act memory | ~8.6 GB | **~MB** |
| Wall-clock | 1× | Large win vs loop; **Triton vs bmm is host-dependent** (probe) |

## Tests

- `tests/test_kernel_parity_bmm.py` — frea backend vs loop on CPU (PyTorch FREA)
- `tests/test_triton_kernels.py::TestFreaParity` — CUDA Triton vs bmm (fp16 loose tol)
- `tests/test_run_findings_fixes.py` — tile choice, backend set/get, CLI flags

## Acceptance (done)

- [x] Triton + PyTorch paths
- [x] Layout-agnostic via F4
- [x] Safe fallback / no hard dependency on triton import at package import
- [x] Shared-mem auto-tile + opt-in budget
- [x] Profitability probe + CLI `--frea-backend`
