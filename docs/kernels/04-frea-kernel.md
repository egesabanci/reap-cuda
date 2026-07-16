# 04 — Phase 3: FREA — Fused Routed Expert Activation

> **Status: LANDED**  
> Dispatch: `kernels/frea.py` → `triton_frea.frea_triton_activations` or  
> `bmm.routed_expert_activations_grouped`  
> Triton source: `kernels/triton_frea.py` (`@triton.jit` SwiGLU)  
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

- One launch **per expert** with `n_e > 0` tokens (CSR segment from F5)
- Tiled SwiGLU in fp32 accum, write back model dtype
- **Gates:** CUDA + triton package; SiLU only; `H ≥ 16`, `I ≥ 16`; weights on CUDA
- On any failure → **automatic** grouped PyTorch fallback (`log_triton_fallback`)

Tiny models in unit tests (H=8) always use PyTorch.

## PyTorch path

`routed_expert_activations_grouped`: `index_select` inputs, then per-expert
`apply_swiglu` (`F.linear` × 3).

## Integration

```txt
observe_moe_batch(..., backend in {bmm,frea,f2})
  use_triton = backend in {frea,f2} and triton_runtime_available()
  pair_out = frea_activations(..., use_triton=use_triton)
```

## Expected impact vs loop

| | Loop | FREA (Triton or bmm) |
|---|---|---|
| Expert FLOPs | E×T×… | top_k×T×… (**~16× less**) |
| Act memory | ~8.6 GB | **~MB** |
| Wall-clock (proj.) | 1× | **~15–25×** expert portion; Triton may add **~1.2–2×** over bmm |

## Tests

- `tests/test_kernel_parity_bmm.py` — frea backend vs loop on CPU (PyTorch FREA)
- `tests/test_triton_kernels.py::TestFreaParity` — CUDA Triton vs bmm when available

## Acceptance (done)

- [x] Triton + PyTorch paths
- [x] Layout-agnostic via F4
- [x] Safe fallback / no hard dependency on triton import at package import
