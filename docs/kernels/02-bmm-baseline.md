# 02 — Phase 1: bmm Baseline (Pure-PyTorch Parity Oracle)

> **Status: LANDED** — `src/reap/kernels/bmm.py`  
> Entry: `routed_expert_activations_grouped` via `--observe-backend bmm`  
> (also fallback for FREA/F2 when Triton is off or ineligible).

> **Concern:** correct, kernel-free reference that eliminates the expert-all-tokens
> loop and `(E,T,H)` materialization. Runs on CPU/MPS/CUDA without Triton.

## Goal

Compute SwiGLU expert outputs only for **routed pairs** from F5, using pure
PyTorch `F.linear` per active expert segment.

## Implementation (current)

```txt
F5 pairs (sorted by expert, CSR expert_offsets)
  → index_select flat_input → routed_x (n_pairs, H)
  → for each expert e with n_e > 0:
        apply_swiglu(xe, W_gate[e], W_up[e], W_down[e])  # F4 linear weights
  → out (n_pairs, H)
```

- Weights come from **F4** (`get_stacked_expert_weights`) — always linear
  convention `(E,I,H)` / `(E,H,I)`.
- **Not** the naive “gather full weight stack per pair” anti-pattern (that would
  be hundreds of GB). Grouped-by-expert form only.

Code: `bmm.routed_expert_activations_grouped`, `weight_cache.apply_swiglu`.

## Reductions

Phase-1 style observe uses the same **F2 routed update** as other backends
(`update_pruning_state_routed`) after pair outputs exist — not the dense
`(E,T,H)` `update_pruning_state` (that remains for `loop` / merge dense needs).

## Backend selection

```python
# kernels/backend.py
auto → "f2" if triton_runtime_available() else "bmm"
```

## Parity

`tests/test_kernel_parity_bmm.py`: tiny fused Qwen3, `loop` vs `bmm` / `frea`
on consumed prune metrics (atol ~1e-4).

## Expected improvement vs loop

| Metric | Loop | bmm |
|---|---|---|
| Expert FLOPs | T × E × … | T × top_k × … (**~16× less**, E=128) |
| `(E,T,H)` | ~8.6 GB | **~0** (MB-scale pair buffers) |
| Launches | ~384/layer | ~O(active experts) linears |

Projected observe wall-clock: **~10–15×** vs loop when expert MLP dominates
(unmeasured; see `08-expected-improvements.md`).

## Acceptance (done)

- [x] Grouped implementation in tree
- [x] Used as Triton fallback
- [x] Parity tests on CPU
