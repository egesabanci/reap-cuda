# 04 — Phase 3: FREA — Fused Routed Expert Activation (headline kernel)

> **Concern:** the single Triton kernel that replaces the expert-execution
> loop. It computes expert activations for **only routed `(token, top_k)`
> pairs**, in **one fused kernel per layer**, with **no `(E, T, H)`
> materialization**. FREA is the headline win; F2 (Phase 4) extends it by fusing
> the saliency reductions in-register.

## 1. What FREA replaces

The two expert loops that dominate the observer today:

**Compute loop** — `src/reap/observer.py:378` (and the mirror in
`layerwise_observer.py`):
```python
for idx, expert in enumerate(module.experts):
    activations[idx] = expert(flat_input).to(device)   # (E, T, H) materialized
```

**Reduce loop** — `src/reap/pruning_metrics.py:178`:
```python
for i in range(num_experts):
    active_mask = (pruning_batch.selected_experts == i).any(dim=-1).to(device)
    if not active_mask.any(): continue
    selected_activations = pruning_batch.activations[i, active_mask, :]
    ean_norm = torch.linalg.norm(selected_activations, dim=-1)
    ean_sum[i] = ean_norm.sum()
    ...
```

FREA fuses the **compute** (gate/up/down + SiLU) with the start of the
**reduce** (the `ean_norm` and per-expert accumulators). Phase 4 (F2) finishes
the job by fusing the remaining reductions too.

## 2. Inputs (from F5 + the adapter)

From F5 (`03-f5-router-fusion.md`):
- `pair_token_idx` `(T*k,)`, `pair_expert_idx` `(T*k,)`, sorted by expert
- `expert_offsets` `(E+1,)` CSR boundaries
- `pair_router_w` `(T*k,)` — per-pair router weight (renorm-aware)

From the model / F4 (`06-f4-weight-stacking.md`):
- `flat_input` `(T, H)` — the layer's hidden states
- `W_gate`, `W_up` `(E, I, H)`, `W_down` `(E, H, I)` — stacked expert weights
  (Phase 5 caches these; FREA reads them)

## 3. The math FREA computes (per routed pair)

For a routed pair `(t, e)` with input `x = flat_input[t]` and router weight
`w = pair_router_w`:

```
g = x @ W_gate[e].T      # (I,)    -- gate_proj
u = x @ W_up[e].T        # (I,)    -- up_proj
h = silu(g) * u          # (I,)    -- SwiGLU activation
y = h @ W_down[e].T      # (H,)    -- down_proj  (the "expert activation")
n = ||y||_2              # scalar  -- ean_norm (matches pruning_metrics.py:193)
```

Then FREA emits (to be reduced by F2 / atomically here):
- `expert_frequency[e] += 1`
- `ean_sum[e] += n`
- `max_activations[e] = max(max_activations[e], max_over_H(y))`  *(see note)*
- `weighted_ean_sum[e] += n * w`
- `reap[e] += n * w`  (then `/= count` for the mean — F2 does the division)

> **Note on `max_activations`**: the existing code
> (`pruning_metrics.py:200`) does `selected_activations.max()` — the max over
> the **`(n_e, H)` block** of routed activations for expert `e`, not the max
> of the norm. Read `pruning_metrics.py:200` carefully:
> ```python
> selected_activations_max = selected_activations.max().to(device="cpu")
> ```
> `selected_activations` is `(n_e, H)`, so `.max()` is over both dims → the
> single largest activation **magnitude** across all routed tokens × hidden
> for that expert. FREA must reduce `max(|y_h|)` across the expert's pairs and
> hidden dim. This is a per-element abs-max, **not** a per-pair norm-max.
> **The parity test must assert this exactly.**

## 4. Kernel structure (Triton)

**Grid**: `(num_expert_blocks,)` — one program per expert (or per expert-tile).
Each program owns a contiguous `[s, t)` slice of the F5-sorted pair array for
**one expert** `e` (given by `expert_offsets`), so it loads `W_gate[e]`,
`W_up[e]`, `W_down[e]` **once** into SRAM and streams its routed tokens.

```
program p (expert e, pair block [s, t)):

  # Load expert weights ONCE into SRAM (small: I×H = 768×2048 bf16 ≈ 3 MB total
  # for the three linears; fits L40S SRAM in tiles).
  Wg = load W_gate[e]   # (I, H) tiled over (I_blk, H_blk)
  Wu = load W_up[e]     # (I, H)
  Wd = load W_down[e]   # (H, I)

  # Accumulators in SRAM / registers (tiny):
  freq_e    = 0
  ean_sum_e = 0
  max_e     = -inf
  wean_e    = 0
  reap_e    = 0
  # (E,H) routed_characteristic_activation accumulator (only for ean_ca):
  ca_e[H_blk] = 0

  for pair in [s, t):                       # stream this expert's routed tokens
      x = flat_input[pair_token_idx[pair]]  # (H,)   -- coalesced if token-sorted
      w = pair_router_w[pair]               # scalar

      # 3 matmuls in SRAM (tile over I and H):
      g = x @ Wg.T      # (I,)
      u = x @ Wu.T      # (I,)
      h = silu(g) * u   # (I,)
      y = h @ Wd.T      # (H,)

      n = sqrt(sum(y*y))                    # ean_norm
      freq_e    += 1
      ean_sum_e += n
      wean_e    += n * w
      reap_e    += n * w                     # mean taken later (F2)
      max_e     = max(max_e, max(|y|))       # per-element abs-max (see §3 note)
      ca_e      += y                         # for routed_characteristic_activation (ean_ca only)

  # Scatter-reduce to global (E,) / (E,H) buffers with atomics:
  atomic_add(expert_frequency, e, freq_e)
  atomic_add(ean_sum,           e, ean_sum_e)
  atomic_add(weighted_ean_sum,  e, wean_e)
  atomic_add(reap,              e, reap_e)
  atomic_max(max_activations,    e, max_e)
  if ean_ca: atomic_add(routed_characteristic_activation, e, ca_e[:])
```

### Why this is correct
Every consumed metric (`00-cost-model.md` §6) is a per-routed-pair reduction.
FREA computes exactly those reductions in-register/atomically. The `(E, T, H)`
activation tensor is **never written to HBM** — only the `(E,)` and `(E,H)`
statistic buffers (≤ 1 MB total) touch HBM.

### Why expert-sorted pairs matter
Because F5 sorts pairs by expert, each FREA program reads one expert's weights
once and streams its tokens — coalesced. Without the sort, FREA would have to
gather weights per pair (the 205 GB anti-pattern from `02-bmm-baseline.md` §8).

## 5. Memory profile

| Buffer | Size (E=128, H=2048) | Lives in |
|---|---|---|
| `expert_frequency` (E,) int64 | 1 KB | HBM (atomic) |
| `ean_sum`, `weighted_ean_sum`, `reap`, `weighted_expert_frequency_sum` (E,) fp32/fp64 | < 4 KB | HBM (atomic) |
| `max_activations` (E,) fp32 | 0.5 KB | HBM (atomic max) |
| `routed_characteristic_activation` (E,H) fp32 (ean_ca only) | 1 MB | HBM (atomic add) |
| Expert weights per program (SRAM, tiled) | ~3 MB bf16 (the 3 linears) | SRAM |
| Per-pair activation `y` (H,) | 8 KB fp32 | registers/SRAM |
| **`(E, T, H)` activation tensor** | **0 (eliminated)** | — |

Peak HBM traffic ≈ read `flat_input` once (T×H) + read each expert's weights
once (E×3×H×I) + write the tiny stat buffers. No 8.6 GB transient.

## 6. Fused vs non-fused layouts

FREA is **layout-agnostic** because it reads stacked weights from F4, not the
HF module:
- **Non-fused** (Qwen3 transformers 4.55): F4 stacks the `ModuleList`'s
  `gate_proj/up_proj/down_proj` into `(E, …)`. FREA uses them.
- **Fused** (Llama4, Qwen3.5/3.6 in transformers ≥5.x): the stacked
  `gate_up_proj` `(E, 2*I, H)` and `down_proj` `(E, H, I)` already exist; F4
  splits `gate_up_proj` into `gate`/`up` halves and passes them to FREA.

This is how FREA also **unblocks issue #4** (fused Qwen3.5/3.6): the observer
no longer calls the stock `module.experts(routed_in)` API that stock-HF fused
experts don't expose; it calls FREA with stacked weights directly.

## 7. Fallback (Mac / no-Triton)

`frea_observe_pytorch` = the **grouped-bmm** from `02-bmm-baseline.md` §8
(segmented `nn.functional.linear` per active expert over routed tokens, with
scatter-add reductions). This is the parity oracle; FREA-Triton must match it.

## 8. Parity contract

`tests/test_kernel_parity_frea.py` (EC2, Triton required; skipped on Mac):
- FREA-Triton vs `frea_observe_pytorch` on a tiny Qwen3-MoE: all consumed
  metrics bit-for-bit (atol=1e-5), per layer.
- Then on Qwen3-30B-A3B (real): compare FREA per-layer stats to the loop
  observer on a small calibration subset (atol=1e-3, real-model fp16 paths).

## 9. Expected improvement (vs loop baseline `00-cost-model.md` §7)

| Metric | Loop | bmm (Ph.1) | FREA (Ph.3) |
|---|---|---|---|
| Expert matmul launches / layer | 384 | 3 (per-expert grouped) | **1 fused** |
| Expert matmul launches / forward | 18,432 | 144 | **48** |
| `(E,T,H)` transient / layer | 8.6 GB | ~0 (grouped, ~MB) | **0** |
| Saliency update kernels / layer | ~6 (reduce loop) | ~6 (scatter) | **fused inline** |
| HBM activation traffic / layer | 8.6 GB write + read | small | **~0** |

Wall-clock on the expert-MLP portion: **~15–25× faster** for E=128 (the 16×
FLOP cut + the elimination of the 8.6 GB HBM traffic, which dominates on the
L40S's 864 GB/s memory). For E=256, the FLOP cut is 32× → **~20–40×**.

## 10. Acceptance

- FREA-Triton matches `frea_observe_pytorch` (the Phase-1 grouped bmm) on all
  consumed metrics (parity test).
- `--observe-backend frea` runs through `python -m reap.prune` and
  `python -m reap.layerwise_prune` on the L40S.
- Peak VRAM during a single layer's observation (layerwise mode) is dominated
  by the block's weights, not by any activation transient (< 100 MB stat
  buffers).
- Observer-only wall-clock on Qwen3-30B-A3B (`run_observer_only=True`) is
  ≥ 10× faster than the loop baseline.