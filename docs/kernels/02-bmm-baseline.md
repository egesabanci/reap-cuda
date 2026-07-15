# 02 — Phase 1: bmm Baseline (Pure-PyTorch Parity Oracle)

> **Concern:** a correct, kernel-free reference implementation that already
> eliminates the Python expert loop and the `(E, T, H)` materialization. Runs
> on Apple MPS. This is the **parity oracle** every Triton kernel (Phases 2–4)
> must match bit-for-bit. Land this first — it is a real speedup and the safest
> thing to ship.

## 1. Goal

Replace the `for idx, expert in enumerate(module.experts)` loop
(`src/reap/observer.py:378` and the matching site in `layerwise_observer.py`)
with a **routed-only batched matmul** that:

1. computes expert activations for **only the routed `(token, top_k)` pairs**,
2. never materializes the `(E, T, H)` `activations` tensor,
3. runs in **pure PyTorch** (`torch.bmm` / `torch.matmul` broadcast) so it
   executes on MPS with no Triton.

It is the *reference output* against which FREA/F2 are validated.

## 2. Inputs (all already available in the hook)

From `_hook_factory` (`src/reap/observer.py:315`) we already have:

| Name | Shape | Source |
|---|---|---|
| `flat_input` | (T, H) | `input.view(-1, hidden_dim)` (`observer.py:339`) |
| `router_logits` | (T, E) | from `output` tuple (`observer.py:377`) |
| `selected_experts` | (T, top_k) | `torch.topk(router_logits, top_k)` (`observer.py:378`) |
| `routing_weights` | (T, E) | `F.softmax(router_logits)` (computed in `update_pruning_state`) |

The expert weights are accessed via the adapter:
`adapter.expert_weight_attrs()` (`src/reap/model_adapters.py:191`) returns the
per-expert weight attribute names:

```python
# src/reap/model_adapters.py:197  (Qwen3)
return {
    "experts": "experts", "gate": "gate", "fused": False,
    "gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj",
}
```

So for a non-fused `Qwen3MoeSparseMoeBlock`, `moe.experts[i].gate_proj.weight`
is `(I, H)`, `moe.experts[i].up_proj.weight` is `(I, H)`,
`moe.experts[i].down_proj.weight` is `(H, I)`.

## 3. Weight stacking (the F4-lite step inlined here)

For the bmm we need the weights as stacked `(E, …)` tensors. For the baseline
this is a one-time `torch.stack` per layer per calibration run (F4 in Phase 5
caches it; here we just build it):

```python
def stack_expert_weights(moe, attrs):
    """Stack per-expert Linear weights into (E, out, in) tensors. Non-fused path."""
    experts = getattr(moe, attrs["experts"])          # ModuleList
    W_gate = torch.stack([getattr(e, attrs["gate_proj"]).weight for e in experts])  # (E, I, H)
    W_up   = torch.stack([getattr(e, attrs["up_proj"]).weight   for e in experts])  # (E, I, H)
    W_down = torch.stack([getattr(e, attrs["down_proj"]).weight for e in experts]) # (E, H, I)
    return W_gate, W_up, W_down
```

> ⚠️ `nn.Linear.weight` is stored as `(out_features, in_features)`, so
> `expert(x) = x @ weight.T + bias`. For the bmm we either transpose to
> `(E, H, I)` or use `.T` semantics carefully. The snippets below use the
> `(out, in)` convention and matmul accordingly.

## 4. The routed-only bmm

```python
def routed_expert_activations_bmm(
    flat_input,        # (T, H)
    selected_experts,  # (T, k)
    routing_weights,   # (T, E)   -- softmax over router_logits
    W_gate, W_up, W_down,  # (E, I, H), (E, I, H), (E, H, I)  [Linear.weight shape, out×in]
    top_k,
):
    T, H = flat_input.shape
    E = W_gate.shape[0]

    # 1. Build the routed (token, expert) pair list. For each token t, its top_k
    #    experts are selected_experts[t]. Flatten to (T*k,) pair indices.
    #    pair_token_idx[t*k + j] = t,  pair_expert_idx[t*k + j] = selected_experts[t, j]
    pair_token_idx = torch.arange(T, device=flat_input.device).repeat_interleave(top_k)  # (T*k,)
    pair_expert_idx = selected_experts.reshape(-1)                                     # (T*k,)

    # 2. Gather the routed inputs: (T*k, H). Use index_select (no 2GB copy).
    routed_x = flat_input.index_select(0, pair_token_idx)                              # (T*k, H)

    # 3. Per-pair expert matmuls via batched gather + bmm.
    #    nn.Linear: y = x @ W.T  -> for the bmm we use W of shape (E, in, out) so
    #    that  Wg[e] @ routed_x[p]  =  (I, H) @ (H,) = (I,). We transpose once.
    Wg = W_gate.transpose(-1, -2)   # (E, H, I)
    Wu = W_up.transpose(-1, -2)     # (E, H, I)
    Wd = W_down                     # (E, H, I) already (out=H, in=I) -> need (E, I, H)
    Wd = W_down.transpose(-1, -2)   # (E, I, H)

    # Gather weights per pair: (T*k, H, I) by indexing the (E, H, I) stack.
    Wg_p = Wg.index_select(0, pair_expert_idx)   # (T*k, H, I)
    Wu_p = Wu.index_select(0, pair_expert_idx)   # (T*k, H, I)
    Wd_p = Wd.index_select(0, pair_expert_idx)   # (T*k, I, H)

    # bmm: (T*k, 1, H) @ (T*k, H, I) -> (T*k, 1, I)
    g = torch.bmm(routed_x.unsqueeze(1), Wg_p).squeeze(1)            # (T*k, I)
    u = torch.bmm(routed_x.unsqueeze(1), Wu_p).squeeze(1)           # (T*k, I)
    act = torch.nn.functional.silu(g) * u                           # (T*k, I)   SwiGLU
    out = torch.bmm(act.unsqueeze(1), Wd_p).squeeze(1)               # (T*k, H)

    # 4. Per-pair router weight: routing_weights[t, e] for each pair.
    pair_router_w = routing_weights[pair_token_idx, pair_expert_idx]  # (T*k,)

    return out, pair_expert_idx, pair_token_idx, pair_router_w
```

### What this replaces

| Site | Current code | Replaced by |
|---|---|---|
| `observer.py:378` | `for idx, expert in enumerate(module.experts): activations[idx] = expert(flat_input)` | `routed_expert_activations_bmm(...)` |
| `layerwise_observer.py` non-fused branch | same loop | same |
| `pruning_metrics.py:178` reduce loop | `for i in range(num_experts): active_mask = ...; selected_activations = activations[i, active_mask, :]` | reductions over the `(T*k,)` pair tensors directly |

### Reductions (Phase 1 keeps them in Python, F2 fuses them later)

Phase 1 still calls `update_pruning_state`, but we must hand it the routed-pair
view. The cleanest approach for the baseline is to **keep the existing
`(E, T, H)` contract** at the `update_pruning_state` boundary and have the bmm
write routed outputs into a sparse `(E, T, H)`-shaped buffer only at the routed
positions. But that reintroduces the 8.6 GB tensor.

**Better**: introduce a `update_pruning_state_routed(...)` variant in
`pruning_metrics.py` that consumes the pair tensors directly:

```python
def update_pruning_state_routed(
    layer_state, *, out, pair_expert_idx, pair_router_w,
    router_logits, selected_experts, num_experts, valid_token_mask=None,
    renormalize_router_weights=False,
):
    # expert_frequency, pairwise_expert_frequency, total_tokens (unchanged logic)
    # ean_norm = ||out||_2 per pair  -> (T*k,)
    ean_norm = torch.linalg.norm(out, dim=-1)                       # (T*k,)
    # scatter-add into per-expert accumulators (index_add)
    ean_sum = torch.zeros(num_experts, dtype=torch.float64, device=out.device)
    ean_sum.index_add_(0, pair_expert_idx, ean_norm)
    weighted_ean_sum = torch.zeros(num_experts, dtype=torch.float64, device=out.device)
    weighted_ean_sum.index_add_(0, pair_expert_idx, ean_norm * pair_router_w)
    weighted_freq = torch.zeros(num_experts, dtype=torch.float64, device=out.device)
    weighted_freq.index_add_(0, pair_expert_idx, pair_router_w)
    # max_activations via scatter-reduce (index_reduce_ on torch>=2.0, or a loop)
    max_a = layer_state["max_activations"]
    per_pair_max = ean_norm                                     # use norm (matches existing .max over H)
    # ... (full mirroring of pruning_metrics.py:178)
```

> The **exact** reduction choice must mirror `pruning_metrics.py:178` *exactly*:
> `ean_norm = torch.linalg.norm(selected_activations, dim=-1)`, then
> `ean_sum[i] = ean_norm.sum()`, `ean_mean[i] = ean_norm.mean()`,
> `weighted_ean_sum[i] = (ean_norm * active_router_weights).sum()`,
> `reap[i] = (ean_norm * active_router_weights).mean()`,
> `max_activations[i] = max(selected_activations.max())`. The baseline's
> per-pair `ean_norm` is the same `torch.linalg.norm(out, dim=-1)`; the
> scatter-adds reproduce the per-expert sums/means/max. **This fidelity is the
> whole point of the parity test.**

## 5. Backend selection (the pattern every kernel site uses)

```python
# src/reap/kernel_backend.py (new)
import torch
try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

def select_expert_activation_backend():
    if torch.cuda.is_available() and _HAS_TRITON:
        return "frea"          # Phase 3
    return "bmm"               # Phase 1 baseline (MPS, CPU, or no-Triton CUDA)
```

The observer's non-fused branch becomes:

```python
backend = select_expert_activation_backend()
if backend == "frea":
    from reap.kernels.frea import frea_observe   # Phase 3
    frea_observe(self.state[layer_number], moe, flat_input, ...)
else:
    from reap.kernel_bmm import routed_observe_bmm   # Phase 1
    routed_observe_bmm(self.state[layer_number], moe, flat_input, ...)
```

The `loop` backend (current code) is kept behind an explicit `--observe-backend
loop` debug flag for the parity test to compare against.

## 6. Parity contract (the gate for Phase 1)

New file `tests/test_kernel_parity_bmm.py`:

```python
"""The bmm baseline must reproduce the loop observer's per-layer state
bit-for-bit (within fp32 accumulation tolerance) on a tiny Qwen3-MoE. This is
the oracle for all Triton kernels (FREA/F2)."""
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig

CONSUMED = ["total_tokens","expert_frequency","ean_sum","ean_mean","reap",
            "weighted_ean_sum","weighted_expert_frequency_sum","max_activations",
            "pairwise_expert_frequency"]

def _run(backend):
    cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=16, intermediate_size=16,
        moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=False)
    torch.manual_seed(0); model = Qwen3MoeForCausalLM(cfg).eval()
    batch = {"input_ids": torch.randint(0,64,(4,16)), "attention_mask": torch.ones(4,16,dtype=torch.long)}
    adapter = infer_model_adapter(model, model.config)
    hc = MoETransformerObserverConfig(module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=False, record_pruning_metrics_only=True, observe_backend=backend)
    obs = MoETransformerObserver(model, hook_config=hc, adapter=adapter)
    with obs.set_attention_mask(batch["attention_mask"]): _ = model(**batch)
    s = obs.report_state(); obs.close_hooks()
    return s

def test_bmm_matches_loop():
    loop = _run("loop"); bmm = _run("bmm")
    for layer in loop:
        for k in CONSUMED:
            a, b = loop[layer][k], bmm[layer][k]
            if isinstance(a, torch.Tensor):
                assert torch.allclose(a.to(torch.float32), b.to(torch.float32), atol=1e-5), \
                    f"layer {layer} key {k}: {a} vs {b}"
```

## 7. Expected improvement (vs the loop, per `00-cost-model.md` §7)

| Metric | Loop | bmm baseline | Delta |
|---|---|---|---|
| Expert matmul launches / layer | 384 | **3** | 128× fewer |
| Expert matmul launches / forward | 18,432 | **144** | 128× fewer |
| Expert FLOPs / layer | T × 1.21 GFLOP | T × 75.7 MFLOP | **16× less** (routed) |
| `(E,T,H)` transient / layer | 8.6 GB (T=8192) | ~50 MB (`(T*k, I)` = 65k×768×4) | **~170× less** |
| Expert loops / layer | 2 | 0 (fused into bmm + scatter) | both gone |

The bmm baseline is **already** the largest single win and the safest: it is
pure PyTorch, runs on MPS, and its correctness is provable against the loop.

## 8. Memory caveat: the `index_select` weight gather

`Wg.index_select(0, pair_expert_idx)` produces a `(T*k, H, I)` tensor — for
T=8192, k=8, H=2048, I=768, bf16: 8192×8×2048×768×2 ≈ **205 GB**. **This is
worse than the loop.** The naive bmm trades the `(E,T,H)` activation tensor
for an even larger `(T*k, H, I)` weight-gather tensor.

### Fix: don't gather weights — gather nothing; use grouped bmm by expert

The correct Phase-1 implementation **groups pairs by expert** (a sort by
`pair_expert_idx`) and runs one `bmm` per *expert that actually received
tokens* — but with the stacked `W_gate[e]` reused, and the routed inputs for
that expert batched. This keeps peak memory at max-over-expert
`(|pairs_e|, H) @ (H, I)` which is tiny.

```python
# Group routed pairs by expert, run one bmm per active expert (no weight gather).
order = torch.argsort(pair_expert_idx)                       # sort pairs by expert
pair_expert_idx = pair_expert_idx[order]
routed_x = routed_x[order]
pair_router_w = pair_router_w[order]
# segment boundaries where expert changes:
boundaries = torch.searchsorted(pair_expert_idx, torch.arange(E+1, device=...))
out = torch.empty(T*k, H, device=...)
for e in range(E):
    s, t = boundaries[e].item(), boundaries[e+1].item()
    if s == t: continue
    xe = routed_x[s:t]                                        # (n_e, H)
    g = torch.nn.functional.linear(xe, W_gate[e])             # (n_e, I)  -- uses (I,H) weight
    u = torch.nn.functional.linear(xe, W_up[e])
    a = torch.nn.functional.silu(g) * u
    out[s:t] = torch.nn.functional.linear(a, W_down[e])       # (n_e, H)
```

This is the form that actually wins: **per-expert grouped `nn.functional.linear`**
over only routed tokens, with stacked weights. Peak memory ≈ max-over-e
`(|pairs_e|, I)` ≈ a few MB. FREA (Phase 3) fuses this grouping + the matmuls +
the reductions into one kernel, removing the per-expert launches entirely.

## 9. Acceptance

- `tests/test_kernel_parity_bmm.py` passes (bmm vs loop, bit-for-bit on
  consumed metrics).
- The grouped-bmm form is used (not the naive weight-gather form), so peak
  activation memory is < 100 MB/layer for T=8192.
- `--observe-backend bmm` runs end-to-end through `python -m reap.prune` on
  the Mac (MPS) without CUDA/Triton.
- `--observe-backend loop` is still available for the parity comparison.