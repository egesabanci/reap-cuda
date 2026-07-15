# 06 — Phase 5: F4 Expert Weight Pre-Stacking Cache

> **Concern:** present per-expert `gate_proj`/`up_proj`/`down_proj` weights as
> contiguous `(E, …)` stacked tensors, cached for the lifetime of a layer's
> calibration, so FREA/F2 tiles can load one expert's weights once into SRAM
> with coalesced reads. F4 is the **layout adapter** between the HF module
> structure and the kernels. It is pure-PyTorch (runs on Mac) and is a
> prerequisite for FREA/F2 performance (not correctness — the kernels work
> without it but are slow).

## 1. Why F4 exists

FREA (Phase 3) loads one expert's three Linear weights into SRAM and streams
its routed tokens. If the weights are **scattered** — one `nn.Linear` per
expert in a `ModuleList`, each a separate tensor in arbitrary HBM locations —
the SRAM load is a gather: slow and uncoalesced. F4 stacks them once into a
contiguous `(E, …)` buffer so the load is a single coalesced read.

This is the fix for the anti-pattern called out in `02-bmm-baseline.md` §8
(the 205 GB weight-gather): F4 makes the stacked weights a **shared cache**,
read contiguously by every kernel tile.

## 2. The two layouts (from the adapter)

`adapter.expert_weight_attrs()` (`src/reap/model_adapters.py:191,333`) returns
the per-layout weight names. F4 reads these and builds the stacked tensors.

### Non-fused (Qwen3 in transformers 4.55, Mixtral) — `model_adapters.py:197`
```python
# src/reap/model_adapters.py:197
return {
    "experts": "experts", "gate": "gate", "fused": False,
    "gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj",
}
```
`moe.experts` is a `ModuleList[Qwen3MoeMLP]`; each `expert.gate_proj.weight` is
`(I, H)`, `expert.up_proj.weight` `(I, H)`, `expert.down_proj.weight` `(H, I)`.

### Fused (Llama4, Qwen3.5/3.6 in transformers ≥5.x) — `model_adapters.py:333`
```python
# src/reap/model_adapters.py:333
return {
    "experts": "experts", "gate": "gate", "fused": True,
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "down_proj": "down_proj",
}
```
`moe.experts` is a single fused module with `gate_up_proj` `(E, 2*I, H)` and
`down_proj` `(E, H, I)` — **already stacked**. F4 just splits `gate_up_proj`
into gate/up halves (views, no copy).

## 3. The F4 API

```python
# src/reap/kernels/weight_cache.py (new)
from typing import Any
import torch.nn as nn

_STACK_CACHE: dict[int, dict[str, torch.Tensor]] = {}

def get_stacked_expert_weights(moe: nn.Module, adapter: Any) -> dict[str, torch.Tensor]:
    """Return contiguous stacked expert weights for FREA/F2.

    Non-fused: stacks the ModuleList's gate_proj/up_proj/down_proj into
        W_gate (E, I, H), W_up (E, I, H), W_down (E, H, I)
    Fused:    views gate_up_proj into gate/up halves (no copy)
        W_gate (E, I, H), W_up (E, I, H), W_down (E, H, I)

    Cached on id(moe) for the layer's calibration lifetime; freed by free_cache().
    """
    attrs = adapter.expert_weight_attrs()
    key = id(moe)
    if key in _STACK_CACHE:
        return _STACK_CACHE[key]

    if attrs["fused"]:
        gate_up = getattr(moe, attrs["experts"]).gate_up_proj   # (E, 2*I, H) -- already stacked
        I = gate_up.shape[1] // 2
        W_gate = gate_up[:, :I, :]      # view (E, I, H)
        W_up   = gate_up[:, I:, :]       # view (E, I, H)
        W_down = getattr(moe, attrs["experts"]).down_proj          # (E, H, I)
    else:
        experts = getattr(moe, attrs["experts"])   # ModuleList
        W_gate = torch.stack([getattr(e, attrs["gate_proj"]).weight for e in experts])  # (E, I, H)
        W_up   = torch.stack([getattr(e, attrs["up_proj"]).weight   for e in experts])  # (E, I, H)
        W_down = torch.stack([getattr(e, attrs["down_proj"]).weight for e in experts])  # (E, H, I)

    stacked = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "fused": attrs["fused"]}
    _STACK_CACHE[key] = stacked
    return stacked

def free_cache(moe: nn.Module | None = None):
    """Drop the stacked-weight cache (call after a layer finishes calibrating)."""
    if moe is None:
        _STACK_CACHE.clear()
    else:
        _STACK_CACHE.pop(id(moe), None)
```

## 4. Where it plugs in

In the observer's FREA/F2 path (replacing the `module.experts` loop):

```python
# in _hook_factory (observer.py) or _process_moe_activations (layerwise_observer.py):
stacked = get_stacked_expert_weights(moe, self.adapter)   # F4: cached, contiguous
f2_observe(
    self.state[layer_number],
    flat_input,
    f5_router_outputs,
    stacked["W_gate"], stacked["W_up"], stacked["W_down"],   # FREA/F2 read these
    layer_cfg,
    compute_routed_ca=(prune_method == "ean_ca"),
)
```

And the layerwise observer calls `free_cache(moe)` after the block finishes
(`_forward_block` epilogue) to drop the cache before the next block moves to
GPU.

## 5. Memory cost & lifetime

For Qwen3-30B-A3B (E=128, H=2048, I=768), **bf16** weights:
- `W_gate + W_up + W_down` = E × (I×H + I×H + H×I) × 2 bytes
  = 128 × (768×2048 + 768×2048 + 2048×768) × 2
  = 128 × 4.72 M × 2 ≈ **1.2 GB per layer** in the cache.

This is **one layer at a time** in layerwise mode (the cache is keyed by `id(moe)`
and freed when the block offloads). In standard mode it is one layer's worth at
a time too (one hook fires per layer per forward; the cache persists across the
calibration batches for that layer, which is the point — avoid re-stacking).

**Trade-off**: the cache adds ~1.2 GB/layer transiently, but it **replaces**
the per-expert `Linear` weight reads that the loop would scatter. Net VRAM
during a layer's calibration is: block weights (~1–2 GB) + F4 cache (~1.2 GB)
+ stat buffers (~1 MB) ≈ well under the 46 GB L40S budget. For a 256-expert
model the cache is ~2.4 GB/layer — still fine one-layer-at-a-time.

> **Optional refinement**: for the non-fused path, the stacked tensors can
> **share storage** with the original `Linear.weight` via a custom
> `nn.Parameter` view if the experts are contiguous in memory — but stock HF
> `ModuleList` does not guarantee contiguity across experts, so F4 copies.
> The copy is a one-time ~ms cost per layer per calibration run, amortized over
> hundreds of batches.

## 6. Correctness (F4 does not change results)

F4 only changes **memory layout**, not values:
- Non-fused: `torch.stack` of identical `Linear.weight` tensors — the stacked
  `W_gate[e]` is bit-identical to `expert.gate_proj.weight`. Verified by the
  Phase-1 parity test (which uses the stacked weights).
- Fused: `gate_up_proj[:, :I, :]` is a **view** of the existing tensor — zero
  copy, bit-identical by construction.

So F4 has **no parity contract of its own** — it is validated transitively by
the FREA/F2 parity tests. The only F4-specific test is a shape/layout test:

```python
def test_f4_stacked_shapes():
    # tiny Qwen3 (non-fused) + tiny Llama4 (fused)
    # assert W_gate/W_up (E,I,H), W_down (E,H,I); assert fused path is a view (shares storage)
```

## 7. Interaction with the merge path

F4 is **prune-path only**. The merge path (`merge_pipeline.py`) reads expert
weights via `MoEExpertMerger` using `model_attrs` from `adapter.expert_weight_attrs()`
directly (it mutates weights in place — it must read/write the original
`Linear.weight`, not a stacked copy). F4's cache must be **freed** before merge
runs on a layer, to avoid the merger mutating a stale stacked copy. The
observer `close_hooks` / layerwise block-offload sites already call
`free_cache`.

## 8. Acceptance

- `get_stacked_expert_weights` returns `(E, I, H)` / `(E, I, H)` / `(E, H, I)`
  for both non-fused (Qwen3) and fused (Llama4) layouts.
- Fused path is a **view** (shares storage with `gate_up_proj`); non-fused
  path is a stack (copy).
- Cache is freed after each layerwise block (`free_cache(moe)`).
- FREA/F2 parity tests pass using F4-cached weights (transitive validation).