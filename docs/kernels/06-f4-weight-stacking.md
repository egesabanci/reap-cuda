# 06 — Phase 5: F4 Expert Weight Pre-Stacking Cache

> **Status: LANDED** — `src/reap/kernels/weight_cache.py`  
> Used by every non-loop observe path and by fused loop activations helper.

> **Concern:** present expert weights as contiguous linear-convention stacks so
> FREA (Triton or PyTorch) never branches on HF layout.

## API

```python
get_stacked_expert_weights(moe, adapter, device=None, dtype=None)
→ { "W_gate": (E,I,H), "W_up": (E,I,H), "W_down": (E,H,I), "fused", "weight_convention": "linear" }

free_cache(moe|None)
```

Cache key: `id(moe)`. Layerwise observer frees after each MoE block process.

## Layouts

| Source | Native | F4 output |
|---|---|---|
| Non-fused ModuleList | per-expert Linear | `torch.stack` weights |
| Qwen/LFM2 fused | `gate_up (E, 2I, H)`, `down (E, H, I)` | split gate/up views + down |
| Llama4 fused | `gate_up (E, H, 2I)`, `down (E, I, H)` **bmm** | transpose to linear |

Adapter provides `weight_convention` and `expert_weight_attrs(moe)`.

## Memory

~**1.2 GB/layer** bf16 for E=128, H=2048, I=768 (stack copy for non-fused;
fused linear may share storage on splits). One layer at a time in layerwise mode.

## Correctness

Does not change numeric values. Validated by FREA/bmm parity and
`tests/test_f4_weight_cache.py`.

## Merge path

F4 is **observe-only**. Merge/permute read original parameters via
`expert_weight_attrs` and must not rely on a stale stack — call `free_cache`
before mutating experts (observer close / layerwise free already do this).
