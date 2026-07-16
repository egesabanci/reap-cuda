# Model Adapters

Adapters isolate **HuggingFace MoE layout** from the rest of REAP. Adding a
family should not require rewriting observers or kernels.

Implementation: `src/reap/model_adapters.py`.

## `MoeLayerConfig`

Frozen metadata returned by `get_layer_config(layer, config)`:

| Field | Meaning |
| --- | --- |
| `num_experts` | Routed expert count (prefer live stack length) |
| `top_k` | Experts per token (clamped to `num_experts`) |
| `norm_topk_prob` | Whether top-k scores are renormalized |
| `adapter_name` | String id (`qwen3_moe`, `llama4_moe`, …) |
| `fused_experts` | Stacked params vs `ModuleList` |
| `use_expert_bias` | LFM2-style per-expert router bias |
| `weight_convention` | `"linear"` (`F.linear` / out×in) or `"bmm"` (in×out) |

## Supported adapters

### Qwen3 (`Qwen3MoeModelAdapter`)

- MoE at `layer.mlp` (`Qwen3MoeSparseMoeBlock`)
- Router: `.gate` (often returns `(logits, scores, indices)`)
- Experts: fused `gate_up_proj (E, 2I, H)`, `down_proj (E, H, I)` on transformers≥5;
  legacy `ModuleList` of MLPs still detected
- Config keys: `num_experts`, `num_experts_per_tok`
- Convention: **linear**

### Qwen3.5 / 3.6 (`Qwen3_5MoeModelAdapter`)

- Subclass of Qwen3; hook class `Qwen3_5MoeSparseMoeBlock`
- **Shared expert** + `shared_expert_gate` — never sliced by prune
- Same fused stacks as Qwen3

### Llama4 (`Llama4MoeModelAdapter`)

- MoE at `layer.feed_forward` (`Llama4TextMoe`)
- Router attribute: **`.router`** (not `.gate`)
- Experts: `gate_up_proj (E, H, 2I)`, `down_proj (E, I, H)` — **bmm** convention
- F4 transposes into Linear form for kernels
- Shared expert present; not pruned

### Mixtral / PhiMoE (`MixtralMoeModelAdapter`)

- MoE at `layer.block_sparse_moe`
- Non-fused `ModuleList` experts with `gate_proj` / `up_proj` / `down_proj`
- Config: `num_local_experts`

### LFM2.5 (`Lfm2MoeModelAdapter`)

- MoE at `layer.feed_forward` (`Lfm2MoeSparseMoeBlock`)
- Router `.gate`; fused linear stacks
- Optional `expert_bias` on the MoE block — **must** be sliced with experts
- Dense early layers use MLP without experts; `is_moe_layer` filters them out

## Inference (`infer_model_adapter`)

1. If a live model is provided, **layout inspection wins** over `config.model_type`
   (order: LFM2 → Llama4 feed_forward → Mixtral → Qwen3.5 → Qwen3).
2. Config-only path uses `model_type` / `architectures` strings.
3. Returns `None` if no supported MoE layout is found.

## `slice_experts` contract

After slicing `keep_indices`:

1. Expert weights reduced on expert axis (dim 0).
2. Router weight (and bias if any) sliced.
3. Live counters updated: `experts.num_experts`, router `num_experts`,
   `top_k = min(top_k, retained)`, block `num_experts` if present.
4. LFM2 `expert_bias` sliced when present.
5. Shared experts left intact.

Without (3), fused Qwen forward fails (`one_hot(num_classes=self.num_experts)`
mismatch). Covered by `tests/test_fused_slice_forward.py`.

## `expert_weight_attrs`

Used by merge / permute / F4:

```python
{
  "experts": "experts",
  "gate": "<router attr>",
  "fused": bool,
  "gate_proj": "gate_proj" | "gate_up_proj",
  "up_proj": "...",
  "down_proj": "down_proj",
  "weight_convention": "linear" | "bmm",
}
```

Pass the live `moe` when available so Qwen fused stacks report `fused=True`
(not the historical non-fused default).

## Extending to a new family

1. Implement an adapter class with the methods above.
2. Register detection order in `infer_model_adapter`.
3. Ensure `hook_regex()` matches the MoE **module class name** exactly
   (observer match is exact class name).
4. Document weight convention for F4.
5. Add adapter unit tests (mock modules OK) and, if possible, a tiny forward +
   slice test.
6. Do **not** special-case the family inside `kernels/` — normalize via F4.

## Related

- [observation-and-metrics.md](observation-and-metrics.md)
- [gpu-and-backends.md](gpu-and-backends.md)
- [pruning.md](pruning.md)
