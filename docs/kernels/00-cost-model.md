# 00 — Cost Model: The Current Bottleneck

> **Concern:** quantify exactly what the stock observer does today, so each
> kernel doc can claim a concrete improvement against this baseline. No code
> changes here — this is the reference the whole kernel suite is measured
> against.

## 1. The two observer paths

REAP collects per-layer MoE statistics during a calibration forward pass.
There are two observer implementations, both of which contain the bottleneck:

| Observer | File | Entry point | Memory mode |
|---|---|---|---|
| Standard | `src/reap/observer.py` (`MoETransformerObserver`) | `python -m reap.prune`, `python -m reap.merge_pipeline` | whole model on GPU |
| Layerwise | `src/reap/layerwise_observer.py` (`LayerwiseMoEObserver`) | `python -m reap.layerwise_prune` | one decoder block on GPU |

Both call into the **same** `pruning_metrics.update_pruning_state` for the
saliency reductions, and both contain the **same expert-execution branch**
(fused vs non-fused). The bottleneck lives in that branch.

## 2. The non-fused expert loop (the bottleneck)

### Standard observer

`src/reap/observer.py:378` — inside `_hook_factory`'s `_hook_fn`:

```python
# src/reap/observer.py:378
else:  # loop based MoE execution
    *_, router_logits = output  # (total_tokens, num_experts)
    _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    for idx, expert in enumerate(module.experts):
        activations[idx] = expert(flat_input).to(
            device
        )  # (num_experts, total_seq_len, hidden_dim)
```

### Layerwise observer

`src/reap/layerwise_observer.py` (inside `_process_moe_activations`):

```python
# layerwise_observer.py, _process_moe_activations, non-fused branch:
            _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
            # Compute activations for all experts
            for idx, expert in enumerate(moe_module.experts):
                activations[idx] = expert(flat_input).to(device)
```

Both are identical in structure: a Python loop over **all E experts**, calling
each expert's forward (3 matmuls: `gate_proj`, `up_proj`, `down_proj` + SiLU),
writing into a pre-allocated `activations` tensor of shape
`(num_experts, total_seq_len, hidden_dim)`.

**Crucially**, the loop computes an expert's output for **every token**, not
just the tokens routed to it. The routing-aware masking happens *later*, in
`update_pruning_state`. So the loop does **E × T × (3 × 2 × H × I)** FLOPs of
expert work even though only **top_k × T** pairs are ever read.

## 3. Quantified cost (Qwen3-30B-A3B)

Per layer, per calibration forward (T = total tokens in the batch, e.g.
batch_size=4 × seq_len=2048 → T = 8192; but activations persist across batches
during a calibration run):

| Quantity | Formula | Value (E=128, top_k=8, H=2048, I=768) |
|---|---|---|
| Expert matmul launches / layer | E × 3 | **384** |
| Expert matmul launches / forward (48 layers) | 384 × 48 | **18,432** |
| Expert FLOPs / layer | T × E × 3 × 2 × H × I | T × 128 × 3 × 2 × 2048 × 768 = T × 1.21 GFLOP |
| Expert FLOPs / forward | × 48 | T × 58.1 TFLOP |
| **Routed** expert FLOPs / layer (what REAP needs) | T × top_k × 3 × 2 × H × I | T × 75.7 MFLOP |
| **Waste ratio** (loop / routed) | E / top_k | **16×** (32× for E=256) |
| `activations` tensor size / layer (fp32) | E × T × H × 4 | 128 × T × 2048 × 4 = T × 1.05 MB/token |
| `activations` tensor for T=8192 | | **8.6 GB / layer** |

Across a 1024-batch calibration run (T_total ≈ 8.4 M tokens):
- **Launches:** ~18,432 × 1024 ≈ **18.9 M launches**. At ~5–10 µs fixed launch
  overhead on L40S, that is **~1.5–3 minutes of pure launch tax** before any
  compute.
- **Wasted expert FLOPs:** 16× over the routed-only work; for a 256-expert
  model, 32×.

### The (E, T, H) materialization

The `activations` tensor (`src/reap/observer.py:351`,
`torch.zeros((num_experts, *flat_input.shape), device=device)`) is the single
largest transient. For E=128, T=8192, H=2048, fp32:

> 128 × 8192 × 2048 × 4 bytes ≈ **8.6 GB per layer**

The layerwise path only keeps **one block** on GPU, so it never holds all 48
layers' tensors — but even one 8.6 GB transient is the dominant VRAM consumer
and the first thing to OOM on a 256-expert model (17 GB / layer).

This tensor is exactly what FREA/F2 eliminate.

## 4. The fused branch (for reference)

`src/reap/observer.py:353` handles fused experts (Llama4-style, and stock-HF
fused Qwen3 in transformers ≥5.x):

```python
# src/reap/observer.py:353
if self.hook_config.fused_experts:
    _, router_scores = output  # (num_experts, total_tokens)
    router_logits = module.router(flat_input)
    _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    ...
    routed_out = module.experts(routed_in)
    activations = routed_out.view(num_experts, *flat_input.shape)
```

The fused branch is a **single** `module.experts(routed_in)` call (one launch),
but it still:
- re-runs the router (`module.router(flat_input)`) even though `output`
  already contains router scores,
- materializes the same `(E, T, H)` `activations` tensor,
- relies on a custom `module.router` / `module.experts` API that **stock HF
  fused Qwen3.5/3.6 does not provide** (see issue #4). So this branch is
  currently Llama4-only and even then untested at runtime.

FREA subsumes the fused branch by computing routed pairs directly, without
the stock HF fused call — making it work for both fused and non-fused layouts.

## 5. The saliency reductions (what consumes the activations)

`src/reap/pruning_metrics.py:133` `update_pruning_state` reads the
`(E, T, H)` `activations` tensor and reduces it to the per-layer state. The
inner loop (`src/reap/pruning_metrics.py:178`):

```python
# src/reap/pruning_metrics.py:178
for i in range(num_experts):
    active_mask = (pruning_batch.selected_experts == i).any(dim=-1).to(device)
    if not active_mask.any():
        continue
    selected_activations = pruning_batch.activations[i, active_mask, :]
    active_router_weights = routing_weights[active_mask, i]
    ean_norm = torch.linalg.norm(selected_activations, dim=-1)
    ean_sum[i] = ean_norm.sum().to(device)
    ean_mean[i] = ean_norm.mean().to(device)
    weighted_expert_frequency_sum[i] = active_router_weights.sum().to(device)
    weighted_ean_sum[i] = (ean_norm * active_router_weights).sum().to(device)
    reap[i] = (ean_norm * active_router_weights).mean().to(device)

    selected_activations_max = selected_activations.max().to(device="cpu")
    if selected_activations_max > layer_state["max_activations"][i]:
        layer_state["max_activations"][i] = selected_activations_max
```

This is a **second** Python `for i in range(num_experts)` loop — so each layer
pays **two** expert loops (one to compute activations, one to reduce them).
Each iteration rebuilds `active_mask = (selected_experts == i)` (an E-sized
boolean mask over T×top_k) — 128 redundant mask constructions per layer.

F2 (Phase 4) fuses these reductions into the FREA kernel so neither loop nor
the masks exist.

## 6. The consumed-metric contract (why routed-only is enough)

`src/reap/prune.py:52` consumes, per layer:

```python
# src/reap/prune.py:60
for layer in tqdm(observer_data, "Pruning layers..."):
    num_experts = observer_data[layer]["expert_frequency"].shape[0]
    if prune_args.prune_method == "ean_ca":
        ean[i] = torch.linalg.norm(
            observer_data[layer]["routed_characteristic_activation"][i], dim=-1
        ).sum()
    else:
        saliency_data = observer_data[layer].get(prune_method)  # 'reap','ean_sum',...
        _, experts_to_prune = torch.topk(saliency_data, n_experts_to_prune, largest=False)
```

| `--prune_method` | State key read | Reduction over |
|---|---|---|
| `frequency` | `expert_frequency` (E,) | routed token counts |
| `ean_sum` | `ean_sum` (E,) | routed activations |
| `ean_mean` | `ean_mean` (E,) | routed activations |
| `weighted_ean_sum` | `weighted_ean_sum` (E,) | routed × router weight |
| `weighted_frequency_sum` | `weighted_expert_frequency_sum` (E,) | routed router weights |
| `max_activations` | `max_activations` (E,) | routed activations |
| `reap` | `reap` (E,) | routed × router weight |
| `ean_ca` | `routed_characteristic_activation` (E,H) | routed activations |

**Every** consumed metric is a function of **routed `(token, top_k)` pairs only.**
The all-token "merging criteria" — `ttm_similarity_matrix` (E,E),
`characteristic_activation` (E,H), `online_characteristic_activation_dist`
(E,E), `router_logit_similarity` (E,E) — are written in
`src/reap/observer.py:277` (only when `record_pruning_metrics_only=False`) and
read **only** by `src/reap/merge_pipeline.py` / `src/reap/cluster.py` — never by
`prune.py`.

This is the foundational correctness fact for FREA and F2:
**on the prune path, no all-pairs tensor is ever needed.** Phase 0 makes that
contract explicit in code.

## 7. Baseline numbers used by the rest of this guide

| Metric | Loop (current) |
|---|---|
| Expert matmul launches / forward | 18,432 |
| Expert FLOPs / forward | T × 58.1 TFLOP (E=128) |
| Peak activation transient / layer | 8.6 GB (T=8192, E=128) |
| Expert loops / layer | 2 (compute + reduce) |
| Router-stage kernels / layer | ~6 (linear→softmax→topk→where→gather) |

The improvement tables in `08-expected-improvements.md` are all deltas against
these numbers.