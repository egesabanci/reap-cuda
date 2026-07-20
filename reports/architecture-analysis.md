# Architecture Analysis: REAP-Pruned LFM2.5-8B-A1B

## Summary

REAP (Residual Expert Activation Pruning) was applied to LiquidAI's LFM2.5-8B-A1B model (32 experts, 8.47B parameters) with 50% compression rate using 200 calibration samples from evol-codealpaca-v1. The pruned model retains 16 experts per MoE layer, selected independently per layer based on activation importance.

## Key Metrics

| Metric | Base Model | Pruned Model | Reduction |
|--------|-----------|-------------|-----------|
| Parameters | 8.47B | 4.59B | 45.8% |
| Disk Size (bf16) | 16.94 GB | 8.57 GB | 45.5% |
| MoE Experts | 32 (704 total) | 16 (352 total) | 50% |
| Router Params | 1.44M | 721K | 50% |
| Dense Params | 0.72B | 0.72B | 0% |
| Attention Params | Unchanged | Unchanged | 0% |
| Conv Params | Unchanged | Unchanged | 0% |
| Embeddings | Unchanged | Unchanged | 0% |
| LM Head | Unchanged | Unchanged | 0% |

## Architecture Details

### Layer Structure

LFM2.5-8B-A1B has 24 decoder layers with a hybrid architecture:

| Layer Type | Layers | Description |
|-----------|--------|-------------|
| Dense FFN | 0–1 | Standard feed-forward with w1/w3/w2 Linear |
| MoE FFN | 2–23 | Sparse MoE with 16 experts each (was 32) |
| Attention | 2, 5, 8, ... (every 3rd) | GQA: q_proj [2048,2048], k_proj [512,2048], v_proj [512,2048] |
| Conv | 0, 1, 3, 4, 6, 7, ... | ShortConv with in_proj [6144,2048], out_proj [2048,2048] |

### MoE Expert Structure

Each MoE layer uses `Lfm2MoeExperts`:
- `gate_up_proj`: fused w1+w3 → [num_experts, 2×1792, 2048]
- `down_proj`: w2 → [num_experts, 2048, 1792]
- `expert_bias`: [num_experts] router bias

Checkpoint stores individual weights:
- `experts.N.w1.weight`: [1792, 2048]
- `experts.N.w2.weight`: [2048, 1792]
- `experts.N.w3.weight`: [1792, 2048]

### Pruning Method

REAP scores each expert based on calibration data importance:
1. Runs calibration forward passes through all experts
2. Computes activation-based importance scores per expert per layer
3. Selects top 16 (50%) experts independently per layer
4. Copies selected expert weights unchanged (no weight modification)
5. Copies corresponding gate rows and expert_bias values unchanged
6. Reindexes kept experts as experts 0–15

**Key finding**: The pruned model is an exact subset — weights are bit-for-bit identical to the base model. No renormalization is applied to gate weights or expert weights. The "-renorm" flag in the run name refers to router probability normalization (norm_topk_prob), not weight adjustment.

## Per-Layer Expert Pruning Map

Each layer independently selects 16 out of 32 experts. The selection varies significantly across layers:

| Layer | Kept Experts (base indices) |
|-------|---------------------------|
| 2 | 0, 2, 3, 5, 7, 8, 9, 11, 12, 16, 18, 19, 21, 22, 23, 27 |
| 3 | 0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 14, 21, 23, 24, 29, 30 |
| 4 | 3, 4, 9, 12, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30 |
| 5 | 0, 1, 3, 6, 7, 9, 10, 12, 13, 14, 16, 22, 24, 25, 26, 27 |
| 6 | 2, 3, 4, 6, 7, 11, 14, 15, 20, 21, 23, 24, 25, 27, 28, 31 |
| 7 | 0, 1, 3, 6, 8, 9, 10, 11, 12, 13, 18, 19, 23, 25, 26, 27 |
| 8 | 1, 2, 12, 15, 16, 17, 19, 20, 21, 24, 25, 27, 28, 29, 30, 31 |
| 9 | 2, 4, 7, 9, 10, 11, 14, 15, 18, 19, 21, 26, 28, 29, 30, 31 |
| 10 | 2, 3, 9, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24, 26, 27, 30 |
| 11 | 2, 4, 5, 7, 8, 10, 14, 17, 19, 22, 23, 24, 26, 28, 29, 30 |
| 12 | 1, 2, 4, 5, 8, 9, 14, 16, 19, 20, 21, 26, 27, 28, 30, 31 |
| 13 | 2, 3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19, 25, 27, 31 |
| 14 | 0, 1, 2, 4, 5, 6, 9, 11, 15, 16, 20, 22, 25, 27, 28, 31 |
| 15 | 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 25, 28, 30, 31 |
| 16 | 0, 2, 6, 8, 15, 17, 19, 21, 22, 23, 24, 26, 27, 28, 30, 31 |
| 17 | 1, 3, 4, 6, 7, 9, 10, 13, 15, 16, 17, 21, 22, 26, 30, 31 |
| 18 | 0, 1, 2, 4, 6, 12, 13, 15, 16, 17, 18, 20, 21, 25, 26, 31 |
| 19 | 0, 5, 6, 7, 9, 11, 12, 14, 16, 21, 22, 23, 24, 27, 28, 31 |
| 20 | 0, 1, 2, 8, 10, 11, 16, 18, 19, 20, 22, 24, 26, 27, 28, 31 |
| 21 | 0, 1, 3, 4, 5, 12, 14, 15, 18, 19, 20, 23, 24, 27, 29, 30 |
| 22 | 5, 7, 8, 9, 11, 12, 13, 14, 16, 24, 25, 27, 28, 29, 30, 31 |
| 23 | 3, 5, 7, 8, 10, 14, 16, 17, 18, 19, 22, 25, 27, 28, 29, 30 |

## Expert Survival Rates

| Expert | Survived | Rate | Expert | Survived | Rate |
|--------|----------|------|--------|----------|------|
| 0 | 10/22 | 45% | 16 | 15/22 | 68% |
| 1 | 10/22 | 45% | 17 | 9/22 | 41% |
| 2 | 13/22 | 59% | 18 | 9/22 | 41% |
| 3 | 11/22 | 50% | 19 | 13/22 | 59% |
| 4 | 9/22 | 41% | 20 | 9/22 | 41% |
| 5 | 10/22 | 45% | 21 | 11/22 | 50% |
| 6 | 9/22 | 41% | 22 | 12/22 | 55% |
| 7 | 10/22 | 45% | 23 | 10/22 | 45% |
| 8 | 10/22 | 45% | 24 | 12/22 | 55% |
| 9 | 14/22 | 64% | 25 | 11/22 | 50% |
| 10 | 8/22 | 36% | 26 | 11/22 | 50% |
| 11 | 11/22 | 50% | 27 | **16/22** | **73%** |
| 12 | 12/22 | 55% | 28 | 13/22 | 59% |
| 13 | 8/22 | 36% | 29 | **7/22** | **32%** |
| 14 | 12/22 | 55% | 30 | 13/22 | 59% |
| 15 | 11/22 | 50% | 31 | 13/22 | 59% |

**Most preserved**: Expert 27 (73%), Expert 16 (68%), Expert 9 (64%)
**Most pruned**: Expert 29 (32%), Experts 10 & 13 (36%)

## Gate/Router Weight Analysis

Gate weights (`feed_forward.gate.weight`) are exact copies from the base model. For each MoE layer:
- Base gate: [32, 2048] → all 32 expert routing weights
- Pruned gate: [16, 2048] → subset of 16 rows corresponding to kept experts
- **No renormalization applied** — the 16 rows are exact copies of the corresponding base gate rows
- Expert bias (`feed_forward.expert_bias`) similarly preserved — 16 values copied from the 32

This means the router continues to use the same routing logits for the kept experts, just with half the capacity. The `norm_topk_prob` flag (true in the pruned config) ensures routing probabilities are properly normalized across the 4 selected experts.

## Key Architectural Observations

1. **Pure subset pruning**: No weight modification or fine-tuning needed
2. **Per-layer independent selection**: Each layer prunes different experts based on local importance
3. **Dense layers preserved**: Layers 0-1 (dense FFN) are not pruned
4. **Attention/cov layers preserved**: Only MoE expert parameters are reduced
5. **No expert is universally kept/pruned**: Some survive in 73% of layers, others in only 32%
6. **Deterministic pruning**: Same calibration data → same pruned experts
