# 03 — Phase 2: F5 Router (softmax + topk + pair CSR)

> **Status: LANDED** — `src/reap/kernels/router.py`  
> Triton softmax: `triton_softmax.py` (`@triton.jit`)  
> Top-k + renorm + expert-sort CSR: always PyTorch

> **Concern:** produce the routed-pair layout FREA/F2 consume, without rebuilding
> `(selected_experts == i)` masks E times.

## Outputs (`RouterPairOutputs`)

| Field | Shape | Meaning |
|---|---|---|
| `selected_experts` | `(T, k)` or filtered | top-k ids (after optional padding filter) |
| `router_weights_full` | `(T, E)` | softmax (+ optional renorm) |
| `pair_token_idx` | `(n_pairs,)` | token index per pair (into original `flat_input`) |
| `pair_expert_idx` | `(n_pairs,)` | expert id per pair (**sorted** by expert) |
| `pair_router_w` | `(n_pairs,)` | weight per pair |
| `expert_offsets` | `(E+1,)` | CSR segment bounds |
| `pair_perm` | `(n_pairs,)` | sort permutation |

`n_pairs = T_valid * k` after attention-mask filtering.

## Implementation

1. **Softmax** — `softmax_rows(logits)`:
   - CUDA + Triton + `E ≤ 1024`: Triton online row softmax (fp32)
   - else: `F.softmax(..., dtype=torch.float32)`
2. **Top-k** — `torch.topk` on probabilities (monotone vs logits)
3. **Renorm** — if `norm_topk_prob`, divide full distribution by top-k mass (matches historical pruning_metrics)
4. **Pairs + CSR** — `repeat_interleave` token ids, `argsort` experts, `bincount` → offsets

Router logits from `extract_router_logits` (unwraps Qwen tuple `(logits, scores, indices)` → element 0).

## Why not full Triton top-k

Variable `top_k` and stable parity with PyTorch `topk` are simpler in PyTorch;
softmax is the fused win that matters for bandwidth. Pair sort is cheap vs MLP.

## Integration

Called from `observe_moe_batch` for all non-`loop` backends. Padding masks drop
invalid tokens’ pairs while keeping `pair_token_idx` into the **original**
flat sequence for FREA gather.

## Tests

- `tests/test_triton_kernels.py` — F5 shapes/CSR; CUDA softmax parity when available
- End-to-end metrics via `test_kernel_parity_bmm.py`

## Expected impact

| | Effect |
|---|---|
| Perf | Small absolute; large **layout** win for FREA |
| Mem | `O(T·k)` pairs — MB-scale |
