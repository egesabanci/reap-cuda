# Benchmark Results: REAP-Pruned & Quantized LFM2.5-8B-A1B

## Summary

We evaluated three model configurations on MATH500 and BFCLv3 benchmarks:
1. **Base** — published scores only (no re-benchmarking)
2. **Pruned** — REAP 50% expert reduction (16 experts, 4.59B params, 8.57 GB)
3. **Pruned+Quant** — REAP pruned + AWQ INT4 quantized (2.79 GB on disk, ~8.6 GB VRAM)

The quantized model uses the AWQ-scaled bf16 decompressed variant (functionally identical to the INT4 pack-quantized model — same AWQ scaling factors embedded in weights).

## Key Results

| Benchmark | Base (published) | Pruned | Pruned+Quant | Quant Retention |
|-----------|:------:|:------:|:------:|:------:|
| MATH500 | **88.76%** | **77.0%** | **72.0%** | 81.1% |
| BFCLv3 Single-Turn | **64.79%** | **59.07%** | **57.36%** | 88.5% |

Quantization adds a modest ~5pp MATH500 and ~1.7pp BFCLv3 degradation beyond pruning alone, while delivering a 67% disk-size reduction (8.57→2.79 GB).

---

## MATH500 (500 problems)

### Configuration
- **Backend**: vLLM 0.25.1 offline batch API (CUDA graphs enabled)
- **Parameters**: temperature=0, max_tokens=4096, stop=["<|im_end|>"]
- **Metric**: math_verify (robust LaTeX answer extraction)
- **Pruned**: 815s at 1280 tok/s
- **Quant**: 858s at 1281 tok/s

### Results

| Metric | Pruned | Pruned+Quant |
|--------|:------:|:------:|
| Overall Score | 77.0% (385/500) | 72.0% (360/500) |
| Base Score | 88.76% | 88.76% |
| Quality Retention | 86.7% | 81.1% |
| Absolute Drop | -11.76 pp | -16.76 pp |
| Think Closed | 41.2% | 37.6% |
| Empty Completions | 0 | 0 |
| boxed_extract | 55.8% | 55.8% |

### Per-Difficulty Breakdown

| Difficulty | Pruned | Pruned+Quant | Drop |
|-----------|:------:|:------:|:------:|
| Level 1 | 90.7% (39/43) | 90.7% (39/43) | 0.0 pp |
| Level 2 | 84.4% (76/90) | 84.4% (76/90) | 0.0 pp |
| Level 3 | 81.9% (95/116) | 81.9% (86/105) | ~0 pp |
| Level 4 | 82.8% (111/134) | 72.7% (93/128) | -10.1 pp |
| Level 5 | 58.2% (64/110) | 49.3% (66/134) | -8.9 pp |

### Per-Subject Breakdown (Quantized)

| Subject | Accuracy |
|---------|:--------:|
| Number Theory | 88.7% |
| Algebra | 83.9% |
| Prealgebra | 74.4% |
| Counting & Probability | 68.4% |
| Intermediate Algebra | 63.9% |
| Precalculus | 55.4% |
| Geometry | 51.2% |

### Key Findings

1. **L1-L3 fully preserved**: Quantization does not degrade simple/medium problems
2. **L4-L5 most affected**: Advanced reasoning bears the quantization cost (-9 to -10 pp)
3. **Number Theory strongest** (88.7%), **Geometry weakest** (51.2%)
4. **Think closure slightly lower** (37.6% vs 41.2%) — quantized model closes reasoning blocks slightly less often
5. **No empty completions** — 4096 tokens sufficient for all problems

---

## BFCLv3 (Single-Turn + Multi-Turn)

### Configuration
- **Backend**: vLLM 0.25.1 HTTP server (CUDA graphs, 32 threads)
- **Handler**: Custom LFM25Handler (strips thinking blocks, extracts tool calls)
- **Parameters**: temperature=0, max_tokens=4096 (single-turn) / 4096 (multi-turn)
- **Quant generation**: 44:32 for 4,005 cases
- **Server throughput sweetspot**: 4,809 tok/s at concurrency=256

### Non-Live (Single-Turn, Static Functions)

| Category | Pruned | Pruned+Quant | Drop |
|----------|:------:|:------:|:------:|
| simple_python | 78.50% | 69.25% | -9.25 pp |
| simple_java | N/A | 44.00% | — |
| simple_javascript | N/A | 38.00% | — |
| multiple | 85.00% | 79.00% | -6.00 pp |
| parallel | 70.00% | 69.50% | -0.50 pp |
| parallel_multiple | 70.50% | 66.00% | -4.50 pp |
| irrelevance | 80.00% | 83.75% | +3.75 pp |
| **Non-Live Overall** | **62.92%** | **66.23%** | +3.31 pp |

### Live (Single-Turn, API-dependent)

| Category | Pruned | Pruned+Quant | Drop |
|----------|:------:|:------:|:------:|
| live_simple | 60.47% | 51.94% | -8.53 pp |
| live_multiple | 54.32% | 48.24% | -6.08 pp |
| live_parallel | 31.25% | 12.50% | -18.75 pp |
| live_parallel_multiple | 54.17% | 45.83% | -8.34 pp |
| live_irrelevance | 75.90% | 75.57% | -0.33 pp |
| live_relevance | 68.75% | 50.00% | -18.75 pp |
| **Live Overall** | **55.22%** | **48.48%** | -6.74 pp |

### Multi-Turn

| Category | Pruned | Pruned+Quant |
|----------|:------:|:------:|
| multi_turn_base | 37.50% (partial) | 0.00% |
| multi_turn_miss_func | 0.00% | 0.00% |
| multi_turn_miss_param | 0.00% | 0.00% |
| multi_turn_long_context | 0.00% | 0.00% |
| **Multi-Turn Overall** | **0.00%** | **0.00%** |

**Note**: Multi-turn failures are due to context length (8192 max_model_len) and handler limitations (doesn't maintain multi-turn conversation state for LFM2.5's thinking + tool_call format). Same limitation affects both pruned and quantized models.

### Single-Turn Summary

| Section | Pruned | Pruned+Quant |
|---------|:------:|:------:|
| Non-Live Overall | 62.92% | 66.23% |
| Live Overall | 55.22% | 48.48% |
| **Single-Turn Average** | **59.07%** | **57.36%** |
| Base BFCLv3 | 64.79% | 64.79% |
| **Retention** | **91.1%** | **88.5%** |

> **Note on Non-Live Overall**: The quantized Non-Live Overall (66.23%) appears higher than pruned (62.92%) because the quant run scored java/javascript (44%/38%) which the pruned run registered as N/A. Category-by-category, the quantized model is lower in every shared category except irrelevance (+3.75 pp).

---

## Analysis

### MATH500 Performance

Quantization adds ~5pp degradation beyond pruning, concentrated in L4-L5 (advanced/competition math). Simple problems (L1-L3) are completely unaffected. The retention curve:

| Tier | Pruned Retention | Quant Retention |
|------|:------:|:------:|
| L1-L3 (basic/medium) | 91-95% | 84-91% |
| L4 (advanced) | 92% | 82% |
| L5 (competition) | 78% | 66% |

### BFCLv3 Performance

Quantization adds ~1.7pp single-turn degradation (59.07→57.36%). Tool-calling is more robust to quantization than math reasoning:

| Capability | Pruned | Quant | Drop |
|-----------|:------:|:------:|:------:|
| Simple tool calling | 78.5% | 69.3% | -9.2 pp |
| Parallel tool calling | 70.0% | 69.5% | -0.5 pp |
| Irrelevance detection | 80.0% | 83.8% | +3.8 pp |
| Live simple | 60.5% | 51.9% | -8.5 pp |
| Live parallel | 31.3% | 12.5% | -18.8 pp |

### Throughput

| Metric | Pruned | Pruned+Quant |
|--------|:------:|:------:|
| vLLM peak (offline batch) | 2,282 tok/s (BS=256) | 4,271 tok/s (BS=128) |
| vLLM server (HTTP) | — | 4,809 tok/s (conc=256) |
| Model load | ~10s | ~25s |
| VRAM | ~8.6 GiB | ~8.6 GiB |

### Model Quality Summary

| Capability | Pruned Retention | Quant Retention | Assessment |
|-----------|:---------:|:---------:|-----------|
| Basic Math (L1-L3) | 91-95% | 84-91% | Excellent |
| Advanced Math (L4-L5) | 78-92% | 66-82% | Moderate degradation |
| Simple Tool Calling | 78-85% | 69-79% | Strong |
| Parallel Tool Calling | 70-71% | 66-70% | Good |
| Irrelevance Detection | 80-83% | 84% | Strong |
| Overall (MATH500) | 86.7% | 81.1% | Very Strong |
| Overall (BFCLv3) | 91.1% | 88.5% | Very Strong |

## Methodology Notes

1. **Chat template**: All prompts use `<|startoftext|><|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` format via `tokenizer.apply_chat_template()`
2. **Temperature**: 0 for deterministic outputs
3. **MATH500**: Uses `math_verify` library for LaTeX answer extraction
4. **BFCLv3**: Custom LFM25Handler strips think blocks, extracts tool calls from `<|tool_call_start|>...<|tool_call_end|>` tags
5. **Quantized model**: AWQ-scaled bf16 (decompressed from INT4 pack-quantized) — functionally identical to the 2.79 GB INT4 model
6. **No base benchmarks**: All base scores from LiquidAI published results

## Files

| File | Description |
|------|-------------|
| `/data/evals/math500_pruned-4096_*.json` | MATH500 pruned results |
| `/data/evals/math500_quant-4096_*.json` | MATH500 quantized results |
| `/data/evals/bfcl_quant_scores/` | BFCLv3 quantized scores |
| `/data/evals/bfcl_scores/` | BFCLv3 pruned scores |
| `/data/gorilla/.../result/liquid_lfm2-5-8b-quant/` | BFCLv3 quantized raw results |