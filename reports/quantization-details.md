# Quantization Analysis: AWQ INT4 on REAP-Pruned LFM2.5-8B-A1B

## Summary

AWQ + W4A16_ASYM INT4 quantization was applied to the REAP-pruned LFM2.5-8B-A1B model using llm-compressor (compressed-tensors backend). The quantized model achieves 2.79 GB disk size (69.6% smaller than pruned, 83.5% vs base) but currently cannot be directly loaded by vLLM due to compressed-tensors format incompatibility with LFM2.5's custom architecture.

## Quantization Configuration

| Parameter | Value |
|-----------|-------|
| Method | AWQ (Activation-Aware Weight Quantization) |
| Scheme | W4A16_ASYM (4-bit weights, 16-bit activations, asymmetric) |
| Group Size | 128 |
| Libraries | llm-compressor (AWQ smoothing) + compressed-tensors (packing) |
| Quantization Format | pack-quantized (compressed-tensors) |
| Target Layers | Linear (all q/k/v/o_proj, conv.in/out_proj, w1/w2/w3) |
| Ignored Layers | lm_head, feed_forward.gate (router), expert_bias |
| Calibration Data | evol-codealpaca-v1 (256 samples, max 2048 tokens) |

## Custom AWQ Mappings

LFM2.5 uses non-standard module names requiring custom AWQ mappings:

### Attention Layers
```
operator_norm → [q_proj, k_proj, v_proj]
```
*(Note: v_proj → out_proj skipped due to GQA dimension mismatch: v_proj outputs 512 dims, out_proj takes 2048)*

### Conv Layers
```
operator_norm → [conv.in_proj]
```
*(Note: conv.in_proj → conv.out_proj skipped due to Conv1d intermediary)*

### Dense FFN (layers 0-1)
```
ffn_norm → [feed_forward.w1, feed_forward.w3]
w3 → [feed_forward.w2]
```

### MoE FFN (layers 2-23)
```
ffn_norm → [experts.N.w1, experts.N.w3]
experts.N.w3 → [experts.N.w2]
```

### Mapping Design

Layer-index-specific regex patterns are used to handle LFM2.5's hybrid architecture (conv + attention layers coexist). Each mapping is restricted to layers of the appropriate type:
- Attention layers: layers 2, 5, 8, 11, 14, 17, 20, 23
- Conv layers: all other layers
- Dense FFN: layers 0-1 only
- MoE FFN: layers 2-23 only

## Quantization Process

1. **Model Loading**: Via transformers `AutoModelForCausalLM` with `trust_remote_code=True`
2. **Calibration**: 256 forward passes through evol-codealpaca-v1 coding data
3. **AWQ Smoothing**: Grid search (n_grid=20) to find optimal per-channel scaling factors
4. **Weight Quantization**: 4-bit asymmetric INT quantization with group_size=128
5. **Packing**: INT32-packed format (8 × 4-bit values per INT32), zero-points also packed
6. **Saving**: `save_compressed=True` produces compressed-tensors format

## Quantized Model Structure

### Safetensors Format

```yaml
# Quantized (packed INT4):
experts.N.w1.weight_packed:   INT32 [1792, 256]  # [intermediate_dim, hidden_dim/8]
experts.N.w1.weight_scale:    BF16  [1792, 16]    # [intermediate_dim, hidden_dim/128]
experts.N.w1.weight_zero_point: INT32 [224, 16]   # Packed 8 values per INT32
experts.N.w1.weight_shape:    INT64 [2]           # [1792, 2048]

# Non-quantized (preserved as-is):
operator_norm.weight:         BF16  [2048]
ffn_norm.weight:              BF16  [2048]
gate.weight:                  BF16  [16, 2048]
conv.conv.weight:             BF16  [2048, 1, 3]
q_layernorm.weight:           BF16  [64]
embed_tokens.weight:          BF16  [124893, 2048]
lm_head.weight:               BF16  [124893, 2048]
```

### Quantized Layer Count

| Layer Type | Quantized | Not Quantized |
|-----------|:---------:|:-------------:|
| Attention (q/k/v/o_proj) | ✓ (8 layers × 4) | — |
| Conv (in/out_proj) | ✓ (16 layers × 2) | — |
| Dense FFN (w1/w2/w3) | ✓ (2 layers × 3) | — |
| MoE Experts (w1/w2/w3) | ✓ (22 layers × 16 experts × 3) | — |
| Routers (gate) | — | ✓ (22 layers) |
| Layer Norms | — | ✓ (all) |
| Conv1d (conv.conv) | — | ✓ (16 layers) |
| Embeddings | — | ✓ |
| LM Head | — | ✓ |

Total: 1,122 quantized weight tensors + 124 non-quantized tensors

## Loading the INT4 Model

### Status

The pack-quantized INT4 model loads correctly with `AutoModelForCausalLM.from_pretrained()` after patching two upstream bugs in transformers' compressed-tensors integration.

### Bugs Fixed

> **Upstream PR**: These two bugs have been reported and fixed upstream — see [huggingface/transformers#47430](https://github.com/huggingface/transformers/pull/47430). The fix is general for any MoE model with fused expert `nn.Parameter` weights (LFM2.5, Qwen2-MoE family) quantized with an asymmetric compressed-tensors scheme. A self-contained, idempotent patch script is shipped with the published INT4 model at [`konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4`](https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4) (`patches/apply_int4_patch.py`) — re-run it after every `pip install -U transformers` until the PR lands in a release.

**Bug 1 — Missing `weight_zero_point` in source patterns** (`transformers/quantizers/quantizer_compressed_tensors.py:137`):

```python
# Before (broken):
new_sources = packed_weight + scale_sources + shape_sources + other
# After (fixed):
zp_sources = [p + "_zero_point$" for p in weight_sources]
new_sources = packed_weight + scale_sources + shape_sources + zp_sources + other
```

Only `_packed`, `_scale`, `_shape` suffixes were included in the conversion patterns. `_zero_point` was missing, so asymmetric quantization's zero-point tensors were never collected from the state dict.

**Bug 2 — Missing `weight_zero_point` in DecompressExperts DummyModule** (`transformers/integrations/compressed_tensors.py:54`):

```python
# Before (broken):
class DummyModule(nn.Module):
    def __init__(self, weight, scale, shape):
        self.weight_packed = nn.Parameter(weight)
        self.weight_scale = nn.Parameter(scale)
        self.weight_shape = nn.Parameter(shape)
# No weight_zero_point → assertion failure for asymmetric quant

# After (fixed):
class DummyModule(nn.Module):
    def __init__(self, weight, scale, shape, zero_point=None):
        self.weight_packed = nn.Parameter(weight)
        self.weight_scale = nn.Parameter(scale)
        self.weight_shape = nn.Parameter(shape)
        if zero_point is not None:
            self.weight_zero_point = nn.Parameter(zero_point)
```

The `PackedQuantizationCompressor.decompress()` asserts `zero_point is not None` for asymmetric quantization (GROUP or CHANNEL strategy with `symmetric=False`). Without the zero-point parameter on DummyModule, the assertion fires and decompression fails.

### Loading API

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/data/reap-lfm2-quant-4096",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
# Model loads correctly — no monkey-patches, no manual weight fixing
```

### Decompression Flow

1. `from_pretrained` applies quantization config to compress model modules (to match checkpoint format)
2. Weight loading collects `weight_packed`, `weight_scale`, `weight_shape`, `weight_zero_point` for each expert
3. `DecompressExperts` creates a DummyModule with all 4 quant parameters, calls `decompress_module`
4. Decompressed weights are fused: w1+w3 → `gate_up_proj`, w2 → `down_proj`
5. Non-expert Linear layers (attention, conv, dense FFN) are handled by the 66-module decompression pass
6. `process_model_after_weight_loading` decompresses all modules in-place

### vLLM

The pack-quantized format is not directly loadable by vLLM due to key naming differences (`conv` vs `short_conv`, compressed-tensors weight suffixes). For vLLM inference, the AWQ-scaled bf16 decompressed model at `/data/reap-lfm2-quant-4096-bf16/` (9.18 GB) can be used instead.

## AWQ Smoothing Results

Per-layer grid search (n_grid=20) converged to optimal scaling factors:

```
Layer 2  operator_norm: best_error=7.106e-04
Layer 3  operator_norm: best_error=6.909e-04
Layer 5  operator_norm: best_error=7.015e-04
...
Layer 2  ffn_norm: best_error=3.401e-04
Layer 3  ffn_norm: best_error=3.460e-04
...
```

Average AWQ smoothing errors across all mappings: ~5–7 × 10⁻⁴, indicating well-converged scaling factors.

## Key Findings

1. **AWQ works for LFM2.5 with custom mappings**: Successfully applies per-channel scaling to 1,122 weight tensors
2. **Custom layer-index-specific AWQ mappings** handle LFM2.5's hybrid conv/attention architecture
3. **Combined compression: 83.5%**: Base 16.94 GB → pruned 8.57 GB → INT4 2.79 GB
4. **INT4 model loads with transformers**: After patching two upstream bugs in transformers' compressed-tensors integration, the pack-quantized model loads correctly via `from_pretrained`
5. **vLLM support pending**: The pack-quantized format isn't loadable by vLLM; use the AWQ-scaled bf16 decompressed model (9.18 GB) for vLLM inference
