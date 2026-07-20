# Published Model Artifacts

The following model artifacts have been published to the `konic-labs` HuggingFace organization. They are the reproducible outputs of the REAP + AWQ compression pipeline documented in the other reports in this folder.

## HuggingFace Repos

| Model | HF Repo | Disk Size | Description |
|-------|---------|----------:|-------------|
| Pruned bf16 | [`konic-labs/LFM2.5-8B-A1B-REAP-50`](https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50) | 8.57 GB | REAP 50% expert pruning (intermediate stage) |
| **INT4 Quantized** (flagship) | [`konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4`](https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4) | **2.79 GB** | REAP pruning + AWQ INT4 quantization (full pipeline) |

## Lineage

```
LiquidAI/LFM2.5-8B-A1B  (base, 16.94 GB)
        │
        ▼  REAP prune (reap-cuda, 200 calibration samples, 50% compression)
        │
konic-labs/LFM2.5-8B-A1B-REAP-50  (8.57 GB, 16 experts/layer)
        │
        ▼  AWQ INT4 quantization (llm-compressor, W4A16_ASYM, group_size=128)
        │
konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4  (2.79 GB, 83.5% total compression)
```

## Benchmark Summary

| Benchmark | Base (published) | REAP-50 (pruned) | REAP-50-AWQ-INT4 (quantized) |
|-----------|:------:|:------:|:------:|
| MATH500 | 88.76% | 77.0% | 72.0% |
| BFCLv3 (single-turn) | 64.79% | 59.07% | 57.36% |
| **Quality retention** | — | 86.7–91.1% | 81.1–88.5% |
| **Total compression** | — | 45.5% | **83.5%** |

## Loading

### Pruned bf16 (`REAP-50`) — works out of the box

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "konic-labs/LFM2.5-8B-A1B-REAP-50",
    trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
)
```

Also loads natively in **vLLM** (the `Lfm2MoeForCausalLM` architecture is supported).

### INT4 Quantized (`REAP-50-AWQ-INT4`) — requires a one-time patch

The compressed-tensors `pack-quantized` format has two bugs in current `transformers`
that break asymmetric INT4 MoE expert loading. The repo ships a self-contained patch:

```bash
# Clone or download, then:
python patches/apply_int4_patch.py          # patches installed transformers (idempotent)
python patches/apply_int4_patch.py --check  # verify without modifying
python patches/apply_int4_patch.py --revert # restore from .orig backup
```

Then load normally:

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4",
    trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
)
```

> **Note**: The INT4 format saves **disk** space (2.79 GB vs 8.57 GB). At runtime
> both models decompress to ~8.6 GB bf16 in VRAM — there is no VRAM savings without
> vLLM INT4 kernel support (future work). For vLLM inference today, use the
> `REAP-50` bf16 model, or decompress the INT4 checkpoint to bf16 (see
> `patches/README.md` in the INT4 repo).

## Patch Details

The two bugs fixed by `apply_int4_patch.py` affect **any** MoE model with fused
expert `nn.Parameter` weights (e.g. LFM2.5, Qwen2-MoE family) quantized with an
asymmetric compressed-tensors scheme:

1. **Missing `weight_zero_point` in conversion source patterns**
   (`transformers/quantizers/quantizer_compressed_tensors.py`) — the MoE expert
   weight-conversion patterns included `_packed`, `_scale`, `_shape` but not
   `_zero_point`, so asymmetric zero-point tensors were never collected.

2. **Missing `weight_zero_point` in `DecompressExperts.DummyModule`**
   (`transformers/integrations/compressed_tensors.py`) — the temporary module
   used to decompress each expert had no zero-point parameter, so the
   decompressor asserted `zero_point is not None` and crashed with
   `"Asymmetric quant requires zero-point values"`.

Both are upstream bugs; the patch is a stopgap until they land in a transformers
release. Re-run the patch after every `pip install -U transformers`.

## Related Reports

| Report | Content |
|--------|---------|
| [`architecture-analysis.md`](./architecture-analysis.md) | Per-layer expert pruning maps, survival rates |
| [`benchmark-results.md`](./benchmark-results.md) | Full MATH500 + BFCLv3 scores (pruned & quantized) |
| [`compression-metrics.md`](./compression-metrics.md) | Size/VRAM/throughput numbers, throughput curve |
| [`quantization-details.md`](./quantization-details.md) | AWQ config, custom LFM2.5 mappings, patch details |

## Tools Used

| Stage | Tool | Repo |
|-------|------|------|
| REAP pruning | `reap-cuda` | https://github.com/egesabanci/reap-cuda |
| AWQ quantization | `llm-compressor` | https://github.com/vllm-project/llm-compressor |
| BFCLv3 evaluation | `berkeley-function-call-leaderboard` | https://github.com/ShishirPatil/gorilla |