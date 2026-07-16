"""Diagnose GPU memory of LFM2.5-8B forward (no observer) at a few seq lengths."""
import os, sys, pathlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import torch
from reap.residency import load_causal_lm, plan_load, estimate_model_bytes_from_module

MODEL = "/data/models/LiquidAI/LFM2.5-8B-A1B"
tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print(f"after tokenizer: alloc={torch.cuda.memory_allocated()/1024**3:.3f} GiB reserved={torch.cuda.memory_reserved()/1024**3:.3f} GiB")

model = load_causal_lm(MODEL, plan_load("gpu_full"), local_files_only=True)
wb = estimate_model_bytes_from_module(model)
print(f"after load: weights={wb/1024**3:.3f} GiB alloc={torch.cuda.memory_allocated()/1024**3:.3f} GiB reserved={torch.cuda.memory_reserved()/1024**3:.3f} GiB")

# Find biggest per-module param buffers
print("\nTop param buffers:")
params = [(n, p.numel()*p.element_size(), tuple(p.shape), p.dtype) for n,p in model.named_parameters() if p.is_cuda]
params.sort(key=lambda x: -x[1])
for n,sz,shape,dt in params[:8]:
    print(f"  {sz/1024**2:8.1f} MiB  {dt}  {shape}  {n}")

for L in (512, 1024, 2048):
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, tok.vocab_size, (1, L), device="cuda")
    mask = torch.ones_like(ids)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        del out
        torch.cuda.synchronize()
        print(f"fwd seq={L:4d}: peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.3f} GiB alloc={torch.cuda.memory_allocated()/1024**3:.3f} GiB reserved={torch.cuda.memory_reserved()/1024**3:.3f} GiB")
    except torch.OutOfMemoryError as e:
        print(f"fwd seq={L:4d}: OOM ({e})")
        torch.cuda.empty_cache()
        break