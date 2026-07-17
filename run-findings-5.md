# run-findings-5 — First **CLI-driven** REAP prune on LFM2.5-8B-A1B (4096-example calibration, compression 0.5)

> Fifth instrumented run. Unlike runs 1–4 (which drove the pipeline through
> `scripts/reap_lfm2_run.py`), this run executed the full **observe → prune →
> save** pipeline through the Typer CLI (`reap prune full`) on a **4096-example**
> random calibration subset at **compression 0.5** (32 → 16 experts). It is the
> first run to exercise the CLI end-to-end, and it surfaced three latent
> CLI-path defects that the script-driven runs never hit (one fixed in
> `ec8b926`, two filed as #50 / #51).

## 1. Scope

First production CLI run. Goals:

1. Drive the whole pipeline with `reap prune full` (not a hand-rolled script).
2. Scale calibration from 100 → **4096** examples and confirm the REAP saliency
   signature is stable across a 40× token-count increase.
3. Prune at **compression 0.5** (remove 16 of 32 experts) and verify the
   16-expert checkpoint loads + runs in a fresh process.
4. Shake out CLI-path bugs that the script-driven run-findings could not expose.

## 2. Configuration

| field | value |
| --- | --- |
| model | `/data/models/LiquidAI/LFM2.5-8B-A1B` (`Lfm2MoeForCausalLM`, 8.47B total params, 32 experts, top_k=4, 22 MoE + 2 dense layers, `use_expert_bias=true`, bf16) |
| dataset | `theblackcat102/evol-codealpaca-v1` via local `--dataset-path` |
| calib subset | **4096 random examples** (`/data/datasets/evol-codealpaca-v1/calib-4096.jsonl`, seed `20260717`, `{instruction,output}`) |
| batch_size | 4 |
| model_max_length | 1024 |
| batches_per_category | 1024 (caps at available → 398 batches) |
| compression_ratio | 0.5 (keep 16 of 32 experts) |
| prune_method | `reap` (router-weighted L2 norm, Welford online) |
| observe_backend | `auto` → `f2` (Triton) |
| frea_backend | `auto` (probe enabled, `REAP_FREA_PROBE=1`) |
| residency | **`gpu_full` (forced)** — `auto` mis-estimates LFM2 at ~73 GiB and wrongly picks `layerwise` (see #50) |
| seed | 42 |
| env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `REAP_FREA_BACKEND=auto`, `REAP_FREA_PROBE=1` |
| entrypoint | `.venv/bin/reap prune full` (Typer CLI → `reap.prune.run`) |
| stack | `.venv` — Python 3.12.13, torch 2.13.0+cu130, transformers 5.14.1, triton 3.7.1 |
| GPU | NVIDIA L4 (23 GiB, 48 KiB default / 99 KiB opt-in per-block shared mem, cc 8.9) |

> **Env note (important):** the run **must** use the project `.venv`
> (transformers 5.14), not the system python. In transformers 5.14 LFM2's
> `moe.gate` is a `Lfm2MoeTopKRouter` whose `forward` returns
> `(logits, routing_weights, selected_experts)` — exactly what the F5
> router-fusion path (`f5_router_from_module`) expects. In the system
> transformers 5.12, `moe.gate` is a plain `nn.Linear` returning only logits,
> and the F5 path raises `f5_router_from_module expects a router returning
> (logits, weights, selected_experts)`. The CLI is installed editable
> (`uv pip install --editable '.[cuda]'`), so `reap` resolves to the venv
> interpreter.

## 3. Outcome

```
outcome    = success
n_experts  = 32 -> 16 (compression 0.5)
total_wall = 749.6 s   (~12 min 30 s)
checkpoint = 6 files, 9.183 GiB (model.safetensors 8.55 GiB pruned weights)
verify     = OK (fresh-process load + forward, 4.592B params, 9.42 GiB peak VRAM)
```

Pruned checkpoint at
`/data/reap-lfm2-cli-4096/LFM2.5-8B-A1B/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.50/`:
`config.json`, `generation_config.json`, `chat_template.jinja`,
`tokenizer.json`, `tokenizer_config.json`, `model.safetensors` (8.55 GiB).

## 4. Per-phase performance

Timestamps from `run.log` (10:08:07 → 10:20:37):

| phase | wall | gpu peak (alloc) | cpu rss | notes |
| --- | ---: | ---: | ---: | --- |
| setup + artifacts dir | ~1.3 s | — | ~1 GiB | |
| model load + profile (max_len 1024) | 148.5 s | 17.39 GiB | low | bf16 ~16 GiB to GPU; FREA profitability probe ran here |
| dataset load + observer hook setup | 11.9 s | 17.39 GiB | — | 4096 examples → 398 packed batches; 22 MoE layers hooked (L2–L23) |
| **5_observe** | **517.5 s (8:37)** | **17.55 GiB** | ~1.2 GiB | **f2: 8800 Triton / 0 PyTorch; frea: 0 Triton / 8800 PyTorch** |
| prune (slice 32→16) | <1 s | — | — | `n_experts_to_prune = int(32 * 0.5) = 16` |
| save (stream, hooks stripped) | 69.83 s | — | ~0.8 GiB | `stream_save_pretrained`; no CPU OOM, no wedge |
| **total** | **749.6 s** | **17.55 GiB** | | |

`nvidia-smi` spot samples during observe: 16.36 / 17.39 / 17.50 / 17.11 / 17.55 GiB
(no 1 Hz board sampler this run; peak alloc ~17.55 GiB, consistent with run-3's
17.73 GiB board peak).

Observe throughput: **1,626,381 tokens / 517.5 s ≈ 3,143 tok/s** — ~3× run-3's
1,022 tok/s, accounted for by `batch_size=4` vs run-3's `batch_size=1` (better
GPU util, fewer kernel launches per token).

## 5. REAP saliency & routing metrics (from `observations_1024_cosine.pt`)

Routing correctness (the native-router integration via `Lfm2MoeTopKRouter`):

```
layers observed = 22 (L2..L23); dense L0,L1 correctly skipped
total_tokens    = 1,626,381
sum(expert_freq)[L2] = 6,505,524 = total_tokens * top_k(4) = 6,505,524   ratio = 4.000 (exact)
```

REAP saliency (`reap` metric) by layer — mean grows monotonically with depth:

```
  L  reap_mean  reap_min  reap_max   freq_min  freq_max  imbalance
  2    0.06667   0.03093    0.1282     11483    474723      41.3x
  3    0.06914   0.04279    0.0983     68934    994363      14.4x
  4    0.07133   0.04421    0.1098    106894    376302       3.5x
  5    0.21545   0.04382    4.4765     81136    583408       7.2x
  ...
 17    0.71911   0.25487    2.6217     14416    347535      24.1x
 20    1.14459   0.35604    3.0271     10166    880843      86.7x
 22    1.49295   0.43502    4.0715     36465    995856      27.3x
 23    2.09719   0.58746    5.8241      9402    737603      78.5x

reap_mean: min=0.06667 (L2)  max=2.0972 (L23)  ratio = 31.5x
```

Expert-frequency imbalance (`freq_max / freq_min`) ranges **3.5×–360×** across
layers — several near-dead experts per layer, i.e. strong REAP prune candidates.

**Stability vs prior runs:** the saliency signature reproduces the 200-example
run (L2≈0.065, L23≈1.97) and run-3 (100-example) almost exactly at 40× the token
count: **L2≈0.067, L23≈2.10, 31.5× depth ratio.** The pruning decision is
well-grounded and calib-size-stable.

## 6. Functional verification (fresh process)

Independent load + forward of the saved checkpoint
(`scripts/verify_pruned.py`, venv interpreter, `device_map=cuda`,
`expandable_segments=True`):

```
VERIFY_OK: 4.592B params | MoE=22 layers | first MoE 16 experts (was 32 -> 16)
           | cfg.num_experts=16 | expert_bias=(16,)
logits (1, 21, 128000) | next-token top1=21744 -> '"""'
peakVRAM = 9.42 GiB
```

- **`expert_bias=(16,)`** — the per-expert router bias was sliced alongside the
  experts (the `slice_experts` fix holds at 50% compression; without it the
  checkpoint would carry 32 entry biases vs `num_experts=16` and fail to load).
- Param count 8.47B → 4.59B (~46% reduction; dense layers + embeddings are
  retained, so it is not exactly 50%).
- Forward is coherent: for the prompt
  `def is_palindrome(s):\n    """Return True if s is a palindrome..."""\n    `
  the top-1 next token is `"""` (closes the docstring) — a sensible continuation.

> **No downstream quality eval was run.** `--no-eval` was set because `eval.py`
> is a stub (issue #40). The one-token spot-check above is **not** a real
> evaluation; quantifying the cost of 50% compression needs the eval harness.

## 7. CLI-path defects surfaced by this run

Runs 1–4 used `scripts/reap_lfm2_run.py`, which loads observations from the
saved `.pt` file. The CLI path (`reap.prune.run` → `record_activations` →
in-memory `report_state()`) is a different code path and exposed three latent
defects:

1. **`report_state()` after `close_hooks()` — FIXED in `ec8b926`.**
   `record_activations` called `observer.report_state()` *after*
   `observer.close_hooks()`, but `close_hooks()` → `reset()` does
   `del self.state; self.state = {}`. The saved `.pt` was valid (22 layers),
   but the in-memory dict handed to prune was `{}` → `StopIteration` at
   `observer_data[next(iter(observer_data))]["expert_frequency"]`. Reordered to
   capture `report_state()` before `close_hooks()`. This is why the CLI prune
   path crashed while the script path sailed.

2. **Auto-residency over-estimate — filed #50.**
   `estimate_model_bytes_from_config` returns ~73.2 GiB for LFM2-8B (real
   16.94 GiB, ~4.3× over). The fallback formula charges expert params to all
   24 layers (incl. the 2 dense) and uses a `max(inter, hidden*4)` floor that
   inflates the expert FFN when `intermediate_size` is absent/misread.
   `resolve_residency("auto")` then sees 73 GiB > 0.85×22 GiB GPU budget and
   picks `layerwise` — contradicting the `--residency` help text that literally
   describes "~16GB model on g6.xlarge" as the `gpu_full` case. Workaround:
   `--residency gpu_full`.

3. **`layerwise_prune.run` `tcr` UnboundLocalError — filed #51.**
   When `prune.run` delegates to the layerwise run with a pre-resolved
   residency (`_residency_resolved is not None`), `tcr` is referenced at line
   280 before its (only) assignment at line 269/310. This is the exact path
   triggered by defect #2's wrong `layerwise` resolution. Workaround: same
   `--residency gpu_full` (avoids the delegate).

#50 and #51 together make the default `reap prune full --residency auto`
unusable on LFM2 without an explicit `--residency gpu_full` override.

## 8. Alignment with run-findings-3 / -4

| aspect | run-3 (100 ex, script) | this run (4096 ex, CLI) | aligned |
| --- | --- | --- | --- |
| residency | `gpu_full` (`device_map=auto`) | `gpu_full` (forced; `auto` buggy) | ✅ (see #50) |
| observe backend | `auto → f2` | `auto → f2`: 8800 Triton / 0 PyTorch | ✅ |
| FREA backend | probe → PyTorch (cuBLAS) on L4 | probe → PyTorch (8800 PyTorch) | ✅ (matches run-3/4: Triton FREA 2.3× slower on L4) |
| throughput | 1022 tok/s (bs=1) | ~3143 tok/s (bs=4) | ✅ (larger batches) |
| peak VRAM | 16.30 GiB alloc / 17.73 GiB board | ~17.55 GiB | ✅ |
| save | stream, hooks stripped, 75.6 s | stream, hooks stripped, 69.8 s | ✅ |
| compression | 0.5 → 16/32 | 0.5 → 16/32 (`int(32*0.5)=16`) | ✅ |
| `expert_bias` slice | — | `(16,)` | ✅ |
| routing math | — | `sum(freq)=tokens×top_k` = 4.000 | ✅ |
| saliency pattern | L2≈0.065, L23≈1.97 | L2≈0.067, L23≈2.10 (31.5×) | ✅ reproduces |

**What's aligned:** backend selection (f2 Triton + FREA→PyTorch on L4), the
gpu_full / stream-save path, VRAM envelope (~17.5 GiB), save time (~70 s), the
compression-ratio math, and the REAP saliency depth-pattern all match the
documented findings. The 4096-example calibration reproduces the 100/200-example
saliency signature at 40× the token count — a strong stability signal.

**What's new / not in prior findings:** the three CLI-path defects (§7) — the
script-driven runs structurally could not surface them.

## 9. Honest caveats

- **No model-quality eval.** "Good metrics" = pipeline + REAP saliency metrics,
  **not** downstream quality. `--no-eval` (eval stub, #40). The one-token
  spot-check is not a real evaluation.
- **`batch_size=4` vs run-3's `batch_size=1`** — intentional deviation for
  throughput with 4096 examples; well within VRAM.
- **`--residency gpu_full` forced** — required because of #50; `--residency auto`
  crashes via #50 → #51.
- **No 1 Hz board sampler** this run; VRAM figures are `nvidia-smi` spot samples
  + torch alloc peak (~17.55 GiB).
- **`frea: 0 Triton / 8800 PyTorch`** — the probe correctly never launched
  Triton FREA on the L4 (cuBLAS wins for these shapes); the "Triton frea never
  launched successfully" warning is expected on L4, not a regression (run-3/4).

## 10. Residual gaps / things to watch

- **#50 / #51** — default `reap prune full --residency auto` is broken on LFM2
  until the residency estimator is fixed and `tcr` is hoisted. Both are High
  severity for first-run CLI usability.
- **Resume-skip overwrites observations with empty state** — if
  `record_activations` is re-run with an existing observations file and
  `--keep-observations`, the per-category loop is skipped but
  `observer.save_state(aggregate_path)` still runs on an empty (freshly
  initialized) observer state, **overwriting the valid file with an empty
  one**, and `report_state()` returns `{}`. Not hit this run (fresh artifacts),
  but a latent resume bug worth filing.
- **`eval.py` stub (#40)** — the missing piece for quantifying compression cost.
- **Wrong-env footgun** — running `reap` from the system python (transformers
  5.12) crashes at the F5 router. The CLI should either pin/announce the
  required transformers version or guard `f5_router_from_module` against
  non-tuple routers with a clear error.

## 11. Reproduce

```bash
cd /home/ubuntu/reap-cuda && source .venv/bin/activate  # or: .venv/bin/reap directly
# editable install already present; re-install only after a pull:
# uv pip install --editable '.[cuda]'

# 4096-example random subset (seed 20260717), one-time:
python3 - <<'PY'
import json, random; random.seed(20260717)
rows=[json.loads(l) for l in open('/data/datasets/evol-codealpaca-v1/train.jsonl') if l.strip()]
random.shuffle(rows)
import io
with open('/data/datasets/evol-codealpaca-v1/calib-4096.jsonl','w') as f:
    for r in rows[:4096]:
        f.write(json.dumps({'instruction':r['instruction'],'output':r['output']},ensure_ascii=False)+'\n')
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REAP_FREA_BACKEND=auto
export REAP_FREA_PROBE=1
.venv/bin/reap prune full \
  --model /data/models/LiquidAI/LFM2.5-8B-A1B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --dataset-path /data/datasets/evol-codealpaca-v1/calib-4096.jsonl \
  --compression-ratio 0.5 --prune-method reap \
  --batch-size 4 --model-max-length 1024 --batches-per-category 1024 \
  --seed 42 --no-eval --no-smoke-test \
  --residency gpu_full \
  --artifacts-dir /data/reap-lfm2-cli-4096
# artifacts -> /data/reap-lfm2-cli-4096/LFM2.5-8B-A1B/evol-codealpaca-v1/
#   all/observations_1024_cosine.pt
#   pruned_models/reap-renorm_true-seed_42-0.50/{model.safetensors,config.json,...}
```

Fresh-process verify:

```bash
.venv/bin/python - <<'PY'
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
d='/data/reap-lfm2-cli-4096/LFM2.5-8B-A1B/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_42-0.50'
tok=AutoTokenizer.from_pretrained(d)
m=AutoModelForCausalLM.from_pretrained(d,torch_dtype=torch.bfloat16,device_map='cuda',low_cpu_mem_usage=True)
ids=tok('def is_palindrome(s):\n    """Return True if s is a palindrome, ignoring spaces and case."""\n    ',return_tensors='pt').to('cuda')
with torch.no_grad(): out=m(**ids)
print(sum(p.numel() for p in m.parameters())/1e9,'B params | peakVRAM',torch.cuda.max_memory_allocated()/1e9,'GB')
PY
```