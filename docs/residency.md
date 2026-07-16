# Weight Residency

**Weight residency** controls *where model parameters live* during load, observe,
mutate, and save — host RAM vs GPU VRAM vs disk offload. It is **orthogonal** to:

| Concern | Documented in | What it is |
| --- | --- | --- |
| **CLI memory mode** (`full` vs `layerwise`) | [layerwise.md](layerwise.md), [cli.md](cli.md) | How observation is scheduled (full forward vs one block) |
| **Observe backends** (`bmm` / `frea` / `f2`) | [gpu-and-backends.md](gpu-and-backends.md) | How expert matmuls / saliency are computed |
| **Saliency device residency** | [observation-and-metrics.md](observation-and-metrics.md) | Where *metrics tensors* live during accumulate |

This document covers **weight residency only**: the policy in `src/reap/residency.py`,
CLI `--residency`, pipeline delegation, and stream-save.

---

## Why it exists

Historically, REAP layerwise paths loaded the **entire** model with
`device_map="cpu"` for calibration, then reloaded with `device_map="auto"` for
prune/save. That works when host RAM ≫ model size (e.g. workstation with
128 GiB RAM). It **OOMs on small-RAM GPU instances** even when the model fits
VRAM:

| Host class | Host RAM | GPU | Typical model | Failure without residency |
| --- | --- | --- | --- | --- |
| **g6.xlarge** (example) | ~16 GiB | L4 24 GiB | LFM2-8B FP16 ~15–16 GiB | `device_map="cpu"` materializes full weights in RAM → host OOM |
| L40S box | ~200 GiB+ | 46 GiB | Qwen3-30B-A3B ~60 GiB bf16 | Layerwise observe OK; full pin may still stress if RAM tight |
| Multi-GPU A100 | ample | 80 GiB×N | 30B+ | Full GPU map fine |

**Goal:** prefer **GPU-resident weights + stream save** when VRAM is enough and
host RAM is the bottleneck; prefer **block-wise observe + disk offload** when the
model does not fit VRAM; never force a full CPU pin unless the user asks for
`cpu_full` and RAM can hold it.

---

## Modes

CLI / `ReapArgs.residency` accepts:

| Mode | Meaning |
| --- | --- |
| `auto` | Pick among the three concrete modes from host/GPU memory + optional model-size estimate |
| `gpu_full` | Load with `device_map="auto"`, keep weights on GPU (or multi-GPU map), stream-save without full CPU materialize |
| `layerwise` | Block-wise observe; load with `device_map="auto"` **+ disk offload folder** (not a full host pin) |
| `cpu_full` | Explicit full model on CPU (`device_map="cpu"`) — only when host RAM is comfortably larger than the model |

Concrete modes after resolve never remain `"auto"`. `plan_load("auto")` raises.

### `gpu_full` (detail)

| Knob | Value |
| --- | --- |
| `device_map` | `"auto"` (accelerate places shards on CUDA / multi-GPU) |
| `low_cpu_mem_usage` | `True` by default |
| `offload_folder` | unset |
| `stream_save_from_gpu` | `True` |
| `avoid_cpu_materialize` | `True` |

**Intended use:** model ≈ fits GPU budget, host RAM is tight relative to model
size (LFM2-8B on g6.xlarge; smaller MoEs on 16–32 GiB RAM boxes).

**Observe path:** full-model hooks (`MoETransformerObserver`) with weights on
device — same as classical “full prune”, but load never forced through a full
CPU state dict when `device_map="auto"` + low_cpu_mem works.

**Save path:** `stream_save_pretrained` (see [Stream save](#stream-save)).

### `layerwise` (detail)

| Knob | Value |
| --- | --- |
| `device_map` | `"auto"` |
| `low_cpu_mem_usage` | `True` |
| `offload_folder` | temp dir or `artifacts/.../.offload` |
| `stream_save_from_gpu` | `True` (mutate/save reload uses `gpu_full` plan) |
| `avoid_cpu_materialize` | `True` |

**Intended use:** model larger than single-GPU budget; calibration must walk
one decoder block at a time ([layerwise.md](layerwise.md)).

**Important change vs older docs:** layerwise **no longer defaults to**
`device_map="cpu"`. Weights are loaded via accelerate **auto + offload_folder**
so host RAM is not required to hold every parameter at once. The layerwise
observer still moves **one block** to CUDA for forward/metrics.

**Mutate/save (prune):** after observe, the CPU/offload model is deleted,
memory cleaned, and the model is **reloaded with `plan_load("gpu_full")`** for
`slice_experts` + stream save. That step still needs enough **VRAM** for the
full (or multi-GPU-mapped) model at save time.

### `cpu_full` (detail)

| Knob | Value |
| --- | --- |
| `device_map` | `"cpu"` |
| `stream_save_from_gpu` | `False` |
| `avoid_cpu_materialize` | `False` |

**Intended use:** debugging without GPU, or hosts with large RAM and no / small
GPU. **Dangerous** on 16 GiB RAM boxes with 8B+ FP16 models.

`preflight_or_warn` logs a warning (or raises if `strict=True`) when
`model_bytes > HOST_SAFE_FRACTION * host_total`.

### `auto` (resolution algorithm)

Implementation: `resolve_residency(requested, model_bytes=..., mem=..., cli_prefers_layerwise=...)`.

```txt
if requested != auto:
    return requested

if model_bytes is None:
    if cli_prefers_layerwise:
        if GPU and host_available < 20 GiB → layerwise
        else → layerwise
    elif GPU present → gpu_full
    else → cpu_full

else:
    fits_gpu   = model_bytes <= GPU_FIT_FRACTION * gpu_total
    host_tight = model_bytes >= HOST_TIGHT_FRAC * host_total
    fits_host  = model_bytes <= HOST_SAFE_FRACTION * host_total

    if fits_gpu and host_tight → gpu_full   # g6.xlarge-style
    if fits_gpu and not layerwise_cli → gpu_full
    if layerwise_cli or (GPU and not fits_gpu) → layerwise
    if fits_host → cpu_full
    if GPU → gpu_full fallback
    else → layerwise fallback
```

**CLI bias:**

| Entrypoint | `cli_prefers_layerwise` |
| --- | --- |
| `reap prune full` / `reap merge full` | `False` |
| `reap prune layerwise` / `reap merge layerwise` | `True` |

So `reap prune layerwise --residency auto` still prefers layerwise observe for
large models, but **can resolve to `gpu_full`** and **delegate** to the full
pipeline when the auto heuristic says the model fits VRAM and host is tight
(delegation section below).

### Worked example: g6.xlarge + LFM2-8B-class

| Input | Value |
| --- | --- |
| Host total | 16 GiB |
| GPU total | 24 GiB |
| Model estimate | ~15.5 GiB |
| Requested | `auto` |
| CLI | `prune full` (`cli_prefers_layerwise=False`) |

```txt
fits_gpu   = 15.5 ≤ 0.85 * 24 ≈ 20.4  → True
host_tight = 15.5 ≥ 0.50 * 16 = 8     → True
→ resolved = gpu_full
reason ≈ "auto: model~15.5GiB fits GPU (24.0GiB) but is large vs host (16.0GiB)"
```

Load plan: `device_map=auto`, stream save, avoid CPU materialize.

---

## Tunable thresholds (env)

| Environment variable | Default | Role |
| --- | --- | --- |
| `REAP_RESIDENCY_HOST_FRAC` | `0.55` | Max fraction of host total for “safe” `cpu_full` |
| `REAP_RESIDENCY_GPU_FRAC` | `0.85` | Max fraction of GPU total for “fits GPU” |
| `REAP_RESIDENCY_HOST_TIGHT` | `0.50` | Model ≥ this × host total ⇒ host considered tight |

Raise `REAP_RESIDENCY_GPU_FRAC` only if you accept higher VRAM pressure; lower
`REAP_RESIDENCY_HOST_FRAC` to force more conservative `cpu_full` warnings.

---

## Memory measurement

### Host

`host_memory_bytes()` tries, in order:

1. **psutil** `virtual_memory()` → `(total, available)`
2. **Linux** `/proc/meminfo` (`MemTotal`, `MemAvailable`)
3. **macOS** `sysctl -n hw.memsize` (available ≈ total/2)
4. Fallback **16 GiB / 8 GiB**

### GPU

`gpu_memory_bytes()`:

1. If no CUDA → `(None, None)`
2. `torch.cuda.mem_get_info(0)` → `(total, free)`
3. Else `get_device_properties(0).total_memory`

`snapshot_memory()` packages both into `MemorySnapshot`.

### Model size estimate

`estimate_model_bytes_from_config(model_name)` (no weight download beyond config):

1. `AutoConfig.from_pretrained` (+ `trust_remote_code`)
2. If `num_parameters` / `n_params` present → `n * 2` (assume bf16/fp16 storage)
3. Else rough MoE formula from `hidden_size`, `num_hidden_layers`, `vocab_size`,
   intermediate sizes, expert counts → `params * 2`

`estimate_model_bytes_from_module(model)` sums `numel * element_size` for
parameters and buffers (used after load for logging).

Failures return `None`; auto still works via the “no estimate” branch.

---

## Load plans

`plan_load(resolved, offload_root=..., low_cpu_mem_usage=...)` → frozen `LoadPlan`:

```python
@dataclass(frozen=True)
class LoadPlan:
    resolved: str
    device_map: str
    low_cpu_mem_usage: bool
    offload_folder: str | None
    stream_save_from_gpu: bool
    avoid_cpu_materialize: bool
    reason: str
```

`load_causal_lm(model_name, plan, ...)` calls
`AutoModelForCausalLM.from_pretrained` with:

- `device_map`, `torch_dtype`, `trust_remote_code`, `low_cpu_mem_usage`
- If `plan.offload_folder`: `offload_folder` + `offload_state_dict=True`
- `model.eval()` after load

Layerwise observe uses `offload_root=results_dir / ".offload"` so offload files
live next to artifacts and are easier to clean than anonymous tempdirs.

---

## Stream save

`stream_save_pretrained(model, output_dir)`:

1. `Path(output_dir).mkdir(parents=True, exist_ok=True)`
2. `accelerate.hooks.remove_hook_from_module(model, recurse=True)`  
   Prevents accelerate from assembling a giant CPU state dict on save.
3. **Does not** call `model.to("cpu")`.
4. `model.save_pretrained(output_dir)` — safetensors can stream CUDA tensors
   shard-wise.

Used from:

- `prune.prune_model` (after slice)
- `merge_pipeline` save path

Older “materialize full CPU then save” behavior is what killed g6.xlarge-class
hosts after a successful GPU prune.

---

## Pipeline integration and delegation

Every prune/merge `run()` resolves residency **once** at entry (unless
`_residency_resolved` is already set from a peer call).

### Prune

| Call site | File | On resolve… |
| --- | --- | --- |
| `reap.prune.run` | `prune.py` | If `layerwise` → call `layerwise_prune.run(..., _residency_resolved=...)` |
| `reap.layerwise_prune.run` | `layerwise_prune.py` | If `gpu_full` or `cpu_full` → call `prune.run(..., _residency_resolved=...)` |

`_residency_resolved` **stops infinite bounce**: the callee skips re-resolution
and does not re-delegate opposite direction for `"auto"`.

```txt
User: reap prune full --residency auto
  prune.run resolves → layerwise (huge model)
    → layerwise_prune.run(_residency_resolved="layerwise")
       stays on layerwise path (no re-auto)

User: reap prune layerwise --residency auto
  layerwise_prune.run resolves → gpu_full (fits VRAM, host tight)
    → prune.run(_residency_resolved="gpu_full")
       full observe + stream save
```

Explicit overrides always win:

```bash
reap prune layerwise --residency gpu_full   # force full path via layerwise CLI
reap prune full --residency layerwise       # force layerwise observe via full CLI
reap prune full --residency cpu_full        # force CPU pin (warn if unsafe)
```

### Merge

Same pattern:

| Call site | Delegates to when |
| --- | --- |
| `merge_pipeline.run` | resolved `layerwise` → `layerwise_merge.run` |
| `layerwise_merge.run` | resolved `gpu_full` / `cpu_full` → `merge_pipeline.run` |

### Layerwise observe load vs mutate reload

For resolved **`layerwise`** prune:

```txt
1. plan_load("layerwise", offload_root=artifacts/.offload)
2. load_causal_lm → block-wise observe
3. del model; cleanup_memory()
4. plan_load("gpu_full"); load_causal_lm (local_files_only first)
5. prune_model → stream_save_pretrained
```

If the model does not fit GPU at step 4, mutate/save will OOM even if observe
succeeded — same class of constraint as before, but observe no longer requires
full host RAM.

---

## CLI surface

Shared Typer option (`src/reap/cli/options.py` → `Residency`):

```bash
--residency auto|gpu_full|layerwise|cpu_full   # default: auto
```

Present on:

- `reap prune full`
- `reap prune layerwise`
- `reap merge full`
- `reap merge layerwise`

Wired into `ReapArgs.residency` via `build_reap_args(residency=...)`.

Help panel name: **Residency**.

### Recommended commands by host

| Host | Model | Command sketch |
| --- | --- | --- |
| g6.xlarge (16 GiB RAM, L4 24 GiB) | ~8B MoE FP16 | `reap prune full --residency auto` (or `gpu_full`) |
| Same | 30B MoE | `reap prune layerwise --residency auto` (observe offload; save still needs VRAM budget) |
| L40S 46 GiB, large RAM | 30B | `reap prune layerwise --residency auto` |
| Multi-GPU 80 GiB+ | 30B+ | `reap prune full --residency gpu_full` |
| CPU-only debug | tiny | `reap prune full --residency cpu_full` |

Example (g6-friendly explicit):

```bash
uv run reap prune full \
  --model LiquidAI/LFM2.5-8B-A1B \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.5 \
  --residency gpu_full \
  --observe-backend bmm \
  --batches-per-category 8 \
  --batch-size 1
```

---

## Preflight checks

`preflight_or_warn(resolved, model_bytes, mem=None, strict=False)`:

| Condition | Action |
| --- | --- |
| `cpu_full` and model > safe host fraction | WARNING (or `RuntimeError` if `strict`) |
| `gpu_full` and model > GPU fit fraction | WARNING (or raise if `strict`) |
| `model_bytes is None` | no-op |

Pipelines call this after resolve so misconfiguration surfaces early in logs
without always aborting (non-strict).

---

## Logging

Typical log lines:

```text
Residency resolved: gpu_full (auto: model~15.5GiB fits GPU (24.0GiB) but is large vs host (16.0GiB))
Loading ... with residency=gpu_full device_map=auto offload=None (...)
Saved model to ... (stream path; hooks stripped)
```

Delegation:

```text
Delegating to layerwise prune (residency=layerwise)
Delegating to full prune path (residency=gpu_full) — avoids full-CPU pin
```

---

## Module API reference

| Symbol | Role |
| --- | --- |
| `RESIDENCY_MODES` | `("auto", "gpu_full", "layerwise", "cpu_full")` |
| `validate_residency(mode)` | Normalize + ValueError on unknown |
| `MemorySnapshot` | Host/GPU totals and availables |
| `snapshot_memory()` | Live snapshot |
| `host_memory_bytes` / `gpu_memory_bytes` | Raw probes |
| `estimate_model_bytes_from_config` | Config-only size |
| `estimate_model_bytes_from_module` | Loaded module size |
| `resolve_residency` | auto → concrete + reason string |
| `plan_load` | Concrete mode → `LoadPlan` |
| `preflight_or_warn` | OOM-risk logging / strict raise |
| `load_causal_lm` | `from_pretrained` honoring plan |
| `stream_save_pretrained` | Hook-strip + GPU-friendly save |

Package: **`reap.residency`** (`src/reap/residency.py`).

Config field: **`ReapArgs.residency`** (`src/reap/args.py`, default `"auto"`).

---

## Interaction with other memory systems

```txt
┌─────────────────────────────────────────────────────────────┐
│  --residency (this doc)                                     │
│    where weights live: GPU map / offload / CPU pin          │
└───────────────────────────┬─────────────────────────────────┘
                            │
     ┌──────────────────────┼──────────────────────┐
     ▼                      ▼                      ▼
 full observe          layerwise observe      stream save
 (all layers)          (one block GPU)        (no CPU dump)
     │                      │
     └──────────┬───────────┘
                ▼
     --observe-backend (bmm/frea/f2)
     saliency tensors on activation device
```

- Residency does **not** change saliency formulas or backend math.
- Residency **does** change whether `full` vs `layerwise` `run()` is used when
  `auto` (or explicit) disagrees with the CLI subcommand — via delegation.
- Layerwise **replay cache** of hidden states remains on **CPU** regardless of
  weight residency (activation I/O trade-off; see layerwise.md).

---

## Failure modes

| Symptom | Cause | Mitigation |
| --- | --- | --- |
| Host OOM during load with old mental model | Still using `cpu_full` or external scripts that pin CPU | `--residency gpu_full` or `auto` |
| Host OOM on save | accelerate hooks materializing CPU state dict | Ensure stream path (`stream_save_pretrained`); update to current prune/merge |
| GPU OOM on full observe | Model > VRAM; auto wrongly chose `gpu_full` | `--residency layerwise` |
| GPU OOM on layerwise mutate/save | Full reload for slice needs VRAM | Multi-GPU, larger instance, or prune offline on bigger box after observe-only |
| `cpu_full` warning in logs | Model large vs host | Prefer `gpu_full` on GPU hosts |
| Infinite recursion (should not happen) | Missing `_residency_resolved` | Fixed by keyword; do not call `run` peers without it when re-entering |

---

## Tests

| File | Coverage |
| --- | --- |
| `tests/test_residency.py` | Mode validation, auto heuristics (g6-like), plans, preflight, stream_save mock, CLI wiring, full↔layerwise delegation |
| `tests/test_cli.py` | Broader CLI (includes residency when exercised via help / shared options) |

Run:

```bash
uv run pytest tests/test_residency.py tests/test_cli.py -q
```

Hermetic: no Hub downloads; memory via synthetic `MemorySnapshot`; pipelines
mocked for CLI/delegation tests.

---

## Design non-goals

- Not a general multi-node FSDP / ZeRO trainer.
- Not automatic multi-host offload orchestration beyond accelerate `device_map`
  + `offload_folder`.
- Does not change expert **routing** or **pruning math**.
- Does not replace layerwise **activation** caching strategy.

---

## Related

- [layerwise.md](layerwise.md) — block replay schedule
- [pipeline.md](pipeline.md) — phase tables with residency-aware load
- [cli.md](cli.md) — `--residency` flag tables
- [gpu-and-backends.md](gpu-and-backends.md) — device policy for activations / kernels
- [pruning.md](pruning.md) — save / slice
- [architecture.md](architecture.md) — module map
- [setup.md](setup.md) — first-run examples on small RAM hosts
