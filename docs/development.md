# Development

## Environment

```bash
git clone <repo> && cd reap-cuda
uv venv .venv --seed --python 3.12
uv pip install --editable .
uv pip install pytest

# CUDA host with optional Triton
uv pip install -e '.[cuda]'
uv pip install -e '.[eval]'   # lm-eval
```

Requires **Python ≥ 3.12**, **torch ≥ 2.10**, **transformers ≥ 5.5**.

## Tests

```bash
uv run pytest tests/ -q
```

Hermetic suite (no Hub downloads):

| Area | Files |
| --- | --- |
| Adapters / slice | `test_model_adapters.py`, `test_fused_slice_forward.py` |
| Observer / layerwise | `test_layerwise_*.py` |
| Merge / skip layers | `test_merge_pipeline.py`, `test_skip_first_last.py` |
| Kernels / contract | `test_kernel_parity_bmm.py`, `test_pruning_metrics_only_contract.py`, `test_f4_weight_cache.py` |
| Weight residency | `test_residency.py` (heuristics, plans, stream_save, delegation) |
| CLI | `test_cli.py` (mocked pipelines; residency wiring) |

## Project layout (src)

```txt
src/reap/
  cli/           # Typer
  kernels/       # observe backends
  residency.py   # weight load/save policy
  *.py           # pipeline modules
tests/
docs/            # this documentation
docs/kernels/    # kernel design SoC docs
docs/residency.md
```

## Extension checklist

### New model family

1. Adapter in `model_adapters.py` + `infer_model_adapter` branch
2. Weight convention for F4
3. Unit tests with mock or tiny config
4. Document in `docs/model-adapters.md` and README table

### New prune metric

1. Accumulate in `pruning_metrics` / observe path
2. Map CLI name in `PRUNE_METHOD_KEY_MAP` and Typer help
3. Contract test: prune-only path still excludes merge keys
4. Document in `observation-and-metrics.md` / `pruning.md`

### New observe backend

1. Implement under `kernels/`
2. Register in `backend.select_observe_backend` / CLI choices
3. Parity test vs `loop` or `bmm` on tiny Qwen3
4. Note in `gpu-and-backends.md` and `docs/kernels/`

### Weight residency changes

1. Update `residency.py` heuristics / `LoadPlan` only — keep pipelines thin
2. Preserve `_residency_resolved` when delegating full ↔ layerwise
3. Prefer `stream_save_pretrained` over manual CPU materialize
4. Extend `tests/test_residency.py` + document in `docs/residency.md`

## Coding conventions

- Prefer `run(dataclasses...)` APIs over parsing inside libraries
- Keep architecture branches in adapters, not kernels
- Do not `.to("cpu")` in observation hot paths
- Do not force full-model `device_map="cpu"` on low-RAM GPU hosts — use residency
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, …)

## CLI smoke (no model)

```bash
uv run reap --help
uv run reap prune full --help
uv run reap version
uv run pytest tests/test_cli.py -q
```

## EC2 checklist

1. Confirm `torch.cuda.is_available()`
2. `pytest` green
3. Layerwise observe-only on target model with small N
4. Full prune + smoke
5. Optional lm-eval

## Related

- [index.md](index.md)
- [architecture.md](architecture.md)
- [cli.md](cli.md)
