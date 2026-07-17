# Development

## Environment

```bash
git clone <repo> && cd reap-cuda
uv venv .venv --seed --python 3.12
uv sync --locked --group dev

# CUDA host with optional Triton and lm-eval
uv sync --locked --group dev --extra cuda --extra eval
```

Requires **Python ≥ 3.12**, **torch ≥ 2.10**, **transformers ≥ 5.5**.

## Tests

```bash
uv run pytest tests/ -q
uv run ruff check src tests
```

GitHub Actions runs this locked CPU quality gate on Python 3.12 and 3.13.
`.github/workflows/gpu.yml` is a scheduled/manual self-hosted `linux,gpu` lane
for CUDA/Triton parity tests; register that runner label to enable it.

Hermetic suite (no Hub downloads):

| Area | Files |
| --- | --- |
| Adapters / slice | `test_model_adapters.py`, `test_fused_slice_forward.py` |
| Observer / layerwise | `test_layerwise_*.py` |
| Merge / skip layers | `test_merge_pipeline.py`, `test_skip_first_last.py` |
| Kernels / contract | `test_kernel_parity_bmm.py`, `test_pruning_metrics_only_contract.py`, `test_f4_weight_cache.py`, `test_triton_kernels.py` |
| Weight residency | `test_residency.py` (heuristics, plans, stream_save, delegation) |
| EC2 run-findings | `test_run_findings_fixes.py` (router, F4 bound, FREA tiles, probe CLI, smoke, artifacts) |
| Dataset / offline | `test_dataset_loading.py` (composite `@path`, columns, offline guards, path threading) |
| CLI | `test_cli.py` (mocked pipelines; residency / frea-backend / dataset-path wiring) |

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

### FREA / Triton policy changes

1. Prefer policy in `triton_frea.py` (`set_frea_backend`, probe, tile choose) over pipeline branches
2. Keep launches SKU-agnostic (query SM; probe timings)
3. Always leave a correct PyTorch fallback
4. Document ops in `docs/frea-throughput.md` and update `gpu-and-backends.md`
5. Extend `tests/test_run_findings_fixes.py` / `test_triton_kernels.py`

### New calibration dataset

1. Processor class in `data.py` + `DATASET_REGISTRY`
2. Optional `_PROCESSOR_REQUIRED_COLUMNS` for offline column checks
3. Hermetic test in `tests/test_dataset_loading.py` if field map is non-trivial
4. Document in `docs/calibration.md`

## Coding conventions

- Prefer `run(dataclasses...)` APIs over parsing inside libraries
- Keep architecture branches in adapters, not kernels
- Do not `.to("cpu")` in observation hot paths
- Do not force full-model `device_map="cpu"` on low-RAM GPU hosts — use residency
- Do not assume Triton is faster than cuBLAS — use `--frea-backend auto` probe
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
