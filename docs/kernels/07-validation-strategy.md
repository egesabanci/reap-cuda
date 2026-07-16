# 07 — Validation Strategy (Current Tests)

> **Status: PARTIAL vs original mega-plan**  
> Shipped tests cover F3, F4, bmm/frea parity, and Triton unit paths.  
> No separate EC2-only `test_kernel_parity_f5.py` file names from the old plan —
> F5/FREA/F2 Triton checks live in `tests/test_triton_kernels.py`.

## Parity ladder (as implemented)

```txt
loop  ←── parity ──→  bmm / frea (PyTorch)     [CPU CI: test_kernel_parity_bmm]
bmm   ←── parity ──→  FREA Triton               [CUDA: test_triton_kernels]
PyTorch scatter ←──→  F2 Triton scatter         [CUDA: test_triton_kernels]
F.softmax       ←──→  F5 Triton softmax         [CUDA: test_triton_kernels]
```

## Consumed keys (prune)

```python
total_tokens, expert_frequency, pairwise_expert_frequency,
ean_sum, ean_mean, reap, weighted_ean_sum,
weighted_expert_frequency_sum, max_activations
# + routed_characteristic_activation when ean_ca / merge metrics
```

## Test files (actual)

| File | Runs on | Checks |
|---|---|---|
| `tests/test_pruning_metrics_only_contract.py` | CPU | F3 keys |
| `tests/test_kernel_parity_bmm.py` | CPU | loop vs bmm/frea metrics |
| `tests/test_f4_weight_cache.py` | CPU | F4 shapes / Llama layout |
| `tests/test_triton_kernels.py` | CPU always; CUDA if Triton | softmax, F5 CSR, FREA, reduce, backend select |
| `tests/test_cli.py` | CPU | `reap kernels` command |

```bash
uv run pytest tests/test_triton_kernels.py tests/test_kernel_parity_bmm.py -q
# On EC2 with triton:
uv run pytest tests/test_triton_kernels.py -q   # fewer skips
```

## Tolerances

| Setting | Tolerance |
|---|---|
| Tiny model, fp32, CPU bmm vs loop | atol/rtol ~1e-4 |
| CUDA fp16 Triton vs bmm | atol/rtol ~1e–2 … 2e-2 (accum order) |

## Still open (nice-to-have)

- [ ] `scripts/bench_observer.py` wall-clock + peak VRAM on 30B
- [ ] E2E retained-expert identity loop vs f2 on small calib subset
- [ ] CI GPU job for Triton tests

## Acceptance (current)

- [x] F3 contract tests
- [x] bmm vs loop parity on Mac/CPU CI
- [x] F4 tests
- [x] Triton unit tests with skip-if-no-CUDA
- [ ] Measured L40S speedup table (update `08` when available)
