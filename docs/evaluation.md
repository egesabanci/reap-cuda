# Evaluation

Post-compression validation options in reap-cuda.

## Smoke generation

Full prune enables `--smoke-test` by default (layerwise CLI defaults smoke off).

- Implementation: `pipeline.smoke_test`
- Applies chat template, `model.generate`, logs decoded text
- Failures are logged; they do not always abort the process (full prune catches
  and logs)

Use smoke to catch broken `slice_experts` / config mismatches before shipping a
checkpoint.

## lm-eval (optional)

```bash
uv pip install -e '.[eval]'
reap prune full ... --eval
```

- Implementation: `reap.eval.run_evaluate`
- Backend: HuggingFace via `lm_eval` / `HFLM`
- Default tasks: winogrande, arc_*, boolq, hellaswag, mmlu, openbookqa, rte
- Writes `results.json` under the checkpoint's `eval/` directory

Typer CLI disables unimplemented eval-plus / LCB / wildbench / math backends by
default even when `--eval` is set (only lm-eval is wired).

## Stubs / not implemented in this tree

| Flag / area | Status |
| --- | --- |
| evalplus | Stub / not run |
| livecodebench | Stub |
| wildbench | Stub |
| math suites | Stub |
| vLLM server path | Explicitly not implemented |

Re-introduce third-party harnesses as separate packages or optional extras if
needed for paper reproduction.

## Suggested validation ladder

1. Unit tests (`pytest`)
2. Tiny in-process observe + prune (no Hub) — already in tests
3. EC2: observe-only small calib on target model
4. EC2: full prune + smoke generate
5. Optional lm-eval subset
6. Downstream task suite of choice (external)

## Related

- [cli.md](cli.md)
- [pruning.md](pruning.md)
- [pipeline.md](pipeline.md)
