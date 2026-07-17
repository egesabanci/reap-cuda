# Evaluation

Post-compression validation is split into structural smoke generation and
optional lm-evaluation-harness task evaluation.

## Smoke generation and publication

`reap prune full` enables `--smoke-test` by default; layerwise prune exposes
it but defaults it off because generation can be expensive on an offloaded
model.

The smoke test runs **before** any checkpoint files are written. A failure
raises, leaves no staged artifact, and preserves any previously published
checkpoint. Successful saves are staged beside the final path and promoted
only after serialization completes.

## lm-eval

Install the optional backend:

```bash
uv sync --extra eval
reap prune full ... --eval \
  --eval-tasks hellaswag,arc_challenge \
  --eval-num-fewshot 5 --eval-batch-size 8 --eval-limit 100
```

The default `--eval-backend hf` loads the saved compressed checkpoint through
lm-eval's `HFLM` wrapper. It writes `eval/results.json` and logs a concise
per-task metric table.

### Baseline deltas

Add `--eval-baseline` to evaluate the source model as well. This writes
`baseline_results.json` and `diff.json`; logs include per-task metric deltas.

```bash
reap prune full ... --eval --eval-baseline --eval-tasks hellaswag
```

### vLLM

`--eval-backend vllm` uses lm-eval's vLLM wrapper when both packages are
installed. If vLLM or the lm-eval wrapper is missing, the command fails with a
clear installation error instead of silently falling back to HF.

### Offline task data

Prime lm-eval/Hugging Face task data once, then pass its cache root with
`--eval-data-path /data/hf-cache` and set `HF_HUB_OFFLINE=1`:

```bash
HF_HUB_OFFLINE=1 reap prune full ... --eval \
  --eval-data-path /data/hf-cache --eval-tasks hellaswag --eval-limit 100
```

The evaluator sets `HF_HOME` and `HF_DATASETS_CACHE` to this local path; it
does not override `HF_HUB_OFFLINE`, so air-gapped operation remains explicit.

## Unsupported optional evaluators

`run_evalplus`, `run_livecodebench`, `run_wildbench`, and `run_math` are not
implemented in this repository. If an API caller enables one, REAP emits a
clear warning rather than silently skipping it. They are intentionally not
surfaced as normal Typer flags.

## Related

- [cli.md](cli.md)
- [pruning.md](pruning.md)
- [pipeline.md](pipeline.md)
