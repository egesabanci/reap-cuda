# Calibration

Calibration supplies token sequences for MoE observation. REAP does not train;
it only runs forward passes and accumulates routing statistics.

Implementation: `src/reap/data.py`, wired through `pipeline.record_activations`
and layerwise prepare helpers.

## Inputs

| Source | CLI / args | Behavior |
| --- | --- | --- |
| Single HF dataset | `--dataset name` | Load split; tokenize via registered processor |
| Composite | `--dataset "a:4096,b[sub]:2048"` | Multi-source batch mix; results dir uses hash name |
| Cached observations | `--dataset combined` | Skip data load; require existing `.pt` |

Also:

- `--split` (default `train`)
- `--dataset-config` / `dataset_config_name` for HF configs
- `--batch-size`, `--batches-per-category`, `--model-max-length`
- `--truncate` for hard truncation policy in processors

## Composite dataset spec

Format (comma-separated components):

```txt
<dataset_name>[subset](split):<num_batches>
```

Examples:

```bash
# Two sources, 64 batches each (batch count, not raw sample count)
reap prune layerwise \
  --dataset "theblackcat102/evol-codealpaca-v1:64,open-r1/Mixture-of-Thoughts[code]:64"
```

Parsed by `parse_composite_dataset_spec`. Layerwise and full paths both honor it.

## Processors

Datasets must be registered in `DATASET_REGISTRY` with a processor that:

1. Loads/iterates rows
2. Extracts text (instruction/output, chat, etc.)
3. Tokenizes with the model tokenizer
4. Returns batches as dicts with at least `input_ids` and usually `attention_mask`

Unsupported dataset names raise a clear `ValueError` listing registered keys.

## Batching and padding

- Batches may include padding; observers use `attention_mask` (or layerwise 2D/4D
  mask handling) so **padding tokens do not contribute** to saliency.
- Prefer realistic sequence lengths (`--model-max-length`, default 2048) for
  meaningful routing; longer sequences cost more VRAM/time.

## How many batches?

| Goal | Suggestion |
| --- | --- |
| CLI smoke / wiring | 4–16 batches, batch size 1–2 |
| Paper-like prune | hundreds–1024 batches (default `batches_per_category=1024`) |
| Debug metrics | small N with `--overwrite-observations` |

Defaults are large because REAP quality depends on stable frequency/EAN
estimates; always override down for first EC2 bring-up.

## Caching

- Full path: per-category files under `artifacts/.../<category>/<output_file_name>`
- Layerwise: `artifacts/.../layerwise/<output_file_name>`
- Skip recompute unless `--overwrite-observations`
- `combined` requires a pre-built cache at the expected path

## Related

- [observation-and-metrics.md](observation-and-metrics.md)
- [pipeline.md](pipeline.md)
- [cli.md](cli.md)
