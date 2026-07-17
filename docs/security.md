# Security and trust boundaries

REAP treats models, tokenizers, calibration data, and cached observation files
as separate trust boundaries.

## Remote model code

Remote model code is **disabled by default**. `--trust-remote-code` is an
explicit opt-in and is passed consistently to config, tokenizer, model, and
evaluation loaders. Enable it only for a repository/revision you trust.

For reproducibility and reduced network exposure, pin `--model-revision` and
use `--local-files-only` when the checkpoint is already available locally.
Those values are persisted with run arguments and observation metadata.

## Observation artifacts

Observation tensors are deserialized with PyTorch's `weights_only=True` mode
on CPU and then checked against a strict layer schema. REAP writes a JSON
manifest beside every newly produced aggregate/layerwise observation artifact.
The manifest includes schema/software version, model/tokenizer provenance,
local model/dataset fingerprints, calibration seed/shuffle settings, and
observer settings.

A cache is reused only when its manifest exactly matches the current run. A
legacy artifact without a compatible manifest is rejected by default. To load
one deliberately, use `--trust-observation-artifact`; this is a compatibility
escape hatch, not a way to bypass tensor-only deserialization.

## Artifact paths

Artifact directories retain a readable normalized full model/dataset identity
and append a stable SHA-256 suffix. Different owners or local paths that share
a basename therefore cannot reuse each other's observations or checkpoints.

## Operational guidance

- Prefer pinned revisions and local files in production/air-gapped runs.
- Treat calibration JSON/Arrow data as untrusted input; inspect it before use.
- Do not enable either trust flag in automated pipelines without an allowlist.
- Preserve manifests alongside copied observations; removing one intentionally
  requires the explicit legacy trust option on reuse.
