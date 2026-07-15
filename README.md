# REAP — Router-weighted Expert Activation Pruning (CUDA)

This repository contains a CUDA implementation of the REAP operation for
Mixture-of-Experts (MoE) LLM compression, as described in the paper
[REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression](https://arxiv.org/abs/2510.13999).

This codebase is maintained as a **reference architecture** for the
REAP algorithm patterns: observer/hook abstractions, pruning saliency
metrics, expert manipulation, and model introspection utilities.

## Directory Structure

```
src/reap/
  __init__.py          # Package init
  args.py              # Argument dataclasses (reference)
  data.py              # Dataset loading and calibration utilities
  eval.py              # Evaluation stub (reference)
  main.py              # Main entry point for observation pipeline
  metrics.py           # Distance/similarity metrics and pruning state
  model_util.py        # Model attribute registry and introspection
  observer.py          # Forward-hook observer for activation collection
  prune.py             # Expert pruning logic
  pruning_metrics.py   # Online pruning metric computation
```

## Related

- [reap-mlx](https://github.com/egesabanci/reap-mlx) — MLX/Aarch64 port
  for Apple Silicon environments
- Original paper: [arXiv 2510.13999](https://arxiv.org/abs/2510.13999)
