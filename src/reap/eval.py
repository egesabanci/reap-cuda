"""Evaluation routines for REAP-compressed models.

This is a reference implementation for running lm-evaluation-harness
on pruned/merged models.
"""

import logging
import pathlib

logger = logging.getLogger(__name__)


def run_evaluate(model_args, results_dir, eval_args, seed):
    """Run lm-evaluation-harness on a model.

    This is a placeholder reference. In the original codebase, this function:
    - Started a vLLM server
    - Ran lm-eval tasks
    - Ran evalplus (MBPP, HumanEval)
    - Ran LiveCodeBench
    - Ran WildBench pairwise evaluation
    - Ran math benchmarks via evalscope

    The custom adapter + Triton kernel implementation will provide its
    own evaluation pipeline.
    """
    logger.info(
        "Evaluation stub: model=%s, results_dir=%s, seed=%d",
        model_args.model_name,
        results_dir,
        seed,
    )
    if not isinstance(results_dir, pathlib.Path):
        results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluation not implemented in this stripped reference.")
