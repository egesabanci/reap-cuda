"""Evaluation routines for REAP-compressed models.

Runs lm-evaluation-harness (HF backend) on pruned/merged models.
vLLM-backed evalplus / livecodebench / wildbench / math are left as
opt-in stubs behind ``EvalArgs.run_*`` flags — re-implement those
in a follow-up if needed.
"""

from __future__ import annotations
import json
import logging
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def run_evaluate(model_args, results_dir, eval_args, seed):
    """Run lm-evaluation-harness on a pruned/merged model.

    Only ``run_lm_eval`` is implemented (HF backend).  The flags
    ``run_evalplus``, ``run_livecodebench``, ``run_wildbench``, and
    ``run_math`` are silently skipped; the original vLLM-based pipeline
    was stripped in the reference-architecture commit.

    Requires ``lm-eval>=0.4.5`` (opt-in ``[eval]`` extra).
    """
    if not isinstance(results_dir, pathlib.Path):
        results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    use_server = getattr(eval_args, "use_server", True)
    if use_server:
        logger.info(
            "vLLM server path not implemented in the reference "
            "codebase; falling back to HF lm-eval backend."
        )

    if not getattr(eval_args, "run_lm_eval", True):
        logger.info("run_lm_eval=False; skipping lm-eval.")
        return

    try:
        from lm_eval import simple_evaluate  # noqa: F401
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error(
            "lm-eval is not installed. Install it with: "
            "pip install -e '.[eval]'"
        )
        return

    torch.manual_seed(seed)
    logger.info("Loading model for evaluation: %s", model_args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name, trust_remote_code=True
    )

    hf_lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    gen_kwargs = {"do_sample": not getattr(eval_args, "greedy", True)}
    if gen_kwargs["do_sample"]:
        gen_kwargs.update(
            temperature=getattr(eval_args, "temperature", 0.7),
            top_p=getattr(eval_args, "top_p", 0.8),
            top_k=getattr(eval_args, "top_k", 20),
        )

    tasks = list(getattr(eval_args, "lm_eval_tasks", ["winogrande", "arc_challenge"]))
    logger.info("Running lm-eval on tasks: %s", tasks)
    results = simple_evaluate(
        model=hf_lm,
        tasks=tasks,
        batch_size=1,
        gen_kwargs=gen_kwargs,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )

    out_file = results_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("lm-eval results written to %s", out_file)
