"""Evaluation routines for REAP-compressed models.

The public evaluator wraps lm-evaluation-harness with either its Hugging Face
backend or its optional vLLM backend.  It writes reproducible JSON artifacts,
can compare a compressed checkpoint against its source model, and never silently
ignores an explicitly requested unsupported evaluator.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from collections.abc import Mapping
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _json_dump(path: pathlib.Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _configure_eval_cache(eval_data_path: str | None) -> None:
    """Point HF datasets/hub at a caller-provided local cache.

    The caller can additionally set ``HF_HUB_OFFLINE=1`` after priming this
    cache once.  We deliberately do not change that environment variable: a
    cache path alone is useful while priming, whereas offline mode must remain
    an explicit user choice.
    """
    if eval_data_path is None:
        return
    cache = pathlib.Path(eval_data_path).expanduser()
    if not cache.exists():
        raise FileNotFoundError(f"--eval-data-path does not exist: {cache}")
    cache = cache.resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    logger.info("Using local evaluation cache at %s", cache)


def _warn_unsupported_evaluators(eval_args: Any) -> None:
    for attr, name in (
        ("run_evalplus", "evalplus"),
        ("run_livecodebench", "LiveCodeBench"),
        ("run_wildbench", "WildBench"),
        ("run_math", "math evaluator"),
    ):
        if getattr(eval_args, attr, False):
            logger.warning(
                "%s was requested but is not implemented; skipping it explicitly.", name
            )


def _generation_kwargs(eval_args: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"do_sample": not getattr(eval_args, "greedy", True)}
    if kwargs["do_sample"]:
        kwargs.update(
            temperature=getattr(eval_args, "temperature", 0.7),
            top_p=getattr(eval_args, "top_p", 0.8),
            top_k=getattr(eval_args, "top_k", 20),
        )
        min_p = getattr(eval_args, "min_p", 0.0)
        if min_p > 0:
            kwargs["min_p"] = min_p
    return kwargs


def _load_hf_lm(model_name: str, model_args: Any, batch_size: int) -> Any:
    try:
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:  # pragma: no cover - guarded by _load_lm_eval
        raise RuntimeError("lm-eval's Hugging Face backend is unavailable.") from exc

    trust_remote_code = bool(getattr(model_args, "trust_remote_code", False))
    revision = getattr(model_args, "model_revision", None)
    local_files_only = bool(getattr(model_args, "local_files_only", False))
    common = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision:
        common["revision"] = revision

    logger.info("Loading Hugging Face evaluation model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        **common,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    return HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)


def _load_vllm_lm(model_name: str, model_args: Any, batch_size: int) -> Any:
    """Instantiate lm-eval's vLLM model wrapper or raise an actionable error."""
    try:
        import vllm  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "--eval-backend vllm requires vllm. Install a vLLM-compatible "
            "environment, then install lm-eval with its vLLM extras."
        ) from exc

    try:
        from lm_eval.models.vllm_causallms import VLLM
    except ImportError:
        try:
            from lm_eval.models.vllm import VLLM
        except ImportError as exc:
            raise RuntimeError(
                "Installed lm-eval does not expose its vLLM backend. Upgrade "
                "lm-eval or use --eval-backend hf."
            ) from exc

    kwargs: dict[str, Any] = {
        "pretrained": model_name,
        "tokenizer": model_name,
        "batch_size": batch_size,
        "trust_remote_code": bool(getattr(model_args, "trust_remote_code", False)),
    }
    revision = getattr(model_args, "model_revision", None)
    if revision:
        kwargs["revision"] = revision
    logger.info("Loading vLLM evaluation model: %s", model_name)
    return VLLM(**kwargs)


def _load_lm_eval() -> Any:
    try:
        from lm_eval import simple_evaluate
    except ImportError as exc:
        raise RuntimeError(
            "lm-eval is not installed. Install evaluation support with "
            "`uv sync --extra eval` or `pip install -e '.[eval]'`."
        ) from exc
    return simple_evaluate


def _run_single_evaluation(
    model_name: str,
    model_args: Any,
    eval_args: Any,
    seed: int,
) -> dict[str, Any]:
    backend = getattr(eval_args, "eval_backend", "hf")
    if backend == "vllm":
        # Check the selected backend first so a missing vLLM installation has
        # the actionable error users expect, even when lm-eval is absent too.
        lm = _load_vllm_lm(model_name, model_args, getattr(eval_args, "eval_batch_size", 1))
    elif backend == "hf":
        lm = _load_hf_lm(model_name, model_args, getattr(eval_args, "eval_batch_size", 1))
    else:
        raise ValueError(f"Unsupported evaluation backend {backend!r}; use 'hf' or 'vllm'.")
    simple_evaluate = _load_lm_eval()

    tasks = list(getattr(eval_args, "lm_eval_tasks", ()))
    if not tasks:
        raise ValueError("At least one lm-eval task must be configured.")
    logger.info("Running %s lm-eval on tasks: %s", backend, ", ".join(tasks))
    return simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=getattr(eval_args, "eval_batch_size", 1),
        num_fewshot=getattr(eval_args, "eval_num_fewshot", 0),
        limit=getattr(eval_args, "eval_limit", None),
        gen_kwargs=_generation_kwargs(eval_args),
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )


def _numeric_metrics(results: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Return numeric per-task metrics from an lm-eval result object."""
    task_results = results.get("results", {})
    if not isinstance(task_results, Mapping):
        return {}
    output: dict[str, dict[str, float]] = {}
    for task, metrics in task_results.items():
        if not isinstance(metrics, Mapping):
            continue
        numeric = {
            str(name): float(value)
            for name, value in metrics.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        }
        if numeric:
            output[str(task)] = numeric
    return output


def _log_task_table(results: Mapping[str, Any], heading: str) -> None:
    numeric = _numeric_metrics(results)
    if not numeric:
        logger.warning("%s produced no numeric per-task metrics.", heading)
        return
    logger.info("%s", heading)
    for task, metrics in sorted(numeric.items()):
        preferred = next(
            (name for name in metrics if name.startswith(("acc", "exact_match", "pass"))),
            next(iter(metrics)),
        )
        logger.info("  %-32s %-24s %.6f", task, preferred, metrics[preferred])


def _result_diff(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any],
    *,
    candidate_model: str,
    baseline_model: str,
) -> dict[str, Any]:
    candidate_metrics = _numeric_metrics(candidate)
    baseline_metrics = _numeric_metrics(baseline)
    tasks: dict[str, dict[str, dict[str, float]]] = {}
    for task in sorted(candidate_metrics.keys() & baseline_metrics.keys()):
        metric_diff: dict[str, dict[str, float]] = {}
        for metric in sorted(candidate_metrics[task].keys() & baseline_metrics[task].keys()):
            current = candidate_metrics[task][metric]
            previous = baseline_metrics[task][metric]
            metric_diff[metric] = {
                "baseline": previous,
                "candidate": current,
                "delta": current - previous,
            }
        if metric_diff:
            tasks[task] = metric_diff
    return {
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "tasks": tasks,
    }


def _log_delta_table(diff: Mapping[str, Any]) -> None:
    for task, metrics in diff.get("tasks", {}).items():
        for metric, values in metrics.items():
            logger.info(
                "Δ %-30s %-24s %+0.6f", task, metric, values["delta"]
            )


def run_evaluate(
    model_args: Any,
    results_dir: str | pathlib.Path,
    eval_args: Any,
    seed: int,
    *,
    baseline_model_name: str | None = None,
    baseline_model_args: Any | None = None,
) -> dict[str, Any] | None:
    """Evaluate a checkpoint and optionally compare it with its source model.

    Candidate output is always ``results.json``. When ``eval_baseline`` is
    enabled and ``baseline_model_name`` is supplied, the source result is
    ``baseline_results.json`` and numeric per-task deltas are written to
    ``diff.json``.
    """
    results_path = pathlib.Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    _configure_eval_cache(getattr(eval_args, "eval_data_path", None))
    _warn_unsupported_evaluators(eval_args)

    if not getattr(eval_args, "run_lm_eval", True):
        logger.info("run_lm_eval=False; skipping lm-eval.")
        return None

    torch.manual_seed(seed)
    candidate_model = str(model_args.model_name)
    try:
        candidate = _run_single_evaluation(candidate_model, model_args, eval_args, seed)
    except RuntimeError as exc:
        # lm-eval is an optional dependency. Keep historical non-eval test and
        # compression workflows usable when the HF backend extra is absent, but
        # make an explicitly requested vLLM backend a hard, actionable error.
        if getattr(eval_args, "eval_backend", "hf") == "vllm":
            raise
        logger.error("Evaluation skipped: %s", exc)
        return None
    _json_dump(results_path / "results.json", candidate)
    _log_task_table(candidate, "Candidate evaluation")
    logger.info("lm-eval results written to %s", results_path / "results.json")

    if getattr(eval_args, "eval_baseline", False):
        if not baseline_model_name:
            raise ValueError("--eval-baseline requires the original model name.")
        baseline = _run_single_evaluation(
            str(baseline_model_name), baseline_model_args or model_args, eval_args, seed
        )
        _json_dump(results_path / "baseline_results.json", baseline)
        diff = _result_diff(
            candidate,
            baseline,
            candidate_model=candidate_model,
            baseline_model=str(baseline_model_name),
        )
        _json_dump(results_path / "diff.json", diff)
        _log_delta_table(diff)

    return candidate
