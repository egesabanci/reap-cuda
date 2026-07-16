#!/usr/bin/env python
"""Instrumented REAP prune run: LFM2.5-8B-A1B on 100 local calibration examples.

Full-GPU end-to-end (residency=gpu_full, device_map="auto"), observe backend=auto
(-> f2 Triton kernels: F5 softmax + FREA SwiGLU + F2 scatter-reduce, with
PyTorch fallback on any launch failure). Prune method=reap, compression=0.5
(keep 16 of 32 routed experts).

Uses the codebase's real REAP functions (load_causal_lm, load_category_batches,
_setup_observer, reap.prune.prune, stream_save_pretrained, smoke_test) and
instruments every phase with wall time + peak GPU allocated + GPU reserved +
board-level GPU used (background nvidia-smi sampler) + CPU RSS.

Outputs (under artifacts/LFM2.5-8B-A1B/evol-codealpaca-v1/):
  perf_report.json, perf_report.csv, gpu_timeline.csv, run.log,
  pruned model checkpoint, observer state .pt
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict

import torch

# Force offline (model + dataset are local); avoid any hub round-trip.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Reduce fragmentation on the tight L4 (16GB model + observe recompute).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Make `src/` importable when run directly.
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import psutil

from accelerate.utils import set_seed

from reap.args import (
    ClusterArgs,
    DatasetArgs,
    EvalArgs,
    ModelArgs,
    ObserverArgs,
    PruneArgs,
    ReapArgs,
)
from reap.data import load_category_batches
from reap import data as reap_data
from reap.kernels.backend import select_observe_backend
from reap.model_adapters import infer_model_adapter
from reap.pipeline import _setup_observer, smoke_test
import re as _re


def _str_to_dir(s: str) -> str:
    return _re.sub(r"[^\w\-_.]", "_", s)
from reap.prune import get_pruned_model_dir, prune as reap_prune
from reap.residency import (
    estimate_model_bytes_from_config,
    estimate_model_bytes_from_module,
    load_causal_lm,
    plan_load,
    stream_save_pretrained,
)
from reap.pruning_metrics import initialize_pruning_state

from datasets import Dataset

logger = logging.getLogger("reap_run")

# --------------------------- configuration ----------------------------------
MODEL_PATH = "/data/models/LiquidAI/LFM2.5-8B-A1B"
DATASET_ARROW = "/data/datasets/evol-codealpaca-calib-200/data-00000-of-00001.arrow"
# Artifacts (pruned models, logs, perf reports) live on /data which has room.
ARTIFACTS_BASE = pathlib.Path("/data/reap-artifacts")
N_EXAMPLES = 100
BATCH_SIZE = 1
MODEL_MAX_LENGTH = 1024
COMPRESSION_RATIO = 0.5
PRUNE_METHOD = "reap"
OBSERVE_BACKEND = "auto"
SEED = 42
RESIDENCY = "gpu_full"

# --------------------------- instrumentation -------------------------------
_phase_metrics: list[dict] = []
_proc = psutil.Process()


def _gpu_mem():
    if not torch.cuda.is_available():
        return None, None
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def _peak():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def phase(name: str, **tags):
    """Time + memory-probe a phase; append a metrics row."""
    torch.cuda.is_available() and torch.cuda.synchronize()
    gc.collect()
    _reset_peak()
    alloc0, reserved0 = _gpu_mem()
    rss0 = _proc.memory_info().rss / 1024**3
    t0 = time.perf_counter()
    row = {"phase": name, "start_s": round(t0, 4), **tags}
    logger.info("=== PHASE START: %s ===", name)
    try:
        yield row
    finally:
        torch.cuda.is_available() and torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        alloc1, reserved1 = _gpu_mem()
        peak = _peak()
        rss1 = _proc.memory_info().rss / 1024**3
        row.update(
            wall_time_s=round(elapsed, 4),
            gpu_alloc_start_gib=round(alloc0, 4) if alloc0 is not None else None,
            gpu_alloc_end_gib=round(alloc1, 4) if alloc1 is not None else None,
            gpu_peak_alloc_gib=round(peak, 4) if peak is not None else None,
            gpu_reserved_start_gib=round(reserved0, 4) if reserved0 is not None else None,
            gpu_reserved_end_gib=round(reserved1, 4) if reserved1 is not None else None,
            cpu_rss_start_gib=round(rss0, 4),
            cpu_rss_end_gib=round(rss1, 4),
            cpu_rss_delta_gib=round(rss1 - rss0, 4),
        )
        _phase_metrics.append(row)
        logger.info(
            "=== PHASE END: %s | wall=%.2fs | gpu_peak=%.3f GiB | gpu_res=%.3f GiB | rss=%.2f GiB ===",
            name, elapsed, peak if peak is not None else float("nan"),
            reserved1 if reserved1 is not None else float("nan"), rss1,
        )


# Background board-level GPU sampler -> gpu_timeline.csv
class GpuSampler(threading.Thread):
    def __init__(self, path: pathlib.Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self._stop_event = threading.Event()
        self.f = open(path, "w", newline="")
        self.writer = csv.writer(self.f)
        self.writer.writerow(
            ["t_s", "memory.used_mib", "memory.total_mib", "utilization.gpu_pct"]
        )
        self.f.flush()
        self.t0 = time.perf_counter()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
                used, total, util = [x.strip() for x in out.split(",")]
                self.writer.writerow(
                    [round(time.perf_counter() - self.t0, 4), used, total, util]
                )
                self.f.flush()
            except Exception as e:
                self.writer.writerow([round(time.perf_counter() - self.t0, 4), "err", "", str(e)])
                self.f.flush()
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=3)
        self.f.close()


# --------------------------- local dataset shim ----------------------------
def _install_local_dataset_shim(n_examples: int):
    """Make load_category_batches use the local 100-row arrow subset."""
    full = Dataset.from_file(DATASET_ARROW)
    local_ds = full.select(range(n_examples))
    logger.info("Local dataset shim: %d examples (of %d) from %s",
                len(local_ds), len(full), DATASET_ARROW)

    _orig = reap_data._load_raw_dataset

    def _shim(dataset_name, split, subset=None):
        if dataset_name == "theblackcat102/evol-codealpaca-v1":
            return local_ds
        return _orig(dataset_name, split, subset=subset)

    reap_data._load_raw_dataset = _shim
    return local_ds


# --------------------------- main ------------------------------------------
def main():
    model_clean = _str_to_dir(MODEL_PATH.split("/")[-1])
    dataset_clean = _str_to_dir("theblackcat102/evol-codealpaca-v1".split("/")[-1])
    results_dir = ARTIFACTS_BASE / model_clean / dataset_clean
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts directory: %s", results_dir)
    log_path = results_dir / "run.log"
    gpu_csv = results_dir / "gpu_timeline.csv"
    perf_json = results_dir / "perf_report.json"
    perf_csv = results_dir / "perf_report.csv"

    # logging: DEBUG to file, INFO to console
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    # also surface reap.* library logs into the file
    for name in ("reap", "reap.kernels", "reap.observer", "reap.prune",
                 "reap.residency", "reap.pipeline", "reap.data"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(fh)
        lg.addHandler(ch)

    sampler = GpuSampler(gpu_csv, interval=1.0)
    sampler.start()

    meta = {
        "model": MODEL_PATH,
        "dataset_arrow": DATASET_ARROW,
        "n_examples": N_EXAMPLES,
        "batch_size": BATCH_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "compression_ratio": COMPRESSION_RATIO,
        "prune_method": PRUNE_METHOD,
        "observe_backend": OBSERVE_BACKEND,
        "residency": RESIDENCY,
        "seed": SEED,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    logger.info("Run config: %s", json.dumps(meta, indent=2))

    try:
        # ---- Phase 0: env / backend check ----
        with phase("0_env_backend_check") as r:
            backend = select_observe_backend(OBSERVE_BACKEND)
            r["resolved_backend"] = backend
            import triton
            r["triton_version"] = triton.__version__
            logger.info("CUDA=%s triton=%s backend=auto->%s",
                        torch.cuda.is_available(), triton.__version__, backend)

        set_seed(SEED)

        # Build args dataclasses (the same surface the Typer CLI builds).
        reap_args = ReapArgs(seed=SEED, profile=False, run_observer_only=False,
                             do_eval=False, smoke_test=True, residency=RESIDENCY)
        model_args = ModelArgs(model_name=MODEL_PATH)
        ds_args = DatasetArgs(dataset_name="theblackcat102/evol-codealpaca-v1",
                              split="train", dataset_config_name=None, shuffle=False)
        obs_args = ObserverArgs(
            batches_per_category=N_EXAMPLES,
            batch_size=BATCH_SIZE,
            model_max_length=MODEL_MAX_LENGTH,
            overwrite_observations=True,
            distance_measure="cosine",
            record_pruning_metrics_only=True,
            observe_backend=OBSERVE_BACKEND,
            renormalize_router_weights=True,
        )
        prune_args = PruneArgs(
            prune_method=PRUNE_METHOD,
            n_experts_to_prune=None,
            overwrite_pruned_model=True,
            perserve_super_experts=False,
            perserve_outliers=False,
        )
        cluster_args = ClusterArgs(compression_ratio=COMPRESSION_RATIO)

        # ---- Phase 1: tokenizer load ----
        with phase("1_tokenizer_load") as r:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            r["vocab_size"] = tokenizer.vocab_size
            r["pad_token_id"] = tokenizer.pad_token_id

        # ---- Phase 2: model load (gpu_full) ----
        with phase("2_model_load") as r:
            plan = plan_load(RESIDENCY)
            r["device_map"] = plan.device_map
            r["plan_reason"] = plan.reason
            model = load_causal_lm(MODEL_PATH, plan, local_files_only=True)
            live_bytes = estimate_model_bytes_from_module(model)
            r["model_weight_gib"] = round(live_bytes / 1024**3, 3)
            n_params = sum(p.numel() for p in model.parameters())
            r["n_params"] = n_params
            adapter = infer_model_adapter(model, model.config)
            r["adapter"] = type(adapter).__name__ if adapter else None
            moe_idx = adapter.identify_moe_layers(model)
            r["n_moe_layers"] = len(moe_idx)
            r["moe_layer_indices"] = moe_idx
            cfg = model.config
            r["num_experts"] = int(getattr(cfg, "num_experts", -1))
            r["num_experts_per_tok"] = int(getattr(cfg, "num_experts_per_tok", -1))
            r["use_expert_bias"] = bool(getattr(cfg, "use_expert_bias", False))
            logger.info("Loaded %s | adapter=%s | MoE layers=%d | experts=%d top_k=%d",
                        MODEL_PATH, r["adapter"], r["n_moe_layers"],
                        r["num_experts"], r["num_experts_per_tok"])

        # ---- Phase 3: dataset load + tokenize (local 100) ----
        with phase("3_dataset_load_tokenize") as r:
            _install_local_dataset_shim(N_EXAMPLES)
            category_data = load_category_batches(
                dataset_name=ds_args.dataset_name,
                split=ds_args.split,
                subset=ds_args.dataset_config_name,
                tokenizer=tokenizer,
                model_max_length=obs_args.model_max_length,
                split_by_category=False,
                return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
                truncate=obs_args.truncate,
                batches_per_category=obs_args.batches_per_category,
                batch_size=obs_args.batch_size,
            )
            batches = category_data["all"]
            r["n_batches"] = len(batches)
            total_tokens = 0
            for b in batches:
                am = b.get("attention_mask")
                if am is not None:
                    total_tokens += int(am.sum().item())
                else:
                    total_tokens += int(b["input_ids"].numel())
            r["total_tokens"] = total_tokens
            r["avg_tokens_per_batch"] = round(total_tokens / max(1, len(batches)), 1)
            logger.info("Prepared %d batches, %d total tokens", len(batches), total_tokens)

        # ---- Phase 4: observer setup ----
        with phase("4_observer_setup") as r:
            observer = _setup_observer(model, obs_args)
            r["hook_regex"] = observer.hook_config.module_class_name_to_hook_regex
            r["record_pruning_metrics_only"] = obs_args.record_pruning_metrics_only
            r["observe_backend"] = obs_args.observe_backend
            logger.info("Observer ready (hook=%s, backend=%s)",
                        r["hook_regex"], r["observe_backend"])

        # ---- Phase 5: observe (f2 Triton) ----
        obs_file = results_dir / "observations.pt"
        with phase("5_observe", backend=OBSERVE_BACKEND) as r:
            t0 = time.perf_counter()
            n_done = 0
            with torch.no_grad():
                for i, sample in enumerate(batches):
                    attention_mask = sample.get("attention_mask", None)
                    sample = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in sample.items()
                    }
                    with observer.set_attention_mask(attention_mask):
                        model(**sample)
                    n_done += 1
                    if (i + 1) % 10 == 0 or (i + 1) == len(batches):
                        cur_alloc = torch.cuda.memory_allocated() / 1024**3
                        logger.info("  batch %d/%d | gpu_alloc=%.3f GiB",
                                    i + 1, len(batches), cur_alloc)
            torch.cuda.is_available() and torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            r["n_batches_observed"] = n_done
            r["throughput_batches_per_s"] = round(n_done / elapsed, 4)
            r["throughput_tokens_per_s"] = round(total_tokens / elapsed, 2)
            observer.save_state(obs_file)
            observer.close_hooks()
            r["obs_state_file"] = str(obs_file)

        # reload observer data (same path as record_activations)
        with phase("5b_load_observer_state") as r:
            with open(obs_file, "rb") as f:
                observer_data = torch.load(f, weights_only=False)
            r["n_layers_in_state"] = len(observer_data)
            first_key = next(iter(observer_data))
            r["state_keys"] = list(observer_data[first_key].keys())
            r["total_tokens_observed"] = int(observer_data[first_key]["total_tokens"])
            logger.info("Observer state: %d layers, keys=%s",
                        r["n_layers_in_state"], r["state_keys"])

        # ---- Phase 6: prune (rank + slice + stream save) ----
        total_experts = int(getattr(model.config, "num_experts", 32))
        n_experts_to_prune = int(total_experts * COMPRESSION_RATIO)
        pruned_dir = get_pruned_model_dir(
            results_dir, n_experts_to_prune, total_experts,
            prune_args, SEED, obs_args.renormalize_router_weights,
        )
        with phase("6_prune_slice_save", n_prune=n_experts_to_prune,
                   keep=total_experts - n_experts_to_prune) as r:
            reap_prune(observer_data, model, prune_args, n_experts_to_prune, pruned_dir)
            # prune() slices + stream-saves weights; save tokenizer alongside
            # (mirrors reap.prune.run which calls tokenizer.save_pretrained).
            tokenizer.save_pretrained(pruned_dir)
            r["pruned_model_dir"] = str(pruned_dir)
            r["pruned_config_num_experts"] = int(getattr(model.config, "num_experts", -1))
            r["pruned_config_top_k"] = int(getattr(model.config, "num_experts_per_tok", -1))
            logger.info("Pruned model -> %s (experts=%d top_k=%d)",
                        pruned_dir, r["pruned_config_num_experts"], r["pruned_config_top_k"])

        # ---- Phase 7: smoke test (generate on sliced model) ----
        with phase("7_smoke_test") as r:
            try:
                smoke_test(model, tokenizer)
                r["smoke_ok"] = True
            except Exception as e:
                r["smoke_ok"] = False
                r["smoke_error"] = repr(e)
                logger.error("Smoke test failed: %s", e)

        # ---- Phase 8: checkpoint artifact summary ----
        with phase("8_artifact_summary") as r:
            files = sorted(pruned_dir.glob("**/*"))
            r["n_artifact_files"] = len(files)
            total_bytes = sum(f.stat().st_size for f in files if f.is_file())
            r["artifact_total_gib"] = round(total_bytes / 1024**3, 3)
            r["artifact_files"] = [str(f.relative_to(pruned_dir)) for f in files if f.is_file()]
            logger.info("Checkpoint: %d files, %.3f GiB", len(files), total_bytes / 1024**3)

        outcome = "success"
    except Exception as e:
        outcome = "error"
        logger.exception("RUN FAILED: %s", e)
        raise
    finally:
        sampler.stop()
        # write reports
        report = {"meta": meta, "outcome": outcome, "phases": _phase_metrics}
        with open(perf_json, "w") as f:
            json.dump(report, f, indent=2)
        if _phase_metrics:
            all_keys = sorted({k for row in _phase_metrics for k in row})
            with open(perf_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_keys)
                w.writeheader()
                for row in _phase_metrics:
                    w.writerow(row)
        print(f"\n=== perf_report: {perf_json} ===")
        print(f"=== gpu_timeline: {gpu_csv} ===")
        print(f"=== pruned model: {pruned_dir if 'pruned_dir' in dir() else 'n/a'} ===")


if __name__ == "__main__":
    main()