"""Model weight residency policy for low-RAM / single-GPU hosts.

Modes
-----
``gpu_full``
    Load with ``device_map="auto"``, mutate and stream-save from GPU tensors.
    Prefer when the model fits VRAM but is large relative to host RAM
    (e.g. LFM2-8B FP16 on g6.xlarge: ~16 GB weights, 16 GB RAM, 24 GB VRAM).

``layerwise``
    Block-wise observe. Prefer disk offload over stuffing the full model into
    host RAM when RAM is tight.

``cpu_full``
    Full model on CPU. Only safe when host RAM is comfortably larger than
    the model.

``auto``
    Pick among the above from measured host/GPU memory and an optional
    model-size estimate.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

ResidencyMode = Literal["auto", "gpu_full", "layerwise", "cpu_full"]
RESIDENCY_MODES: tuple[str, ...] = ("auto", "gpu_full", "layerwise", "cpu_full")

# Heuristics (bytes). Tunable via env for experiments.
_HOST_SAFE_FRACTION = float(os.environ.get("REAP_RESIDENCY_HOST_FRAC", "0.55"))
_GPU_FIT_FRACTION = float(os.environ.get("REAP_RESIDENCY_GPU_FRAC", "0.85"))
_HOST_TIGHT_FRAC = float(os.environ.get("REAP_RESIDENCY_HOST_TIGHT", "0.50"))


@dataclass(frozen=True)
class MemorySnapshot:
    host_total: int
    host_available: int
    gpu_total: int | None
    gpu_available: int | None


@dataclass(frozen=True)
class LoadPlan:
    """How to load / save weights under a resolved residency mode."""

    resolved: str
    device_map: str
    low_cpu_mem_usage: bool
    offload_folder: str | None
    stream_save_from_gpu: bool
    avoid_cpu_materialize: bool
    reason: str


def validate_residency(mode: str) -> str:
    m = (mode or "auto").lower().strip()
    if m not in RESIDENCY_MODES:
        raise ValueError(
            f"Unknown residency {mode!r}; expected one of {RESIDENCY_MODES}"
        )
    return m


def host_memory_bytes() -> tuple[int, int]:
    """Return ``(total, available)`` host RAM in bytes."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    # Linux
    try:
        total = available = 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
        if total > 0:
            return total, available or total // 2
    except Exception:
        pass
    # macOS fallback via sysctl
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        total = int(out.strip())
        return total, total // 2
    except Exception:
        return 16 * 1024**3, 8 * 1024**3


def gpu_memory_bytes() -> tuple[int | None, int | None]:
    """Return ``(total, free)`` for CUDA device 0, or ``(None, None)``."""
    if not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(0)
        return int(total), int(free)
    except Exception:
        try:
            props = torch.cuda.get_device_properties(0)
            total = int(props.total_memory)
            return total, total
        except Exception:
            return None, None


def snapshot_memory() -> MemorySnapshot:
    host_total, host_avail = host_memory_bytes()
    gpu_total, gpu_free = gpu_memory_bytes()
    return MemorySnapshot(
        host_total=host_total,
        host_available=host_avail,
        gpu_total=gpu_total,
        gpu_available=gpu_free,
    )


def _try_meta_parameter_count(cfg, *, trust_remote_code: bool) -> int | None:
    """Construct model metadata-only to get exact parameter count.

    Uses ``accelerate.init_empty_weights`` + ``from_config`` so no
    checkpoint tensors are loaded.  Falls back to ``None`` when the
    architecture is unknown or construction fails.
    """
    try:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        with init_empty_weights(include_buffers=True):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
        return sum(p.numel() for p in model.parameters())
    except Exception as exc:
        logger.debug("Meta-parameter count unavailable: %s", exc)
        return None


def _fallback_config_heuristic(cfg) -> int | None:
    """Fallback heuristic for unknown architectures.

    Unlike the old formula, this only charges expert params to the layers
    that the config explicitly identifies as MoE (``num_experts`` attribute
    exists per layer or is non-zero).  When the config does not distinguish
    MoE from dense layers, it returns ``None`` so the caller can warn and
    default to conservative behavior.
    """
    try:
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        vocab = int(getattr(cfg, "vocab_size", 0) or 0)
        if hidden <= 0 or layers <= 0:
            return None

        # Embedding + LM head.
        embed = vocab * hidden * 2  # tie weights by default

        num_experts = int(getattr(cfg, "num_experts", 0) or getattr(cfg, "num_local_experts", 0) or 0)
        inter = int(
            getattr(cfg, "intermediate_size", 0)
            or getattr(cfg, "moe_intermediate_size", 0)
            or 0
        )

        if num_experts > 0 and inter > 0:
            # Architecture exposes MoE structure — estimate per-layer breakdown.
            # Count expert-only FFN vs dense-attn-only layers transparently.
            moe_layers = int(getattr(cfg, "num_experts_per_layer", 0) or getattr(cfg, "num_moe_layers", 0) or layers)
            dense_layers = layers - moe_layers

            # Attention (QKV+O): ~4 * hidden^2 per layer.
            attn_per_layer = 4 * hidden * hidden

            # Dense FFN: ~3 * hidden * inter.
            dense_ffn_per_layer = 3 * hidden * inter if inter > 0 else 0

            # MoE FFN: experts * 3 * hidden * inter (up/gate/down).
            moe_ffn_per_layer = num_experts * 3 * hidden * inter

            params = (
                embed
                + dense_layers * (attn_per_layer + dense_ffn_per_layer)
                + moe_layers * (attn_per_layer + moe_ffn_per_layer)
            )
            return int(params)

        # No MoE info — attn-only estimate (dense model).
        per_layer = 4 * hidden * hidden + 3 * hidden * max(inter, hidden * 4) if inter > 0 else 12 * hidden * hidden
        params = embed + layers * per_layer
        return int(params)
    except Exception:
        return None


def estimate_model_bytes_from_config(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
    local_files_only: bool = False,
) -> int | None:
    """Estimate parameter bytes from HF config.

    Resolution order:
    1. Explicit config field (``num_parameters`` / ``n_params``).
    2. Meta-device model construction via ``init_empty_weights``.
    3. Conservative config heuristic (only for architectures with
       known MoE/dense layer layout).

    Returns ``None`` when no estimate is possible, allowing the caller
    to decide conservatively.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        logger.debug("Could not load config for size estimate (%s): %s", model_name, exc)
        return None

    # Stage 1 — explicit config field.
    n = getattr(cfg, "num_parameters", None)
    if n is None:
        n = getattr(cfg, "n_params", None)
    if isinstance(n, int) and n > 0:
        logger.debug("Size estimate from config field num_parameters=%d", n)
        return int(n) * 2

    # Stage 2 — meta-device model construction.
    meta_params = _try_meta_parameter_count(cfg, trust_remote_code=trust_remote_code)
    if meta_params is not None:
        logger.debug("Size estimate from meta-device model: %d params", meta_params)
        return meta_params * 2

    # Stage 3 — conservative heuristic.
    heuristic = _fallback_config_heuristic(cfg)
    if heuristic is not None:
        logger.warning(
            "Heuristic size estimate: %d params; auto-residency may be "
            "conservative for unknown architectures.",
            heuristic,
        )
        return heuristic * 2

    return None


def estimate_model_bytes_from_module(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def resolve_residency(
    requested: str,
    *,
    model_bytes: int | None = None,
    mem: MemorySnapshot | None = None,
    cli_prefers_layerwise: bool = False,
) -> tuple[str, str]:
    """Resolve ``auto``/explicit mode → concrete mode + human reason."""
    requested = validate_residency(requested)
    mem = mem or snapshot_memory()

    if requested != "auto":
        return requested, f"explicit residency={requested}"

    mb = model_bytes
    host_total = max(mem.host_total, 1)
    host_avail = max(mem.host_available, 0)
    gpu_total = mem.gpu_total

    # No size estimate: prefer GPU if present, else CPU; honor layerwise CLI bias.
    if mb is None:
        if cli_prefers_layerwise:
            if gpu_total and host_avail < 20 * 1024**3:
                return (
                    "layerwise",
                    "auto: layerwise CLI + limited host RAM (no size estimate)",
                )
            return "layerwise", "auto: layerwise CLI (no size estimate)"
        if gpu_total:
            return "gpu_full", "auto: CUDA present (no size estimate)"
        return "cpu_full", "auto: no CUDA (no size estimate)"

    fits_gpu = bool(gpu_total and mb <= _GPU_FIT_FRACTION * gpu_total)
    host_tight = mb >= _HOST_TIGHT_FRAC * host_total
    fits_host_safe = mb <= _HOST_SAFE_FRACTION * host_total

    if fits_gpu and host_tight:
        return (
            "gpu_full",
            f"auto: model~{mb / 1024**3:.1f}GiB fits GPU "
            f"({(gpu_total or 0) / 1024**3:.1f}GiB) but is large vs host "
            f"({host_total / 1024**3:.1f}GiB)",
        )
    if fits_gpu and not cli_prefers_layerwise:
        return (
            "gpu_full",
            f"auto: model~{mb / 1024**3:.1f}GiB fits GPU "
            f"({(gpu_total or 0) / 1024**3:.1f}GiB)",
        )
    if cli_prefers_layerwise or (gpu_total and not fits_gpu):
        mode = "layerwise"
        why = "layerwise CLI" if cli_prefers_layerwise else "model larger than GPU budget"
        return mode, f"auto: {why} (model~{mb / 1024**3:.1f}GiB)"
    if fits_host_safe:
        return (
            "cpu_full",
            f"auto: model~{mb / 1024**3:.1f}GiB fits host safely "
            f"({host_total / 1024**3:.1f}GiB)",
        )
    # Fallback: if GPU exists use it; else layerwise offload mindset
    if gpu_total:
        return "gpu_full", "auto: fallback to gpu_full (host tight, GPU present)"
    return "layerwise", "auto: fallback to layerwise (no safe full-CPU fit)"


def plan_load(
    resolved: str,
    *,
    offload_root: str | Path | None = None,
    low_cpu_mem_usage: bool = True,
) -> LoadPlan:
    """Translate a resolved residency mode into HF load/save knobs."""
    resolved = validate_residency(resolved)
    if resolved == "auto":
        raise ValueError("plan_load expects a resolved mode, not 'auto'")

    if resolved == "gpu_full":
        return LoadPlan(
            resolved=resolved,
            device_map="auto",
            low_cpu_mem_usage=low_cpu_mem_usage,
            offload_folder=None,
            stream_save_from_gpu=True,
            avoid_cpu_materialize=True,
            reason="GPU-resident weights; stream save from device tensors",
        )

    if resolved == "cpu_full":
        return LoadPlan(
            resolved=resolved,
            device_map="cpu",
            low_cpu_mem_usage=low_cpu_mem_usage,
            offload_folder=None,
            stream_save_from_gpu=False,
            avoid_cpu_materialize=False,
            reason="Full model on CPU",
        )

    # layerwise: prefer disk offload when possible so host is not forced to hold all weights
    offload: str | None = None
    if offload_root is not None:
        path = Path(offload_root)
        path.mkdir(parents=True, exist_ok=True)
        offload = str(path)
    else:
        offload = tempfile.mkdtemp(prefix="reap_offload_")

    # device_map="auto" with offload_folder lets accelerate page layers to disk.
    # Layerwise observer still moves one block to CUDA at a time when needed.
    return LoadPlan(
        resolved=resolved,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder=offload,
        stream_save_from_gpu=True,
        avoid_cpu_materialize=True,
        reason="Layerwise observe; weights via auto+disk offload (avoid full CPU pin)",
    )


def preflight_or_warn(
    resolved: str,
    model_bytes: int | None,
    mem: MemorySnapshot | None = None,
    *,
    strict: bool = False,
) -> None:
    """Log warnings (or raise if strict) when residency is likely to OOM."""
    mem = mem or snapshot_memory()
    if model_bytes is None:
        return
    if resolved == "cpu_full" and model_bytes > _HOST_SAFE_FRACTION * mem.host_total:
        msg = (
            f"residency=cpu_full but model~{model_bytes / 1024**3:.1f}GiB may exceed "
            f"safe host RAM ({mem.host_total / 1024**3:.1f}GiB total). "
            "Prefer --residency gpu_full on small-RAM GPU instances."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    if (
        resolved == "gpu_full"
        and mem.gpu_total is not None
        and model_bytes > _GPU_FIT_FRACTION * mem.gpu_total
    ):
        msg = (
            f"residency=gpu_full but model~{model_bytes / 1024**3:.1f}GiB may exceed "
            f"GPU capacity ({mem.gpu_total / 1024**3:.1f}GiB). "
            "Prefer --residency layerwise."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)


def load_causal_lm(
    model_name: str,
    plan: LoadPlan,
    *,
    torch_dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = False,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """``from_pretrained`` honoring a :class:`LoadPlan`."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "device_map": plan.device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        "low_cpu_mem_usage": plan.low_cpu_mem_usage,
        "local_files_only": local_files_only,
    }
    if plan.offload_folder:
        kwargs["offload_folder"] = plan.offload_folder
        # accelerate uses this when device_map is auto and offload is set
        kwargs.setdefault("offload_state_dict", True)

    logger.info(
        "Loading %s with residency=%s device_map=%s offload=%s (%s)",
        model_name,
        plan.resolved,
        plan.device_map,
        plan.offload_folder,
        plan.reason,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def log_model_residency(model: Any, *, phase: str) -> None:
    """Log Accelerate placement and CUDA peak allocation without moving weights."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        placements = sorted({str(device) for device in device_map.values()})
        logger.info("%s device placement: %s", phase, ", ".join(placements))
    else:
        try:
            logger.info("%s model device: %s", phase, next(model.parameters()).device)
        except (StopIteration, AttributeError):
            logger.info("%s model device: unavailable", phase)
    if torch.cuda.is_available():
        logger.info(
            "%s CUDA memory allocated=%.2f GiB peak=%.2f GiB",
            phase,
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.max_memory_allocated() / 1024**3,
        )


def stream_save_pretrained(model: Any, output_dir: str | Path) -> None:
    """Save without materializing a full CPU state dict when possible."""
    from accelerate.hooks import remove_hook_from_module

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        remove_hook_from_module(model, recurse=True)
    except Exception as exc:
        logger.debug("remove_hook_from_module: %s", exc)
    # Do not model.to("cpu") — safetensors can stream CUDA tensors shard-wise.
    model.save_pretrained(output_dir)
    logger.info("Saved model to %s (stream path; hooks stripped)", output_dir)
