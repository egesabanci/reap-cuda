"""Hermetic tests for EC2 run-findings fixes (#14–#23)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from reap.kernels.router import (
    f5_router_from_module,
    prefers_native_router,
)
from reap.kernels.triton_frea import (
    choose_frea_block_sizes,
    estimate_frea_shared_bytes,
)
from reap.kernels.triton_utils import (
    clear_triton_disable_memo,
    format_triton_usage_summary,
    log_triton_fallback,
    prefer_triton_for,
    record_triton_ok,
    reset_triton_usage,
    shared_mem_feasible,
    triton_usage_snapshot,
)
from reap.kernels.weight_cache import cache_size, free_cache, get_stacked_expert_weights
from reap.pipeline import create_results_directory, resolve_artifacts_root, smoke_test


# ---------------------------------------------------------------------------
# #14 native router detection
# ---------------------------------------------------------------------------


class _SigmoidBiasRouter(nn.Module):
    def __init__(self, h=8, e=4, top_k=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(e, h))
        self.top_k = top_k
        self.use_expert_bias = True
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0

    def forward(self, hidden_states, expert_bias=None):
        logits = F.linear(hidden_states, self.weight)
        routing_weights = logits.sigmoid()
        if self.use_expert_bias:
            scores = routing_weights + (
                expert_bias if expert_bias is not None else 0.0
            )
            _, selected = torch.topk(scores, self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, 1, selected)
        else:
            routing_weights, selected = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
        if self.norm_topk_prob:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-6
            )
        routing_weights = routing_weights * self.routed_scaling_factor
        return logits, routing_weights, selected


class _MoeWithBias(nn.Module):
    def __init__(self, h=8, e=4, k=2):
        super().__init__()
        self.gate = _SigmoidBiasRouter(h, e, k)
        self.expert_bias = nn.Parameter(torch.zeros(e))
        self.num_experts = e


class _SoftmaxAdapter:
    adapter_name = "mock_softmax"

    def router_attr(self):
        return "gate"


class _Lfm2LikeAdapter:
    adapter_name = "lfm2_moe"

    def router_attr(self):
        return "gate"


def test_prefers_native_router_on_expert_bias():
    moe = _MoeWithBias()
    assert prefers_native_router(moe, _SoftmaxAdapter()) is True


def test_prefers_native_router_on_adapter_name():
    moe = _MoeWithBias()
    assert prefers_native_router(moe, _Lfm2LikeAdapter()) is True


def test_prefers_native_router_false_for_plain_softmax_gate():
    """Bare Linear gate without expert_bias must stay on F5 softmax path."""

    class _Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 4, bias=False)

    assert prefers_native_router(_Plain(), _SoftmaxAdapter()) is False


def test_f5_router_from_module_uses_bias():
    torch.manual_seed(0)
    moe = _MoeWithBias()
    flat = torch.randn(6, 8)
    logits, pairs = f5_router_from_module(
        moe, _Lfm2LikeAdapter(), flat, top_k=2
    )
    assert logits.shape == (6, 4)
    assert pairs.selected_experts.shape == (6, 2)
    assert pairs.pair_token_idx.numel() == 12
    assert pairs.expert_offsets.shape == (5,)


# ---------------------------------------------------------------------------
# #15 F4 single-entry cache
# ---------------------------------------------------------------------------


class _FusedExperts(nn.Module):
    def __init__(self, e=2, h=4, i=4):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.randn(e, h, i))


class _QwenLikeMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _FusedExperts()
        self.gate = nn.Linear(4, 2, bias=False)


def test_f4_cache_bounded_to_one_entry():
    from reap.model_adapters import Qwen3MoeModelAdapter

    free_cache()
    adapter = Qwen3MoeModelAdapter()
    m1, m2, m3 = _QwenLikeMoe(), _QwenLikeMoe(), _QwenLikeMoe()
    get_stacked_expert_weights(m1, adapter)
    assert cache_size() == 1
    get_stacked_expert_weights(m2, adapter)
    assert cache_size() == 1
    get_stacked_expert_weights(m3, adapter)
    assert cache_size() == 1
    free_cache()
    assert cache_size() == 0


# ---------------------------------------------------------------------------
# #16 / #23 smoke_test hermetic
# ---------------------------------------------------------------------------


def test_smoke_test_batch_encoding_path():
    """transformers 5.14-style: apply_chat_template returns a mapping."""

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, *a, **k):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def batch_decode(self, outs, skip_special_tokens=False):
            return ["ok"]

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        @property
        def device(self):
            return torch.device("cpu")

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]])

    smoke_test(Mod(), Tok())  # must not raise


def test_smoke_test_fallback_plain_tokenize():
    class Tok:
        pad_token_id = 0

        def apply_chat_template(self, *a, **k):
            raise RuntimeError("no chat template")

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[5, 6]])}

        def batch_decode(self, outs, skip_special_tokens=False):
            return ["plain"]

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            return torch.tensor([[5, 6, 7]])

    smoke_test(Mod(), Tok())


# ---------------------------------------------------------------------------
# #17 / #20 FREA shared-mem scaling (hardware-agnostic)
# ---------------------------------------------------------------------------


def test_estimate_frea_shared_bytes_known_shape():
    # Reverse-engineered L4 failure case: 128/128/16 → 139264
    assert estimate_frea_shared_bytes(16, 128, 128) == 2 * 128 * 128 * 4 + 16 * 128 * 4 + 4096


def test_choose_frea_blocks_respects_tight_shared_mem(monkeypatch):
    # Simulate L4 opt-in budget (99 KiB); default on L4 is 48 KiB, opt-in 99 KiB
    def _smem(device=None, prefer_optin=False):
        return 101376

    monkeypatch.setattr("reap.kernels.triton_frea.device_shared_memory_bytes", _smem)
    monkeypatch.setattr("reap.kernels.triton_utils.device_shared_memory_bytes", _smem)
    blocks = choose_frea_block_sizes(2048, 1792)
    assert blocks is not None
    bh, bi, bn = blocks
    need = estimate_frea_shared_bytes(bn, bh, bi)
    ok, reason = shared_mem_feasible(need, device=None)
    assert ok, reason
    # Must not pick 128/128 on 99 KiB
    assert not (bh == 128 and bi == 128)


def test_choose_frea_blocks_none_when_impossible(monkeypatch):
    def _smem(device=None, prefer_optin=False):
        return 1024  # tiny

    monkeypatch.setattr("reap.kernels.triton_frea.device_shared_memory_bytes", _smem)
    assert choose_frea_block_sizes(2048, 1792) is None


def test_prefer_triton_min_numel():
    t = torch.zeros(2, dtype=torch.float32)
    # Without CUDA this is False either way; just ensure API accepts min_numel
    assert prefer_triton_for(t, min_numel=100) is False


# ---------------------------------------------------------------------------
# #18 Triton usage accounting
# ---------------------------------------------------------------------------


def test_triton_usage_summary_counters():
    reset_triton_usage()
    clear_triton_disable_memo()
    record_triton_ok("f2_reduce")
    record_triton_ok("f2_reduce")
    log_triton_fallback("frea", "shared mem test")
    snap = triton_usage_snapshot()
    assert snap["f2_reduce"]["ok"] == 2
    assert snap["frea"]["fallback"] == 1
    text = format_triton_usage_summary()
    assert "f2_reduce" in text and "frea" in text
    reset_triton_usage()


# ---------------------------------------------------------------------------
# #19 F2 scatter dtype contract (PyTorch path)
# ---------------------------------------------------------------------------


def test_scatter_pytorch_fp64():
    from reap.kernels.triton_reduce import scatter_pair_stats

    torch.manual_seed(0)
    pair_out = torch.randn(12, 32)
    idx = torch.randint(0, 4, (12,))
    w = torch.rand(12)
    out = scatter_pair_stats(pair_out, idx, w, num_experts=4)
    assert out["ean_sum"].dtype == torch.float64
    assert out["weighted_ean_sum"].dtype == torch.float64
    assert out["weighted_freq"].dtype == torch.float64
    assert out["batch_max"].dtype == torch.float32


# ---------------------------------------------------------------------------
# #21 local dataset path
# ---------------------------------------------------------------------------


def test_load_local_arrow(tmp_path: Path):
    from datasets import Dataset

    from reap.data import _load_local_dataset, load_category_batches

    ds = Dataset.from_dict(
        {
            "instruction": ["print hello"] * 4,
            "output": ["print('hello')"] * 4,
        }
    )
    arrow = tmp_path / "data.arrow"
    # datasets Dataset.from_file expects arrow table; use save + from_file via map
    ds.save_to_disk(str(tmp_path / "disk"))
    loaded = _load_local_dataset(str(tmp_path / "disk"))
    assert len(loaded) == 4

    class FakeTok:
        model_max_length = 32
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, *a, **k):
            return {
                "input_ids": torch.ones(1, 8, dtype=torch.long),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }

        def apply_chat_template(self, *a, **k):
            return "x"

        def encode(self, *a, **k):
            return [1, 2, 3]

    # Processor path with dataset_path
    batches = load_category_batches(
        dataset_name="theblackcat102/evol-codealpaca-v1",
        split="train",
        subset=None,
        tokenizer=FakeTok(),
        model_max_length=32,
        batch_size=2,
        split_by_category=False,
        return_vllm_tokens_prompt=False,
        truncate=True,
        batches_per_category=1,
        dataset_path=str(tmp_path / "disk"),
    )
    assert "all" in batches or len(batches) >= 1


# ---------------------------------------------------------------------------
# #22 artifacts dir
# ---------------------------------------------------------------------------


def test_create_results_directory_custom_base(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("REAP_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("REAP_OUTPUT_DIR", raising=False)
    out = create_results_directory(
        "org/MyModel", "theblackcat102/evol-codealpaca-v1", base=tmp_path / "arts"
    )
    assert out.parent.parent == (tmp_path / "arts").resolve()
    assert out.exists()


def test_resolve_artifacts_root_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REAP_ARTIFACTS_DIR", str(tmp_path / "from_env"))
    assert resolve_artifacts_root(None) == (tmp_path / "from_env").resolve()


# ---------------------------------------------------------------------------
# CLI wiring for new flags
# ---------------------------------------------------------------------------


def test_cli_help_shows_dataset_path_and_artifacts():
    from typer.testing import CliRunner

    from reap.cli.app import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["prune", "full", "--help"],
        color=False,
        env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"},
    )
    assert result.exit_code == 0
    low = result.stdout.lower()
    assert "dataset-path" in low or "dataset_path" in low
    assert "artifacts-dir" in low or "artifacts" in low
    assert "frea-backend" in low or "frea_backend" in low


def test_frea_backend_set_and_get():
    from reap.kernels.triton_frea import (
        get_frea_backend,
        reset_frea_probe_cache,
        set_frea_backend,
    )

    reset_frea_probe_cache()
    assert set_frea_backend("pytorch") == "pytorch"
    assert get_frea_backend() == "pytorch"
    set_frea_backend("auto")
    assert get_frea_backend() == "auto"


def test_tile_profitable_heuristic():
    from reap.kernels.triton_frea import _tile_profitable

    assert _tile_profitable((128, 64, 16)) is True
    assert _tile_profitable((64, 64, 16)) is False
    assert _tile_profitable(None) is False


@patch("reap.prune.run")
def test_cli_passes_dataset_path_and_artifacts(mock_run: MagicMock):
    from typer.testing import CliRunner

    from reap.cli.app import app

    mock_run.return_value = None
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "prune",
            "full",
            "--observe-only",
            "--dataset-path",
            "/tmp/local.arrow",
            "--artifacts-dir",
            "/data/out",
            "--residency",
            "gpu_full",
        ],
        color=False,
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code == 0, result.stdout
    reap_args = mock_run.call_args.args[0]
    ds_args = mock_run.call_args.args[1]
    assert reap_args.artifacts_dir == "/data/out"
    assert ds_args.dataset_path == "/tmp/local.arrow"


# ---------------------------------------------------------------------------
# FREA probe/SM state scoping + bulk offset extraction
# ---------------------------------------------------------------------------


def test_frea_probe_cache_is_dtor_scoped():
    """Probe choices must be scoped by device type, index, and dtype."""
    from reap.kernels.triton_frea import _probe_key, reset_frea_probe_cache

    reset_frea_probe_cache()
    wg_fp16 = torch.randn(4, 32, 64, dtype=torch.float16)
    wg_fp32 = torch.randn(4, 32, 64, dtype=torch.float32)
    key_fp16 = _probe_key(torch.empty(1, dtype=torch.float16), wg_fp16)
    key_fp32 = _probe_key(torch.empty(1, dtype=torch.float32), wg_fp32)
    assert key_fp16 != key_fp32
    key_cpu = _probe_key(torch.empty(1, 1, dtype=torch.float32), wg_fp32)
    assert key_cpu[0] == "cpu"
    reset_frea_probe_cache()


def test_frea_smem_optin_is_per_device():
    """SM opt-in state must be tracked per-device, not globally."""
    from reap.kernels.triton_frea import (
        _get_smem_optin,
        _set_smem_optin,
        reset_frea_probe_cache,
    )

    reset_frea_probe_cache()
    dev_a = torch.device("cpu")
    dev_b_str = "cuda:0"

    _set_smem_optin(dev_a, True)
    _set_smem_optin(dev_b_str, False)
    assert _get_smem_optin(dev_a) is True
    assert _get_smem_optin(dev_b_str) is False
    assert _get_smem_optin(torch.device("cpu")) is True
    reset_frea_probe_cache()
    assert _get_smem_optin(dev_a) is None


def test_frea_scoped_disable_isolates_devices():
    """A scoped disable on one device must not affect another."""
    from reap.kernels.triton_utils import (
        clear_triton_disable_memo,
        disable_component,
        is_component_disabled,
    )

    clear_triton_disable_memo()
    assert is_component_disabled("frea", scope="cuda:0") is None
    disable_component("frea", "SM OOM", scope="cuda:0")
    assert is_component_disabled("frea", scope="cuda:0") is not None
    assert is_component_disabled("frea", scope="cuda:1") is None
    assert is_component_disabled("frea") is None
    clear_triton_disable_memo()


def test_frea_global_disable_still_works():
    """Backward compat: disable_component without scope still works globally."""
    from reap.kernels.triton_utils import (
        clear_triton_disable_memo,
        disable_component,
        is_component_disabled,
    )

    clear_triton_disable_memo()
    disable_component("frea", "global SM failure")
    assert is_component_disabled("frea") is not None
    assert is_component_disabled("frea", scope="cuda:0") is not None
    clear_triton_disable_memo()


def test_bmm_bulk_offset_extraction_correctness():
    """bmm.routed_expert_activations_grouped must produce correct results with bulk offsets."""
    from reap.kernels.bmm import routed_expert_activations_grouped
    from reap.kernels.router import f5_router_pytorch

    torch.manual_seed(42)
    t, e, k, h, i = 16, 4, 2, 32, 32
    flat = torch.randn(t, h)
    logits = torch.randn(t, e)
    pairs = f5_router_pytorch(logits, k, use_triton_softmax=False)
    W_gate = torch.randn(e, i, h)
    W_up = torch.randn(e, i, h)
    W_down = torch.randn(e, h, i)

    out = routed_expert_activations_grouped(flat, pairs, W_gate, W_up, W_down)
    assert out.shape == (t * k, h)

    offsets = pairs.expert_offsets.tolist()
    for eid in range(e):
        start, end = offsets[eid], offsets[eid + 1]
        if start == end:
            continue
        xe = flat[pairs.pair_token_idx[start:end]]
        g = torch.nn.functional.linear(xe, W_gate[eid])
        u = torch.nn.functional.linear(xe, W_up[eid])
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(g) * u, W_down[eid]
        )
        assert torch.allclose(out[start:end], expected, atol=1e-5), (
            f"Expert {eid} mismatch"
        )


def test_frea_usage_summary_handles_scoped_disables():
    """format_triton_usage_summary must handle scoped disable entries gracefully."""
    from reap.kernels.triton_utils import (
        clear_triton_disable_memo,
        disable_component,
        format_triton_usage_summary,
        reset_triton_usage,
    )

    reset_triton_usage()
    clear_triton_disable_memo()
    disable_component("frea", "SM OOM", scope="cuda:0")
    text = format_triton_usage_summary()
    assert "frea" in text
    assert "scope=cuda:0" in text
    clear_triton_disable_memo()
    reset_triton_usage()


# ---------------------------------------------------------------------------
# FREA capability-scoped disable memoization (per device/dtype/H/I/tile)
# ---------------------------------------------------------------------------


def test_frea_capability_key_distinguishes_dtype():
    """The FREA capability key must distinguish input dtype."""
    from reap.kernels.triton_frea import _capability_key, reset_frea_probe_cache

    reset_frea_probe_cache()
    wg_fp16 = torch.randn(4, 32, 64, dtype=torch.float16)
    wg_fp32 = torch.randn(4, 32, 64, dtype=torch.float32)
    xi_fp16 = torch.empty(1, 64, dtype=torch.float16)
    xi_fp32 = torch.empty(1, 64, dtype=torch.float32)
    blocks = (64, 64, 16)
    key_fp16 = _capability_key(xi_fp16, wg_fp16, blocks)
    key_fp32 = _capability_key(xi_fp32, wg_fp32, blocks)
    assert key_fp16 != key_fp32


def test_frea_capability_key_distinguishes_tile():
    """The FREA capability key must distinguish the tile tuple."""
    from reap.kernels.triton_frea import _capability_key, reset_frea_probe_cache

    reset_frea_probe_cache()
    wg = torch.randn(4, 32, 64, dtype=torch.float16)
    xi = torch.empty(1, 64, dtype=torch.float16)
    key_a = _capability_key(xi, wg, (128, 128, 16))
    key_b = _capability_key(xi, wg, (64, 64, 16))
    assert key_a != key_b


def test_frea_capability_key_distinguishes_h_i():
    """The FREA capability key must distinguish H and I dimensions."""
    from reap.kernels.triton_frea import _capability_key, reset_frea_probe_cache

    reset_frea_probe_cache()
    wg_a = torch.randn(4, 32, 64, dtype=torch.float16)
    wg_b = torch.randn(4, 64, 128, dtype=torch.float16)
    xi = torch.empty(1, 64, dtype=torch.float16)
    key_a = _capability_key(xi, wg_a, (64, 32, 16))
    key_b = _capability_key(xi, wg_b, (64, 64, 16))
    assert key_a != key_b


def test_frea_capability_scoped_disable_isolates_dtype():
    """Disabling one (device, dtype, H, I, tile) must not disable a different dtype."""
    from reap.kernels.triton_frea import _capability_key, reset_frea_probe_cache
    from reap.kernels.triton_utils import (
        clear_triton_disable_memo,
        disable_component,
        is_component_disabled,
    )

    reset_frea_probe_cache()
    clear_triton_disable_memo()
    wg_fp16 = torch.randn(4, 32, 64, dtype=torch.float16)
    wg_fp32 = torch.randn(4, 32, 64, dtype=torch.float32)
    xi_fp16 = torch.empty(1, 64, dtype=torch.float16)
    xi_fp32 = torch.empty(1, 64, dtype=torch.float32)
    blocks = (64, 64, 16)
    key_fp16 = _capability_key(xi_fp16, wg_fp16, blocks)
    key_fp32 = _capability_key(xi_fp32, wg_fp32, blocks)
    assert key_fp16 != key_fp32

    disable_component("frea", "SM OOM fp16", scope=key_fp16)
    assert is_component_disabled("frea", scope=key_fp16) is not None
    # The fp32 capability on the same device must remain eligible.
    assert is_component_disabled("frea", scope=key_fp32) is None
    clear_triton_disable_memo()
    reset_frea_probe_cache()


def test_frea_capability_scoped_disable_isolates_tile():
    """Disabling one tile configuration must not disable a different tile."""
    from reap.kernels.triton_frea import _capability_key, reset_frea_probe_cache
    from reap.kernels.triton_utils import (
        clear_triton_disable_memo,
        disable_component,
        is_component_disabled,
    )

    reset_frea_probe_cache()
    clear_triton_disable_memo()
    wg = torch.randn(4, 32, 64, dtype=torch.float16)
    xi = torch.empty(1, 64, dtype=torch.float16)
    key_big = _capability_key(xi, wg, (128, 128, 16))
    key_small = _capability_key(xi, wg, (64, 64, 16))
    assert key_big != key_small

    disable_component("frea", "SM OOM 128x128", scope=key_big)
    assert is_component_disabled("frea", scope=key_big) is not None
    # The small-tile capability on the same device/dtype/H/I must remain eligible.
    assert is_component_disabled("frea", scope=key_small) is None
    clear_triton_disable_memo()
    reset_frea_probe_cache()


def test_frea_capability_scoped_disable_memoized_on_repeat():
    """A repeated invocation of the same capability observes the memoized disable."""
    from reap.kernels.triton_frea import _capability_key, reset_frea_probe_cache
    from reap.kernels.triton_utils import (
        clear_triton_disable_memo,
        disable_component,
        is_component_disabled,
    )

    reset_frea_probe_cache()
    clear_triton_disable_memo()
    wg = torch.randn(4, 32, 64, dtype=torch.float16)
    xi = torch.empty(1, 64, dtype=torch.float16)
    key = _capability_key(xi, wg, (64, 64, 16))

    disable_component("frea", "SM OOM", scope=key)
    # Repeated checks for the same capability must observe the memo.
    assert is_component_disabled("frea", scope=key) is not None
    assert is_component_disabled("frea", scope=key) == "SM OOM"
    clear_triton_disable_memo()
    reset_frea_probe_cache()


def test_frea_capability_scoped_disable_via_support_check(monkeypatch):
    """_triton_frea_supported must honor the capability-scoped disable.

    Mocks the structural checks to pass so only the disable memo determines
    the result. Two different dtypes on the same device must be isolated.
    """
    import reap.kernels.triton_frea as frea_mod
    from reap.kernels.triton_frea import (
        _capability_key,
        _triton_frea_supported,
        reset_frea_probe_cache,
    )
    from reap.kernels.triton_utils import (
        clear_triton_disable_memo,
        disable_component,
    )

    reset_frea_probe_cache()
    clear_triton_disable_memo()

    # Mock structural prerequisites so the support check reaches the
    # capability-scoped disable memo.
    monkeypatch.setattr(frea_mod, "_is_silu", lambda fn: True)
    monkeypatch.setattr(frea_mod, "triton_runtime_available", lambda: True)
    monkeypatch.setattr(
        frea_mod, "prefer_triton_for", lambda t, min_numel=None: True
    )
    monkeypatch.setattr(
        frea_mod, "choose_frea_block_sizes", lambda h, i, device=None: (64, 64, 16)
    )

    xi_fp16 = torch.empty(1, 64, dtype=torch.float16)
    xi_fp32 = torch.empty(1, 64, dtype=torch.float32)
    # Use mock W_gate objects so .is_cuda can be True on a CPU test machine.
    from unittest.mock import MagicMock

    wg_fp16 = MagicMock()
    wg_fp16.shape = (4, 64, 64)
    wg_fp16.dtype = torch.float16
    wg_fp16.is_cuda = True
    wg_fp32 = MagicMock()
    wg_fp32.shape = (4, 64, 64)
    wg_fp32.dtype = torch.float32
    wg_fp32.is_cuda = True

    key_fp16 = _capability_key(xi_fp16, wg_fp16, (64, 64, 16))
    disable_component("frea", "SM OOM", scope=key_fp16)

    ok_fp16, reason_fp16 = _triton_frea_supported(
        xi_fp16, wg_fp16, act_fn=F.silu
    )
    assert not ok_fp16
    assert "disabled" in reason_fp16

    # The fp32 capability must remain eligible (not disabled).
    ok_fp32, reason_fp32 = _triton_frea_supported(
        xi_fp32, wg_fp32, act_fn=F.silu
    )
    assert ok_fp32, f"fp32 capability should be eligible but got: {reason_fp32}"

    clear_triton_disable_memo()
    reset_frea_probe_cache()


# ---------------------------------------------------------------------------
# Native-router loop-vs-routed weighted-metric parity (Fix #2)
# and prune-only no-full-router-weight (Fix #5)
# ---------------------------------------------------------------------------


class _NonSoftmaxRouter(nn.Module):
    """Router that returns deliberately non-softmax-compatible outputs.

    Returns zero logits with a fixed selected weight of ``0.9`` so that a
    softmax reconstruction (which would give ``1/E``) clearly differs from the
    native selected weight.
    """

    def __init__(self, h=8, e=4, top_k=2, weight=0.9):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(e, h))
        self.top_k = top_k
        self.use_expert_bias = True
        self.norm_topk_prob = False
        self.routed_scaling_factor = 1.0
        self._fixed_weight = weight

    def forward(self, hidden_states, expert_bias=None):
        logits = F.linear(hidden_states, self.weight)
        scores = logits.sigmoid()
        if expert_bias is not None:
            scores = scores + expert_bias
        _, selected = torch.topk(scores, self.top_k, dim=-1)
        t = hidden_states.shape[0]
        routing_weights = torch.full(
            (t, self.top_k), self._fixed_weight, dtype=torch.float32
        )
        return logits, routing_weights, selected


class _NativeFusedMoe(nn.Module):
    def __init__(self, e=4, h=8, i=8, k=2, weight=0.9):
        super().__init__()
        self.gate = _NonSoftmaxRouter(h, e, k, weight=weight)
        self.expert_bias = nn.Parameter(torch.zeros(e))
        self.experts = _FusedExperts(e, h, i)
        self.num_experts = e
        self.top_k = k


class _Lfm2FusedAdapter:
    adapter_name = "lfm2_moe"

    def router_attr(self):
        return "gate"

    def experts_attr(self):
        return "experts"

    def expert_weight_attrs(self, moe=None):
        return {
            "experts": "experts",
            "gate": "gate",
            "fused": True,
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "down_proj": "down_proj",
            "weight_convention": "linear",
        }

    def weight_convention(self):
        return "linear"


_WEIGHTED_KEYS = [
    "weighted_expert_frequency_sum",
    "weighted_ean_sum",
    "reap",
]


def _observe_native(backend: str, record_only: bool = True):
    from reap.kernels.observe import observe_moe_batch
    from reap.pruning_metrics import initialize_pruning_state

    torch.manual_seed(42)
    moe = _NativeFusedMoe(e=4, h=8, i=8, k=2, weight=0.9)
    adapter = _Lfm2FusedAdapter()
    state = initialize_pruning_state(4, device="cpu")
    flat = torch.randn(12, 8)
    out = observe_moe_batch(
        state,
        moe,
        adapter,
        flat,
        num_experts=4,
        top_k=2,
        backend=backend,
        record_pruning_metrics_only=record_only,
        fused=True,
    )
    return state, out


def test_native_router_loop_vs_bmm_weighted_parity():
    """Loop and bmm must agree on router-weighted saliency for native routers."""
    state_loop, _ = _observe_native("loop")
    state_bmm, _ = _observe_native("bmm")
    for k in _WEIGHTED_KEYS:
        a = state_loop[k]
        b = state_bmm[k]
        if isinstance(a, torch.Tensor):
            assert torch.allclose(
                a.to(torch.float32).cpu(),
                b.to(torch.float32).cpu(),
                atol=1e-5,
                rtol=1e-5,
            ), f"loop vs bmm mismatch for {k}:\n{a}\nvs\n{b}"
        elif hasattr(a, "mean"):
            assert torch.allclose(
                a.mean.to(torch.float32).cpu(),
                b.mean.to(torch.float32).cpu(),
                atol=1e-5,
                rtol=1e-5,
            ), f"loop vs bmm mismatch for {k}.mean:\n{a.mean}\nvs\n{b.mean}"
            assert torch.equal(a.count, b.count), (
                f"count mismatch for {k}: {a.count} vs {b.count}"
            )


def test_native_router_selected_weights_differ_from_softmax_reconstruction():
    """Demonstrate that softmax reconstruction would produce a different value.

    The native router returns a fixed selected weight of 0.9. A softmax on the
    zero-biased logits would give 1/E = 0.25 per expert. The loop backend must
    use the native 0.9, not the softmax-derived 0.25.
    """
    state, _ = _observe_native("loop")
    e = 4
    # Each token selects top_k=2 experts with weight 0.9. With 12 tokens,
    # the total weighted frequency per expert is (count_of_selections * 0.9).
    # The softmax reconstruction would give 0.25 per selection instead.
    wf = state["weighted_expert_frequency_sum"]
    # The native weight is 0.9 per selected pair. Verify at least one expert
    # has a weighted frequency that is a multiple of 0.9 (not 0.25).
    total = wf.sum().item()
    # 12 tokens * 2 selections * 0.9 = 21.6 total weight.
    assert abs(total - 12 * 2 * 0.9) < 1e-4, (
        f"Expected total weighted frequency 21.6 (native 0.9), got {total}"
    )
    # Softmax reconstruction would have given 12 * 2 * 0.25 = 6.0.
    assert abs(total - 6.0) > 1e-4, (
        "Total weighted frequency matches softmax reconstruction (0.25); "
        "native selected weights were not used."
    )


def test_native_router_unweighted_metrics_unchanged():
    """Unweighted metrics must preserve existing behavior (loop vs bmm)."""
    state_loop, _ = _observe_native("loop")
    state_bmm, _ = _observe_native("bmm")
    for k in ["total_tokens", "expert_frequency", "ean_sum", "max_activations"]:
        a = state_loop[k]
        b = state_bmm[k]
        if isinstance(a, torch.Tensor):
            if a.dtype.is_floating_point:
                assert torch.allclose(
                    a.to(torch.float64).cpu(), b.to(torch.float64).cpu(), atol=1e-4
                ), f"{k}: {a} vs {b}"
            else:
                assert torch.equal(a.cpu(), b.cpu()), f"{k}: {a} vs {b}"


def test_native_router_loop_vs_frea_weighted_parity():
    """Loop and frea (PyTorch fallback on CPU) must agree on native-router weighted metrics."""
    from reap.kernels.triton_frea import reset_frea_probe_cache

    reset_frea_probe_cache()
    state_loop, _ = _observe_native("loop")
    state_frea, _ = _observe_native("frea")
    for k in _WEIGHTED_KEYS:
        a = state_loop[k]
        b = state_frea[k]
        if isinstance(a, torch.Tensor):
            assert torch.allclose(
                a.to(torch.float32).cpu(),
                b.to(torch.float32).cpu(),
                atol=1e-5,
                rtol=1e-5,
            ), f"loop vs frea mismatch for {k}: {a} vs {b}"
        elif hasattr(a, "mean"):
            assert torch.allclose(
                a.mean.to(torch.float32).cpu(),
                b.mean.to(torch.float32).cpu(),
                atol=1e-5,
                rtol=1e-5,
            ), f"loop vs frea mismatch for {k}.mean"
    reset_frea_probe_cache()


def test_native_router_prune_only_no_full_weights():
    """Prune-only routed mode must not allocate the (T, E) full router weights."""
    from reap.kernels.router import f5_router_from_module

    torch.manual_seed(42)
    moe = _NativeFusedMoe(e=4, h=8, i=8, k=2, weight=0.9)
    adapter = _Lfm2FusedAdapter()
    flat = torch.randn(12, 8)

    # Prune-only mode: router_weights_full must be None.
    _, pairs_prune = f5_router_from_module(
        moe, adapter, flat, top_k=2, include_router_weights_full=False
    )
    assert pairs_prune.router_weights_full is None
    assert pairs_prune.selected_experts.shape == (12, 2)
    assert pairs_prune.pair_router_w.numel() == 24
    assert pairs_prune.expert_offsets.shape == (5,)

    # Full mode: router_weights_full must be (T, E).
    _, pairs_full = f5_router_from_module(
        moe, adapter, flat, top_k=2, include_router_weights_full=True
    )
    assert pairs_full.router_weights_full is not None
    assert pairs_full.router_weights_full.shape == (12, 4)

    # Selected experts and pair weights must match between modes.
    assert torch.equal(pairs_prune.selected_experts, pairs_full.selected_experts)
    assert torch.allclose(
        pairs_prune.pair_router_w, pairs_full.pair_router_w, atol=1e-6
    )
    assert torch.equal(pairs_prune.expert_offsets, pairs_full.expert_offsets)


def test_native_router_prune_only_metrics_match_full_mode():
    """Prune-only and full-weight routed metrics must be equivalent."""
    from reap.kernels.observe import observe_moe_batch
    from reap.pruning_metrics import initialize_pruning_state

    def _run(record_only: bool):
        torch.manual_seed(42)
        moe = _NativeFusedMoe(e=4, h=8, i=8, k=2, weight=0.9)
        adapter = _Lfm2FusedAdapter()
        state = initialize_pruning_state(4, device="cpu")
        flat = torch.randn(12, 8)
        observe_moe_batch(
            state, moe, adapter, flat,
            num_experts=4, top_k=2, backend="bmm",
            record_pruning_metrics_only=record_only,
            fused=True,
        )
        return state

    state_prune = _run(True)
    state_full = _run(False)
    for k in _WEIGHTED_KEYS + ["total_tokens", "expert_frequency", "ean_sum"]:
        a = state_prune[k]
        b = state_full[k]
        if isinstance(a, torch.Tensor):
            assert torch.allclose(
                a.to(torch.float32), b.to(torch.float32), atol=1e-5
            ), f"prune vs full mismatch for {k}: {a} vs {b}"
        elif hasattr(a, "mean"):
            assert torch.allclose(
                a.mean.to(torch.float32), b.mean.to(torch.float32), atol=1e-5
            ), f"prune vs full mismatch for {k}.mean"


def _observe_native_masked(backend: str):
    """Like ``_observe_native`` but with an alternating ``valid_token_mask``.

    12 token positions are supplied with a nontrivial alternating boolean
    mask so six tokens remain after masking. This exercises the native-router
    *loop* path that reaches ``update_pruning_state`` with full-token
    ``selected_router_weights`` alongside the routed (bmm) reference path.
    """
    from reap.kernels.observe import observe_moe_batch
    from reap.pruning_metrics import initialize_pruning_state

    torch.manual_seed(42)
    moe = _NativeFusedMoe(e=4, h=8, i=8, k=2, weight=0.9)
    adapter = _Lfm2FusedAdapter()
    state = initialize_pruning_state(4, device="cpu")
    flat = torch.randn(12, 8)
    # Alternating mask: tokens 0,2,4,6,8,10 survive -> 6 of 12 tokens.
    mask = torch.tensor([True, False] * 6, dtype=torch.bool)
    observe_moe_batch(
        state,
        moe,
        adapter,
        flat,
        num_experts=4,
        top_k=2,
        backend=backend,
        record_pruning_metrics_only=True,
        valid_token_mask=mask,
        fused=True,
    )
    return state


def test_native_router_loop_masked_vs_routed_parity():
    """Masked native-router loop must match routed (bmm) pruning state.

    Regression for the blocker where ``update_pruning_state`` filtered
    ``selected_experts`` via ``valid_token_mask`` but left the native
    ``selected_router_weights`` unfiltered, raising
    ``(12, 2) != (6, 2)``. After the fix the loop path must succeed and agree
    with the routed reference on the masked tokens.
    """
    # Fixture sanity: 12 pre-mask tokens, 6 post-mask tokens.
    mask = torch.tensor([True, False] * 6, dtype=torch.bool)
    assert mask.numel() == 12
    assert int(mask.sum().item()) == 6
    assert not bool(mask.all()) and not bool((~mask).all())

    state_loop = _observe_native_masked("loop")
    state_bmm = _observe_native_masked("bmm")

    # total_tokens reflects the post-mask token count for both paths.
    assert int(state_loop["total_tokens"].item()) == 6
    assert int(state_bmm["total_tokens"].item()) == 6

    for k in _WEIGHTED_KEYS + ["total_tokens", "expert_frequency", "ean_sum"]:
        a = state_loop[k]
        b = state_bmm[k]
        if isinstance(a, torch.Tensor):
            assert torch.allclose(
                a.to(torch.float32).cpu(),
                b.to(torch.float32).cpu(),
                atol=1e-5,
                rtol=1e-5,
            ), f"loop vs bmm mismatch for {k}: {a} vs {b}"
        elif hasattr(a, "mean"):
            assert torch.allclose(
                a.mean.to(torch.float32).cpu(),
                b.mean.to(torch.float32).cpu(),
                atol=1e-5,
                rtol=1e-5,
            ), f"loop vs bmm mismatch for {k}.mean: {a.mean} vs {b.mean}"
            assert torch.equal(a.count, b.count), (
                f"count mismatch for {k}: {a.count} vs {b.count}"
            )

    # The native fixed weight is 0.9 per selected pair; with 6 surviving
    # tokens * 2 selections = 12 pairs, total weighted frequency is 10.8.
    total_wf = state_loop["weighted_expert_frequency_sum"].sum().item()
    assert abs(total_wf - 6 * 2 * 0.9) < 1e-4, (
        f"Expected total weighted frequency 10.8 (6 tokens * 2 * 0.9), "
        f"got {total_wf}"
    )
