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
