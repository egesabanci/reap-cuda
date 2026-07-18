"""F4 stacked weights: shapes + Linear-convention normalization."""
from __future__ import annotations

import torch
import torch.nn as nn

from reap.kernels.weight_cache import free_cache, get_stacked_expert_weights
from reap.model_adapters import Llama4MoeModelAdapter, Qwen3MoeModelAdapter


class _FusedExperts(nn.Module):
    def __init__(self, e, h, i, convention="linear"):
        super().__init__()
        self.num_experts = e
        if convention == "linear":
            self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * i, h))
            self.down_proj = nn.Parameter(torch.randn(e, h, i))
        else:
            self.gate_up_proj = nn.Parameter(torch.randn(e, h, 2 * i))
            self.down_proj = nn.Parameter(torch.randn(e, i, h))


class _QwenMoe(nn.Module):
    def __init__(self, e=4, h=8, i=4):
        super().__init__()
        self.experts = _FusedExperts(e, h, i, "linear")
        self.gate = nn.Linear(h, e, bias=False)
        self.num_experts = e


class _LlamaMoe(nn.Module):
    def __init__(self, e=4, h=8, i=4):
        super().__init__()
        self.experts = _FusedExperts(e, h, i, "bmm")
        self.router = nn.Linear(h, e, bias=False)
        self.num_experts = e
        self.top_k = 2


def test_f4_qwen_linear_shapes():
    free_cache()
    moe = _QwenMoe()
    adapter = Qwen3MoeModelAdapter()
    stacked = get_stacked_expert_weights(moe, adapter)
    e, two_i, h = moe.experts.gate_up_proj.shape
    i = two_i // 2
    assert stacked["W_gate"].shape == (e, i, h)
    assert stacked["W_up"].shape == (e, i, h)
    assert stacked["W_down"].shape == (e, h, i)
    free_cache(moe)


def test_f4_llama_bmm_normalized():
    free_cache()
    moe = _LlamaMoe()
    adapter = Llama4MoeModelAdapter()
    stacked = get_stacked_expert_weights(moe, adapter)
    e, h, two_i = moe.experts.gate_up_proj.shape
    i = two_i // 2
    assert stacked["W_gate"].shape == (e, i, h)
    assert stacked["W_up"].shape == (e, i, h)
    assert stacked["W_down"].shape == (e, h, i)
    # Values match transpose of native bmm layout.
    assert torch.allclose(
        stacked["W_gate"], moe.experts.gate_up_proj[..., :i].transpose(-1, -2)
    )
    free_cache()


# ---------------------------------------------------------------------------
# Cache representation safety (device + dtype)
# ---------------------------------------------------------------------------


def test_f4_cache_dtype_mismatch_rebuilds():
    """A dtype change must not return a stale cached representation."""
    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()

    # First request: native dtype (float32 from nn.Parameter).
    s1 = get_stacked_expert_weights(moe, adapter)
    assert s1["W_gate"].dtype == torch.float32

    # Second request: explicit float16 — must rebuild, not return fp32.
    s2 = get_stacked_expert_weights(moe, adapter, dtype=torch.float16)
    assert s2["W_gate"].dtype == torch.float16
    assert s2["_resolved_dtype"] == torch.float16

    # Third request: back to float32 — must rebuild again.
    s3 = get_stacked_expert_weights(moe, adapter, dtype=torch.float32)
    assert s3["W_gate"].dtype == torch.float32

    free_cache()


def test_f4_cache_dtype_none_accepts_cached():
    """When dtype=None, accept whatever is cached (caller does not care)."""
    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()

    s1 = get_stacked_expert_weights(moe, adapter, dtype=torch.float16)
    assert s1["W_gate"].dtype == torch.float16

    # dtype=None should accept the cached fp16 representation.
    s2 = get_stacked_expert_weights(moe, adapter)
    assert s2["W_gate"].dtype == torch.float16

    free_cache()


def test_f4_cache_stays_bounded_after_dtype_changes():
    """Cache size remains <= 1 even after multiple dtype-triggered rebuilds."""
    from reap.kernels.weight_cache import cache_size

    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()

    for dt in (torch.float16, torch.float32, torch.float16, torch.bfloat16):
        get_stacked_expert_weights(moe, adapter, dtype=dt)
        assert cache_size() <= 1

    free_cache()


def test_f4_cache_resolved_metadata_present():
    """Cache entries carry _resolved_device and _resolved_dtype metadata."""
    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()
    stacked = get_stacked_expert_weights(moe, adapter, dtype=torch.float16)
    assert "_resolved_device" in stacked
    assert "_resolved_dtype" in stacked
    assert stacked["_resolved_dtype"] == torch.float16
    free_cache()
