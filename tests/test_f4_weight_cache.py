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


def test_f4_cache_dtype_none_resolves_to_source_native():
    """An omitted dtype after a converted entry must rebuild to source-native.

    Replaces the prior wildcard-cache regression: a ``dtype=torch.float16``
    call followed by ``dtype=None`` must return the source-weight dtype
    (fp32 from ``nn.Parameter``) rather than reusing the cached fp16 stack.
    """
    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()

    # First request: explicit float16 — builds and caches the converted stack.
    s_fp16 = get_stacked_expert_weights(moe, adapter, dtype=torch.float16)
    assert s_fp16["W_gate"].dtype == torch.float16
    assert s_fp16["_resolved_dtype"] == torch.float16
    assert s_fp16["W_gate"].device == moe.experts.gate_up_proj.device

    # Second request: dtype=None resolves to the source-weight dtype (fp32).
    # It must NOT reuse the cached fp16 representation.
    s_native = get_stacked_expert_weights(moe, adapter)
    assert s_native["W_gate"].dtype == torch.float32
    assert s_native["_resolved_dtype"] == torch.float32
    assert s_native["W_gate"].device == moe.experts.gate_up_proj.device
    # The native stack is not the converted representation.
    assert s_native is not s_fp16
    assert not torch.equal(
        s_native["W_gate"], s_fp16["W_gate"].to(torch.float32)
    ) or s_native["W_gate"].dtype != s_fp16["W_gate"].dtype

    free_cache()


def test_f4_cache_native_converted_native_sequence():
    """Native -> converted -> native must produce fp32, fp16, fp32 in order."""
    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()

    s1 = get_stacked_expert_weights(moe, adapter)
    assert s1["W_gate"].dtype == torch.float32

    s2 = get_stacked_expert_weights(moe, adapter, dtype=torch.float16)
    assert s2["W_gate"].dtype == torch.float16

    s3 = get_stacked_expert_weights(moe, adapter)
    assert s3["W_gate"].dtype == torch.float32
    # The final native stack is not the converted representation object.
    assert s3 is not s2

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


# ---------------------------------------------------------------------------
# Fix 2: .detach() — stacked weights must not carry requires_grad
# ---------------------------------------------------------------------------


def test_f4_stacked_weights_are_detached():
    """Stacked weights must be detached from autograd for memory safety."""
    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()
    stacked = get_stacked_expert_weights(moe, adapter)
    assert not stacked["W_gate"].requires_grad
    assert not stacked["W_up"].requires_grad
    assert not stacked["W_down"].requires_grad
    free_cache()


# ---------------------------------------------------------------------------
# Fix 3: free_cache with specific moe argument
# ---------------------------------------------------------------------------


def test_f4_free_cache_specific_moe():
    """free_cache(moe) must evict only the matching entry."""
    free_cache()
    moe_a = _QwenMoe(e=4, h=8, i=4)
    moe_b = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()

    stacked_a = get_stacked_expert_weights(moe_a, adapter)
    assert stacked_a is not None
    free_cache(moe_a)
    # Fresh request on moe_b must build from scratch (no stale hit from moe_a).
    from reap.kernels.weight_cache import cache_size
    assert cache_size() == 0
    stacked_b = get_stacked_expert_weights(moe_b, adapter)
    assert stacked_b is not None
    assert not torch.equal(stacked_a["W_gate"], stacked_b["W_gate"])
    free_cache()


def test_f4_free_cache_none_clears_all():
    """free_cache() without argument must clear the single-entry cache."""
    free_cache()
    moe = _QwenMoe(e=4, h=8, i=4)
    adapter = Qwen3MoeModelAdapter()
    get_stacked_expert_weights(moe, adapter)
    from reap.kernels.weight_cache import cache_size
    assert cache_size() == 1
    free_cache()
    assert cache_size() == 0
