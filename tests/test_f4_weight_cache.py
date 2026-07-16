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
