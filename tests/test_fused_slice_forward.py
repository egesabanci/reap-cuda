"""Fused Qwen3 slice_experts must leave a runnable live module."""
from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.model_adapters import Qwen3MoeModelAdapter


def test_fused_slice_updates_counts_and_forward():
    cfg = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
    )
    model = Qwen3MoeForCausalLM(cfg).eval()
    adapter = Qwen3MoeModelAdapter()
    moe = adapter.get_moe(adapter.layers(model)[0])
    assert adapter._is_fused_experts(moe.experts)
    adapter.slice_experts(moe, [0, 2])
    assert moe.experts.gate_up_proj.shape[0] == 2
    assert moe.experts.num_experts == 2
    assert moe.gate.weight.shape[0] == 2
    assert moe.gate.top_k == 2  # still 2 <= retained
    adapter.slice_experts(moe, [0])  # keep 1
    assert moe.experts.num_experts == 1
    assert moe.gate.top_k == 1
    adapter.update_config(model.config, 1, 2)
    assert model.config.num_experts == 1
    assert model.config.num_experts_per_tok == 1
    x = torch.randn(1, 3, 8)
    out = moe(x)
    assert out.shape == (1, 3, 8)
