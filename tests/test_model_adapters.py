"""Lightweight CPU-only tests for the reap model-adapter layer.

These build mock ``nn.Module`` trees shaped like Qwen3-MoE (with a DENSE layer 0
to exercise the first-MoE-layer lookup), Mixtral, and Llama4 — no weights are
downloaded and everything runs on CPU. They guard the regressions fixed in
``aa59833``:

* ``MoETransformerObserverConfig`` import + kwarg acceptance,
* first-MoE-layer (not ``layers[0]``) config reads,
* Mixtral ``num_local_experts`` config-key dispatch,
* guarded ``vllm`` import (module importability).
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn

from reap.model_adapters import (
    Qwen3MoeModelAdapter,
    Llama4MoeModelAdapter,
    MixtralMoeModelAdapter,
    infer_model_adapter,
)
from reap.main import _setup_observer

H, E, K = 16, 8, 2


class MockConfig:
    """Supports both ``getattr`` and ``.get()``/``[]`` like HF PretrainedConfig."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def __contains__(self, key):
        return hasattr(self, key)


# --- Mock MoE blocks (class names MUST match adapter.hook_regex()) ---


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, num_experts=E, hidden=H):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden * 4) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):
        return x, torch.zeros(self.gate.out_features, x.shape[0])


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, num_experts=E, hidden=H):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden * 4) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):
        return x, torch.zeros(self.gate.out_features, x.shape[0])


class Llama4TextMoe(nn.Module):
    def __init__(self, num_experts=E, hidden=H):
        super().__init__()
        self.num_experts = num_experts
        self.experts = types.SimpleNamespace(
            gate_up_proj=nn.Parameter(torch.zeros(num_experts, hidden * 2)),
            down_proj=nn.Parameter(torch.zeros(num_experts, hidden)),
        )
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.gate = self.router  # alias for is_moe_layer check

    def forward(self, x):
        return x, torch.zeros(self.num_experts, x.shape[0])


class DenseMLP(nn.Module):
    def __init__(self, hidden=H):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden * 4)

    def forward(self, x):
        return self.fc(x)


class Qwen3Layer(nn.Module):
    def __init__(self, moe=None):
        super().__init__()
        self.mlp = moe if moe is not None else DenseMLP()


class MixtralLayer(nn.Module):
    def __init__(self, moe=None):
        super().__init__()
        self.block_sparse_moe = moe  # may be None (dense layer)


class Llama4Layer(nn.Module):
    def __init__(self, moe=None):
        super().__init__()
        if moe is not None:
            self.feed_forward = moe


class Inner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class MockCausalLM(nn.Module):
    def __init__(self, layers, config):
        super().__init__()
        self.model = Inner(layers)
        self.config = config


# --- builders ---


def build_qwen3():
    # layer 0 DENSE, layer 1 MoE -> exercises first-MoE-layer fix
    cfg = MockConfig(
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=E,
        num_experts_per_tok=K,
        norm_topk_prob=True,
    )
    layers = [Qwen3Layer(moe=None), Qwen3Layer(moe=Qwen3MoeSparseMoeBlock())]
    return MockCausalLM(layers, cfg)


def build_mixtral():
    cfg = MockConfig(
        model_type="mixtral",
        architectures=["MixtralForCausalLM"],
        num_local_experts=E,  # Mixtral uses num_local_experts, NOT num_experts
        num_experts_per_tok=K,
        norm_topk_prob=False,
    )
    return MockCausalLM([MixtralLayer(moe=MixtralSparseMoeBlock())], cfg)


def build_llama4():
    cfg = MockConfig(
        model_type="llama4",
        architectures=["Llama4ForCausalLM"],
        num_local_experts=E,
        num_experts_per_tok=K,
    )
    return MockCausalLM([Llama4Layer(moe=Llama4TextMoe())], cfg)


def _obs_args():
    return types.SimpleNamespace(
        renormalize_router_weights=True,
        record_pruning_metrics_only=False,
    )


# --- tests ---


def test_infer_adapter_qwen3_layout():
    q = build_qwen3()
    adapter = infer_model_adapter(q, q.config)
    assert isinstance(adapter, Qwen3MoeModelAdapter)
    assert adapter.identify_moe_layers(q) == [1]  # layer 0 is dense


def test_qwen3_get_layer_config():
    q = build_qwen3()
    adapter = infer_model_adapter(q, q.config)
    lc = adapter.get_layer_config(adapter.layers(q)[1], q.config)
    assert lc.num_experts == E
    assert lc.top_k == K
    assert lc.norm_topk_prob is True
    assert lc.fused_experts is False


def test_qwen3_slice_experts_and_update_config():
    q = build_qwen3()
    adapter = infer_model_adapter(q, q.config)
    moe = adapter.get_moe(adapter.layers(q)[1])
    keep = [0, 2, 4, 6]
    adapter.slice_experts(moe, keep)
    assert len(moe.experts) == 4
    assert moe.gate.out_features == 4
    assert moe.gate.weight.shape[0] == 4
    adapter.update_config(q.config, 4, K)
    assert q.config.num_experts == 4
    assert q.config.num_experts_per_tok == 2


def test_qwen3_setup_observer_with_dense_layer_zero():
    q = build_qwen3()
    obs = _setup_observer(q, _obs_args())
    assert obs is not None
    assert len(obs.hooks) > 0  # dense layer 0 skipped, MoE layer 1 hooked


def test_mixtral_get_layer_config_uses_num_local_experts():
    m = build_mixtral()
    adapter = infer_model_adapter(m, m.config)
    assert isinstance(adapter, MixtralMoeModelAdapter)
    lc = adapter.get_layer_config(adapter.layers(m)[0], m.config)
    assert lc.num_experts == E  # reads num_local_experts, not num_experts
    assert lc.top_k == K


def test_mixtral_slice_and_update_config():
    m = build_mixtral()
    adapter = infer_model_adapter(m, m.config)
    moe = adapter.get_moe(adapter.layers(m)[0])
    adapter.slice_experts(moe, [1, 3, 5, 7])
    assert len(moe.experts) == 4
    adapter.update_config(m.config, 4, K)
    assert m.config.num_local_experts == 4


def test_mixtral_setup_observer():
    m = build_mixtral()
    obs = _setup_observer(m, _obs_args())
    assert obs is not None
    assert len(obs.hooks) > 0


def test_llama4_fused_layer_config():
    l4 = build_llama4()
    adapter = infer_model_adapter(l4, l4.config)
    assert isinstance(adapter, Llama4MoeModelAdapter)
    lc = adapter.get_layer_config(adapter.layers(l4)[0], l4.config)
    assert lc.fused_experts is True
    assert lc.num_experts == E


def test_config_only_inference():
    assert isinstance(
        infer_model_adapter(None, MockConfig(model_type="qwen3_moe")),
        Qwen3MoeModelAdapter,
    )
    assert isinstance(
        infer_model_adapter(None, MockConfig(model_type="mixtral")),
        MixtralMoeModelAdapter,
    )
    assert isinstance(
        infer_model_adapter(None, MockConfig(model_type="llama4")),
        Llama4MoeModelAdapter,
    )
    assert infer_model_adapter(None, MockConfig(model_type="gpt2")) is None


def test_dense_model_returns_none_adapter():
    dense = MockCausalLM(
        [Qwen3Layer(moe=None), Qwen3Layer(moe=None)],
        MockConfig(model_type="qwen3_moe", num_experts=E, num_experts_per_tok=K),
    )
    assert infer_model_adapter(dense, dense.config) is None