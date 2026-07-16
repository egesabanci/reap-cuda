"""bmm / frea backends must match the loop observer on consumed prune metrics."""
from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig

CONSUMED = [
    "total_tokens",
    "expert_frequency",
    "ean_sum",
    "ean_mean",
    "reap",
    "weighted_ean_sum",
    "weighted_expert_frequency_sum",
    "max_activations",
    "pairwise_expert_frequency",
]


def _run(backend: str):
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).eval()
    batch = {
        "input_ids": torch.randint(0, 64, (4, 16)),
        "attention_mask": torch.ones(4, 16, dtype=torch.long),
    }
    adapter = infer_model_adapter(model, model.config)
    fused = adapter.get_layer_config(adapter.layers(model)[0], model.config).fused_experts
    hc = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=fused,
        record_pruning_metrics_only=True,
        observe_backend=backend,
    )
    obs = MoETransformerObserver(model, hook_config=hc, adapter=adapter)
    with obs.set_attention_mask(batch["attention_mask"]):
        _ = model(**batch)
    s = obs.report_state()
    obs.close_hooks()
    return s


def test_bmm_matches_loop():
    loop = _run("loop")
    bmm = _run("bmm")
    for layer in loop:
        for k in CONSUMED:
            a, b = loop[layer][k], bmm[layer][k]
            if isinstance(a, torch.Tensor):
                assert torch.allclose(
                    a.to(torch.float32).cpu(),
                    b.to(torch.float32).cpu(),
                    atol=1e-4,
                    rtol=1e-4,
                ), f"layer {layer} key {k}:\n{a}\nvs\n{b}"


def test_frea_matches_loop():
    loop = _run("loop")
    frea = _run("frea")
    for layer in loop:
        for k in CONSUMED:
            a, b = loop[layer][k], frea[layer][k]
            if isinstance(a, torch.Tensor):
                assert torch.allclose(
                    a.to(torch.float32).cpu(),
                    b.to(torch.float32).cpu(),
                    atol=1e-4,
                    rtol=1e-4,
                ), f"layer {layer} key {k}:\n{a}\nvs\n{b}"
