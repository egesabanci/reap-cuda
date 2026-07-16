"""Regression tests for skip_first / skip_last budget calculation in
``merge_pipeline.run_merge``.

Background: an earlier implementation redistributed a global cluster budget
across only the merged layers:

    num_clusters = int(total_experts * (1 - ratio)) / merged_layers

That overflows ``num_clusters`` past ``experts_per_layer`` when skipping is
aggressive (e.g. 3 layers, skip both ends -> 1 merged layer), which then
crashes ``linkage_to_labels`` with ``ValueError: num_clusters > n_samples``.

The fix uses the per-layer formula unchanged regardless of skip; skipped
layers just receive identity cluster labels. These tests pin that behavior.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.args import (
    ClusterArgs,
    DatasetArgs,
    EvalArgs,
    MergeArgs,
    ModelArgs,
    ObserverArgs,
    ReapArgs,
)
from reap.merge_pipeline import run_merge
from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig


def _make_model(num_experts: int = 4, num_hidden_layers: int = 3):
    cfg = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=num_experts,
        num_experts_per_tok=1,
        norm_topk_prob=False,
    )
    return Qwen3MoeForCausalLM(cfg).eval()


def _expert_weights(moe, i, adapter):
    """Snapshot expert i's weights, fused-aware.

    Returns dict with ``gate_up`` and ``down`` tensors so the identity-cluster
    preservation check works for both the non-fused ModuleList layout
    (transformers 4.55) and the fused stacked-param layout (transformers >=5.x,
    where ``moe.experts[i]`` is not subscriptable).
    """
    if adapter._is_fused_experts(moe.experts):
        return {
            "gate_up": moe.experts.gate_up_proj[i].detach().clone(),
            "down": moe.experts.down_proj[i].detach().clone(),
        }
    return {
        "gate_up": moe.experts[i].gate_proj.weight.detach().clone(),
        "down": moe.experts[i].down_proj.weight.detach().clone(),
    }


def _collect_observer_data(model, num_experts: int, hidden_dim: int):
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9], [1, 2, 3, 4]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.long,
        ),
    }
    adapter = infer_model_adapter(model, model.config)
    first_layer = adapter.layers(model)[0]
    fused = adapter.get_layer_config(first_layer, model.config).fused_experts
    hook_config = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=fused,
        record_pruning_metrics_only=False,  # merge needs merging-criteria metrics
    )
    observer = MoETransformerObserver(model, hook_config=hook_config, adapter=adapter)
    with observer.set_attention_mask(batch["attention_mask"]):
        _ = model(**batch)
    data = observer.report_state()
    observer.close_hooks()
    # sanity: merging-criteria metrics present
    assert data[0]["characteristic_activation"].shape == (num_experts, hidden_dim)
    assert data[0]["router_logit_similiarity"].shape == (num_experts, num_experts)
    return data


def _run_merge(model, observer_data, merge_args: MergeArgs, results_dir):
    return run_merge(
        model,
        None,
        observer_data,
        ReapArgs(),
        ModelArgs(),
        DatasetArgs(dataset_name="test"),
        ObserverArgs(distance_measure="angular", record_pruning_metrics_only=False),
        ClusterArgs(
            expert_sim="characteristic_activation",
            cluster_method="agglomerative",
            compression_ratio=0.5,
            linkage_method="average",
        ),
        merge_args,
        EvalArgs(),
        results_dir,
    )


@pytest.mark.parametrize("skip_first,skip_last", [(True, False), (False, True), (True, True)])
def test_skip_first_last_does_not_crash(skip_first, skip_last):
    """Aggressive skipping must not overflow num_clusters past experts_per_layer.

    3 layers, 4 experts, compression 0.5 -> num_clusters=2 per merged layer.
    With skip_first+skip_last that left 1 merged layer; the old global-budget
    formula computed num_clusters=6 (>4 experts) and crashed. This test guards
    against regression by running the full merge for each skip combination.
    """
    torch.manual_seed(0)
    model = _make_model(num_experts=4, num_hidden_layers=3)
    observer_data = _collect_observer_data(model, num_experts=4, hidden_dim=8)

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = pathlib.Path(tmp)
        out = _run_merge(
            model,
            observer_data,
            MergeArgs(
                merge_method="frequency_weighted_average",
                skip_first=skip_first,
                skip_last=skip_last,
            ),
            results_dir,
        )
        saved = list(out.glob("*.safetensors"))
        assert saved, "merged model was not saved"


def test_skip_all_layers_raises():
    """skip_first + skip_last on a 2-layer model excludes everything -> ValueError."""
    torch.manual_seed(0)
    model = _make_model(num_experts=4, num_hidden_layers=2)
    observer_data = _collect_observer_data(model, num_experts=4, hidden_dim=8)

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = pathlib.Path(tmp)
        with pytest.raises(ValueError, match="nothing to merge"):
            _run_merge(
                model,
                observer_data,
                MergeArgs(
                    merge_method="frequency_weighted_average",
                    skip_first=True,
                    skip_last=True,
                ),
                results_dir,
            )


def test_skip_identity_clusters_preserve_experts():
    """Skipped layers must keep all experts as singletons (identity labels).

    After a merge with skip_first on a 3-layer model, the first layer's experts
    should be untouched relative to the pre-merge model (identity clustering),
    while the merged layers collapse their cluster members to equal weights.
    """
    torch.manual_seed(1)
    num_experts, num_layers = 4, 3
    model = _make_model(num_experts=num_experts, num_hidden_layers=num_layers)
    observer_data = _collect_observer_data(model, num_experts, hidden_dim=8)

    # Snapshot first-layer expert weights before merge.
    first_layer = model.model.layers[0]
    moe = first_layer.mlp
    adapter = infer_model_adapter(model, model.config)
    pre_weights = [_expert_weights(moe, i, adapter) for i in range(num_experts)]

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = pathlib.Path(tmp)
        _run_merge(
            model,
            observer_data,
            MergeArgs(
                merge_method="frequency_weighted_average",
                skip_first=True,
                skip_last=False,
            ),
            results_dir,
        )

    # Skipped first layer: experts unchanged (identity clusters -> singletons).
    for i in range(num_experts):
        post = _expert_weights(moe, i, adapter)
        assert torch.allclose(
            pre_weights[i]["gate_up"], post["gate_up"]
        ), f"skipped layer expert {i} gate_up was modified by merge"
        assert torch.allclose(
            pre_weights[i]["down"], post["down"]
        ), f"skipped layer expert {i} down was modified by merge"