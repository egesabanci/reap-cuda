"""End-to-end merge pipeline test: observer (with merging-criteria metrics)
-> cluster -> merge -> save.

Merging in REAP does not shrink the expert ModuleList; it overwrites/ties
expert weights within each cluster so cluster members become equal. The
``assert_merge`` check inside ``run_merge`` verifies that equality. This test
confirms the full merge wiring (ported off MODEL_ATTRS onto adapters) runs and
saves a merged model on a tiny real ``Qwen3MoeForCausalLM``.
"""
from __future__ import annotations

import pathlib
import tempfile

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.args import (
    DatasetArgs,
    ObserverArgs,
    ClusterArgs,
    MergeArgs,
    ModelArgs,
    EvalArgs,
    ReapArgs,
)
from reap.merge_pipeline import run_merge
from reap.model_adapters import infer_model_adapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig


def _make_model(num_experts: int = 4, num_hidden_layers: int = 2):
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


def test_merge_pipeline_end_to_end():
    torch.manual_seed(0)
    model = _make_model(num_experts=4, num_hidden_layers=2)

    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9], [1, 2, 3, 4]], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long
        ),
    }

    # Collect observer state WITH merging-criteria metrics (record_pruning_metrics_only=False)
    # so clustering has characteristic_activation / router_logit_similarity to work with.
    adapter = infer_model_adapter(model, model.config)
    first_layer = adapter.layers(model)[0]
    fused = adapter.get_layer_config(first_layer, model.config).fused_experts
    hook_config = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=fused,
        record_pruning_metrics_only=False,
    )
    observer = MoETransformerObserver(model, hook_config=hook_config, adapter=adapter)
    with observer.set_attention_mask(batch["attention_mask"]):
        _ = model(**batch)
    observer_data = observer.report_state()
    observer.close_hooks()

    # sanity: merging-criteria metrics present and tensor-shaped (report_state
    # unwraps OnlineStatsTracker -> .mean tensors)
    assert observer_data[0]["characteristic_activation"].shape == (4, 8)
    assert observer_data[0]["router_logit_similiarity"].shape == (4, 4)

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = pathlib.Path(tmp)
        out = run_merge(
            model,
            None,  # tokenizer optional in save_merged_model
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
            MergeArgs(merge_method="frequency_weighted_average"),
            EvalArgs(),
            results_dir,
        )
        # run_merge calls assert_merge internally; reaching here means cluster
        # experts were merged (weights equal within each cluster) without error.
        saved = list(out.glob("*.safetensors"))
        assert saved, "merged model was not saved"
        assert (out / "reap_args.yaml").exists(), "merge args were not dumped"
        assert (out / "clusters" / "clusters.pkl").exists(), "cluster labels not saved"