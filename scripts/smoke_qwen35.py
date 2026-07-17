#!/usr/bin/env python3
"""Targeted 3-layer calibration smoke test for the fused Qwen3.6-35B-A3B MoE.

The full 35B model (67 GB bf16) cannot fit on this box (61 GB RAM, 46 GB VRAM),
and REAP's `qwen3_5_moe` support is new.  This smoke verifies the *wiring* of
the real fused-expert pipeline on a tiny, runnable slice of the real model:

  1. Build a 50-example calibration subset from the downloaded
     ``theblackcat102/evol-codealpaca-v1`` (saved to ``data/smoke_50.jsonl``).
  2. Construct a 3-decoder-layer ``Qwen3_5MoeForCausalLM`` (text tower only)
     and materialise layers 0-2 + embeddings + final norm + lm_head from the
     REAL 35B safetensors (fused ``gate_up_proj``/``down_proj`` + shared expert).
  3. Run the REAP pipeline on GPU: adapter detection -> observer (pruning
     metrics) -> prune -> fused expert slicing.
  4. Assert: 256 -> 56 routed experts, shared expert weights byte-identical,
     config patched, router sliced.

This is the bridge to the Triton kernel work (#13): the observer's fused
branch is exactly the "Phase-1 bmm grouped" pattern the FREA kernel replaces.

Usage:
  HF_HOME=/data/.hf-cache python3 scripts/smoke_qwen35.py
"""
from __future__ import annotations

import json
import os
import sys
import pathlib

# Use the pre-populated HF cache on /data *before* importing datasets/torch.
os.environ.setdefault("HF_HOME", "/data/.hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "/data/.hf-cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/.hf-cache/datasets")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

# Make `reap` importable from the src-layout without an editable install.
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import torch
from transformers import AutoTokenizer, AutoConfig, Qwen3_5MoeForCausalLM

from reap.model_adapters import infer_model_adapter, Qwen3_5MoeModelAdapter
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig
from reap.args import PruneArgs
from reap.prune import apply_pruning, publish_pruned_model

MODEL_PATH = "/data/models/unsloth/Qwen3.6-35B-A3B"
DATASET_JSONL = (
    "/data/.hf-cache/hub/datasets--theblackcat102--evol-codealpaca-v1/"
    "snapshots/c75242318519b5470635e84064baed7c78594020/train.jsonl"
)
OUT_DIR = _REPO / "artifacts" / "smoke_qwen35"
N_LAYERS = 3
N_EXAMPLES = 50
MAX_LEN = 128
N_EXPERTS_TO_PRUNE = 200  # 256 -> 56 kept


def build_calibration_subset() -> list[dict]:
    """Read 50 records from evol-codealpaca-v1 and dump a small jsonl."""
    records = []
    with open(DATASET_JSONL, "r") as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
            if len(records) >= N_EXAMPLES:
                break
    out = _REPO / "data" / "smoke_50.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[data] wrote {len(records)} calibration examples -> {out}")
    return records


def build_real_fused_model():
    """Construct a 3-layer text tower with REAL 35B fused-expert weights."""
    print(f"[model] loading config from {MODEL_PATH}")
    full_cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tcfg = full_cfg.text_config
    orig_ntypes = len(tcfg.layer_types)
    tcfg.num_hidden_layers = N_LAYERS
    tcfg.layer_types = list(tcfg.layer_types[:N_LAYERS])
    print(
        f"[model] text_config: num_experts={tcfg.num_experts} "
        f"top_k={tcfg.num_experts_per_tok} layers={N_LAYERS} (of {orig_ntypes}) "
        f"layer_types={tcfg.layer_types} hidden={tcfg.hidden_size} "
        f"moe_inter={tcfg.moe_intermediate_size} shared_inter="
        f"{getattr(tcfg, 'shared_expert_intermediate_size', None)}"
    )

    model = Qwen3_5MoeForCausalLM(tcfg).to(torch.bfloat16)
    model.eval()
    print(f"[model] instantiated {model.__class__.__name__} on CPU "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # Load real weights for layers 0-2 + embed + norm + lm_head from the
    # safetensors index. Source keys live under ``model.language_model.*``;
    # the text-only CausalLM expects ``model.*`` (strip ``language_model.``).
    idx_path = pathlib.Path(MODEL_PATH) / "model.safetensors.index.json"
    with idx_path.open() as handle:
        idx = json.load(handle)
    wmap = idx["weight_map"]

    want_prefix = (
        tuple(f"model.language_model.layers.{i}." for i in range(N_LAYERS))
        + ("model.language_model.embed_tokens",)
        + ("model.language_model.norm",)
        + ("lm_head",)
    )

    def wanted(src_key: str) -> bool:
        return any(src_key == p or src_key.startswith(p + ".") or src_key.startswith(p)
                   for p in want_prefix)

    def remap(src_key: str) -> str:
        if src_key.startswith("model.language_model."):
            return "model." + src_key[len("model.language_model."):]
        return src_key  # lm_head

    needed_shards = sorted({wmap[k] for k in wmap if wanted(k)})
    print(f"[model] loading {len(needed_shards)} shard(s) for layers 0-{N_LAYERS-1}")

    from safetensors.torch import load_file
    remapped = {}
    for shard in needed_shards:
        spath = str(pathlib.Path(MODEL_PATH) / shard)
        blk = load_file(spath)
        for k, v in blk.items():
            if wanted(k):
                remapped[remap(k)] = v.to(torch.bfloat16)
        del blk

    result = model.load_state_dict(remapped, strict=False)
    n_loaded = len(remapped)
    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)
    print(f"[model] load_state_dict: loaded={n_loaded} missing={n_missing} "
          f"unexpected={n_unexpected}")
    # The only acceptable "missing" keys are non-persistent buffers (rotary,
    # conv caches) that live on the module but not in the checkpoint.
    if n_unexpected:
        print("[model] WARNING unexpected keys:", result.unexpected_keys[:5])

    return model


def tokenize_subset(tokenizer, records):
    """Apply chat template to 50 records -> one padded batch (input_ids, mask)."""
    seqs = []
    for r in records:
        messages = [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer(text, truncation=True, max_length=MAX_LEN)["input_ids"]
        seqs.append(ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    print(f"[data] tokenised {len(seqs)} examples, padded to seq_len={maxlen} "
          f"(total real tokens={int(attn.sum())})")
    return input_ids, attn


def run_smoke():
    torch.manual_seed(0)
    records = build_calibration_subset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_real_fused_model()
    device = "cuda"
    model = model.to(device)

    # ---- 1. adapter detection ------------------------------------------------
    adapter = infer_model_adapter(model, model.config)
    assert adapter is not None, "infer_model_adapter returned None"
    assert isinstance(adapter, Qwen3_5MoeModelAdapter), (
        f"expected Qwen3_5MoeModelAdapter, got {type(adapter).__name__}"
    )
    moe_idx = adapter.identify_moe_layers(model)
    print(f"[adapter] {type(adapter).__name__} hook_regex={adapter.hook_regex()!r} "
          f"moe_layers={moe_idx}")
    assert moe_idx == list(range(N_LAYERS)), moe_idx

    first_moe_layer = adapter.layers(model)[moe_idx[0]]
    first_moe = adapter.get_moe(first_moe_layer)
    layer_cfg = adapter.get_layer_config(first_moe_layer, model.config)
    assert layer_cfg.fused_experts, "experts should be detected as FUSED"
    assert layer_cfg.num_experts == 256, layer_cfg.num_experts
    assert layer_cfg.top_k == 8, layer_cfg.top_k
    print(f"[adapter] layer_config: {layer_cfg}")

    # ---- 2. observer (pruning metrics only) ---------------------------------
    obs_cfg = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=adapter.hook_regex(),
        fused_experts=True,
        record_pruning_metrics_only=True,
    )
    observer = MoETransformerObserver(model=model, hook_config=obs_cfg, adapter=adapter)
    print(f"[observer] registered {len(observer.hooks)} hooks "
          f"(expect {N_LAYERS})")
    assert len(observer.hooks) == N_LAYERS, len(observer.hooks)

    input_ids, attn = tokenize_subset(tokenizer, records)
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    # snapshot shared expert BEFORE prune (must survive unchanged)
    se_before = {
        i: (adapter.layers(model)[i].mlp.shared_expert.down_proj.weight.clone(),
            adapter.layers(model)[i].mlp.shared_expert.gate_proj.weight.clone(),
            adapter.layers(model)[i].mlp.shared_expert_gate.weight.clone())
        for i in moe_idx
    }
    gup_shape_before = tuple(
        adapter.layers(model)[i].mlp.experts.gate_up_proj.shape for i in moe_idx
    )

    # ---- 3. calibration forward (observer hooks fire per MoE block) ----------
    with torch.no_grad(), observer.set_attention_mask(attn):
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    print("[observer] forward pass complete")

    observer_data = observer.report_state()
    print(f"[observer] state layers: {sorted(observer_data.keys())}")
    for li in moe_idx:
        st = observer_data[li]
        ef = st["expert_frequency"]
        print(f"[observer] layer {li}: expert_frequency shape={tuple(ef.shape)} "
              f"sum={float(ef.sum()):.1f} nonzero={int((ef > 0).sum())}")
        assert ef.shape[0] == 256, ef.shape
        assert float(ef.sum()) > 0, "no expert firings recorded"
    observer.close_hooks()

    # ---- 4. prune (fused slice + config patch) -------------------------------
    prune_args = PruneArgs(prune_method="frequency", overwrite_pruned_model=True)
    apply_pruning(
        observer_data=observer_data,
        model=model,
        prune_args=prune_args,
        n_experts_to_prune=N_EXPERTS_TO_PRUNE,
    )

    # ---- 5. assertions -------------------------------------------------------
    for i in moe_idx:
        moe = adapter.get_moe(adapter.layers(model)[i])
        assert moe.experts.gate_up_proj.shape[0] == 56, (
            f"layer {i} gate_up_proj dim0={moe.experts.gate_up_proj.shape[0]} != 56"
        )
        assert moe.experts.down_proj.shape[0] == 56, (
            f"layer {i} down_proj dim0={moe.experts.down_proj.shape[0]} != 56"
        )
        assert moe.gate.weight.shape[0] == 56, (
            f"layer {i} router gate dim0={moe.gate.weight.shape[0]} != 56"
        )
        # shared expert untouched
        se_after = (
            moe.shared_expert.down_proj.weight,
            moe.shared_expert.gate_proj.weight,
            moe.shared_expert_gate.weight,
        )
        for name, a, b in zip(
            ["down_proj", "gate_proj", "shared_expert_gate"],
            se_after, se_before[i],
        ):
            assert torch.equal(a, b), f"layer {i} shared expert {name} changed!"
    assert getattr(model.config, "num_experts", None) == 56, model.config.num_experts
    print(f"[prune] config.num_experts={model.config.num_experts} "
          f"(was 256) -- OK")
    print(f"[prune] gate_up_proj shapes before={gup_shape_before} "
          f"after={tuple(adapter.layers(model)[i].mlp.experts.gate_up_proj.shape for i in moe_idx)}")
    publish_pruned_model(
        model,
        tokenizer,
        OUT_DIR,
        smoke_test_fn=lambda: smoke_test(model, tokenizer),
    )

    print("\n=== SMOKE PASSED ===")
    print("Wiring verified on real fused Qwen3.6-35B-A3B experts:")
    print("  - Qwen3_5MoeModelAdapter detected Qwen3_5MoeSparseMoeBlock")
    print("  - observer fused branch captured pruning metrics (router tuple + "
          "single-tensor output handled)")
    print("  - prune sliced fused gate_up_proj/down_proj 256->56")
    print("  - shared expert + shared_expert_gate preserved byte-identical")
    print("  - config.num_experts patched to 56")
    print("  - pruned 3-layer model saved to", OUT_DIR)


if __name__ == "__main__":
    run_smoke()