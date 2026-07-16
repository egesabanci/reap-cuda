"""Triton kernel unit tests.

* Always run pure-PyTorch fallbacks (CPU CI).
* CUDA+Triton paths run only when ``triton_runtime_available()``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from reap.kernels.backend import select_observe_backend, triton_available, triton_status
from reap.kernels.bmm import routed_expert_activations_grouped
from reap.kernels.router import RouterPairOutputs, f5_router, f5_router_pytorch
from reap.kernels.triton_softmax import softmax_rows
from reap.kernels.triton_utils import triton_runtime_available

requires_triton = pytest.mark.skipif(
    not triton_runtime_available(),
    reason="CUDA + triton runtime required",
)


class TestBackendSelection:
    def test_status_dict_keys(self):
        s = triton_status()
        assert "package" in s and "runtime" in s

    def test_auto_backend_is_known(self):
        b = select_observe_backend("auto")
        assert b in ("bmm", "f2")

    def test_explicit_backends(self):
        for name in ("loop", "bmm", "frea", "f2"):
            assert select_observe_backend(name) == name

    def test_disable_env(self, monkeypatch):
        monkeypatch.setenv("REAP_DISABLE_TRITON", "1")
        # Clear cached runtime check
        from reap.kernels import triton_utils

        triton_utils.triton_runtime_available.cache_clear()
        assert triton_runtime_available() is False
        assert select_observe_backend("auto") == "bmm"
        monkeypatch.delenv("REAP_DISABLE_TRITON", raising=False)
        triton_utils.triton_runtime_available.cache_clear()


class TestSoftmaxParity:
    def test_softmax_rows_matches_torch_cpu(self):
        torch.manual_seed(0)
        x = torch.randn(32, 64)
        y = softmax_rows(x)
        ref = F.softmax(x, dim=-1, dtype=torch.float32)
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)

    @requires_triton
    def test_softmax_rows_matches_torch_cuda(self):
        torch.manual_seed(0)
        x = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        y = softmax_rows(x)
        ref = F.softmax(x, dim=-1, dtype=torch.float32)
        assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3)


class TestF5Router:
    def test_f5_pair_shapes_and_csr(self):
        torch.manual_seed(0)
        t, e, k = 16, 8, 2
        logits = torch.randn(t, e)
        out = f5_router(logits, k, norm_topk_prob=False)
        assert out.selected_experts.shape == (t, k)
        assert out.pair_token_idx.numel() == t * k
        assert out.expert_offsets.shape == (e + 1,)
        assert int(out.expert_offsets[-1]) == t * k
        # CSR is sorted by expert
        assert torch.all(out.pair_expert_idx[1:] >= out.pair_expert_idx[:-1])

    def test_f5_with_mask(self):
        logits = torch.randn(4, 6)
        mask = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
        out = f5_router(logits, 2, valid_token_mask=mask)
        assert out.selected_experts.shape[0] == 3
        assert out.pair_token_idx.numel() == 6


class TestFreaParity:
    def _make_pairs(self, t=8, e=4, k=2, h=32, i=32, device="cpu", dtype=torch.float32):
        torch.manual_seed(0)
        flat = torch.randn(t, h, device=device, dtype=dtype)
        logits = torch.randn(t, e, device=device, dtype=dtype)
        pairs = f5_router_pytorch(
            logits, k, use_triton_softmax=False
        )
        W_gate = torch.randn(e, i, h, device=device, dtype=dtype)
        W_up = torch.randn(e, i, h, device=device, dtype=dtype)
        W_down = torch.randn(e, h, i, device=device, dtype=dtype)
        return flat, pairs, W_gate, W_up, W_down

    def test_grouped_bmm_runs_cpu(self):
        flat, pairs, wg, wu, wd = self._make_pairs()
        out = routed_expert_activations_grouped(flat, pairs, wg, wu, wd)
        assert out.shape == (pairs.pair_token_idx.numel(), flat.shape[-1])

    @requires_triton
    def test_triton_frea_matches_bmm(self):
        from reap.kernels.triton_frea import frea_triton_activations

        flat, pairs, wg, wu, wd = self._make_pairs(
            t=32, e=8, k=2, h=64, i=64, device="cuda", dtype=torch.float16
        )
        ref = routed_expert_activations_grouped(flat, pairs, wg, wu, wd, act_fn=F.silu)
        got = frea_triton_activations(flat, pairs, wg, wu, wd, act_fn=F.silu)
        assert got.shape == ref.shape
        assert torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2)


class TestScatterReduce:
    def test_scatter_pytorch_path(self):
        from reap.kernels.triton_reduce import scatter_pair_stats

        torch.manual_seed(0)
        n, h, e = 20, 16, 5
        pair_out = torch.randn(n, h)
        idx = torch.randint(0, e, (n,))
        w = torch.rand(n)
        stats = scatter_pair_stats(pair_out, idx, w, e)
        assert stats["ean_sum"].shape == (e,)
        assert stats["batch_max"].shape == (e,)
        # Manual check one expert
        for ei in range(e):
            mask = idx == ei
            if not mask.any():
                assert stats["ean_sum"][ei] == 0
                continue
            norms = torch.linalg.norm(pair_out[mask].float(), dim=-1)
            assert torch.allclose(
                stats["ean_sum"][ei].float(), norms.sum().double().float(), atol=1e-4
            )

    @requires_triton
    def test_scatter_triton_matches_pytorch(self):
        from reap.kernels.triton_reduce import _scatter_pytorch, scatter_pair_stats

        torch.manual_seed(0)
        n, h, e = 64, 64, 8
        pair_out = torch.randn(n, h, device="cuda", dtype=torch.float16)
        idx = torch.randint(0, e, (n,), device="cuda")
        w = torch.rand(n, device="cuda")
        ref = _scatter_pytorch(pair_out, idx, w, e)
        got = scatter_pair_stats(pair_out, idx, w, e)
        assert torch.allclose(
            got["ean_sum"].float(), ref["ean_sum"].float(), atol=1e-2, rtol=1e-2
        )
        assert torch.allclose(
            got["weighted_ean_sum"].float(),
            ref["weighted_ean_sum"].float(),
            atol=1e-2,
            rtol=1e-2,
        )


class TestEndToEndObserveBackend:
    """Tiny observe_moe_batch smoke without Hub (CPU / optional CUDA)."""

    def test_observe_bmm_on_mock_fused(self):
        import torch.nn as nn

        from reap.kernels.observe import observe_moe_batch
        from reap.pruning_metrics import initialize_pruning_state

        class Exp(nn.Module):
            def __init__(self, e=4, h=16, i=16):
                super().__init__()
                self.num_experts = e
                self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * i, h))
                self.down_proj = nn.Parameter(torch.randn(e, h, i))

        class Moe(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Exp()
                self.gate = nn.Linear(16, 4, bias=False)

        class Adapter:
            def router_attr(self):
                return "gate"

            def experts_attr(self):
                return "experts"

            def expert_weight_attrs(self, moe=None):
                return {
                    "experts": "experts",
                    "gate": "gate",
                    "fused": True,
                    "gate_proj": "gate_up_proj",
                    "up_proj": "gate_up_proj",
                    "down_proj": "down_proj",
                    "weight_convention": "linear",
                }

            def weight_convention(self):
                return "linear"

        moe = Moe()
        adapter = Adapter()
        state = initialize_pruning_state(4, device="cpu")
        flat = torch.randn(12, 16)
        out = observe_moe_batch(
            state,
            moe,
            adapter,
            flat,
            num_experts=4,
            top_k=2,
            backend="bmm",
            record_pruning_metrics_only=True,
            fused=True,
        )
        assert "selected_experts" in out
        assert state["total_tokens"].item() == 12
        assert state["ean_sum"].shape == (4,)
