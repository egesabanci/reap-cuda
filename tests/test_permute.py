"""Regression coverage for advertised expert permutation modes."""
from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from reap.permute import DirectAlignmentPermuter


class _Expert(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 7):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


def _attrs(*, fused: bool = False) -> dict[str, object]:
    return {
        "up_proj": "up_proj",
        "gate_proj": "gate_proj",
        "down_proj": "down_proj",
        "fused": fused,
    }


def test_direct_permuter_is_concrete_and_forward_equivalent():
    torch.manual_seed(0)
    experts = [_Expert(), _Expert(), _Expert()]
    inputs = torch.randn(5, 4)
    before = experts[1](inputs).detach().clone()
    untouched = experts[2](inputs).detach().clone()

    permuter = DirectAlignmentPermuter(_attrs())
    assert not inspect.isabstract(type(permuter))
    permuter.permute(experts, expert_indices=[0, 1], dom_expert_idx=0)

    assert torch.allclose(before, experts[1](inputs), atol=1e-6, rtol=1e-6)
    assert torch.allclose(untouched, experts[2](inputs), atol=1e-6, rtol=1e-6)


def test_direct_permuter_rejects_fused_layout_actionably():
    permuter = DirectAlignmentPermuter(_attrs(fused=True))
    with pytest.raises(NotImplementedError, match="does not support fused"):
        permuter.permute(nn.Identity(), expert_indices=[0], dom_expert_idx=0)
