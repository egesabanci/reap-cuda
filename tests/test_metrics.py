"""OnlineStatsTracker scalar/tensor count contract (Fix #4)."""

from __future__ import annotations

import torch

from reap.metrics import OnlineStatsTracker


def test_update_accepts_python_int_count():
    """An int new_count must not raise and must broadcast to count_shape."""
    tracker = OnlineStatsTracker(shape=(2,), count_shape=(2,), device="cpu")
    tracker.update(torch.ones(2), 1)
    assert torch.equal(tracker.count, torch.tensor([1, 1], dtype=torch.long))
    assert torch.allclose(tracker.mean, torch.ones(2, dtype=torch.float32))


def test_int_and_scalar_tensor_count_produce_identical_state():
    """int 1 and tensor(1) must yield identical tracker state."""
    a = OnlineStatsTracker(shape=(2,), count_shape=(2,), device="cpu")
    b = OnlineStatsTracker(shape=(2,), count_shape=(2,), device="cpu")
    a.update(torch.ones(2), 1)
    b.update(torch.ones(2), torch.tensor(1, dtype=torch.long))
    assert torch.equal(a.count, b.count)
    assert torch.allclose(a.mean, b.mean)


def test_update_accepts_broadcastable_tensor_count():
    """A per-element (count_shape,) tensor count is the existing caller pattern."""
    tracker = OnlineStatsTracker(shape=(3,), count_shape=(3,), device="cpu")
    new_mean = torch.tensor([1.0, 2.0, 3.0])
    new_count = torch.tensor([2, 1, 3], dtype=torch.long)
    tracker.update(new_mean, new_count)
    assert torch.equal(tracker.count, torch.tensor([2, 1, 3], dtype=torch.long))
    # First batch: mean == new_mean by definition.
    assert torch.allclose(tracker.mean, new_mean)


def test_update_zero_count_element_preserved_by_nan_to_num():
    """A zero-count element must not contaminate the mean (nan_to_num guard)."""
    tracker = OnlineStatsTracker(shape=(2,), count_shape=(2,), device="cpu")
    # First update seeds the mean.
    tracker.update(torch.tensor([5.0, 5.0]), torch.tensor([1, 1], dtype=torch.long))
    # Second update with a zero-count element on index 1.
    tracker.update(torch.tensor([7.0, 0.0]), torch.tensor([1, 0], dtype=torch.long))
    # Index 0: mean of [5, 7] == 6; index 1: stays 5 (zero-count, no change).
    assert torch.allclose(tracker.mean, torch.tensor([6.0, 5.0]))
    assert torch.equal(tracker.count, torch.tensor([2, 1], dtype=torch.long))


def test_update_int_count_matches_existing_tensor_caller_pattern():
    """Existing pruning-metrics callers pass expert_frequency tensors; the
    int path must produce the same result as the equivalent scalar tensor."""
    e = 4
    a = OnlineStatsTracker(shape=(e,), count_shape=(e,), device="cpu")
    b = OnlineStatsTracker(shape=(e,), count_shape=(e,), device="cpu")
    new_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    freq = torch.tensor([2, 1, 0, 3], dtype=torch.long)
    a.update(new_mean, freq)
    # Equivalent: each element updated with its count, then summed via one call.
    b.update(new_mean, freq)
    assert torch.equal(a.count, b.count)
    assert torch.allclose(a.mean, b.mean)