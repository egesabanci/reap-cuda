"""Regression tests for merge clustering option validation."""
from __future__ import annotations

import math

import pytest

from reap.args import ClusterArgs
from reap.merge_pipeline import _resolve_num_clusters, validate_cluster_args_static


def test_explicit_num_clusters_works_without_ratio():
    args = ClusterArgs(num_clusters=3, compression_ratio=None)
    validate_cluster_args_static(args)
    assert _resolve_num_clusters(args.num_clusters, args.compression_ratio, 8) == 3


@pytest.mark.parametrize(
    ("explicit", "ratio", "experts"),
    [
        (0, None, 8),
        (9, None, 8),
        (None, None, 8),
        (None, -0.1, 8),
        (None, 1.0, 8),
        (None, math.inf, 8),
    ],
)
def test_invalid_cluster_count_or_ratio_is_rejected(explicit, ratio, experts):
    with pytest.raises(ValueError):
        _resolve_num_clusters(explicit, ratio, experts)


def test_static_validation_rejects_unimplemented_cluster_method():
    with pytest.raises(ValueError, match="Unsupported cluster_method"):
        validate_cluster_args_static(ClusterArgs(cluster_method="spectral"))


def test_num_clusters_documented_precedence_allows_default_ratio():
    args = ClusterArgs(num_clusters=2, compression_ratio=0.5)
    validate_cluster_args_static(args)
    assert _resolve_num_clusters(args.num_clusters, args.compression_ratio, 4) == 2
