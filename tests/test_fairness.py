"""Smoke tests for src/fairness.py."""

import numpy as np
import pytest

from src.config import PROTECTED_ATTRIBUTES
from src.fairness import (
    compare_aware_vs_unaware,
    cost_optimal_threshold,
    group_metrics,
    run_fairness_audit,
)


def test_group_metrics_basic_properties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    proba = rng.random(200)
    groups = rng.choice(["a", "b"], size=200)

    table = group_metrics(y, proba, groups, threshold=0.5)
    assert set(table["group"]) == {"a", "b"}
    assert (table["approval_rate"].between(0, 1)).all()
    assert (table["disparate_impact_ratio_vs_max"] <= 1.0 + 1e-9).all()
    assert table["disparate_impact_ratio_vs_max"].max() == pytest.approx(1.0)


def test_run_fairness_audit_covers_all_protected_attributes():
    tables = run_fairness_audit(threshold=0.5)
    assert set(tables) == set(PROTECTED_ATTRIBUTES)
    for attr, table in tables.items():
        assert table["n"].sum() == 1000
        assert not table["approval_rate"].isna().any()


def test_cost_optimal_threshold_in_range_and_beats_default():
    sweep = cost_optimal_threshold()
    assert 0 < sweep["best_threshold"] < 1
    assert sweep["best_cost"] <= sweep["default_cost"]  # sweep must be at least as good


def test_aware_vs_unaware_variants_present():
    comparison = compare_aware_vs_unaware(threshold=0.5)
    assert len(comparison["variants"]) == 2
    for entry in comparison["variants"].values():
        assert 0.5 < entry["test_auc"] < 1.0  # better than chance on this dataset
        for attr in PROTECTED_ATTRIBUTES:
            assert 0 <= entry[f"{attr}_min_disparate_impact_ratio"] <= 1.0
