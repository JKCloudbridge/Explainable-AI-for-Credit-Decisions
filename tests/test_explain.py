"""Smoke tests for src/explain.py (SHAP + LIME)."""

import numpy as np
import pytest

from src.config import FEATURE_COLUMNS
from src.explain import (
    _pick_examples,
    explain_local,
    global_importance,
    reason_codes,
)


@pytest.fixture(scope="module")
def examples():
    return _pick_examples()  # (approve_applicant, deny_applicant)


def test_shap_additivity(examples):
    """base_value + sum(shap contributions) must equal predict_proba."""
    _, deny = examples
    result = explain_local(deny, run_lime=False)
    reconstructed = result["base_value"] + sum(c["shap"] for c in result["shap"])
    assert reconstructed == pytest.approx(result["probability"], abs=1e-6)
    assert len(result["shap"]) == len(FEATURE_COLUMNS)


def test_explain_local_respects_threshold(examples):
    _, deny = examples
    lenient = explain_local(deny, run_lime=False, threshold=0.99)
    strict = explain_local(deny, run_lime=False, threshold=0.01)
    assert lenient["decision"] == "APPROVE"
    assert strict["decision"] == "DENY"
    assert lenient["probability"] == pytest.approx(strict["probability"])  # same score either way


def test_lime_runs_and_returns_conditions(examples):
    _, deny = examples
    result = explain_local(deny, run_lime=True)
    assert len(result["lime"]) > 0
    assert all({"feature", "condition", "weight"} <= item.keys() for item in result["lime"])


def test_reason_codes_only_lists_adverse_features(examples):
    _, deny = examples
    rc = reason_codes(deny, k=4)
    assert rc["decision"] == "DENY"
    assert 0 < len(rc["reasons"]) <= 4
    assert all(r["impact_on_p_default"] > 0 for r in rc["reasons"])


def test_reason_codes_empty_for_confident_approval(examples):
    approve, _ = examples
    rc = reason_codes(approve, k=4)
    assert rc["decision"] == "APPROVE"


def test_global_importance_is_sorted_and_covers_all_features():
    importance = global_importance()
    assert len(importance) == len(FEATURE_COLUMNS)
    assert list(importance.values) == sorted(importance.values, reverse=True)
    assert (importance.values >= 0).all()
    assert importance.index[0] == "checking_status"  # documented top global driver
