"""Smoke tests for src/adverse_action.py."""

import pytest

from src.adverse_action import generate_notice
from src.explain import _pick_examples


@pytest.fixture(scope="module")
def examples():
    return _pick_examples()


def test_denied_notice_lists_principal_reasons(examples):
    _, deny = examples
    notice = generate_notice(deny, applicant_ref="test-deny")
    assert notice["decision"] == "DENY"
    assert len(notice["reasons"]) > 0
    assert "Principal reason" in notice["body"]
    assert "DENIED" in notice["body"]


def test_approved_notice_has_no_principal_reasons_section(examples):
    """An approval never gets a Reg B reasons-for-denial section, even though a
    strongly-approved applicant can still have a few small positive-SHAP features
    (see test_explain.py) -- reason_codes() surfaces those regardless of decision."""
    approve, _ = examples
    notice = generate_notice(approve, applicant_ref="test-approve")
    assert notice["decision"] == "APPROVE"
    assert "APPROVED" in notice["body"]
    assert "Principal reason" not in notice["body"]


def test_notice_respects_threshold(examples):
    _, deny = examples
    lenient = generate_notice(deny, threshold=0.99)
    assert lenient["decision"] == "APPROVE"
