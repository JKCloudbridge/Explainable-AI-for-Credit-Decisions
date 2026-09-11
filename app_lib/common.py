"""Cached loaders, session-state, and form-building helpers shared by every page.

Every Streamlit entry file (streamlit_app.py and pages/*.py) inserts the project
root onto sys.path *before* importing this module, so the plain ``from src...``
imports below always resolve regardless of which directory Streamlit treats as
the working directory.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from src.data import get_codebook, get_splits, load_data
from src.explain import load_artifacts

# Human-readable labels for form fields (feature name -> label, with units).
FEATURE_LABELS = {
    "checking_status": "Checking account status",
    "duration_months": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)",
    "savings_status": "Savings account balance",
    "employment_since": "Employment length",
    "installment_rate_pct": "Instalment rate (% of disposable income)",
    "personal_status_sex": "Personal status & sex",
    "other_debtors": "Other debtors / guarantors",
    "residence_since": "Years at present residence",
    "property_type": "Property",
    "age_years": "Age (years)",
    "other_installment_plans": "Other instalment plans",
    "housing": "Housing",
    "existing_credits_count": "Existing credits at this bank",
    "job": "Job category",
    "liable_maintenance_count": "Dependants",
    "telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# =====================================================================
# Cached loaders — data.py / explain.py already lru_cache their own heavy
# lifting (model, SHAP explainer, LIME explainer); these wrappers just save
# the DataFrame decoding / split work on every Streamlit rerun.
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """(model, preprocessor, metadata) — persisted Phase 1 artefacts."""
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_full_data() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner=False)
def get_test_split():
    """(X_test, y_test) — the canonical hold-out split, index reset to 0..N-1."""
    _, _, X_test, _, y_test = get_splits()
    return X_test.reset_index(drop=True), y_test


@st.cache_data(show_spinner=False)
def get_codebook_cached() -> dict:
    return get_codebook()


# =====================================================================
# Scoring
# =====================================================================
def score_applicant(applicant: dict, threshold: float) -> dict:
    model, preprocessor, _ = get_model_bundle()
    frame = pd.DataFrame([applicant])[FEATURE_COLUMNS]
    proba = float(model.predict_proba(preprocessor.transform(frame))[0, 1])
    return {"probability": proba, "decision": "DENY" if proba >= threshold else "APPROVE"}


def default_applicant() -> dict:
    """A dataset-median / most-common-value applicant, used to seed forms."""
    df = get_full_data()
    row: dict = {}
    for f in NUMERIC_FEATURES:
        row[f] = int(df[f].median())
    for f in CATEGORICAL_FEATURES:
        row[f] = df[f].mode().iloc[0]
    return row


# =====================================================================
# Session state
# =====================================================================
def ensure_state() -> None:
    """Guarantee threshold/applicant exist, however the user reached this page."""
    st.session_state.setdefault("threshold", DECISION_THRESHOLD)
    if "applicant" not in st.session_state:
        st.session_state.applicant = default_applicant()
        st.session_state.applicant_source = "dataset default (median / most common value)"


def numeric_range(df: pd.DataFrame, feature: str) -> tuple[int, int]:
    return int(df[feature].min()), int(df[feature].max())


# =====================================================================
# Form building
# =====================================================================
def render_applicant_form(container, current: dict, key_prefix: str, features: list[str]) -> dict:
    """Render one widget per feature in ``features``, seeded from ``current``.

    Uses a stable ``key`` per widget (``f"{key_prefix}_{feature}"``) so Streamlit
    persists edits across reruns; callers that need to force a reseed should
    delete those keys from ``st.session_state`` first (see ``reseed_form``).
    """
    df = get_full_data()
    out: dict = {}
    for f in features:
        key = f"{key_prefix}_{f}"
        if f in CATEGORICAL_FEATURES:
            options = sorted(df[f].unique().tolist())
            default = current.get(f, options[0])
            index = options.index(default) if default in options else 0
            out[f] = container.selectbox(label(f), options, index=index, key=key)
        else:
            lo, hi = numeric_range(df, f)
            default = int(current.get(f, (lo + hi) // 2))
            default = min(max(default, lo), hi)
            out[f] = container.slider(label(f), min_value=lo, max_value=hi, value=default, key=key)
    return out


def reseed_form(key_prefix: str, values: dict) -> None:
    """Force render_applicant_form's widgets to pick up new values on next render."""
    for f in FEATURE_COLUMNS:
        key = f"{key_prefix}_{f}"
        if key in st.session_state:
            del st.session_state[key]


def sync_whatif_defaults(key_prefix: str, base_applicant: dict) -> None:
    """Reseed a form's widgets whenever the underlying base applicant changes.

    Without this, a form bound to fixed widget ``key``s would keep showing stale
    values after the user picks a different applicant on another page.
    """
    signature = repr(sorted(base_applicant.items()))
    sig_key = f"{key_prefix}_base_signature"
    if st.session_state.get(sig_key) != signature:
        reseed_form(key_prefix, base_applicant)
        st.session_state[sig_key] = signature
