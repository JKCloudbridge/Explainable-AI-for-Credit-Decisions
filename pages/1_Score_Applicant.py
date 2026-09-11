"""Page 1 — pick or build an applicant, choose a threshold, see the decision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from app_lib.common import (
    default_applicant,
    ensure_state,
    get_test_split,
    reseed_form,
    render_applicant_form,
    score_applicant,
)
from src.config import FEATURE_COLUMNS

st.set_page_config(page_title="Score an Applicant", page_icon="\U0001F4CB", layout="wide")
ensure_state()

st.title("\U0001F4CB Score an applicant")

st.session_state.threshold = st.slider(
    "Decision threshold — an applicant is DENIED at or above this P(default)",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.threshold,
    step=0.01,
)

mode = st.radio("Applicant source", ["Pick from test set", "Enter manually"], horizontal=True)

if mode == "Pick from test set":
    X_test, y_test = get_test_split()
    idx = st.number_input(
        "Test-set row", min_value=0, max_value=len(X_test) - 1, value=0, step=1, key="test_row_idx"
    )
    applicant = X_test.iloc[int(idx)].to_dict()
    st.session_state.applicant = applicant
    st.session_state.applicant_source = f"test-set row #{idx}"

    true_label = "defaulted" if y_test[int(idx)] == 1 else "repaid"
    st.caption(f"Historical outcome for this applicant (**not** seen by the model): **{true_label}**")
    st.dataframe(pd.DataFrame([applicant]), use_container_width=True)
else:
    b1, b2 = st.columns(2)
    if b1.button("Reset to dataset default"):
        st.session_state.applicant = default_applicant()
        reseed_form("manual", st.session_state.applicant)
    if b2.button("Load current applicant into the form"):
        reseed_form("manual", st.session_state.applicant)

    st.caption("Starting point: your current applicant. Edit any field below.")
    base = st.session_state.applicant
    left, right = st.columns(2)
    half = len(FEATURE_COLUMNS) // 2
    applicant = {}
    applicant.update(render_applicant_form(left, base, "manual", FEATURE_COLUMNS[:half]))
    applicant.update(render_applicant_form(right, base, "manual", FEATURE_COLUMNS[half:]))
    st.session_state.applicant = applicant
    st.session_state.applicant_source = "manual entry"

result = score_applicant(st.session_state.applicant, st.session_state.threshold)

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("P(default)", f"{result['probability']:.1%}")
m2.metric("Threshold", f"{st.session_state.threshold:.0%}")
badge = "\U0001F534 DENY" if result["decision"] == "DENY" else "\U0001F7E2 APPROVE"
m3.metric("Decision", badge)

st.info("Open **Why this decision** in the sidebar to see the SHAP / LIME explanation for this applicant.")
