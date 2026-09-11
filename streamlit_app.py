"""Explainable AI for Credit / Loan Decisions — Streamlit app (Phase 3).

Run from the project root, inside the venv:

    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import streamlit as st

from app_lib.common import ensure_state, get_full_data, get_model_bundle
from src.config import METRICS_PATH, TARGET

st.set_page_config(page_title="Explainable Credit AI", page_icon="\U0001F4B3", layout="wide")
ensure_state()

st.title("\U0001F4B3 Explainable AI for Credit / Loan Decisions")
st.caption(
    "A credit-risk model you can interrogate: per-applicant explanations (SHAP + LIME) "
    "and what-if analysis today; a fairness audit and adverse-action notices arrive in Phase 4."
)

model, preprocessor, metadata = get_model_bundle()
df = get_full_data()
metrics = json.loads(METRICS_PATH.read_text())
xgb_test = metrics["models"]["xgboost"]["test"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Applicants", f"{len(df):,}")
c2.metric("Historical default rate", f"{df[TARGET].mean():.0%}")
c3.metric("Model ROC-AUC (test)", f"{xgb_test['roc_auc']:.3f}")
c4.metric("Deployed model", "XGBoost")

st.divider()
st.markdown(
    """
### How to use this app

1. **Score an applicant** — pick one from the hold-out test set, or fill in a form by hand.
2. **Why this decision** — the SHAP and LIME explanation behind that applicant's score.
3. **What-if** — nudge the most influential features and watch the decision update live.
4. **Global insights** — what drives the model overall, and how well it performs.

The applicant you're scoring, and your chosen decision threshold, follow you between
pages via the sidebar navigation — pick one on **Score an applicant** first.
"""
)

st.divider()
st.subheader("Dataset")
st.markdown(
    f"**{metadata['dataset']}** — {metadata['n_samples']} applicants, "
    f"{len(metadata['feature_columns'])} attributes, trained/tested "
    f"{metadata['n_train']}/{metadata['n_test']}. See `data/README.md` for the full codebook."
)
st.dataframe(df.head(10), use_container_width=True)
