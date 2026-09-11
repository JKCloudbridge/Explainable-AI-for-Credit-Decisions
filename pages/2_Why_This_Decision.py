"""Page 2 — SHAP + LIME explanation and plain-language reason codes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st

from app_lib.common import ensure_state
from src.explain import explain_local, local_lime_figure, local_shap_figure, reason_codes

st.set_page_config(page_title="Why This Decision", page_icon="\U0001F50D", layout="wide")
ensure_state()

st.title("\U0001F50D Why this decision")
st.caption(f"Applicant: {st.session_state.applicant_source}")

with st.spinner("Computing SHAP + LIME explanation ..."):
    result = explain_local(st.session_state.applicant, run_lime=True)

badge = "\U0001F534 DENY" if result["decision"] == "DENY" else "\U0001F7E2 APPROVE"
c1, c2, c3 = st.columns(3)
c1.metric("P(default)", f"{result['probability']:.1%}")
c2.metric("Baseline P(default)", f"{result['base_value']:.1%}", help="Average prediction over the SHAP background sample")
c3.metric("Decision", badge, help=f"threshold {st.session_state.threshold:.0%}")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("SHAP — exact attribution")
    st.caption("Sums exactly to the prediction: base value + all contributions = P(default).")
    fig = local_shap_figure(result)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.subheader("LIME — local surrogate")
    st.caption("A locally-fit linear model around this one applicant.")
    fig2 = local_lime_figure(result)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

st.divider()
st.subheader("Reason codes")
rc = reason_codes(st.session_state.applicant, k=4)
if rc["reasons"]:
    st.markdown("Top factors pushing this application toward **denial**:")
    for r in rc["reasons"]:
        st.markdown(f"- **{r['statement']}**  (+{r['impact_on_p_default']:.3f} to P(default))")
else:
    st.markdown("No feature pushed this application toward denial — a clean approval.")

st.caption(
    "On a 40-applicant sample of the hold-out test set, SHAP and LIME agree on the "
    "*direction* of a feature's effect 97% of the time (see `reports/shap_vs_lime.md`); "
    "expect some disagreement on exact ranking — that's why both are shown above."
)
