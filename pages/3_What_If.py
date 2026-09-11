"""Page 3 — nudge the most influential features and watch the decision update live."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st

from app_lib.common import (
    ensure_state,
    render_applicant_form,
    reseed_form,
    score_applicant,
    sync_whatif_defaults,
)
from src.explain import explain_local, global_importance, local_shap_figure

st.set_page_config(page_title="What-If", page_icon="\U0001F39B️", layout="wide")
ensure_state()

st.title("\U0001F39B️ What-if analysis")
st.caption(f"Base applicant: {st.session_state.applicant_source}")

# Reseed the what-if widgets whenever the base applicant changed on another page.
sync_whatif_defaults("whatif", st.session_state.applicant)

top_features = global_importance().head(8).index.tolist()
st.markdown(
    "Adjust the model's **8 most influential features** (by mean |SHAP|) and see how "
    "the prediction moves. Everything else stays fixed at the base applicant's values."
)

if st.button("Reset sliders to base applicant"):
    reseed_form("whatif", st.session_state.applicant)

left, right = st.columns(2)
half = len(top_features) // 2 + len(top_features) % 2
whatif = dict(st.session_state.applicant)
whatif.update(render_applicant_form(left, st.session_state.applicant, "whatif", top_features[:half]))
whatif.update(render_applicant_form(right, st.session_state.applicant, "whatif", top_features[half:]))

original = score_applicant(st.session_state.applicant, st.session_state.threshold)
modified = score_applicant(whatif, st.session_state.threshold)

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Original P(default)", f"{original['probability']:.1%}", help=f"decision: {original['decision']}")
c2.metric(
    "What-if P(default)",
    f"{modified['probability']:.1%}",
    delta=f"{modified['probability'] - original['probability']:+.1%}",
    delta_color="inverse",
    help=f"decision: {modified['decision']}",
)
badge = "\U0001F534 DENY" if modified["decision"] == "DENY" else "\U0001F7E2 APPROVE"
c3.metric("What-if decision", badge)

if st.button("Use this what-if scenario as my current applicant"):
    st.session_state.applicant = whatif
    st.session_state.applicant_source = "what-if scenario"
    st.success("Updated — open **Why this decision** to see the new explanation.")

with st.expander("SHAP explanation for the what-if scenario"):
    res = explain_local(whatif, run_lime=False)
    fig = local_shap_figure(res)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
