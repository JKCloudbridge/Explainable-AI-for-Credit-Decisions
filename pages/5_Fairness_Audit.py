"""Page 5 — group fairness audit at the app's live threshold, plus the
protected-attribute-free comparison model and the cost-optimal threshold finding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app_lib.common import ensure_state
from src.config import PROTECTED_ATTRIBUTES
from src.fairness import (
    DISPARATE_IMPACT_FLOOR,
    compare_aware_vs_unaware,
    cost_optimal_threshold,
    group_bar_figure,
    run_fairness_audit,
)

st.set_page_config(page_title="Fairness Audit", page_icon="⚖️", layout="wide")
ensure_state()

st.title("⚖️ Fairness audit")
st.caption(
    "DEMONSTRATION — evaluated on the full 1,000-row dataset for stable subgroup "
    "estimates (some rows were used in training); see `reports/fairness_report.md` "
    "for the full write-up and limitations."
)

threshold = st.session_state.threshold
st.info(
    f"Using the threshold set on **Score an applicant**: **{threshold:.0%}**. "
    "Change it there to see how these tables move."
)

tables = run_fairness_audit(threshold=threshold)

for attr in PROTECTED_ATTRIBUTES:
    table = tables[attr]
    st.subheader(f"`{attr}`")
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.dataframe(table.round(3), use_container_width=True, hide_index=True)

    min_ratio = float(table["disparate_impact_ratio_vs_max"].min())
    if min_ratio < DISPARATE_IMPACT_FLOOR:
        c2.metric("Min disparate-impact ratio", f"{min_ratio:.2f}", help="Below the 0.8 four-fifths rule")
        c2.error(f"Below the {DISPARATE_IMPACT_FLOOR:.0%} four-fifths floor")
    else:
        c2.metric("Min disparate-impact ratio", f"{min_ratio:.2f}")
        c2.success("Within the four-fifths rule")
    small_groups = table[table["n"] < 50]
    if not small_groups.empty:
        c3.warning(
            f"Small group(s): {', '.join(f'{g} (n={n})' for g, n in zip(small_groups['group'], small_groups['n']))} "
            "— treat with caution."
        )

    p1, p2 = st.columns(2)
    fig1 = group_bar_figure(table, "approval_rate", f"Approval rate by {attr}")
    p1.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
    fig2 = group_bar_figure(table, "fpr_deny_given_repaid", f"Wrongly-denied rate (qualified) by {attr}")
    p2.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.divider()

st.subheader("Cost-optimal threshold")
st.caption(
    "German Credit's documented cost matrix: missing a bad risk is 5x costlier than "
    "wrongly denying a good one. Swept on the 200-row hold-out test set."
)
sweep = cost_optimal_threshold()
c1, c2, c3 = st.columns(3)
c1.metric("Default threshold", f"{sweep['default_threshold']:.0%}", help=f"cost {sweep['default_cost']:.0f}")
c2.metric(
    "Cost-optimal threshold",
    f"{sweep['best_threshold']:.0%}",
    delta=f"{sweep['cost_reduction_pct']:.0%} lower cost",
    delta_color="normal",
)
c3.metric("Your current threshold", f"{threshold:.0%}")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sweep["grid"], sweep["costs"], color="#4C72B0")
ax.axvline(sweep["best_threshold"], color="#C44E52", ls="--", label="cost-optimal")
ax.axvline(threshold, color="#55A868", ls="-", label="your threshold")
ax.set_xlabel("decision threshold")
ax.set_ylabel("total cost")
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption(
    "A lower threshold denies more applicants and typically **widens** approval-rate "
    "gaps — re-check the tables above before adopting a cost-optimal threshold."
)

st.divider()
st.subheader("Does removing sex and age fix it?")
st.caption("A second XGBoost model, retrained on the same split without `personal_status_sex` or `age_years`.")
comparison = compare_aware_vs_unaware(threshold=threshold)
rows = []
for name, entry in comparison["variants"].items():
    row = {"variant": name, "test AUC": round(entry["test_auc"], 3)}
    for attr in PROTECTED_ATTRIBUTES:
        row[f"{attr} gap"] = round(entry[f"{attr}_approval_gap"], 3)
        row[f"{attr} min DI"] = round(entry[f"{attr}_min_disparate_impact_ratio"], 3)
    rows.append(row)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.markdown(
    "Removing the protected attributes barely moves the gaps — the model reconstructs "
    "much of that signal from correlated features (`job`, `credit_history`, "
    "`savings_status`, ...). Removing a protected attribute from the model's inputs is "
    "**necessary but not sufficient** for fairness; auditing outcomes (the tables above) "
    "is what actually catches this."
)
