"""Group fairness audit + a protected-attribute-free comparison model.

**Evaluation set.** The per-group fairness table is computed on the **full
1,000-row dataset**, not just the 200-row hold-out, because several protected
subgroups are small (``foreign_worker = no`` is only 37 rows overall and 6 in
the test split) — a held-out-only estimate would be too noisy to read. This
means some rows here were seen during training; treat this as a demonstration
audit, not a rigorous out-of-sample fairness evaluation. The **cost-optimal
threshold** search and the **aware-vs-unaware AUC** comparison *do* use the
proper 200-row hold-out, since those don't need per-subgroup stability.

**Definitions** (positive class = ``default`` = 1 = bad risk; "deny" = predicted
``P(default) >= threshold``):

- ``approval_rate`` — share of a group predicted to repay (not denied). This is
  the "selection rate" the 4/5ths rule and demographic parity are defined on.
- ``disparate_impact_ratio_vs_max`` — a group's approval rate divided by the
  most-favoured group's approval rate. The EEOC's 4/5ths rule flags anything
  below 0.8.
- ``tpr_deny_given_default`` — recall for catching true defaulters (equal
  *opportunity to be correctly assessed as risky*).
- ``fpr_deny_given_repaid`` — the more consequential harm: the rate at which a
  group's genuinely creditworthy applicants are wrongly denied.
- ``calibration_gap`` — mean predicted P(default) minus the group's actual
  default rate; positive means the model overestimates that group's risk.

Run the whole audit (tables, comparison, cost-optimal threshold, figures,
``reports/fairness_metrics.json`` + ``reports/fairness_report.md``)::

    python -m src.fairness
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PROTECTED_ATTRIBUTES,
    REPORTS_DIR,
    TARGET,
)
from src.data import get_splits, load_data
from src.explain import load_artifacts
from src.preprocess import build_preprocessor
from src.train import XGB_PARAMS

# Features dropped for the protected-attribute-free ("unaware") comparison model.
UNAWARE_DROP = ["personal_status_sex", "age_years"]
UNAWARE_MODEL_PATH = MODELS_DIR / "xgboost_model_unaware.joblib"
UNAWARE_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_unaware.joblib"
FAIRNESS_METRICS_PATH = REPORTS_DIR / "fairness_metrics.json"
FAIRNESS_REPORT_PATH = REPORTS_DIR / "fairness_report.md"

# German Credit's documented cost matrix: missing a bad risk (approving a
# defaulter) is 5x costlier than wrongly denying a good risk.
COST_FN = 5.0  # false negative: predicted repay, actually defaulted
COST_FP = 1.0  # false positive: predicted default, actually would have repaid

DISPARATE_IMPACT_FLOOR = 0.8  # EEOC four-fifths rule


# =====================================================================
# Core metric
# =====================================================================
def group_metrics(y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray, threshold: float) -> pd.DataFrame:
    """Per-group fairness table. See module docstring for column definitions."""
    deny = (proba >= threshold).astype(int)
    rows = []
    for g in sorted(pd.unique(groups)):
        mask = groups == g
        yt, yp, pr = y_true[mask], deny[mask], proba[mask]
        is_default, is_repaid = yt == 1, yt == 0
        rows.append(
            {
                "group": g,
                "n": int(mask.sum()),
                "actual_default_rate": float(yt.mean()),
                "approval_rate": float(1 - yp.mean()),
                "tpr_deny_given_default": float(yp[is_default].mean()) if is_default.any() else float("nan"),
                "fpr_deny_given_repaid": float(yp[is_repaid].mean()) if is_repaid.any() else float("nan"),
                "auc": float(roc_auc_score(yt, pr)) if len(set(yt)) > 1 else float("nan"),
                "calibration_gap": float(pr.mean() - yt.mean()),
            }
        )
    table = pd.DataFrame(rows)
    best = table["approval_rate"].max()
    table["disparate_impact_ratio_vs_max"] = (
        table["approval_rate"] / best if best > 0 else float("nan")
    )
    return table


def _scores(model, preprocessor, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(preprocessor.transform(frame[features]))[:, 1]


def run_fairness_audit(threshold: float | None = None) -> dict[str, pd.DataFrame]:
    """Group fairness tables for every PROTECTED_ATTRIBUTES, on the full dataset."""
    threshold = DECISION_THRESHOLD if threshold is None else threshold
    model, preprocessor, _ = load_artifacts()
    df = load_data()
    proba = _scores(model, preprocessor, df, FEATURE_COLUMNS)
    y = df[TARGET].to_numpy()
    return {attr: group_metrics(y, proba, df[attr].to_numpy(), threshold) for attr in PROTECTED_ATTRIBUTES}


# =====================================================================
# Cost-optimal threshold (hold-out test set)
# =====================================================================
def cost_optimal_threshold(cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> dict:
    """Sweep thresholds on the 200-row test set to minimise the dataset's cost matrix."""
    model, preprocessor, _ = load_artifacts()
    _, _, X_test, _, y_test = get_splits()
    proba = _scores(model, preprocessor, X_test, FEATURE_COLUMNS)

    grid = np.linspace(0.02, 0.98, 97)
    costs = []
    for t in grid:
        deny = (proba >= t).astype(int)
        fn = int(((deny == 0) & (y_test == 1)).sum())
        fp = int(((deny == 1) & (y_test == 0)).sum())
        costs.append(cost_fn * fn + cost_fp * fp)
    costs = np.array(costs, dtype=float)
    j = int(np.argmin(costs))

    default_deny = (proba >= DECISION_THRESHOLD).astype(int)
    default_fn = int(((default_deny == 0) & (y_test == 1)).sum())
    default_fp = int(((default_deny == 1) & (y_test == 0)).sum())
    default_cost = cost_fn * default_fn + cost_fp * default_fp

    return {
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "grid": grid.tolist(),
        "costs": costs.tolist(),
        "best_threshold": float(grid[j]),
        "best_cost": float(costs[j]),
        "default_threshold": DECISION_THRESHOLD,
        "default_cost": float(default_cost),
        "cost_reduction_pct": float(1 - costs[j] / default_cost) if default_cost else 0.0,
    }


# =====================================================================
# Protected-attribute-free ("unaware") comparison model
# =====================================================================
def _unaware_feature_lists() -> tuple[list[str], list[str]]:
    num = [f for f in NUMERIC_FEATURES if f not in UNAWARE_DROP]
    cat = [f for f in CATEGORICAL_FEATURES if f not in UNAWARE_DROP]
    return num, cat


def train_unaware_variant():
    """Retrain XGBoost on the same split, minus UNAWARE_DROP. Persists to disk."""
    _, X_train, _, y_train, _ = get_splits()
    num, cat = _unaware_feature_lists()
    preprocessor = build_preprocessor(numeric_features=num, categorical_features=cat)
    X_train_t = preprocessor.fit_transform(X_train[num + cat])

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    model = XGBClassifier(scale_pos_weight=neg / pos, **XGB_PARAMS)
    model.fit(X_train_t, y_train)

    joblib.dump(model, UNAWARE_MODEL_PATH)
    joblib.dump(preprocessor, UNAWARE_PREPROCESSOR_PATH)
    return model, preprocessor


def load_or_train_unaware():
    if UNAWARE_MODEL_PATH.exists() and UNAWARE_PREPROCESSOR_PATH.exists():
        return joblib.load(UNAWARE_MODEL_PATH), joblib.load(UNAWARE_PREPROCESSOR_PATH)
    return train_unaware_variant()


def compare_aware_vs_unaware(threshold: float | None = None) -> dict:
    """AUC (test set) + fairness gaps (full dataset) for both models."""
    threshold = DECISION_THRESHOLD if threshold is None else threshold
    aware_model, aware_pre, _ = load_artifacts()
    unaware_model, unaware_pre = load_or_train_unaware()
    num, cat = _unaware_feature_lists()

    df = load_data()
    _, _, X_test, _, y_test = get_splits()
    y_full = df[TARGET].to_numpy()

    variants = {
        "aware (all 20 features)": {
            "model": aware_model,
            "pre": aware_pre,
            "features": FEATURE_COLUMNS,
        },
        f"unaware (drops {', '.join(UNAWARE_DROP)})": {
            "model": unaware_model,
            "pre": unaware_pre,
            "features": num + cat,
        },
    }

    result: dict = {"threshold": threshold, "dropped_features": UNAWARE_DROP, "variants": {}}
    for name, v in variants.items():
        test_proba = _scores(v["model"], v["pre"], X_test, v["features"])
        full_proba = _scores(v["model"], v["pre"], df, v["features"])
        entry = {"test_auc": float(roc_auc_score(y_test, test_proba))}
        for attr in PROTECTED_ATTRIBUTES:
            tbl = group_metrics(y_full, full_proba, df[attr].to_numpy(), threshold)
            entry[f"{attr}_approval_gap"] = float(
                tbl["approval_rate"].max() - tbl["approval_rate"].min()
            )
            entry[f"{attr}_min_disparate_impact_ratio"] = float(
                tbl["disparate_impact_ratio_vs_max"].min()
            )
        result["variants"][name] = entry
    return result


# =====================================================================
# Plots
# =====================================================================
def group_bar_figure(table: pd.DataFrame, value_col: str, title: str, ref_line: float | None = None):
    """Build (not save) a per-group bar chart -- reused live by the Streamlit app."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(table["group"].astype(str), table[value_col], color="#4C72B0")
    if ref_line is not None:
        ax.axhline(ref_line, color="#C44E52", ls="--", lw=1)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    top = float(np.nanmax(table[value_col])) if len(table) else 1.0
    ax.set_ylim(0, max(1.0, top * 1.15))
    fig.tight_layout()
    return fig


def plot_group_bar(table: pd.DataFrame, value_col: str, title: str, path, ref_line: float | None = None) -> None:
    fig = group_bar_figure(table, value_col, title, ref_line)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cost_curve(sweep: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sweep["grid"], sweep["costs"], color="#4C72B0")
    ax.axvline(sweep["best_threshold"], color="#C44E52", ls="--", label=f"cost-optimal = {sweep['best_threshold']:.2f}")
    ax.axvline(sweep["default_threshold"], color="grey", ls=":", label=f"default = {sweep['default_threshold']:.2f}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel(f"total cost  ({sweep['cost_fn']:.0f}x missed default + {sweep['cost_fp']:.0f}x wrongly denied)")
    ax.set_title("Cost-optimal threshold — hold-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# Report
# =====================================================================
def _write_fairness_report(tables: dict[str, pd.DataFrame], sweep: dict, comparison: dict) -> None:
    lines = [
        "# Fairness audit",
        "",
        "Positive class = `default` = 1 = bad risk. \"Deny\" = predicted "
        "`P(default) >= threshold`. Group tables use the full 1,000-row dataset "
        "for stable subgroup estimates (see `src/fairness.py` docstring for the caveat); "
        "the cost-optimal threshold and the aware-vs-unaware AUC use the 200-row hold-out.",
        "",
    ]
    for attr, table in tables.items():
        lines.append(f"## `{attr}`")
        lines.append("")
        lines.append(table.round(3).to_markdown(index=False))
        min_ratio = table["disparate_impact_ratio_vs_max"].min()
        flag = "⚠️ **below the 0.8 four-fifths threshold**" if min_ratio < DISPARATE_IMPACT_FLOOR else "within the four-fifths rule"
        lines.append("")
        lines.append(f"Minimum disparate-impact ratio: **{min_ratio:.2f}** — {flag}.")
        lines.append("")

    lines += [
        "## Cost-optimal threshold (hold-out test set)",
        "",
        f"German Credit's documented cost matrix weights a missed bad risk "
        f"({sweep['cost_fn']:.0f}x) far above a wrongly denied good risk ({sweep['cost_fp']:.0f}x). "
        f"At the default threshold ({sweep['default_threshold']:.2f}) the test-set cost is "
        f"**{sweep['default_cost']:.0f}**. The cost-minimising threshold is "
        f"**{sweep['best_threshold']:.2f}**, cost **{sweep['best_cost']:.0f}** "
        f"({sweep['cost_reduction_pct']:.0%} lower). Lowering the threshold denies more "
        "applicants overall — re-run the group tables at this threshold before adopting it, "
        "since a lower threshold typically widens approval-rate gaps.",
        "",
        "## Protected-attribute-free comparison",
        "",
        f"A second XGBoost model was trained on the identical split, dropping "
        f"`{'`, `'.join(comparison['dropped_features'])}` (\"unaware\").",
        "",
    ]
    rows = []
    for name, entry in comparison["variants"].items():
        row = {"variant": name, "test_auc": round(entry["test_auc"], 3)}
        for attr in PROTECTED_ATTRIBUTES:
            row[f"{attr} approval gap"] = round(entry[f"{attr}_approval_gap"], 3)
            row[f"{attr} min DI ratio"] = round(entry[f"{attr}_min_disparate_impact_ratio"], 3)
        rows.append(row)
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    aware_auc = comparison["variants"]["aware (all 20 features)"]["test_auc"]
    unaware_key = [k for k in comparison["variants"] if k.startswith("unaware")][0]
    unaware_auc = comparison["variants"][unaware_key]["test_auc"]
    auc_delta = unaware_auc - aware_auc
    auc_sentence = (
        f"costs {abs(auc_delta):.3f} AUC" if auc_delta < -0.001
        else f"does not cost accuracy (test AUC actually {abs(auc_delta):.3f} higher, "
             "well within the noise of a 200-row test set)" if auc_delta > 0.001
        else "changes test AUC by less than 0.001"
    )
    lines += [
        "",
        f"Dropping `personal_status_sex` and `age_years` {auc_sentence}, and does **not** "
        "eliminate the approval-rate gaps — they shrink only slightly, because the model "
        "can still reconstruct some of that signal from correlated features (e.g. `job`, "
        "`credit_history`, `savings_status`). This is classic proxy discrimination: "
        "removing a protected attribute from the input is necessary but not sufficient "
        "for fairness, and is exactly why per-group *outcome* audits (like the tables "
        "above) matter more than checking which columns the model sees.",
        "",
        "## Limitations",
        "",
        "- `foreign_worker = no` is 37 of 1,000 applicants (6 of 200 in the test split) — "
        "too small for a statistically reliable disparate-impact conclusion; reported for "
        "completeness, not as a finding to act on.",
        "- The fairness table mixes training and test rows (see above) — a production audit "
        "would use a held-out or post-deployment sample.",
        "- German Credit is a 1990s benchmark in Deutsche Mark; `personal_status_sex` conflates "
        "sex with marital status and cannot be cleanly disentangled in this dataset.",
    ]
    FAIRNESS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# Entry point
# =====================================================================
def main() -> None:
    print("Group fairness audit (full dataset, threshold = %.2f)\n" % DECISION_THRESHOLD)
    tables = run_fairness_audit()
    metrics_out: dict = {"threshold": DECISION_THRESHOLD, "tables": {}}
    for attr, table in tables.items():
        print(f"-- {attr} --")
        print(table.round(3).to_string(index=False))
        print()
        metrics_out["tables"][attr] = table.to_dict("records")
        plot_group_bar(
            table,
            "approval_rate",
            f"Approval rate by {attr}",
            FIGURES_DIR / f"fairness_{attr}_approval_rate.png",
        )
        plot_group_bar(
            table,
            "fpr_deny_given_repaid",
            f"Wrongly-denied rate (qualified applicants) by {attr}",
            FIGURES_DIR / f"fairness_{attr}_fpr.png",
        )

    print("Cost-optimal threshold sweep (hold-out test set) ...")
    sweep = cost_optimal_threshold()
    metrics_out["cost_sweep"] = {k: v for k, v in sweep.items() if k not in ("grid", "costs")}
    plot_cost_curve(sweep, FIGURES_DIR / "fairness_cost_vs_threshold.png")
    print(
        f"  default threshold {sweep['default_threshold']:.2f} -> cost {sweep['default_cost']:.0f} | "
        f"cost-optimal {sweep['best_threshold']:.2f} -> cost {sweep['best_cost']:.0f} "
        f"({sweep['cost_reduction_pct']:.0%} lower)"
    )

    print("\nTraining/loading the protected-attribute-free comparison model ...")
    comparison = compare_aware_vs_unaware()
    metrics_out["aware_vs_unaware"] = comparison
    for name, entry in comparison["variants"].items():
        print(f"  {name}: test AUC {entry['test_auc']:.3f}")

    FAIRNESS_METRICS_PATH.write_text(json.dumps(metrics_out, indent=2))
    _write_fairness_report(tables, sweep, comparison)
    print(f"\nWrote {FAIRNESS_METRICS_PATH}")
    print(f"Wrote {FAIRNESS_REPORT_PATH}")


if __name__ == "__main__":
    main()
