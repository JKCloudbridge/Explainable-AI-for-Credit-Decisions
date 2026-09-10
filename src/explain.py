"""Explainability engine: SHAP (global + local) and LIME (local).

Everything here runs against the persisted Phase 1 artefacts and works in the
original 20-feature space, so the Streamlit app (Phase 3) and the fairness /
adverse-action code (Phase 4) can consume it unchanged.

SHAP values are computed in **probability space**: a contribution of ``+0.08``
means the feature pushed ``P(default)`` up by 0.08 from the baseline. The one-hot
columns produced by the preprocessor are summed back into their parent
categorical feature, so an explanation always has exactly 20 rows.

Run the whole engine (global figures + two worked local examples + the
SHAP-vs-LIME agreement report)::

    python -m src.explain
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from src.config import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    EXPLAIN_BACKGROUND_SIZE,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    SHAP_CACHE_PATH,
    SHAP_VS_LIME_PATH,
)
from src.data import get_splits

# --- Plain-language templates for reason codes ------------------------
# {value} is filled with the applicant's (decoded) feature value.
_REASON_TEMPLATES = {
    "checking_status": "Checking account status: {value}",
    "duration_months": "Long loan duration ({value} months)",
    "credit_history": "Credit history: {value}",
    "purpose": "Loan purpose: {value}",
    "credit_amount": "Credit amount requested ({value} DM)",
    "savings_status": "Savings account balance: {value}",
    "employment_since": "Employment length: {value}",
    "installment_rate_pct": "Instalment is {value}% of disposable income",
    "personal_status_sex": "Personal status: {value}",
    "other_debtors": "Other debtors / guarantors: {value}",
    "residence_since": "Only {value} year(s) at present residence",
    "property_type": "Property: {value}",
    "age_years": "Applicant age ({value})",
    "other_installment_plans": "Other instalment plans: {value}",
    "housing": "Housing: {value}",
    "existing_credits_count": "{value} existing credit(s) at this bank",
    "job": "Job category: {value}",
    "liable_maintenance_count": "{value} dependant(s)",
    "telephone": "Telephone: {value}",
    "foreign_worker": "Foreign worker: {value}",
}


# =====================================================================
# Artefacts and the canonical split
# =====================================================================
@lru_cache(maxsize=1)
def load_artifacts():
    """(model, preprocessor, metadata) from Phase 1 — cached per process."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, preprocessor, metadata


@lru_cache(maxsize=1)
def _splits():
    return get_splits()  # df, X_train, X_test, y_train, y_test


def _feature_groups(preprocessor) -> dict[str, list[int]]:
    """Original feature name -> its column indices in the transformed matrix."""
    groups: dict[str, list[int]] = {}
    col = 0
    for f in NUMERIC_FEATURES:
        groups[f] = [col]
        col += 1
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    for f, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        groups[f] = list(range(col, col + len(cats)))
        col += len(cats)
    return groups


def _aggregate(shap_2d: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    """(n, n_transformed) -> (n, 20) summed into the original features."""
    out = np.zeros((shap_2d.shape[0], len(FEATURE_COLUMNS)))
    for j, f in enumerate(FEATURE_COLUMNS):
        out[:, j] = shap_2d[:, groups[f]].sum(axis=1)
    return out


# =====================================================================
# SHAP
# =====================================================================
@lru_cache(maxsize=1)
def get_explainer():
    """SHAP TreeExplainer in probability space, cached per process."""
    model, preprocessor, _ = load_artifacts()
    _, X_train, _, _, _ = _splits()
    n = min(EXPLAIN_BACKGROUND_SIZE, len(X_train))
    background = preprocessor.transform(X_train.sample(n, random_state=RANDOM_SEED))
    return shap.TreeExplainer(
        model,
        data=np.asarray(background, dtype=float),
        model_output="probability",
        feature_perturbation="interventional",
    )


def shap_for_frame(frame: pd.DataFrame):
    """Return ``(agg_shap (n,20), base_value)`` for rows in original space."""
    _, preprocessor, _ = load_artifacts()
    explainer = get_explainer()
    matrix = np.asarray(preprocessor.transform(frame[FEATURE_COLUMNS]), dtype=float)
    sv = np.asarray(explainer.shap_values(matrix, check_additivity=False))
    agg = _aggregate(sv, _feature_groups(preprocessor))
    return agg, float(np.asarray(explainer.expected_value).ravel()[-1])


def compute_and_cache_shap(save: bool = True) -> dict:
    """SHAP values for every row, aggregated to 20 features, cached to disk."""
    df, *_ = _splits()
    agg, base = shap_for_frame(df)
    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "base_value": base,
        "agg_shap": agg,
        "feature_values": df[FEATURE_COLUMNS].reset_index(drop=True),
        "mean_abs": np.abs(agg).mean(axis=0),
    }
    if save:
        joblib.dump(payload, SHAP_CACHE_PATH)
    return payload


def load_cached_shap() -> dict:
    if SHAP_CACHE_PATH.exists():
        return joblib.load(SHAP_CACHE_PATH)
    return compute_and_cache_shap(save=True)


def global_importance() -> pd.Series:
    """Mean |SHAP| per original feature, descending."""
    cache = load_cached_shap()
    return (
        pd.Series(cache["mean_abs"], index=cache["feature_columns"])
        .sort_values(ascending=False)
    )


# =====================================================================
# LIME
# =====================================================================
@lru_cache(maxsize=1)
def _lime_setup():
    """LimeTabularExplainer over the original feature space (categoricals coded)."""
    _, preprocessor, _ = load_artifacts()
    df, X_train, _, _, _ = _splits()

    cat_levels = {f: sorted(df[f].unique().tolist()) for f in CATEGORICAL_FEATURES}
    cat_index = {f: {c: i for i, c in enumerate(cat_levels[f])} for f in CATEGORICAL_FEATURES}
    cat_feature_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATURES]
    categorical_names = {
        FEATURE_COLUMNS.index(f): cat_levels[f] for f in CATEGORICAL_FEATURES
    }

    def encode(frame: pd.DataFrame) -> np.ndarray:
        arr = np.zeros((len(frame), len(FEATURE_COLUMNS)))
        for j, f in enumerate(FEATURE_COLUMNS):
            if f in CATEGORICAL_FEATURES:
                arr[:, j] = frame[f].map(cat_index[f]).to_numpy()
            else:
                arr[:, j] = frame[f].to_numpy(dtype=float)
        return arr

    def decode(coded_2d: np.ndarray) -> pd.DataFrame:
        rows = []
        for r in coded_2d:
            row = {}
            for j, f in enumerate(FEATURE_COLUMNS):
                if f in CATEGORICAL_FEATURES:
                    levels = cat_levels[f]
                    k = int(np.clip(round(r[j]), 0, len(levels) - 1))
                    row[f] = levels[k]
                else:
                    row[f] = r[j]
            rows.append(row)
        return pd.DataFrame(rows)[FEATURE_COLUMNS]

    model, _, _ = load_artifacts()

    def predict_fn(coded_2d: np.ndarray) -> np.ndarray:
        frame = decode(np.atleast_2d(coded_2d))
        matrix = preprocessor.transform(frame)
        return model.predict_proba(matrix)

    explainer = LimeTabularExplainer(
        training_data=encode(X_train),
        feature_names=FEATURE_COLUMNS,
        class_names=["repaid", "default"],
        categorical_features=cat_feature_idx,
        categorical_names=categorical_names,
        discretize_continuous=True,
        mode="classification",
        random_state=RANDOM_SEED,
    )
    return explainer, encode, predict_fn


def lime_local(applicant: pd.Series, num_features: int = 10, num_samples: int = 3000):
    """LIME explanation for one applicant -> list of {feature, condition, weight}."""
    explainer, encode, predict_fn = _lime_setup()
    coded = encode(pd.DataFrame([applicant])[FEATURE_COLUMNS])[0]
    exp = explainer.explain_instance(
        coded, predict_fn, num_features=num_features, labels=(1,), num_samples=num_samples
    )
    out = []
    for cond, weight in exp.as_list(label=1):
        base = cond
        for f in sorted(FEATURE_COLUMNS, key=len, reverse=True):
            if f in cond:
                base = f
                break
        out.append({"feature": base, "condition": cond, "weight": float(weight)})
    return out


# =====================================================================
# Local explanation (SHAP + LIME together) and reason codes
# =====================================================================
def explain_local(applicant, run_lime: bool = True) -> dict:
    """Score one applicant and explain the decision with SHAP (+ optionally LIME).

    ``applicant`` is a dict / Series in the original 20-feature space.
    """
    model, preprocessor, _ = load_artifacts()
    if isinstance(applicant, dict):
        applicant = pd.Series(applicant)
    frame = pd.DataFrame([applicant])[FEATURE_COLUMNS]

    proba = float(model.predict_proba(preprocessor.transform(frame))[0, 1])
    decision = "DENY" if proba >= DECISION_THRESHOLD else "APPROVE"

    agg, base = shap_for_frame(frame)
    contribs = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": [frame.iloc[0][f] for f in FEATURE_COLUMNS],
            "shap": agg[0],
        }
    )
    contribs = contribs.reindex(
        contribs["shap"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    result = {
        "probability": proba,
        "decision": decision,
        "threshold": DECISION_THRESHOLD,
        "base_value": base,
        "shap": contribs.to_dict("records"),
    }
    if run_lime:
        result["lime"] = lime_local(applicant)
    return result


def reason_codes(applicant, k: int = 4) -> dict:
    """The k features that pushed the decision most toward DENY, in plain words.

    Precursor to the ECOA / Regulation B adverse-action notice built in Phase 4.
    """
    res = explain_local(applicant, run_lime=False)
    adverse = [c for c in res["shap"] if c["shap"] > 0][:k]
    reasons = []
    for c in adverse:
        template = _REASON_TEMPLATES.get(c["feature"], "{feature} = {value}")
        text = template.format(feature=c["feature"], value=_fmt_value(c["value"]))
        reasons.append(
            {
                "feature": c["feature"],
                "statement": text,
                "impact_on_p_default": round(float(c["shap"]), 4),
            }
        )
    return {
        "decision": res["decision"],
        "probability": round(res["probability"], 4),
        "reasons": reasons,
    }


def _fmt_value(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"
    return str(v)


# =====================================================================
# Plots
# =====================================================================
def _plot_global_bar(importance: pd.Series, path) -> None:
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("mean |SHAP value|  (impact on P(default))")
    ax.set_title("Global feature importance — SHAP")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_beeswarm(path) -> None:
    """True SHAP beeswarm on the transformed columns (all numeric)."""
    _, preprocessor, _ = load_artifacts()
    df, *_ = _splits()
    explainer = get_explainer()
    matrix = np.asarray(preprocessor.transform(df[FEATURE_COLUMNS]), dtype=float)
    sv = np.asarray(explainer.shap_values(matrix, check_additivity=False))
    names = preprocessor.get_feature_names_out()
    shap.summary_plot(
        sv,
        features=matrix,
        feature_names=names,
        max_display=15,
        show=False,
        plot_size=(9, 6),
    )
    plt.title("SHAP beeswarm - transformed features", fontsize=12, loc="left", pad=12)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def _plot_dependence(feature: str, path) -> None:
    """SHAP value of one numeric feature vs its (original) value."""
    cache = load_cached_shap()
    j = cache["feature_columns"].index(feature)
    x = cache["feature_values"][feature].to_numpy(dtype=float)
    y = cache["agg_shap"][:, j]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(x, y, s=14, alpha=0.5, color="#4C72B0")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(f"Dependence — {feature}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_local(result: dict, title: str, path) -> None:
    rows = result["shap"][:12][::-1]
    labels = [f"{r['feature']} = {_fmt_value(r['value'])}" for r in rows]
    vals = [r["shap"] for r in rows]
    colors = ["#C44E52" if v > 0 else "#55A868" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP contribution to P(default)   (red = toward DENY)")
    ax.set_title(
        f"{title}\nbase {result['base_value']:.3f}  ->  "
        f"P(default) {result['probability']:.3f}  ->  {result['decision']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_lime(result: dict, title: str, path) -> None:
    rows = result["lime"][:10][::-1]
    labels = [r["condition"] for r in rows]
    vals = [r["weight"] for r in rows]
    colors = ["#C44E52" if v > 0 else "#55A868" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LIME weight for class 'default'   (red = toward DENY)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# SHAP vs LIME agreement
# =====================================================================
def compare_shap_lime(n: int = 40, top_k: int = 5) -> dict:
    """Compare SHAP and LIME local rankings on a sample of the test set."""
    _, X_test, *_ = _splits()[1:]
    sample = X_test.sample(min(n, len(X_test)), random_state=RANDOM_SEED)

    jaccard, rank_corr, sign_agree = [], [], []
    for _, applicant in sample.iterrows():
        res = explain_local(applicant, run_lime=True)
        shap_rank = [c["feature"] for c in res["shap"]]
        shap_w = {c["feature"]: c["shap"] for c in res["shap"]}
        lime_w: dict[str, float] = {}
        for item in res["lime"]:
            lime_w[item["feature"]] = lime_w.get(item["feature"], 0.0) + item["weight"]
        lime_rank = sorted(lime_w, key=lambda f: abs(lime_w[f]), reverse=True)

        s_top, l_top = set(shap_rank[:top_k]), set(lime_rank[:top_k])
        union = s_top | l_top
        jaccard.append(len(s_top & l_top) / len(union) if union else 0.0)

        shared = [f for f in shap_rank if f in lime_w]
        if len(shared) >= 3:
            sr = [shap_rank.index(f) for f in shared]
            lr = [lime_rank.index(f) for f in shared]
            rho = spearmanr(sr, lr).correlation
            if not np.isnan(rho):
                rank_corr.append(rho)
            sign_agree.append(
                np.mean([np.sign(shap_w[f]) == np.sign(lime_w[f]) for f in shared])
            )

    summary = {
        "n_applicants": int(len(sample)),
        "top_k": top_k,
        "jaccard_top_k_mean": float(np.mean(jaccard)),
        "jaccard_top_k_std": float(np.std(jaccard)),
        "spearman_rank_corr_mean": float(np.mean(rank_corr)),
        "sign_agreement_mean": float(np.mean(sign_agree)),
    }
    _write_compare_report(summary)
    return summary


def _write_compare_report(summary: dict) -> None:
    lines = [
        "# SHAP vs LIME - local explanation agreement",
        "",
        f"Sample: **{summary['n_applicants']}** applicants from the hold-out test set. "
        f"Agreement measured on the top **{summary['top_k']}** features per applicant.",
        "",
        "| Metric | Value | Reading |",
        "|---|---|---|",
        f"| Jaccard overlap of top-{summary['top_k']} feature sets "
        f"| {summary['jaccard_top_k_mean']:.2f} +/- {summary['jaccard_top_k_std']:.2f} "
        "| 1.0 = identical feature sets |",
        f"| Spearman rank correlation (shared features) "
        f"| {summary['spearman_rank_corr_mean']:.2f} "
        "| 1.0 = identical ordering |",
        f"| Sign agreement (shared features) "
        f"| {summary['sign_agreement_mean']:.2f} "
        "| share of features both methods push the same way |",
        "",
        "SHAP (exact TreeExplainer, probability space) is treated as the reference; "
        "LIME is a sparse local linear surrogate, so partial overlap is expected. "
        "High **sign agreement** with lower rank correlation means the two methods "
        "usually agree on *which way* a feature pushes the decision but not always on "
        "its exact rank - acceptable for reason-code generation, which only uses the "
        "direction and the few largest contributions.",
        "",
    ]
    SHAP_VS_LIME_PATH.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# Entry point
# =====================================================================
def _pick_examples():
    """One clear APPROVE and one clear DENY applicant from the test set."""
    model, preprocessor, _ = load_artifacts()
    _, X_test, *_ = _splits()[1:]
    proba = model.predict_proba(preprocessor.transform(X_test[FEATURE_COLUMNS]))[:, 1]
    order = np.argsort(proba)
    return X_test.iloc[order[0]], X_test.iloc[order[-1]]


def main() -> None:
    print("Computing SHAP values for all 1,000 rows ...")
    compute_and_cache_shap(save=True)
    importance = global_importance()
    print("\nTop 10 features (mean |SHAP|):")
    for f, v in importance.head(10).items():
        print(f"  {f:26} {v:.4f}")

    _plot_global_bar(importance, FIGURES_DIR / "shap_global_importance.png")
    _plot_beeswarm(FIGURES_DIR / "shap_beeswarm.png")
    top_numeric = [f for f in importance.index if f in NUMERIC_FEATURES][:3]
    for f in top_numeric:
        _plot_dependence(f, FIGURES_DIR / f"shap_dependence_{f}.png")
    print(f"\nGlobal figures + dependence plots ({', '.join(top_numeric)}) -> {FIGURES_DIR}")

    approve, deny = _pick_examples()
    for tag, applicant in (("approve", approve), ("deny", deny)):
        res = explain_local(applicant, run_lime=True)
        _plot_local(res, f"Local SHAP — {tag.upper()} example", FIGURES_DIR / f"local_{tag}_shap.png")
        _plot_lime(res, f"Local LIME — {tag.upper()} example", FIGURES_DIR / f"local_{tag}_lime.png")
        rc = reason_codes(applicant, k=4)
        print(f"\n{tag.upper()} example  P(default)={rc['probability']}  -> {rc['decision']}")
        for r in rc["reasons"]:
            print(f"  +{r['impact_on_p_default']:+.3f}  {r['statement']}")

    print("\nComparing SHAP vs LIME on 40 test applicants (LIME is slow) ...")
    summary = compare_shap_lime(n=40)
    print(json.dumps(summary, indent=2))
    print(f"\nReport -> {SHAP_VS_LIME_PATH}")
    print(f"SHAP cache -> {SHAP_CACHE_PATH}")


if __name__ == "__main__":
    main()
