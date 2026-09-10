# Explainable AI for Credit / Loan Decisions

A credit-risk model you can **interrogate**: per-applicant explanations, what-if
analysis, and a fairness audit — framed around how lenders must justify adverse
decisions under fair-lending rules (ECOA / Regulation B, EU AI Act).

- **Model:** scikit-learn `LogisticRegression` baseline vs **XGBoost** (deployed)
- **Explainability:** SHAP (global + local) and LIME (local)
- **App:** Streamlit
- **Data:** [Statlog German Credit](data/README.md) — 1,000 applicants, 20 attributes, UCI id 144

---

## Status

| Phase | Scope | State |
|---|---|---|
| **1. Data & baseline model** | reproducible pipeline, LR vs XGBoost, evaluation, persisted artefacts | ✅ done |
| **2. Explainability engine** | SHAP + LIME module, global & local explanations, reason codes, agreement analysis | ✅ done |
| 3. Streamlit decision app | scoring, local explanations, what-if sliders, global insights | ⬜ next |
| 4. Fairness audit + deploy | group fairness metrics, adverse-action codes, model card, Streamlit Cloud | ⬜ |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Python 3.11. `numpy` is pinned `< 2` for Streamlit 1.39 compatibility.

## Run Phase 1

```bash
python -m src.data      # download + decode the dataset (cached in data/raw/)
python -m src.train     # train, evaluate, write models/ + reports/
```

`src/train.py` writes:

| Path | Contents |
|---|---|
| `models/xgboost_model.joblib` | fitted `XGBClassifier` (the deployed model) |
| `models/preprocessor.joblib` | fitted `ColumnTransformer` (scale + one-hot) |
| `models/model_metadata.json` | feature lists, params, split sizes, threshold |
| `reports/metrics.json` | CV + hold-out metrics for both models |
| `reports/figures/*.png` | ROC, precision-recall, confusion matrix |

## Phase 1 results (hold-out test set, n = 200, threshold 0.5)

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Recall | Precision | Brier |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.769 ± 0.053 | 0.806 | 0.633 | 0.80 | 0.56 | 0.182 |
| **XGBoost** (deployed) | **0.790 ± 0.035** | 0.784 | 0.619 | 0.70 | 0.55 | 0.180 |

XGBoost is carried forward: higher and more stable cross-validated AUC, and its
tree structure lets SHAP's `TreeExplainer` produce **exact** Shapley values in
Phase 2. On the 200-row test split the two models are within noise of each other.
`recall` = share of true defaulters the model would have denied; the dataset
authors weight a missed bad risk as 5× costlier than a rejected good one, so the
decision threshold is revisited in Phase 4.

## Run Phase 2

```bash
python -m src.explain   # SHAP values + global figures + 2 local examples + SHAP-vs-LIME report
```

`src/explain.py` works in the original 20-feature space (one-hot columns summed
back to their parent feature) and computes SHAP in **probability space** — a
contribution of `+0.08` means the feature raised `P(default)` by 0.08.

| Function | Purpose |
|---|---|
| `explain_local(applicant)` | score + SHAP contributions + LIME weights for one applicant |
| `reason_codes(applicant, k)` | the k features pushing hardest toward DENY, in plain language (feeds the Phase 4 adverse-action notice) |
| `global_importance()` | mean \|SHAP\| per feature |
| `compare_shap_lime(n)` | SHAP-vs-LIME agreement over a test-set sample → `reports/shap_vs_lime.md` |

Writes: `models/shap_values.joblib` (cached values for all 1,000 rows),
`reports/shap_vs_lime.md`, and `reports/figures/shap_*.png` +
`reports/figures/local_{approve,deny}_{shap,lime}.png`.

**Global drivers (mean |SHAP|):** `checking_status` ≫ `duration_months` >
`credit_amount` > `credit_history` > `savings_status` > `purpose`.
**SHAP vs LIME:** sign agreement **0.97**, top-5 Jaccard 0.43 — the methods almost
always agree on *direction*, less so on exact rank.

## Layout

```
src/
  config.py       paths, seed, feature lists, artefact locations
  data.py         download + decode German Credit; derive target & protected attrs;
                  get_splits() — the canonical stratified 80/20 split
  preprocess.py   ColumnTransformer (StandardScaler + dense OneHotEncoder)
  evaluate.py     metrics + ROC / PR / confusion-matrix plots
  train.py        LR vs XGBoost, 5-fold CV, persist model + preprocessor + metrics
  explain.py      SHAP (TreeExplainer, prob space) + LIME; global/local/reason codes
data/             dataset notes; raw file downloaded on first run (git-ignored)
models/           trained artefacts + shap_values.joblib (committed)
reports/          metrics.json, shap_vs_lime.md, figures/
Phases/           Plan + one completion report per phase
```

## Notes

- The model is trained on all 20 attributes, **including** `personal_status_sex`
  and `age_years`. This is deliberate: Phase 4 audits the resulting disparities
  by `sex`, `age_group`, and `foreign_worker` and contrasts them with a
  protected-attribute-free variant.
- German Credit is a benchmark, not production data (Deutsche Mark, 1990s). It is
  used here to demonstrate method, not to make real lending decisions.
