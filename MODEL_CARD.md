# Model Card — Explainable Credit-Risk Classifier

Format loosely follows Mitchell et al., *Model Cards for Model Reporting* (2019).
This is a **capstone / educational project**, not a production credit model.

## Model details

- **Developed by:** capstone project, M3 "Explainable AI for Credit / Loan Decisions"
- **Model type:** `XGBClassifier` (gradient-boosted trees), 300 estimators, max depth 3
- **Input:** 20 applicant attributes → `ColumnTransformer` (StandardScaler + one-hot) → XGBoost
- **Output:** `P(default)` ∈ [0, 1]; decision = DENY if `P(default) ≥ threshold` (default 0.5)
- **Explainability:** SHAP `TreeExplainer` (exact, probability space) for every prediction;
  LIME as a second, independent local explanation
- **Repository:** `github.com/JKCloudbridge/Explainable-AI-for-Credit-Decisions`
- **License / use:** educational only — see [Intended use](#intended-use)

## Intended use

**Intended:** demonstrating explainable-AI and fairness-auditing techniques for a
credit-scoring use case, in a university capstone report and its accompanying
Streamlit app. Every screen, notice, and figure is clearly a demonstration.

**Not intended:** making, or supporting, a real lending decision about a real
person. The training data is a 1990s benchmark dataset in Deutsche Mark; the
model has never been validated against a real credit portfolio, a real
regulator, or real-world drift.

## Training data

[Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
UCI Machine Learning Repository, id 144. 1,000 applicants, 20 attributes, binary
label (700 good / 300 bad credit risk). See `data/README.md` for the full
codebook and the derived `sex` / `age_group` protected attributes. Split 80/20
(stratified, seed 42) into train/test; see `src/data.py:get_splits()`.

## Evaluation

Hold-out test set (n = 200), decision threshold 0.5:

| Model | ROC-AUC | PR-AUC | Recall | Precision | Brier |
|---|---|---|---|---|---|
| LogisticRegression | 0.806 | 0.633 | 0.80 | 0.56 | 0.182 |
| **XGBoost (deployed)** | 0.784 | 0.619 | 0.70 | 0.55 | 0.180 |

XGBoost is deployed for its higher, lower-variance 5-fold cross-validated AUC
(0.790 ± 0.035 vs 0.769 ± 0.053) and because SHAP's `TreeExplainer` gives exact
attributions for tree models. Full metrics: `reports/metrics.json`.

**Cost-sensitivity.** German Credit's documented cost matrix weights a missed
bad risk 5x above a wrongly denied good risk. At threshold 0.5 the test-set
cost is 124; the cost-minimising threshold is **0.29** (cost 107, 14% lower).
See `reports/fairness_report.md`.

## Explainability

- **Global:** `checking_status` is by far the strongest driver, followed by
  `duration_months`, `credit_amount`, `credit_history`, `savings_status`.
  `personal_status_sex` ranks 9th of 20 (mean |SHAP| 0.027) — material enough
  to warrant the fairness audit below. See `reports/figures/shap_*.png`.
- **Local:** every prediction ships with an exact SHAP decomposition (base
  value + per-feature contributions = the prediction) and an independent LIME
  explanation. On a 40-applicant sample, the two agree on the *direction* of a
  feature's effect 97% of the time (`reports/shap_vs_lime.md`).
- **Reason codes / adverse-action notices:** `src/adverse_action.py` turns the
  top adverse SHAP features into an ECOA/Regulation-B-style statement of
  reasons for a denial (demonstration only — see the module docstring).

## Fairness

Evaluated across `sex`, `age_group` (< 25 / ≥ 25), and `foreign_worker`
(`reports/fairness_report.md`, `python -m src.fairness`):

| Attribute | Disadvantaged group | Min disparate-impact ratio | 4/5ths rule |
|---|---|---|---|
| `sex` | female | 0.88 | ✅ pass |
| `age_group` | age < 25 | 0.77 | ❌ **fail** |
| `foreign_worker` | yes | 0.72 | ❌ **fail** (but `no` is only n=37 — low confidence) |

A protected-attribute-free variant (drops `personal_status_sex`, `age_years`)
was trained on the same split for comparison: it does **not** cost test-set
accuracy, and only marginally narrows the gaps above — the model reconstructs
much of that signal from correlated features (`job`, `credit_history`,
`savings_status`). **Removing a protected attribute from the model's inputs is
necessary but not sufficient for fairness**; outcome audits like the one here
are what actually catch it.

## Ethical considerations & limitations

- `age_group` and `foreign_worker` show disparate impact below the 4/5ths
  guideline on this dataset. In a real deployment this would require a
  documented business-necessity justification or a less-disparate alternative
  model, under a disparate-impact framework (e.g. US fair-lending law).
- `foreign_worker = no` is 37 of 1,000 rows (6 of 200 in the test split) — too
  small for a statistically confident fairness conclusion; reported for
  completeness, not as an actionable finding.
- The fairness table mixes rows the model was trained on with held-out rows
  (done for per-group sample-size stability); a production audit would use a
  held-out or post-deployment sample exclusively.
- `personal_status_sex` conflates sex and marital status in the raw data and
  cannot be cleanly disentangled.
- The dataset is small (1,000 rows), decades old, and in a different currency
  and regulatory regime than any current market — findings illustrate *method*,
  not a transferable real-world conclusion.
- No monitoring for data or concept drift exists or is implied.

## Recommendations for anyone extending this project

1. Do not deploy this model, or the pattern of choices here, against real
   applicants without re-validating on representative, current data.
2. Treat the disparate-impact findings on `age_group` and `foreign_worker` as
   the headline result of this audit, not a footnote.
3. Prefer outcome-based fairness audits over "protected attribute removed"
   claims — this project's own comparison shows why.
4. If a regulator-facing adverse-action notice is ever needed for real,
   involve compliance / legal counsel — `src/adverse_action.py` is a
   structural demonstration, not compliant legal text.
