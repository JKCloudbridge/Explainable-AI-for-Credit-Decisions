# Explainable AI for Credit / Loan Decisions

A credit-risk model you can **interrogate**: per-applicant explanations, what-if
analysis, and a fairness audit — framed around how lenders must justify adverse
decisions under fair-lending rules (ECOA / Regulation B, EU AI Act).

- **Model:** scikit-learn `LogisticRegression` baseline vs **XGBoost** (deployed)
- **Explainability:** SHAP (global + local) and LIME (local)
- **Fairness:** group audit (demographic parity, disparate impact, equal opportunity) + a protected-attribute-free comparison model
- **App:** Streamlit — 5 pages, live threshold, what-if analysis, fairness audit
- **Data:** [Statlog German Credit](data/README.md) — 1,000 applicants, 20 attributes, UCI id 144

**Headline finding:** the model fails the four-fifths disparate-impact rule on
`age_group` (0.77) and `foreign_worker` (0.72, small-sample caveat) — and
removing `sex`/`age` from its inputs barely narrows the gap, because the model
reconstructs that signal from correlated features. See [MODEL_CARD.md](MODEL_CARD.md)
and [reports/fairness_report.md](reports/fairness_report.md).

---

## Status

| Phase | Scope | State |
|---|---|---|
| **1. Data & baseline model** | reproducible pipeline, LR vs XGBoost, evaluation, persisted artefacts | ✅ done |
| **2. Explainability engine** | SHAP + LIME module, global & local explanations, reason codes, agreement analysis | ✅ done |
| **3. Streamlit decision app** | scoring, local explanations, what-if sliders, global insights | ✅ done |
| **4. Fairness audit + deploy** | group fairness metrics, adverse-action notices, model card, tests, Streamlit Cloud | ✅ done |

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

## Run Phase 3 — the app

```bash
streamlit run streamlit_app.py
```

Five pages (sidebar navigation):

| Page | What it does |
|---|---|
| **Home** | dataset + model summary |
| **1 · Score an applicant** | pick a test-set row or fill in a form; set the decision threshold |
| **2 · Why this decision** | SHAP + LIME charts, plain-language reason codes, and (if denied) an adverse-action notice |
| **3 · What-if** | sliders for the 8 most influential features, live re-scoring vs the original |
| **4 · Global insights** | model performance + the Phase 2 SHAP figures |
| **5 · Fairness audit** | group metrics by protected attribute at your chosen threshold, cost-optimal-threshold finder, aware-vs-unaware comparison |

The current applicant and threshold are held in `st.session_state` and carry over
between pages — the Fairness page's tables and the "Why this decision" notice
both use whatever threshold you set on page 1. All heavy lifting (model,
SHAP/LIME explainers) is cached at the process level (`functools.lru_cache` in
`src/explain.py`, `st.cache_resource` / `st.cache_data` in `app_lib/common.py`),
so only the *first* explanation on a freshly-started server is slow.

## Run Phase 4

```bash
python -m src.fairness          # group audit + cost-optimal threshold + aware-vs-unaware
python -m src.adverse_action    # prints a worked APPROVE + DENY notice
pip install -r requirements-dev.txt && pytest   # 21 smoke tests across data/preprocess/explain/fairness/adverse_action
```

| File | Purpose |
|---|---|
| `src/fairness.py` | `group_metrics()`, `run_fairness_audit()`, `cost_optimal_threshold()`, `compare_aware_vs_unaware()` |
| `src/adverse_action.py` | `generate_notice()` — ECOA/Reg-B-style statement of reasons (demonstration) |
| `MODEL_CARD.md` | intended use, training data, evaluation, fairness findings, limitations |
| `reports/report_notes.md` | the report write-up: global vs local explanations, fairness, regulatory framing |
| `reports/fairness_report.md` / `fairness_metrics.json` | the fairness audit's numbers |
| `models/*_unaware.joblib` | the protected-attribute-free comparison model |

**Fairness headline** (full dataset, threshold 0.5): `sex` passes the four-fifths
rule (DI ratio 0.88); `age_group` (0.77) and `foreign_worker` (0.72, n=37 in the
disadvantaged group) fail it. Dropping `personal_status_sex` + `age_years` from
the model costs no test accuracy but barely narrows the gaps — proxy features
(`job`, `credit_history`, `savings_status`) carry much of the same signal.
**Cost-optimal threshold** (200-row test set, the dataset's 5:1 cost matrix):
0.29 vs the default 0.5, 14% lower cost — but a lower threshold denies more
applicants and tends to widen these gaps, so it isn't adopted as the new
default without re-running the fairness tables (the app's Fairness page does
this live).

### Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (already done: `JKCloudbridge/Explainable-AI-for-Credit-Decisions`).
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **New app**.
3. Repository: this repo · Branch: `main` · Main file path: `streamlit_app.py`.
4. Python version: 3.11 (read from `.python-version`; override in the app's
   "Advanced settings" if the platform doesn't pick it up automatically).
5. Deploy. First load downloads the dataset from UCI into the app's ephemeral
   storage (`src/data.py:download_raw()`) — everything else (model, SHAP cache,
   figures) is already committed, so no retraining happens on the server.
6. No secrets are required — the app reads no API keys or credentials.

*(Manual GitHub-OAuth step — not something that can be scripted from here.
Once deployed, add the live URL to this README.)*

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
  fairness.py     group audit, cost-optimal threshold, aware-vs-unaware comparison
  adverse_action.py  ECOA/Reg-B-style adverse-action notice (demonstration)
app_lib/
  common.py       cached loaders, session-state, applicant-form widget builder
streamlit_app.py  app entry point (Home page) — `streamlit run streamlit_app.py`
pages/            the other 4 app pages (Streamlit's file-based multipage routing)
tests/            pytest smoke tests — data, preprocess, explain, fairness, adverse_action
data/             dataset notes; raw file downloaded on first run (git-ignored)
models/           trained artefacts, incl. shap_values.joblib + *_unaware.joblib (committed)
reports/          metrics.json, shap_vs_lime.md, fairness_report.md, report_notes.md, figures/
Phases/           Plan + one completion report per phase
MODEL_CARD.md     intended use, data, evaluation, fairness findings, limitations
```

## Notes

- The model is trained on all 20 attributes, **including** `personal_status_sex`
  and `age_years`. This is deliberate: Phase 4 audits the resulting disparities
  by `sex`, `age_group`, and `foreign_worker` and contrasts them with a
  protected-attribute-free variant (see [MODEL_CARD.md](MODEL_CARD.md)).
- German Credit is a benchmark, not production data (Deutsche Mark, 1990s). It is
  used here to demonstrate method, not to make real lending decisions.
- Nothing in this repository — the model, the adverse-action notice template,
  or the fairness audit — should be used to make a decision about a real person.
