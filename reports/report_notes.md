# Report notes — global vs local explanations, fairness, regulatory framing

Working notes for the dissertation write-up. Draws on `reports/metrics.json`,
`reports/shap_vs_lime.md`, `reports/fairness_metrics.json` /
`reports/fairness_report.md`, and `MODEL_CARD.md` — cite those for full figures.

---

## 1. Problem framing

Credit-scoring models are a canonical "high-stakes, opaque model" case: the
decision materially affects a person, the model is usually a black box to that
person, and (in the US, EU, and elsewhere) the law requires the lender to be
able to say *why* an application was denied. This project builds a model
(XGBoost on the Statlog German Credit dataset), then asks three questions of it:

1. **Global** — what does the model rely on, in aggregate?
2. **Local** — why did *this* applicant get *this* decision, and do two
   independent explanation methods (SHAP, LIME) agree?
3. **Fair** — does the model treat protected groups equivalently, and does
   removing protected attributes from its inputs actually fix that?

## 2. Global explanations

Mean |SHAP| ranks `checking_status` far above everything else, followed by
`duration_months`, `credit_amount`, `credit_history`, `savings_status`,
`purpose` (`reports/figures/shap_global_importance.png`). This matches the
published German Credit literature and is a useful sanity check: a model that
instead leaned on, say, `telephone` or `liable_maintenance_count` would be a
red flag.

The beeswarm (`shap_beeswarm.png`) adds direction: a "no checking account"
applicant is pushed *down* in risk (SHAP < 0) relative to "< 0 DM" (pushed up)
— consistent with domain intuition, since "no checking account" partly reflects
applicants who simply don't bank with this institution rather than applicants
in financial distress.

**A counter-intuitive finding worth flagging explicitly in the report:**
`credit_history = "no credits taken / all paid back duly"` has a *positive*
SHAP value (pushes toward default) — a worse signal than "critical account /
other credits existing elsewhere". This is a known property of this dataset:
applicants with no credit history simply have no track record to vouch for
them, whereas applicants who have already carried and repaid other credit
have demonstrated repayment behaviour. It is a legitimate pattern in the data,
not a model bug — but it is exactly the kind of counter-intuitive driver a
global-importance number alone would hide, and that a beeswarm / dependence
plot surfaces.

Global explanations answer "what does the model generally do" — they do not
tell an individual applicant why *they* were denied, which motivates local
explanation.

## 3. Local explanations: SHAP vs LIME

For any one applicant, SHAP's `TreeExplainer` gives an **exact** decomposition:
`base_value + Σ(per-feature contribution) = P(default)`, verified in
`tests/test_explain.py::test_shap_additivity`. LIME instead fits a sparse
linear surrogate around that one point by perturbing it and re-scoring.

Worked example — the test-set applicant with the highest predicted risk
(`P(default) = 0.968`): both methods agree on 4 of the top 5 factors
(`checking_status`, `credit_history`, `duration_months`, `credit_amount`),
though they rank them slightly differently and LIME's raw weights are larger
in magnitude (linear surrogates tend to overstate a locally-dominant feature).

Systematically, over 40 hold-out applicants (`reports/shap_vs_lime.md`):

| Metric | Value |
|---|---|
| Sign agreement on shared top features | **0.97** |
| Top-5 feature-set Jaccard overlap | 0.43 |
| Spearman rank correlation | 0.37 |

**Interpretation for the report:** SHAP and LIME almost always agree on
*direction* — whether a feature helps or hurts the applicant — and disagree
more on exact ranking and on which features make the "top 5" at all. This is
expected: SHAP is an exact game-theoretic attribution against a fixed
background; LIME is a local linear approximation whose feature selection is
itself a modelling choice. For an adverse-action notice, where only direction
and the few largest contributions matter, this level of agreement is
sufficient; it would *not* be sufficient for a use case that needed a precise,
single ranking of reasons.

## 4. Fairness

Evaluated across three protected attributes on the full 1,000-row dataset
(subgroup-size caveat below), at the default 0.5 threshold:

| Attribute | Min disparate-impact ratio | 4/5ths rule |
|---|---|---|
| `sex` (female vs male) | 0.88 | pass |
| `age_group` (< 25 vs ≥ 25) | **0.77** | **fail** |
| `foreign_worker` (yes vs no) | **0.72** | **fail**, but `no` is n=37 |

The `age_group` finding is the more defensible of the two failures: 149
applicants under 25, a real (if smaller) test-set subgroup too, and a
plausible mechanism (younger applicants have shorter credit histories, fewer
existing accounts, similar to the `checking_status`/`credit_history` global
drivers above — i.e., age likely correlates with the very features the model
already leans on most). The `foreign_worker` finding rests on only 37
non-foreign-worker applicants and should be reported with that caveat
front and centre, not as a headline result on its own.

**Does removing the protected attribute fix it?** A second model, trained
identically minus `personal_status_sex` and `age_years`, was compared
(`reports/fairness_report.md`, "Protected-attribute-free comparison"): test
AUC did not drop (in fact +0.012, within test-set noise), but the
disparate-impact ratios and approval-rate gaps barely moved. The model
recovers much of the missing signal from correlated features — `job`,
`credit_history`, `savings_status` all correlate with age and its proxies.
**This is the report's central fairness argument**: *fairness through
unawareness* (dropping a protected column) is a weak intervention on its own;
an outcome-level audit — measuring actual approval rates and error rates by
group, regardless of what the model was fed — is what actually detects and
would need to fix this.

**Cost-optimal threshold and fairness interact.** The dataset's 5:1 cost
matrix (a missed bad risk costs 5x a wrongly-denied good one) makes 0.29 the
cost-minimising threshold on the test set, 14% cheaper than the default 0.5.
Lowering the threshold denies more applicants across the board — the report
should note that adopting a cost-optimal threshold without re-checking the
fairness tables risks *worsening* disparate impact, since a stricter cutoff
tends to widen (not narrow) the gap between an already-disadvantaged group and
the reference group.

## 5. Regulatory framing

- **ECOA / Regulation B (US), 12 CFR 1002.9** — requires a creditor to give an
  applicant the *specific, principal reasons* for a denial. `reason_codes()`
  (Phase 2) and `src/adverse_action.py` (Phase 4) demonstrate how a model's
  local SHAP attribution maps directly onto this requirement: the top adverse
  features become the "principal reasons" statement. This is the clearest
  practical link between an XAI technique and a specific legal obligation.
- **Disparate impact** — US fair-lending doctrine (and the EEOC's four-fifths
  rule used here as a screening heuristic) evaluates *outcomes* by protected
  group regardless of intent; Section 4 above is structured directly around
  this framework.
- **EU AI Act** — creditworthiness assessment is listed as a high-risk AI use
  case (Annex III), which brings obligations around risk management, data
  governance, technical documentation, human oversight, and — most relevant
  here — the ability to provide "clear and meaningful information" about the
  logic of automated decisions. This project's Streamlit "Why this decision"
  page and the model card are a small-scale analogue of that documentation
  obligation.
- **Model cards** (`MODEL_CARD.md`) are increasingly treated as the baseline
  documentation artefact regulators and auditors expect for exactly this
  reason — intended use, training data, evaluation, and known fairness
  limitations in one place.

## 6. Limitations to state plainly in the report

- German Credit is a 1990s benchmark (Deutsche Mark), not a real, current
  lending portfolio — findings demonstrate *method*, not a transferable
  real-world conclusion.
- The fairness table draws on the full dataset rather than a held-out-only
  sample, trading methodological purity for per-subgroup stability with such
  a small dataset; flagged in `src/fairness.py` and `reports/fairness_report.md`.
- `personal_status_sex` conflates sex and marital status; cannot be
  disentangled in this data.
- `src/adverse_action.py` produces a structurally realistic but legally
  non-binding demonstration notice — not reviewed by counsel, not tied to a
  real institution.
