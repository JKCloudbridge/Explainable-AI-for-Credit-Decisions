# Fairness audit

Positive class = `default` = 1 = bad risk. "Deny" = predicted `P(default) >= threshold`. Group tables use the full 1,000-row dataset for stable subgroup estimates (see `src/fairness.py` docstring for the caveat); the cost-optimal threshold and the aware-vs-unaware AUC use the 200-row hold-out.

## `sex`

| group   |   n |   actual_default_rate |   approval_rate |   tpr_deny_given_default |   fpr_deny_given_repaid |   auc |   calibration_gap |   disparate_impact_ratio_vs_max |
|:--------|----:|----------------------:|----------------:|-------------------------:|------------------------:|------:|------------------:|--------------------------------:|
| female  | 310 |                 0.352 |           0.577 |                    0.881 |                   0.174 | 0.928 |             0.098 |                           0.883 |
| male    | 690 |                 0.277 |           0.654 |                    0.901 |                   0.134 | 0.942 |             0.092 |                           1     |

Minimum disparate-impact ratio: **0.88** — within the four-fifths rule.

## `age_group`

| group     |   n |   actual_default_rate |   approval_rate |   tpr_deny_given_default |   fpr_deny_given_repaid |   auc |   calibration_gap |   disparate_impact_ratio_vs_max |
|:----------|----:|----------------------:|----------------:|-------------------------:|------------------------:|------:|------------------:|--------------------------------:|
| age < 25  | 149 |                 0.409 |           0.503 |                    0.852 |                   0.25  | 0.885 |             0.094 |                           0.772 |
| age >= 25 | 851 |                 0.281 |           0.652 |                    0.904 |                   0.131 | 0.947 |             0.094 |                           1     |

Minimum disparate-impact ratio: **0.77** — ⚠️ **below the 0.8 four-fifths threshold**.

## `foreign_worker`

| group   |   n |   actual_default_rate |   approval_rate |   tpr_deny_given_default |   fpr_deny_given_repaid |   auc |   calibration_gap |   disparate_impact_ratio_vs_max |
|:--------|----:|----------------------:|----------------:|-------------------------:|------------------------:|------:|------------------:|--------------------------------:|
| no      |  37 |                 0.108 |           0.865 |                    1     |                   0.03  | 1     |             0.071 |                           1     |
| yes     | 963 |                 0.307 |           0.621 |                    0.892 |                   0.151 | 0.936 |             0.095 |                           0.718 |

Minimum disparate-impact ratio: **0.72** — ⚠️ **below the 0.8 four-fifths threshold**.

## Cost-optimal threshold (hold-out test set)

German Credit's documented cost matrix weights a missed bad risk (5x) far above a wrongly denied good risk (1x). At the default threshold (0.50) the test-set cost is **124**. The cost-minimising threshold is **0.29**, cost **107** (14% lower). Lowering the threshold denies more applicants overall — re-run the group tables at this threshold before adopting it, since a lower threshold typically widens approval-rate gaps.

## Protected-attribute-free comparison

A second XGBoost model was trained on the identical split, dropping `personal_status_sex`, `age_years` ("unaware").

| variant                                        |   test_auc |   sex approval gap |   sex min DI ratio |   age_group approval gap |   age_group min DI ratio |   foreign_worker approval gap |   foreign_worker min DI ratio |
|:-----------------------------------------------|-----------:|-------------------:|-------------------:|-------------------------:|-------------------------:|------------------------------:|------------------------------:|
| aware (all 20 features)                        |      0.784 |              0.076 |              0.883 |                    0.149 |                    0.772 |                         0.244 |                         0.718 |
| unaware (drops personal_status_sex, age_years) |      0.796 |              0.073 |              0.887 |                    0.146 |                    0.775 |                         0.246 |                         0.716 |

Dropping `personal_status_sex` and `age_years` does not cost accuracy (test AUC actually 0.012 higher, well within the noise of a 200-row test set), and does **not** eliminate the approval-rate gaps — they shrink only slightly, because the model can still reconstruct some of that signal from correlated features (e.g. `job`, `credit_history`, `savings_status`). This is classic proxy discrimination: removing a protected attribute from the input is necessary but not sufficient for fairness, and is exactly why per-group *outcome* audits (like the tables above) matter more than checking which columns the model sees.

## Limitations

- `foreign_worker = no` is 37 of 1,000 applicants (6 of 200 in the test split) — too small for a statistically reliable disparate-impact conclusion; reported for completeness, not as a finding to act on.
- The fairness table mixes training and test rows (see above) — a production audit would use a held-out or post-deployment sample.
- German Credit is a 1990s benchmark in Deutsche Mark; `personal_status_sex` conflates sex with marital status and cannot be cleanly disentangled in this dataset.