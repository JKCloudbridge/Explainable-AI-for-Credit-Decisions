# SHAP vs LIME - local explanation agreement

Sample: **40** applicants from the hold-out test set. Agreement measured on the top **5** features per applicant.

| Metric | Value | Reading |
|---|---|---|
| Jaccard overlap of top-5 feature sets | 0.43 +/- 0.16 | 1.0 = identical feature sets |
| Spearman rank correlation (shared features) | 0.37 | 1.0 = identical ordering |
| Sign agreement (shared features) | 0.97 | share of features both methods push the same way |

SHAP (exact TreeExplainer, probability space) is treated as the reference; LIME is a sparse local linear surrogate, so partial overlap is expected. High **sign agreement** with lower rank correlation means the two methods usually agree on *which way* a feature pushes the decision but not always on its exact rank - acceptable for reason-code generation, which only uses the direction and the few largest contributions.
