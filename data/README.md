# Dataset — Statlog (German Credit Data)

- **Source:** UCI Machine Learning Repository, dataset id **144**
  <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>
- **Raw file used:** `german.data` (space-separated, coded categorical values)
  <https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data>
- **Rows:** 1,000 loan applicants · **Attributes:** 20 (7 numeric, 13 categorical) · **Target:** credit risk

`src/data.py` downloads `german.data` into `data/raw/` on first run (cached afterwards),
renames the columns, decodes every `Axx` code into a human-readable label, and derives:

| Derived column | From | Meaning |
|---|---|---|
| `default` | `credit_class` (2 → 1, 1 → 0) | **model target** — 1 = applicant defaulted / bad credit risk |
| `sex` | `personal_status_sex` | `male` / `female` — used only for the Phase 4 fairness audit |
| `age_group` | `age_years` | `age < 25` / `age >= 25` — used only for the Phase 4 fairness audit |

`foreign_worker` is both a model feature and a protected attribute for the fairness audit.

The raw file is **git-ignored** — re-run `python -m src.data` (or any training script) to fetch it.

## Cost matrix

The dataset authors note that misclassifying a bad risk as good is **5×** worse than the
reverse. This is not enforced in the Phase 1 baseline but is discussed in the Phase 4 report.
