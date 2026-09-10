"""Central configuration: paths, constants, feature lists.

Every other module imports from here so that phases 2-4 (SHAP/LIME, the Streamlit
app, the fairness audit) stay in lock-step with how the model was trained.
"""

from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset -------------------------------------------------------------
DATASET_NAME = "Statlog (German Credit Data) — UCI id 144"
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
RAW_DATA_FILE = RAW_DATA_DIR / "german.data"

# --- Modelling ---------------------------------------------------------
TARGET = "default"  # 1 = applicant defaulted (bad risk), 0 = repaid (good risk)
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5  # P(default) at or above this => loan denied

# 7 numeric attributes
NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since",
    "age_years",
    "existing_credits_count",
    "liable_maintenance_count",
]

# 13 categorical attributes (values are decoded to readable labels in data.py)
CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property_type",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Sliced on in the Phase 4 fairness audit (not all are model inputs).
PROTECTED_ATTRIBUTES = ["sex", "age_group", "foreign_worker"]

# --- Artefacts -------------------------------------------------------
MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --- Explainability (Phase 2) ---------------------------------------
# SHAP explanations run in probability space so a contribution reads as
# "this feature added +0.08 to P(default)".
EXPLAIN_BACKGROUND_SIZE = 200  # training rows used as the SHAP baseline
SHAP_CACHE_PATH = MODELS_DIR / "shap_values.joblib"
SHAP_VS_LIME_PATH = REPORTS_DIR / "shap_vs_lime.md"
