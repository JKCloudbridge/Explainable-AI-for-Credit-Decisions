"""Train the credit-risk model.

Compares a LogisticRegression baseline against XGBoost with stratified 5-fold
CV, then refits XGBoost on the full training split, evaluates on the hold-out
test set, and persists:

    models/xgboost_model.joblib     fitted XGBClassifier
    models/preprocessor.joblib      fitted ColumnTransformer
    models/model_metadata.json      feature lists, params, split sizes
    reports/metrics.json            CV + test metrics for both models
    reports/figures/*.png           ROC, PR, confusion-matrix plots

Run::

    python -m src.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_NAME,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    PROTECTED_ATTRIBUTES,
    RANDOM_SEED,
    REPORTS_DIR,
    FIGURES_DIR,
    TARGET,
    TEST_SIZE,
)
from src.data import load_data
from src.evaluate import compute_metrics, plot_confusion, plot_pr, plot_roc
from src.preprocess import build_preprocessor, get_feature_names, transform_frame

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    min_child_weight=2,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


def _build_models(scale_pos_weight: float) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "xgboost": XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS),
    }


def main() -> None:
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()
    X_train_t = transform_frame(preprocessor.fit(X_train), X_train)
    X_test_t = transform_frame(preprocessor, X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"Transformed matrix: {X_train_t.shape[1]} columns")

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / pos
    models = _build_models(scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results: dict[str, dict] = {}

    print("\n5-fold CV (ROC-AUC on training split):")
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train_t, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"  {name:20} {scores.mean():.3f} +/- {scores.std():.3f}")
        model.fit(X_train_t, y_train)
        proba = model.predict_proba(X_test_t)[:, 1]
        test_metrics = compute_metrics(y_test, proba, DECISION_THRESHOLD)
        results[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
            "test": test_metrics,
        }

    print("\nHold-out test set:")
    for name, res in results.items():
        t = res["test"]
        print(
            f"  {name:20} AUC {t['roc_auc']:.3f} | AP {t['pr_auc']:.3f} | "
            f"recall {t['recall']:.3f} | precision {t['precision']:.3f} | "
            f"Brier {t['brier']:.3f}"
        )

    # XGBoost is the deployed model for phases 2-4.
    best = models["xgboost"]
    best_proba = best.predict_proba(X_test_t)[:, 1]

    plot_roc(y_test, best_proba, FIGURES_DIR / "roc_curve.png")
    plot_pr(y_test, best_proba, FIGURES_DIR / "pr_curve.png")
    plot_confusion(
        y_test, best_proba, DECISION_THRESHOLD, FIGURES_DIR / "confusion_matrix.png"
    )
    print(f"\nFigures written to {FIGURES_DIR}")

    joblib.dump(best, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "model_type": "XGBClassifier",
        "deployed_model": "xgboost",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET_NAME,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": TARGET,
        "positive_class_meaning": "applicant defaulted / bad credit risk",
        "decision_threshold": DECISION_THRESHOLD,
        "class_balance_train": {"repaid_0": neg, "default_1": pos},
        "scale_pos_weight": scale_pos_weight,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "transformed_feature_names": feature_names,
        "protected_attributes": PROTECTED_ATTRIBUTES,
        "xgb_params": {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    metrics_out = {
        "dataset": DATASET_NAME,
        "decision_threshold": DECISION_THRESHOLD,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(metrics_out, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
