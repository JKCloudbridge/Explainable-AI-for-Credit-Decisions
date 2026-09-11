"""Feature preprocessing: a scikit-learn ColumnTransformer.

Numeric columns are standardised; categoricals are one-hot encoded with a
dense output so that SHAP's TreeExplainer (Phase 2) can consume the matrix
directly. The fitted transformer is persisted alongside the model.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer.

    Defaults to all 20 credit features. Phase 4 passes a restricted feature
    list to train the protected-attribute-free ("unaware") comparison model
    (see ``src/fairness.py``).
    """
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = (
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable names for the transformed matrix columns (post-fit)."""
    return preprocessor.get_feature_names_out().tolist()


def transform_frame(preprocessor: ColumnTransformer, frame) -> np.ndarray:
    """Apply a fitted preprocessor and return a plain float array."""
    return np.asarray(preprocessor.transform(frame), dtype=float)
