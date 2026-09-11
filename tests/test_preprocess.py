"""Smoke tests for src/preprocess.py and the persisted Phase 1 model artefacts."""

import joblib
import numpy as np
import pytest

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
)
from src.data import get_splits
from src.preprocess import build_preprocessor, get_feature_names, transform_frame


@pytest.fixture(scope="module")
def splits():
    return get_splits()


def test_build_preprocessor_default_shape(splits):
    _, X_train, _, _, _ = splits
    pre = build_preprocessor()
    matrix = transform_frame(pre.fit(X_train), X_train)
    n_onehot_cols = sum(pre.named_transformers_["cat"].named_steps["onehot"].categories_[i].size for i in range(len(CATEGORICAL_FEATURES)))
    assert matrix.shape == (len(X_train), len(NUMERIC_FEATURES) + n_onehot_cols)
    assert not np.isnan(matrix).any()
    assert len(get_feature_names(pre)) == matrix.shape[1]


def test_build_preprocessor_restricted_feature_list(splits):
    """Used by the Phase 4 protected-attribute-free variant."""
    _, X_train, _, _, _ = splits
    num = [f for f in NUMERIC_FEATURES if f != "age_years"]
    cat = [f for f in CATEGORICAL_FEATURES if f != "personal_status_sex"]
    pre = build_preprocessor(numeric_features=num, categorical_features=cat)
    matrix = pre.fit_transform(X_train[num + cat])
    assert matrix.shape[0] == len(X_train)
    assert "age_years" not in get_feature_names(pre)
    assert not any(name.startswith("personal_status_sex") for name in get_feature_names(pre))


@pytest.fixture(scope="module")
def model_bundle():
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


def test_persisted_model_predicts_valid_probabilities(model_bundle, splits):
    model, preprocessor = model_bundle
    _, _, X_test, _, _ = splits
    proba = model.predict_proba(preprocessor.transform(X_test[FEATURE_COLUMNS]))[:, 1]
    assert proba.shape == (len(X_test),)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert proba.std() > 0  # not a degenerate constant predictor
