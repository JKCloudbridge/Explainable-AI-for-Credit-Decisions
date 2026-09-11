"""Smoke tests for src/data.py."""

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from src.data import get_splits, load_data


def test_load_data_shape_and_types():
    df = load_data()
    assert df.shape == (1000, 23)
    assert set(FEATURE_COLUMNS + [TARGET, "sex", "age_group"]) == set(df.columns)
    assert df[TARGET].isin([0, 1]).all()
    assert not df[FEATURE_COLUMNS].isna().any().any()


def test_categoricals_are_decoded_not_raw_codes():
    df = load_data()
    for col in CATEGORICAL_FEATURES:
        values = set(df[col].unique())
        assert not any(str(v).startswith("A") and str(v)[1:].isdigit() for v in values), (
            f"{col} still has raw Axx codes: {values}"
        )


def test_numeric_features_are_numeric():
    df = load_data()
    for col in NUMERIC_FEATURES:
        assert df[col].dtype.kind in "if"


def test_derived_protected_attributes():
    df = load_data()
    assert set(df["sex"].unique()) == {"male", "female"}
    assert set(df["age_group"].unique()) == {"age < 25", "age >= 25"}


def test_get_splits_is_stratified_and_reproducible():
    df, X_train, X_test, y_train, y_test = get_splits()
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert abs(y_train.mean() - y_test.mean()) < 0.05  # stratified split

    # Calling again must reproduce the exact same split (fixed random_state).
    _, X_train2, X_test2, _, _ = get_splits()
    assert list(X_train.index) == list(X_train2.index)
    assert list(X_test.index) == list(X_test2.index)
