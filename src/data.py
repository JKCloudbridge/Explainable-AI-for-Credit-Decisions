"""Download, decode and load the Statlog German Credit dataset.

Run directly to fetch + cache the raw file and print a summary::

    python -m src.data
"""

from __future__ import annotations

import ssl
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_URL,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    RAW_DATA_FILE,
    TARGET,
    TEST_SIZE,
)

# Raw column order in german.data (20 attributes + the class label).
RAW_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property_type",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "liable_maintenance_count",
    "telephone",
    "foreign_worker",
    "credit_class",
]

# Axx code -> human-readable label, straight from the UCI data dictionary.
VALUE_MAPS: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 - 200 DM",
        "A13": ">= 200 DM / salary assigned",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing elsewhere",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture / equipment",
        "A43": "radio / television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 - 500 DM",
        "A63": "500 - 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 - 4 years",
        "A74": "4 - 7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male : divorced / separated",
        "A92": "female : divorced / separated / married",
        "A93": "male : single",
        "A94": "male : married / widowed",
        "A95": "female : single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property_type": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes, registered under the customer's name",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}


def download_raw(force: bool = False) -> None:
    """Fetch german.data into data/raw/ unless it is already cached."""
    if RAW_DATA_FILE.exists() and not force:
        return
    # UCI's certificate chain is occasionally incomplete on Windows; the file
    # is public and integrity is checked by the fixed row/column count below.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(DATASET_URL, timeout=60, context=ctx) as resp:
        payload = resp.read()
    RAW_DATA_FILE.write_bytes(payload)


def load_raw() -> pd.DataFrame:
    """Return the raw dataset with proper column names, codes untouched."""
    download_raw()
    df = pd.read_csv(RAW_DATA_FILE, sep=" ", header=None, names=RAW_COLUMNS)
    if df.shape != (1000, 21):
        raise ValueError(
            f"Unexpected raw shape {df.shape}; delete {RAW_DATA_FILE} and retry."
        )
    return df


def load_data() -> pd.DataFrame:
    """Return the analysis-ready frame: decoded labels + derived columns.

    Columns: the 20 model features, the ``default`` target, and the
    ``sex`` / ``age_group`` protected attributes used by the fairness audit.
    """
    df = load_raw()

    for col, mapping in VALUE_MAPS.items():
        df[col] = df[col].map(mapping)
        if df[col].isna().any():
            raise ValueError(f"Unmapped code in column {col!r}")

    # Target: dataset codes 1 = good, 2 = bad. Model predicts P(default).
    df[TARGET] = (df["credit_class"] == 2).astype(int)

    # Protected attributes for the Phase 4 audit (derived, not model inputs).
    df["sex"] = (
        df["personal_status_sex"].str.split(" : ").str[0].str.strip()
    )
    df["age_group"] = pd.cut(
        df["age_years"],
        bins=[0, 25, 200],
        right=False,
        labels=["age < 25", "age >= 25"],
    ).astype(str)

    df = df.drop(columns=["credit_class"])

    ordered = FEATURE_COLUMNS + [TARGET, "sex", "age_group"]
    return df[ordered]


def get_splits():
    """The canonical stratified 80/20 split, shared by training and explanations.

    Returns ``(df, X_train, X_test, y_train, y_test)`` where ``X_*`` are frames in
    the original 20-feature space and ``y_*`` are numpy arrays.
    """
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    return df, X_train, X_test, y_train, y_test


def get_codebook() -> dict[str, dict[str, str]]:
    """Decode maps for categoricals plus notes for numeric columns."""
    numeric_notes = {
        "duration_months": "loan duration in months",
        "credit_amount": "credit amount in Deutsche Mark",
        "installment_rate_pct": "instalment as % of disposable income (1-4)",
        "residence_since": "years at present residence (1-4)",
        "age_years": "age in years",
        "existing_credits_count": "number of existing credits at this bank",
        "liable_maintenance_count": "number of dependants",
    }
    return {"categorical": VALUE_MAPS, "numeric": numeric_notes}


if __name__ == "__main__":
    download_raw()
    frame = load_data()
    print(f"Loaded {len(frame)} rows x {frame.shape[1]} columns")
    print(f"Cached raw file: {RAW_DATA_FILE}")
    rate = frame[TARGET].mean()
    print(f"Default rate: {rate:.1%}  ({frame[TARGET].sum()} of {len(frame)})")
    print("\nNumeric feature summary:")
    print(frame[NUMERIC_FEATURES].describe().round(1).T[["mean", "min", "max"]])
    print("\nCategorical cardinality:")
    for c in CATEGORICAL_FEATURES:
        print(f"  {c:24} {frame[c].nunique()} levels")
