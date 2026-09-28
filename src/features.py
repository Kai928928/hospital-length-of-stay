"""Feature matrix construction: encoding and target preparation.

Two encodings are produced:
  - `X_raw` : numeric features as-is + one-hot encoded categorical features.
              Used by tree-based models.
  - `X_scaled` : the same matrix after standardisation of numeric columns.
              Used by linear models (inside a Pipeline in `models.py`).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CAT_COLS,
    ENGINEERED_CAT_COLS,
    ENGINEERED_NUM_COLS,
    NUM_COLS,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
)
from sklearn.model_selection import train_test_split


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the raw feature matrix (before encoding) for a cleaned frame."""
    feature_cols = (
        NUM_COLS
        + ENGINEERED_NUM_COLS
        + CAT_COLS
        + ENGINEERED_CAT_COLS
    )
    X = df[feature_cols].copy()

    # numeric IDs used as categorical codes
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        X[c] = X[c].astype("category")

    return X


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns; keep numeric columns as-is."""
    num_cols = [c for c in X.columns if X[c].dtype in ("int64", "float64", "int32")]
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    # The engineered ordinal columns are numeric already; keep them numeric.
    num_cols = [c for c in num_cols if c not in ENGINEERED_CAT_COLS]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    X_enc = pre.fit_transform(X)
    feature_names = (
        list(num_cols)
        + list(pre.named_transformers_["cat"].get_feature_names_out(cat_cols))
    )
    return pd.DataFrame(X_enc, columns=feature_names)


def prepare_train_test(df: pd.DataFrame):
    """Split the cleaned frame into (X, y) train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, feature_columns
    """
    X = build_features(df)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=None
    )
    return X_train, X_test, y_train, y_test
