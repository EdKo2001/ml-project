"""Basic preprocessing helpers for the shared pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_CANDIDATES = ["Machine failure", "machine_failure", "target", "failure"]
DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def infer_target_column(df: pd.DataFrame, candidates: Iterable[str] = TARGET_CANDIDATES) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError("Could not find a target column. Check the dataset column names.")


def prepare_features_and_target(df: pd.DataFrame, target_column: str | None = None):
    target_column = target_column or infer_target_column(df)
    drop_columns = [column for column in DROP_COLUMNS if column in df.columns and column != target_column]
    X = df.drop(columns=[target_column] + drop_columns, errors="ignore")
    y = df[target_column]
    return X, y, target_column, drop_columns


def split_data(
    df: pd.DataFrame,
    target: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X, y, _, _ = prepare_features_and_target(df, target)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def make_shared_preprocessing_pipeline(df: pd.DataFrame, target_column: str | None = None):
    X, y, target_column, drop_columns = prepare_features_and_target(df, target_column)
    preprocessor = build_preprocessor(X)
    return {
        "target_column": target_column,
        "dropped_columns": drop_columns,
        "X": X,
        "y": y,
        "preprocessor": preprocessor,
    }