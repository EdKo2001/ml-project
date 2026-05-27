"""Shared data pipeline entrypoint for the team."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .preprocessing import (
    load_data,
    make_shared_preprocessing_pipeline,
    split_data,
)


@dataclass(frozen=True)
class DataPipelineArtifacts:
    df: pd.DataFrame
    target_column: str
    dropped_columns: list[str]
    X: pd.DataFrame
    y: pd.Series
    preprocessor: object
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def build_data_pipeline(
    data_path: str | Path,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataPipelineArtifacts:
    """Load data, build shared preprocessing, and split train/test."""
    df = load_data(str(data_path))
    pipeline = make_shared_preprocessing_pipeline(df, target_column=target_column)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target=pipeline["target_column"],
        test_size=test_size,
        random_state=random_state,
    )

    return DataPipelineArtifacts(
        df=df,
        target_column=pipeline["target_column"],
        dropped_columns=pipeline["dropped_columns"],
        X=pipeline["X"],
        y=pipeline["y"],
        preprocessor=pipeline["preprocessor"],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def detect_target_column(
    data_path: str | Path,
    candidates: Optional[Iterable[str]] = None,
) -> str:
    """Detect a target column using case/whitespace-insensitive matching."""
    if candidates is None:
        candidates = ["machine failure", "machine_failure", "target", "label"]

    header_df = pd.read_csv(str(data_path), nrows=1)
    normalized = {str(col).strip().lower(): col for col in header_df.columns}
    for candidate in candidates:
        candidate_key = candidate.strip().lower()
        if candidate_key in normalized:
            return normalized[candidate_key]

    raise KeyError(
        "Target column not found in CSV header. Update the candidate list or set it manually."
    )
