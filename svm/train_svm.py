"""Train an isolated SVM model for predictive maintenance.

This file intentionally lives under ``svm/`` and writes to ``svm/results/`` so
the SVM work stays separate from the shared project code and results.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "predictive_maintenance.csv"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt

TARGET_CANDIDATES = ("Target", "Machine failure", "machine_failure", "failure", "label")
DROP_COLUMNS = ("UDI", "Product ID", "Failure Type")


def detect_target_column(df: pd.DataFrame) -> str:
    """Find the binary target column using simple case-insensitive matching."""
    normalized_columns = {str(column).strip().lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        match = normalized_columns.get(candidate.strip().lower())
        if match is not None:
            return str(match)
    raise ValueError(f"Could not find target column. Columns: {list(df.columns)}")


def build_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split features and labels, dropping IDs and leakage columns."""
    dropped_columns = [
        column for column in DROP_COLUMNS if column in df.columns and column != target_column
    ]
    X = df.drop(columns=[target_column] + dropped_columns)
    y = df[target_column]
    return X, y, dropped_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

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


def train_svm(
    data_path: Path = DEFAULT_DATA_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(data_path)
    target_column = detect_target_column(df)
    X, y, dropped_columns = build_features(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, y_pred)

    run_id = datetime.now().strftime("svm_%Y%m%d_%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_id": run_id,
        "model": "Support Vector Machine",
        "model_params": pipeline.named_steps["model"].get_params(),
        "data_path": str(data_path),
        "data_shape": list(df.shape),
        "target_column": target_column,
        "dropped_columns": dropped_columns,
        "feature_columns": X.columns.tolist(),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            digits=3,
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(pipeline, run_dir / "svm_model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(y_pred),
            "failure_probability": pd.Series(y_proba),
        }
    ).to_csv(run_dir / "predictions.csv", index=False)

    ConfusionMatrixDisplay(confusion_matrix=matrix).plot(values_format="d")
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("SVM ROC Curve")
    plt.tight_layout()
    plt.savefig(run_dir / "roc_curve.png", dpi=150)
    plt.close()

    print(f"SVM results saved to: {run_dir}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Confusion matrix:")
    print(matrix)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_svm(
        data_path=args.data_path,
        results_dir=args.results_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
