"""Tune SVM hyperparameters for the predictive maintenance model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from train_svm import (
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    build_features,
    build_preprocessor,
    detect_target_column,
)


DEFAULT_PARAM_GRID = {
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", 0.01, 0.1],
    "model__kernel": ["rbf"],
}


def tune_svm(
    data_path: Path = DEFAULT_DATA_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 3,
    scoring: str = "roc_auc",
) -> dict:
    """Run grid search tuning, evaluate the best SVM, and save results."""
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
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=DEFAULT_PARAM_GRID,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, y_pred)

    run_id = datetime.now().strftime("svm_tuning_%Y%m%d_%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score",
        ascending=True,
    )
    cv_results.to_csv(run_dir / "cv_results.csv", index=False)

    metrics = {
        "run_id": run_id,
        "model": "Tuned Support Vector Machine",
        "scoring": scoring,
        "param_grid": DEFAULT_PARAM_GRID,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
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

    joblib.dump(best_model, run_dir / "tuned_svm_model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(y_pred),
            "failure_probability": pd.Series(y_proba),
        }
    ).to_csv(run_dir / "predictions.csv", index=False)

    print(f"Tuned SVM results saved to: {run_dir}")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV {scoring}: {search.best_score_:.3f}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Test precision: {metrics['precision']:.3f}")
    print(f"Test recall: {metrics['recall']:.3f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.3f}")
    print("Confusion matrix:")
    print(matrix)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--scoring", default="roc_auc")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_svm(
        data_path=args.data_path,
        results_dir=args.results_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
    )
