"""Train and evaluate one shared baseline model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import RAW_DATA_DIR, RESULTS_DIR
from src.data_pipeline import build_data_pipeline, detect_target_column
from src.models import MODEL_NAMES, get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        choices=sorted(MODEL_NAMES),
        help="Model to train.",
    )
    parser.add_argument(
        "--data-path",
        default=RAW_DATA_DIR / "predictive_maintenance.csv",
        type=Path,
        help="CSV dataset path.",
    )
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_column = detect_target_column(args.data_path)
    artifacts = build_data_pipeline(
        args.data_path,
        target_column=target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = get_model(args.model, random_state=args.random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", artifacts.preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(artifacts.X_train, artifacts.y_train)

    y_pred = pipeline.predict(artifacts.X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(artifacts.X_test)[:, 1]

    metrics = {
        "model": args.model,
        "model_display_name": MODEL_NAMES[args.model],
        "target_column": artifacts.target_column,
        "dropped_columns": artifacts.dropped_columns,
        "train_shape": list(artifacts.X_train.shape),
        "test_shape": list(artifacts.X_test.shape),
        "accuracy": float(accuracy_score(artifacts.y_test, y_pred)),
        "confusion_matrix": confusion_matrix(artifacts.y_test, y_pred).tolist(),
        "classification_report": classification_report(
            artifacts.y_test,
            y_pred,
            digits=3,
            output_dict=True,
        ),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(artifacts.y_test, y_proba))

    run_id = datetime.now().strftime(f"{args.model}_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Model: {MODEL_NAMES[args.model]}")
    print(f"Saved to: {run_dir}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    if "roc_auc" in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(artifacts.y_test, y_pred))
    print(classification_report(artifacts.y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
