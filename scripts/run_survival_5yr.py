from __future__ import annotations

"""Run the latest 5-year survival pipeline and print a clear patient-level summary.

This script loads the most recent `survival_5yr_*` run in `RESULTS_DIR`, reads the
processed survival CSV, selects an example patient row, and prints:
- patient features
- survival probability (5 years)
- event risk (1 - survival probability)
- confidence level and expert-review recommendation

Usage: run as a script. Optionally edit `EXAMPLE_INDEX` below or modify to accept CLI args.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# Ensure the repo root is on sys.path so `src` is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402


EXAMPLE_INDEX = 1500


def find_latest_run(results_dir: Path) -> Path:
    runs = sorted(results_dir.glob("survival_5yr_*"))
    if not runs:
        raise FileNotFoundError("No survival_5yr_* results found. Run the notebook first.")
    return runs[-1]


def load_pipeline(run_dir: Path):
    model_path = run_dir / "survival_5yr_random_forest_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model at {model_path}")
    return joblib.load(model_path)


def pick_example_row(df: pd.DataFrame, feature_cols: list[str], index: int) -> pd.DataFrame:
    if index < 0 or index >= len(df):
        raise IndexError("example index out of range")
    return df[feature_cols].iloc[[index]]


def compute_proba_for_class_one(pipeline, example_row: pd.DataFrame) -> float:
    probs = pipeline.predict_proba(example_row)
    classes = getattr(pipeline, "classes_", None)
    if classes is not None:
        classes_list = list(classes)
        if 1 in classes_list:
            idx_one = classes_list.index(1)
        else:
            idx_one = -1
    else:
        idx_one = 1 if probs.shape[1] > 1 else 0
    return float(probs[0, idx_one])


def confidence_level(c: float) -> str:
    if c >= 0.80:
        return "High Confidence"
    if c >= 0.65:
        return "Medium Confidence"
    return "Low Confidence"


def review_flag(c: float) -> str:
    return "Needs Expert Review" if c < 0.70 else "No Expert Review Needed"


def main(example_index: Optional[int] = None) -> None:
    example_index = EXAMPLE_INDEX if example_index is None else example_index

    run_dir = find_latest_run(RESULTS_DIR)
    pipeline = load_pipeline(run_dir)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = None

    data_path = PROCESSED_DATA_DIR / "breast_cancer_survival_processed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing CSV at {data_path}")

    df = pd.read_csv(data_path)
    feature_cols = [
        c
        for c in df.columns
        if c not in ["survival_months", "status", "survived_5yr"]
    ]

    example_row = pick_example_row(df, feature_cols, example_index)

    proba_survive = compute_proba_for_class_one(pipeline, example_row)
    proba_event = 1 - proba_survive
    confidence = max(proba_survive, proba_event)

    conf_level = confidence_level(confidence)
    recommendation = review_flag(confidence)

    # Nicely print results
    print("=== Patient Summary ===")
    print(example_row.to_string(index=False))
    print()
    if metrics is not None:
        print("Loaded run metrics:")
        print(json.dumps(metrics, indent=2))
        print()

    classes = getattr(pipeline, "classes_", None)
    if classes is not None:
        print(f"Model classes: {list(classes)}")

    print(f"Survival probability (5 yr): {proba_survive * 100:.1f}%")
    print(f"Event risk within 5 yr: {proba_event * 100:.1f}%")
    print(f"Confidence: {confidence * 100:.1f}% ({conf_level})")
    print(f"Recommendation: {recommendation}")
    print()
    print(f"Artifacts folder: {run_dir}")


if __name__ == "__main__":
    main()
