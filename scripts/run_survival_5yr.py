from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Ensure the repo root is on sys.path so `src` is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402


def find_latest_run(results_dir: Path) -> Path:
    runs = sorted(results_dir.glob("survival_5yr_*"))
    if not runs:
        raise FileNotFoundError("No survival_5yr_* results found. Run the notebook first.")
    return runs[-1]


def main() -> None:
    run_dir = find_latest_run(RESULTS_DIR)
    model_path = run_dir / "survival_5yr_pipeline.joblib"
    metrics_path = run_dir / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model at {model_path}")

    pipeline = joblib.load(model_path)

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print("Loaded metrics:")
        print(json.dumps(metrics, indent=2))

    data_path = PROCESSED_DATA_DIR / "breast_cancer_survival_processed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing CSV at {data_path}")

    df = pd.read_csv(data_path)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["survival_months", "status", "survived_5yr"]
    ]

    example_index = 100
    example_row = df[feature_cols].iloc[[example_index]]
    proba = pipeline.predict_proba(example_row)[0, 1]

    print("Patient details:")
    print(example_row.to_string(index=False))
    print(
        f"The patient has an {proba * 100:.1f}% probability of developing breast cancer within the next 5 years."
    )
    print(f"Artifacts folder: {run_dir}")


if __name__ == "__main__":
    main()
