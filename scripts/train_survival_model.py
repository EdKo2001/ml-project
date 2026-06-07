"""Train a survival 5-year classifier on processed survival CSV and save to results/metrics.

Saves: results/metrics/breast_cancer_survival_model.joblib and results/metrics/breast_cancer_survival_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure repo root on path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import make_shared_preprocessing_pipeline
from src.models import get_model
from sklearn.pipeline import Pipeline

RESULTS_METRICS = REPO_ROOT / "results" / "metrics"
RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
PROCESSED = REPO_ROOT / "data" / "processed" / "breast_cancer_survival_processed.csv"
HELDOUT_UIDS = REPO_ROOT / "results" / "metrics" / "heldout_survival_uids.json"

if not PROCESSED.exists():
    raise FileNotFoundError(f"Missing processed survival CSV at {PROCESSED}")

print("Loading processed survival CSV...")
df = pd.read_csv(PROCESSED)
# Ensure a survived_5yr target exists (notebooks derive this from survival_months)
if "survived_5yr" not in df.columns:
    if "survival_months" in df.columns:
        df["survived_5yr"] = (df["survival_months"] >= 60).astype(int)
        print("Derived survived_5yr from survival_months")
    else:
        raise ValueError(
            "Expected column survived_5yr or survival_months not found in processed survival CSV"
        )

# Add stable uid if missing
if 'uid' not in df.columns:
    df = df.reset_index().rename(columns={'index': 'uid'})
    df['uid'] = df['uid'].astype(str)

# Exclude heldout UIDs from training if present
if HELDOUT_UIDS.exists():
    try:
        heldout_uids = set(json.loads(HELDOUT_UIDS.read_text()))
        before = len(df)
        df = df[~df['uid'].isin(heldout_uids)]
        after = len(df)
        print(f"Excluded {before - after} heldout rows from training (uids from {HELDOUT_UIDS})")
    except Exception as e:
        print('Failed to read heldout UIDs:', e)

# Build preprocessing
info = make_shared_preprocessing_pipeline(df, target_column="survived_5yr")
X = info["X"]
y = info["y"]
pre = info["preprocessor"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline
clf = get_model("random_forest", n_estimators=100)
pipeline = Pipeline([("preprocessor", pre), ("model", clf)])

print("Training RandomForest on survival dataset...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
try:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
except Exception:
    y_proba = None

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "report": classification_report(y_test, y_pred, output_dict=True),
}

# Save model and metrics
model_path = RESULTS_METRICS / "breast_cancer_survival_model.joblib"
metrics_path = RESULTS_METRICS / "breast_cancer_survival_metrics.json"
joblib.dump(pipeline, model_path)
metrics_path.write_text(json.dumps(metrics, indent=2))
print("Saved model to", model_path)
print("Saved metrics to", metrics_path)
print("Done")
