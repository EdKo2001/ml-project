"""Pipeline to prepare, train, evaluate, and explain a breast-cancer classifier.

Minimal, runnable script: it will prepare the processed CSV if missing,
train a RandomForest, handle imbalance via resampling or class weights,
report accuracy/precision/recall/F1 (JSON) and save a SHAP summary plot if available.

Keep comments short as requested.
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib

import pandas as pd

from src.prepare_breast_cancer_dataset import (
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_PATH,
    DEFAULT_PROFILE_PATH,
    prepare_dataset,
)
from src.preprocessing import make_shared_preprocessing_pipeline
from src.imbalance import compute_class_weights, resample_dataset
from src.eval import evaluate, save_report
from src.models import get_model


def ensure_processed(raw_path=DEFAULT_RAW_PATH, processed_path=DEFAULT_PROCESSED_PATH, profile_path=DEFAULT_PROFILE_PATH):
    if not Path(processed_path).exists():
        prepare_dataset(Path(raw_path), Path(processed_path), Path(profile_path))


def run(output_dir: str | Path = "results/metrics", resample_method: str | None = "smote"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ensure_processed()
    df = pd.read_csv(DEFAULT_PROCESSED_PATH)

    # Use shared preprocessing to get X/y and a preprocessor
    pipeline = make_shared_preprocessing_pipeline(df, target_column="diagnosis")
    X = pipeline["X"]
    y = pipeline["y"]
    preprocessor = pipeline["preprocessor"]

    # Split manually to keep control; use sklearn train_test_split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optionally resample training set to handle imbalance
    use_resample = resample_method is not None
    if use_resample:
        X_train_res, y_train_res = resample_dataset(X_train, y_train, method=resample_method)
        X_train_fit, y_train_fit = X_train_res, y_train_res
        class_weights = None
    else:
        X_train_fit, y_train_fit = X_train, y_train
        class_weights = compute_class_weights(y_train)

    # Build a pipeline: preprocessor -> model
    from sklearn.pipeline import Pipeline

    model = get_model("random_forest", n_estimators=100, class_weight=class_weights)
    clf = Pipeline([("preprocessor", preprocessor), ("model", model)])

    clf.fit(X_train_fit, y_train_fit)

    y_pred = clf.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    report_path = out / "breast_cancer_metrics.json"
    save_report({"metrics": metrics}, report_path)

    # Save model
    model_path = out / "breast_cancer_model.joblib"
    joblib.dump(clf, model_path)

    # Try to produce a SHAP summary plot if shap is installed
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.Explainer(clf.named_steps["model"], clf.named_steps["preprocessor"].transform(X_train))
        shap_values = explainer(clf.named_steps["preprocessor"].transform(X_test))
        plt.figure(figsize=(6, 4))
        shap.summary_plot(shap_values, feature_names=clf.named_steps["preprocessor"].get_feature_names_out(), show=False)
        plt.tight_layout()
        plt.savefig(out / "shap_summary.png", dpi=150)
        plt.close()
    except Exception:
        # shap not available or explainer failed — skip silently
        pass

    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
