from __future__ import annotations

"""Streamlit UI client for 5-year breast cancer survival prediction.

Features
--------
1. Upload a CSV containing one patient record.
2. Select one of the available trained survival models from the latest results run.
3. Predict 5-year survival probability and event risk.
4. Explain the prediction with SHAP and patient-friendly notes.

Run
---
streamlit run survival_5yr_ui_client.py

Expected project layout
-----------------------
This file follows the same project assumptions as run_survival_5yr.py:
- src/config.py defines PROCESSED_DATA_DIR and RESULTS_DIR
- Results folders are named survival_5yr_*
- Model files are stored as .joblib under the latest survival_5yr_* folder
- Processed training data is breast_cancer_survival_processed.csv
"""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
except ImportError:  # SHAP is optional until explanation is requested
    shap = None

# Same import pattern used by the original script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

TARGET_COLUMNS = ["survival_months", "status", "survived_5yr"]
DEFAULT_DATA_FILE = "breast_cancer_survival_processed.csv"


# ---------------------------
# Loading helpers
# ---------------------------

def find_latest_run(results_dir: Path) -> Path:
    runs = sorted(results_dir.glob("survival_5yr_*"))
    if not runs:
        raise FileNotFoundError("No survival_5yr_* results found. Run the training notebook first.")
    return runs[-1]


def discover_models(run_dir: Path) -> dict[str, Path]:
    """Find available model pipelines in the latest run folder."""
    model_files = sorted(run_dir.glob("*.joblib"))
    models: dict[str, Path] = {}

    for path in model_files:
        name = path.stem
        # Make display names cleaner but keep the original model name recognizable.
        name = name.replace("survival_5yr_", "").replace("_pipeline", "")
        name = name.replace("_", " ").title()
        models[name] = path

    if not models:
        raise FileNotFoundError(f"No .joblib model files found in {run_dir}")
    return models


@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: str):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    data_path = PROCESSED_DATA_DIR / DEFAULT_DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed data file: {data_path}")
    return pd.read_csv(data_path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGET_COLUMNS]


def align_uploaded_record(uploaded_df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Validate and align the uploaded CSV to exactly one patient row."""
    if uploaded_df.empty:
        raise ValueError("The uploaded CSV is empty.")
    if len(uploaded_df) != 1:
        raise ValueError("Please upload a CSV with exactly one patient record.")

    missing = [c for c in expected_features if c not in uploaded_df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Extra columns are ignored so the user can upload a row copied from the processed file.
    row = uploaded_df[expected_features].copy()
    return row


# ---------------------------
# Prediction helpers
# ---------------------------

def class_one_index(pipeline: Any, probs: np.ndarray) -> int:
    classes = getattr(pipeline, "classes_", None)
    if classes is not None:
        classes_list = list(classes)
        if 1 in classes_list:
            return classes_list.index(1)
    return 1 if probs.shape[1] > 1 else 0


def predict_survival_probability(pipeline: Any, patient_row: pd.DataFrame) -> float:
    probs = pipeline.predict_proba(patient_row)
    idx_one = class_one_index(pipeline, probs)
    return float(probs[0, idx_one])


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "High Confidence"
    if confidence >= 0.65:
        return "Medium Confidence"
    return "Low Confidence"


def review_flag(confidence: float) -> str:
    return "Needs Expert Review" if confidence < 0.70 else "No Expert Review Needed"


# ---------------------------
# SHAP helpers
# ---------------------------

def get_preprocessor_and_model(pipeline: Any):
    """Return final estimator and optional preprocessing step from a sklearn Pipeline."""
    if hasattr(pipeline, "steps") and len(pipeline.steps) >= 2:
        preprocessor = pipeline[:-1]
        model = pipeline.steps[-1][1]
        return preprocessor, model
    return None, pipeline


def transformed_feature_names(preprocessor: Any, raw_features: list[str]) -> list[str]:
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        names = list(preprocessor.get_feature_names_out())
        return [n.split("__", 1)[-1] for n in names]
    return raw_features


def compute_shap_table(pipeline: Any, patient_row: pd.DataFrame, background_df: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP explanation for the selected patient.

    Uses TreeExplainer for tree models. If the model is not tree-based, it falls back
    to KernelExplainer using a small background sample.
    """
    if shap is None:
        raise ImportError("SHAP is not installed. Install it with: pip install shap")

    raw_features = list(patient_row.columns)
    preprocessor, model = get_preprocessor_and_model(pipeline)

    if preprocessor is not None:
        x_patient = preprocessor.transform(patient_row)
        x_background = preprocessor.transform(background_df[raw_features])
        names = transformed_feature_names(preprocessor, raw_features)
    else:
        x_patient = patient_row
        x_background = background_df[raw_features]
        names = raw_features

    # Convert sparse matrices if necessary.
    if hasattr(x_patient, "toarray"):
        x_patient = x_patient.toarray()
    if hasattr(x_background, "toarray"):
        x_background = x_background.toarray()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_patient)
    except Exception:
        # Generic fallback. Slower, but works for many sklearn classifiers.
        background_small = shap.sample(x_background, min(50, len(background_df)), random_state=42)
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background_small)
        shap_values = explainer.shap_values(x_patient, nsamples=100)

    # For binary classifiers, SHAP may return a list: [class 0 values, class 1 values].
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 3:
        # Newer SHAP versions may return shape: rows x features x classes.
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    values = values[0]

    patient_values = np.asarray(x_patient)[0]
    table = pd.DataFrame(
        {
            "feature": names[: len(values)],
            "patient_value": patient_values[: len(values)],
            "shap_value": values,
            "impact": np.where(values >= 0, "Increases predicted survival", "Decreases predicted survival"),
            "abs_impact": np.abs(values),
        }
    ).sort_values("abs_impact", ascending=False)

    return table[["feature", "patient_value", "shap_value", "impact"]]


def patient_friendly_summary(shap_table: pd.DataFrame, survival_probability: float) -> str:
    risk_probability = 1 - survival_probability
    top = shap_table.head(5)

    lines = [
        f"The model estimates a 5-year survival probability of {survival_probability * 100:.1f}% "
        f"and an event risk of {risk_probability * 100:.1f}%.",
        "The most important factors for this prediction are listed below. Positive SHAP values push the prediction toward survival; negative SHAP values push it away from survival.",
    ]

    for _, row in top.iterrows():
        direction = "supports higher predicted survival" if row["shap_value"] >= 0 else "supports higher predicted risk"
        lines.append(f"- {row['feature']} = {row['patient_value']}: {direction}.")

    lines.append(
        "This is a decision-support result, not a diagnosis. A clinician should review the patient context before making care decisions."
    )
    return "\n".join(lines)


# ---------------------------
# Streamlit UI
# ---------------------------

def main() -> None:
    st.set_page_config(page_title="5-Year Survival Predictor", layout="wide")
    st.title("5-Year Breast Cancer Survival Prediction")
    st.caption("Upload one patient record, choose a trained model, and review SHAP explanation.")

    try:
        run_dir = find_latest_run(RESULTS_DIR)
        models = discover_models(run_dir)
        training_df = load_training_data()
        expected_features = feature_columns(training_df)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Model")
        selected_model_name = st.selectbox("Choose model", list(models.keys()))
        selected_model_path = models[selected_model_name]
        st.write("Run folder:")
        st.code(str(run_dir))

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with st.expander("Model metrics"):
                st.json(json.loads(metrics_path.read_text()))

    st.subheader("1. Upload one patient CSV record")
    uploaded_file = st.file_uploader("CSV must contain exactly one row and the required feature columns.", type=["csv"])

    with st.expander("Required columns"):
        st.write(expected_features)

    if uploaded_file is None:
        st.info("Upload a one-row CSV to start.")
        st.stop()

    try:
        uploaded_df = pd.read_csv(uploaded_file)
        patient_row = align_uploaded_record(uploaded_df, expected_features)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.write("Uploaded patient record")
    st.dataframe(patient_row, use_container_width=True)

    if st.button("Predict", type="primary"):
        try:
            pipeline = load_pipeline(str(selected_model_path))
            proba_survive = predict_survival_probability(pipeline, patient_row)
            proba_event = 1 - proba_survive
            confidence = max(proba_survive, proba_event)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("5-Year Survival Probability", f"{proba_survive * 100:.1f}%")
            col2.metric("Event Risk within 5 Years", f"{proba_event * 100:.1f}%")
            col3.metric("Confidence", f"{confidence * 100:.1f}%")
            col4.metric("Review", review_flag(confidence))

            st.write(f"Confidence level: **{confidence_level(confidence)}**")

            st.subheader("2. SHAP explanation")
            background = training_df[expected_features].sample(
                n=min(100, len(training_df)), random_state=42
            )
            shap_table = compute_shap_table(pipeline, patient_row, background)

            st.write("Top factors influencing this prediction")
            st.dataframe(shap_table.head(10), use_container_width=True)

            chart_data = shap_table.head(10).set_index("feature")[["shap_value"]]
            st.bar_chart(chart_data)

            st.subheader("Patient-friendly explanation")
            st.markdown(patient_friendly_summary(shap_table, proba_survive))

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
