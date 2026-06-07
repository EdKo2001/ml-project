"""Simple Streamlit UI for patient-level predictions and SHAP explanation prompts.

Run with: `streamlit run src/ui_streamlit.py`
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SURVIVAL = ROOT / "data" / "processed" / "breast_cancer_survival_processed.csv"
PROCESSED_DIAGNOSTIC = ROOT / "data" / "processed" / "breast_cancer_processed.csv"
DEFAULT_MODEL = ROOT / "results" / "metrics" / "breast_cancer_model.joblib"
RESULTS_DIR = ROOT / "results"


def load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def load_shap_local(run_dir: Path, model_name: str):
    p = run_dir / f"{model_name}_shap_example_local.joblib"
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def shap_local_to_top(shap_obj: Any, feature_names: list[str], top_n: int = 5):
    # Attempt to extract local contribution magnitudes
    try:
        if hasattr(shap_obj, "values"):
            vals = np.asarray(shap_obj.values)
            if vals.ndim == 3:
                # (samples, features, classes) -> take class 1 if present
                class_idx = 1 if vals.shape[2] > 1 else 0
                contrib = vals[0, :, class_idx]
            elif vals.ndim == 2:
                contrib = vals[0, :]
            else:
                contrib = np.ravel(vals)
        else:
            arr = np.asarray(shap_obj)
            if arr.ndim == 2:
                contrib = arr[0, :]
            else:
                contrib = np.ravel(arr)
    except Exception:
        return []

    mags = np.abs(contrib)
    idx = np.argsort(mags)[-top_n:][::-1]
    return [(feature_names[i], float(contrib[i])) for i in idx]


st.title("Breast Cancer: Patient-level prediction & explanation helper")

# Dataset selection: survival vs diagnostic processed CSV
dataset_choice = st.sidebar.selectbox(
    "Processed dataset",
    ("Survival", "Diagnostic"),
    index=0,
)

selected_path = PROCESSED_SURVIVAL if dataset_choice == "Survival" else PROCESSED_DIAGNOSTIC
if not selected_path.exists():
    st.warning(f"Selected processed CSV not found: {selected_path}. Run dataset prep first.")

df = None
if selected_path.exists():
    df = pd.read_csv(selected_path)

st.sidebar.header("Model selection")
model_path = None
if DEFAULT_MODEL.exists():
    st.sidebar.write(f"Found default model: {DEFAULT_MODEL.name}")
    if st.sidebar.checkbox("Use default model", value=True):
        model_path = DEFAULT_MODEL

uploaded = st.sidebar.file_uploader("Or upload a model (.joblib)")
if uploaded is not None and model_path is None:
    tmp = Path(".") / "uploaded_model.joblib"
    with tmp.open("wb") as fh:
        fh.write(uploaded.getbuffer())
    model_path = tmp

if model_path is None:
    st.sidebar.info("No model selected. Upload a joblib Pipeline or place a model at results/metrics/breast_cancer_model.joblib")

model = None
if model_path is not None:
    model = load_model(model_path)

st.header("Select patient input")
if df is None:
    st.info("No processed data available to select a sample. You can upload a single-row CSV with matching columns.")
    uploaded_row = st.file_uploader("Upload patient CSV (single row)", type=["csv"])
    patient_df = None
    if uploaded_row is not None:
        patient_df = pd.read_csv(uploaded_row)
else:
    idx = st.number_input("Patient row index", min_value=0, max_value=max(0, len(df) - 1), value=0)
    patient_df = df.iloc[[int(idx)]]

if patient_df is not None:
    st.subheader("Patient data preview")
    st.dataframe(patient_df)

    if model is None:
        st.warning("No model available to predict. Provide a model to enable predictions.")
    else:
        # Verify that the model and patient row have matching features
        expected_feats = None
        pre = None
        try:
            if hasattr(model, "named_steps"):
                pre = model.named_steps.get("preprocessor")
            if pre is not None:
                try:
                    expected_feats = list(pre.get_feature_names_out(patient_df.columns.tolist()))
                except Exception:
                    try:
                        expected_feats = list(pre.get_feature_names_out())
                    except Exception:
                        expected_feats = None
            elif hasattr(model, "feature_names_in_"):
                expected_feats = list(model.feature_names_in_)
        except Exception:
            expected_feats = None

        missing = None
        if expected_feats is not None:
            missing = set(expected_feats) - set(patient_df.columns.tolist())
        if missing:
            st.error("Model evaluation failed: columns are missing: {}".format(sorted(list(missing))))
            st.info("Possible fixes: select the matching processed dataset in the sidebar, upload a one-row CSV matching the model inputs, or run dataset preparation.")
            if st.button("Prepare processed datasets now"):
                try:
                    from src.finalize_dataset import main as finalize_main

                    finalize_main()
                    st.success("Dataset preparation completed. Reload the app or re-select the dataset.")
                except Exception as e:
                    st.error(f"Failed to prepare datasets: {e}")
            proba = None
        else:
            try:
                proba = model.predict_proba(patient_df)[0, 1]
            except Exception:
                try:
                    proba = float(model.predict(patient_df)[0])
                except Exception as e:
                    st.error(f"Model evaluation failed: {e}")
                    proba = None

        if proba is not None:
            pct = int(round(proba * 100))
            st.metric("Predicted 5-year survival (approx)", f"{pct}%")
            risk_label = "low" if proba >= 0.5 else "intermediate-to-higher"
            st.write(f"Risk group (threshold 0.5): **{risk_label}**")

            # Attempt to find SHAP local joblib in latest shap run
            shap_runs = sorted([p for p in RESULTS_DIR.glob("shap_compare_*") if p.is_dir()])
            shap_local = None
            if shap_runs:
                latest = shap_runs[-1]
                shap_local = load_shap_local(latest, "random_forest")

            top_drivers = []
            if shap_local is not None:
                try:
                    # try to reconstruct feature names from pipeline preprocessor
                    pre = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None
                    if pre is not None:
                        try:
                            feat_names = pre.get_feature_names_out(patient_df.columns.tolist())
                        except Exception:
                            feat_names = patient_df.columns.tolist()
                    else:
                        feat_names = patient_df.columns.tolist()
                    top_drivers = shap_local_to_top(shap_local, feat_names, top_n=5)
                except Exception:
                    top_drivers = []

            if top_drivers:
                st.subheader("Top contributing features (local SHAP)")
                for fn, val in top_drivers:
                    st.write(f"- **{fn}**: {val:+.3f}")
            else:
                st.info("No local SHAP artifact found for RandomForest in latest SHAP run; you can run SHAP scripts to generate them.")

            # Generate patient-friendly explanation and LLM prompt
            st.subheader("Patient-facing explanation (suggested)")
            # Build simple natural-language explanation using template
            feat_lines = []
            for fn, val in top_drivers[:3]:
                short = fn.replace("num__", "").replace("cat__", "")
                reason = "increases" if val > 0 else "decreases"
                feat_lines.append(f"{short}: {reason} risk")

            context = []
            for k in ["age", "tumor_size", "regional_node_positive"]:
                if k in patient_df.columns:
                    context.append(f"{k}={patient_df.iloc[0][k]}")

            prob_alive = pct
            rl = "intermediate-to-higher risk" if proba < 0.5 else "lower risk"
            expl = (
                f"Based on the information provided, this model estimates about a {prob_alive}% chance of being alive at 5 years, "
                f"placing this case in a {rl} group. The factors that influenced this prediction most were: {', '.join([f for f,_ in top_drivers[:3]]) or 'clinical and tumor-related measurements'}. "
                "This estimate is probabilistic and not a definitive medical judgment. Treatments, health status, and follow-up care can change outcomes. "
                "Please review this estimate with your oncology team to combine it with imaging, pathology, and treatment options for a personalized plan."
            )
            st.write(expl)

            st.subheader("LLM prompt (copy to your preferred LLM)")
            prompt = {
                "Predicted_probability_alive_5yr": f"{prob_alive}%",
                "Risk_group_label": rl,
                "Top_drivers": [{fn: float(val)} for fn, val in top_drivers[:3]],
                "Example_patient_context": ", ".join(context) or "n/a",
                "Model_limitations": "Model provides probability estimates from historical data and does not determine individual outcomes.",
                "Suggested_next_step": "Review with oncology team; combine with imaging and pathology"
            }
            st.code(json.dumps(prompt, indent=2))

else:
    st.info("Provide a patient row or select an index to proceed.")
