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
PROCESSED_SURVIVAL = (
    ROOT / "data" / "processed" / "breast_cancer_survival_processed.csv"
)
PROCESSED_DIAGNOSTIC = ROOT / "data" / "processed" / "breast_cancer_processed.csv"
HELDOUT_SURVIVAL = ROOT / "data" / "processed" / "heldout_survival.csv"
HELDOUT_DIAGNOSTIC = ROOT / "data" / "processed" / "heldout_diagnostic.csv"
DEFAULT_MODEL = ROOT / "results" / "metrics" / "breast_cancer_model.joblib"
RESULTS_DIR = ROOT / "results"


def load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def get_model_with_cache(path: Path, force_reload: bool = False):
    """Load a model and cache it in session_state keyed by path and mtime.

    This avoids restarting Streamlit when the model file is overwritten. Call
    with `force_reload=True` to force reloading.
    """
    if path is None or not path.exists():
        return None

    mtime = path.stat().st_mtime
    key_path = str(path)
    cache_key = f"model_mtime_{key_path}"
    obj_key = f"model_obj_{key_path}"

    if force_reload:
        st.session_state.pop(cache_key, None)
        st.session_state.pop(obj_key, None)

    if st.session_state.get(cache_key) == mtime and obj_key in st.session_state:
        return st.session_state[obj_key]

    model = load_model(path)
    if model is not None:
        st.session_state[cache_key] = mtime
        st.session_state[obj_key] = model
    return model


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
            if arr.ndim == 3:
                class_idx = 1 if arr.shape[2] > 1 else 0
                contrib = arr[0, :, class_idx]
            elif arr.ndim == 2:
                contrib = arr[0, :]
            else:
                contrib = np.ravel(arr)
    except Exception:
        return []

    mags = np.abs(contrib)
    idx = np.argsort(mags)[-top_n:][::-1]
    return [(feature_names[i], float(contrib[i])) for i in idx]


def input_columns_for_dataset(dataset_choice: str, df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers and target columns before model prediction."""
    drop_cols = ["uid"]
    if dataset_choice == "Survival":
        drop_cols.extend(["survived_5yr"])
    else:
        drop_cols.append("diagnosis")
    return df.drop(columns=drop_cols, errors="ignore")


def compute_live_shap_local(
    model: Any, patient_df: pd.DataFrame, background_df: pd.DataFrame
):
    """Compute a local SHAP explanation directly from the current model.

    This is a fallback when a saved SHAP artifact is missing, stale, or was
    generated for a different dataset/model.
    """
    try:
        import shap
    except Exception:
        return None, []

    if not hasattr(model, "named_steps"):
        return None, []

    pre = model.named_steps.get("preprocessor")
    core = model.named_steps.get("model")
    if pre is None or core is None:
        return None, []

    try:
        feat_names = list(pre.get_feature_names_out(background_df.columns.tolist()))
    except Exception:
        feat_names = background_df.columns.tolist()

    bg = background_df.sample(n=min(100, len(background_df)), random_state=42)
    try:
        bg_trans = pre.transform(bg)
        ex_trans = pre.transform(patient_df)
    except Exception:
        return None, feat_names

    try:
        explainer = shap.Explainer(core, bg_trans)
        return explainer(ex_trans), feat_names
    except Exception:
        try:
            explainer = shap.TreeExplainer(core, data=bg_trans, check_additivity=False)
            return explainer(ex_trans), feat_names
        except Exception:
            try:
                predict_fn = (
                    lambda z: core.predict_proba(z)[:, 1]
                    if hasattr(core, "predict_proba")
                    else core.predict(z)
                )
                explainer = shap.KernelExplainer(predict_fn, bg_trans)
                shap_vals = explainer.shap_values(ex_trans, nsamples=100)
                return shap_vals, feat_names
            except Exception:
                return None, feat_names


st.title("Breast Cancer: Patient-level prediction & explanation helper")

# Dataset selection: survival vs diagnostic processed CSV
dataset_choice = st.sidebar.selectbox(
    "Processed dataset",
    ("Survival", "Diagnostic"),
    index=0,
    key="dataset_choice",
)

selected_path = (
    HELDOUT_SURVIVAL
    if dataset_choice == "Survival" and HELDOUT_SURVIVAL.exists()
    else HELDOUT_DIAGNOSTIC
    if dataset_choice == "Diagnostic" and HELDOUT_DIAGNOSTIC.exists()
    else PROCESSED_SURVIVAL
    if dataset_choice == "Survival"
    else PROCESSED_DIAGNOSTIC
)
if not selected_path.exists():
    st.warning(
        f"Selected processed CSV not found: {selected_path}. Run dataset prep first."
    )

df = None
if selected_path.exists():
    df = pd.read_csv(selected_path)
    if "uid" not in df.columns:
        df = df.reset_index().rename(columns={"index": "uid"})
        df["uid"] = df["uid"].astype(str)

st.sidebar.header("Model selection")
# Discover available model files in results/metrics
model_path = None
metrics_dir = ROOT / "results" / "metrics"
available_models = []
if metrics_dir.exists():
    available_models = sorted([p for p in metrics_dir.glob("*.joblib") if p.is_file()])

model_choices = [p.name for p in available_models]
if DEFAULT_MODEL.exists() and DEFAULT_MODEL.name not in model_choices:
    model_choices.insert(0, DEFAULT_MODEL.name)


# Auto-select a model that matches the chosen dataset when possible
def _auto_select_model_for_dataset(dataset: str, choices: list[str]) -> str | None:
    if not choices:
        return None
    lower = [c.lower() for c in choices]
    if dataset == "Survival":
        surv = [c for c in choices if "survival" in c.lower()]
        if surv:
            return surv[-1]
        # fallback: any model with 'surv' substring
        surv2 = [c for c in choices if "surv" in c.lower()]
        if surv2:
            return surv2[-1]
        return None
    # Diagnostic dataset
    # Prefer the default diagnostic model name, else any non-survival model
    if DEFAULT_MODEL.name in choices:
        return DEFAULT_MODEL.name
    non_surv = [c for c in choices if "survival" not in c.lower()]
    if non_surv:
        return non_surv[0]
    return None


auto_model = _auto_select_model_for_dataset(dataset_choice, model_choices)
sel_model_name = None
if model_choices:
    # selectbox options include an empty option for manual 'none'
    opts = ("",) + tuple(model_choices)
    # determine default index: 0 means empty selection
    default_index = 0
    if auto_model and auto_model in model_choices:
        default_index = model_choices.index(auto_model) + 1
    sel_model_name = st.sidebar.selectbox(
        "Choose a model file", opts, index=default_index
    )
    if sel_model_name == "" and auto_model is not None:
        # if user hasn't manually chosen, show that we auto-selected and allow them to pick
        st.sidebar.write(f"Auto-suggested model for {dataset_choice}: {auto_model}")

if sel_model_name:
    candidate = metrics_dir / sel_model_name
    if not candidate.exists() and Path(sel_model_name).exists():
        candidate = Path(sel_model_name)
    model_path = candidate

if model_path is None:
    st.sidebar.info(
        "No model selected. Upload a joblib Pipeline or place a model in results/metrics"
    )

model = None
if model_path is not None:
    model = get_model_with_cache(model_path)

st.header("Select patient input")
if df is None:
    st.warning(
        "No processed data available to select a sample. Run dataset preparation first and refresh the app."
    )
    patient_df = None
else:
    idx = st.number_input(
        "Patient row index", min_value=0, max_value=max(0, len(df) - 1), value=0
    )
    patient_df = df.iloc[[int(idx)]]

if patient_df is not None:
    st.subheader("Patient data preview")
    st.dataframe(patient_df)

    model_input_df = input_columns_for_dataset(dataset_choice, patient_df)

    if model is None:
        st.warning(
            "No model available to predict. Provide a model to enable predictions."
        )
    else:
        # Verify that the model and patient row have matching features
        expected_feats = None
        pre = None
        try:
            if hasattr(model, "named_steps"):
                pre = model.named_steps.get("preprocessor")
            if pre is not None:
                try:
                    expected_feats = list(
                        pre.get_feature_names_out(model_input_df.columns.tolist())
                    )
                except Exception:
                    try:
                        expected_feats = list(pre.get_feature_names_out())
                    except Exception:
                        expected_feats = None
            elif hasattr(model, "feature_names_in_"):
                expected_feats = list(model.feature_names_in_)
        except Exception:
            expected_feats = None

        # Map transformed feature names back to raw column names by substring matching
        missing_raw = None
        if expected_feats is not None:
            raw_cols = set(model_input_df.columns.tolist())
            unmatched = []
            for feat in expected_feats:
                mapped = False
                for rc in raw_cols:
                    if rc in feat:
                        mapped = True
                        break
                if not mapped:
                    base = feat.split("__", 1)[-1] if "__" in feat else feat
                    for rc in raw_cols:
                        if rc in base or base in rc:
                            mapped = True
                            break
                if not mapped:
                    unmatched.append(feat)
            if unmatched:
                missing_raw = sorted(
                    list({u.split("__", 1)[-1] if "__" in u else u for u in unmatched})
                )

        if missing_raw:
            st.error(
                "Model evaluation failed: columns are missing: {}".format(missing_raw)
            )
            st.info(
                "Possible fixes: select the matching processed dataset in the sidebar, upload a one-row CSV matching the model inputs, or run dataset preparation."
            )
            if st.button("Switch to Diagnostic dataset"):
                try:
                    st.session_state["dataset_choice"] = "Diagnostic"
                    st.experimental_rerun()
                except Exception:
                    st.warning(
                        "Could not switch automatically; please select 'Diagnostic' from the sidebar."
                    )

            if st.button("Prepare processed datasets now"):
                try:
                    from src.finalize_dataset import main as finalize_main

                    finalize_main()
                    st.success(
                        "Dataset preparation completed. Reload the app or re-select the dataset."
                    )
                except Exception as e:
                    st.error(f"Failed to prepare datasets: {e}")
            proba = None
        else:
            try:
                proba = model.predict_proba(model_input_df)[0, 1]
            except Exception:
                try:
                    proba = float(model.predict(model_input_df)[0])
                except Exception as e:
                    st.error(f"Model evaluation failed: {e}")
                    proba = None

        if proba is not None:
            pct = int(round(proba * 100))
            st.metric("Predicted 5-year survival (approx)", f"{pct}%")
            risk_label = "low" if proba >= 0.5 else "intermediate-to-higher"
            st.write(f"Risk group (threshold 0.5): **{risk_label}**")

            # Attempt to find SHAP local joblib in the latest shap run.
            # If that does not work, compute SHAP live from the currently selected model.
            shap_runs = sorted(
                [p for p in RESULTS_DIR.glob("shap_compare_*") if p.is_dir()]
            )
            shap_local = None
            if shap_runs:
                latest = shap_runs[-1]
                shap_model_name = "logistic_regression" if "logistic" in str(model_path).lower() else "random_forest"
                shap_local = load_shap_local(latest, shap_model_name)

            top_drivers = []
            if shap_local is not None:

                if not top_drivers:
                    live_shap, feat_names = compute_live_shap_local(
                        model, model_input_df, model_input_df
                    )
                    if live_shap is not None:
                        try:
                            top_drivers = shap_local_to_top(
                                live_shap, feat_names, top_n=5
                            )
                        except Exception:
                            top_drivers = []

            if top_drivers:
                st.subheader("Top contributing features (local SHAP)")
                for fn, val in top_drivers:
                    st.write(f"- **{fn}**: {val:+.3f}")
            else:
                st.info(
                    "SHAP could not be read from the saved run, so the app could not show local drivers for this model."
                )

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
                "Suggested_next_step": "Review with oncology team; combine with imaging and pathology",
            }
            st.code(json.dumps(prompt, indent=2))

else:
    st.info("Provide a patient row or select an index to proceed.")
