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

from src.preprocessing import SURVIVAL_LEAKAGE_COLUMNS

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
    """Drop identifiers and target/leakage columns before model prediction."""
    drop_cols = ["uid"]
    if dataset_choice == "Survival":
        drop_cols.extend(["survived_5yr", *SURVIVAL_LEAKAGE_COLUMNS])
    else:
        drop_cols.append("diagnosis")
    return df.drop(columns=drop_cols, errors="ignore")


def discover_model_files() -> list[Path]:
    """Collect trained pipeline artifacts from metrics/ and survival run folders."""
    paths: list[Path] = []
    metrics_dir = ROOT / "results" / "metrics"
    if metrics_dir.exists():
        paths.extend(metrics_dir.glob("*.joblib"))
    for pattern in ("survival_5yr_*", "survival_5yr_rf_*"):
        for run_dir in sorted(RESULTS_DIR.glob(pattern)):
            paths.extend(run_dir.glob("*.joblib"))
    if DEFAULT_MODEL.exists():
        paths.append(DEFAULT_MODEL)
    return sorted({p.resolve() for p in paths if p.is_file()})


def model_dataset_kind(path: Path) -> str:
    """Infer whether a saved pipeline targets survival or diagnostic data."""
    token = f"{path.parent.name}/{path.name}".lower()
    if "survival" in token or "surv_5yr" in token:
        return "Survival"
    return "Diagnostic"


def pipeline_input_columns(model: Any) -> list[str] | None:
    """Return raw input column names expected by a fitted sklearn Pipeline."""
    if hasattr(model, "named_steps"):
        pre = model.named_steps.get("preprocessor")
        if pre is not None and hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def models_for_dataset(dataset_choice: str, paths: list[Path]) -> list[Path]:
    return [p for p in paths if model_dataset_kind(p) == dataset_choice]


def model_display_name(path: Path) -> str:
    if path.parent.name in {"metrics", ".", ""}:
        return path.name
    return f"{path.name} ({path.parent.name})"


def validate_model_inputs(model: Any, model_input_df: pd.DataFrame) -> tuple[bool, list[str]]:
    expected = pipeline_input_columns(model)
    if not expected:
        return True, []
    missing = [col for col in expected if col not in model_input_df.columns]
    return not missing, missing


def positive_class_index(model: Any, proba_row: np.ndarray) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model"), "classes_", None)
    if classes is not None:
        classes_list = list(classes)
        for preferred in (1, "M", "malignant", "Dead"):
            if preferred in classes_list:
                return classes_list.index(preferred)
        if len(classes_list) > 1:
            return 1
    return 1 if len(proba_row) > 1 else 0


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
all_models = discover_model_files()
compatible_models = models_for_dataset(dataset_choice, all_models)
model_path = None
model = None

if not all_models:
    st.sidebar.info(
        "No trained models found. Run `python -m src.breast_cancer_pipeline` "
        "or notebook 07 for survival models."
    )
elif not compatible_models:
    st.sidebar.warning(
        f"No **{dataset_choice}** model found for this dataset. "
        f"Switch dataset or train a matching pipeline first."
    )
    other = "Diagnostic" if dataset_choice == "Survival" else "Survival"
    other_models = models_for_dataset(other, all_models)
    if other_models:
        st.sidebar.caption(f"Available {other} models: {', '.join(p.name for p in other_models)}")
else:
    labels = [model_display_name(p) for p in compatible_models]
    preferred = None
    if dataset_choice == "Diagnostic" and DEFAULT_MODEL.exists():
        preferred = DEFAULT_MODEL.resolve()
    elif dataset_choice == "Survival":
        for candidate in compatible_models:
            if "random_forest" in candidate.name.lower():
                preferred = candidate.resolve()
                break
        if preferred is None:
            preferred = compatible_models[-1].resolve()

    default_index = 0
    if preferred is not None:
        for i, path in enumerate(compatible_models):
            if path.resolve() == preferred:
                default_index = i
                break

    selected_label = st.sidebar.selectbox(
        "Choose a model file",
        labels,
        index=default_index,
        key=f"model_select_{dataset_choice}",
    )
    model_path = compatible_models[labels.index(selected_label)]
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
            "No compatible model available for this dataset. "
            "Train a matching pipeline or switch the dataset in the sidebar."
        )
    else:
        ok, missing = validate_model_inputs(model, model_input_df)
        proba = None

        if not ok:
            st.error(
                "Selected model does not match this dataset. Missing columns: "
                + ", ".join(missing)
            )
            st.info(
                "Pick a model from the sidebar that matches the selected dataset, "
                "or switch between Survival and Diagnostic."
            )
        else:
            try:
                probs = model.predict_proba(model_input_df)[0]
                proba = float(probs[positive_class_index(model, probs)])
            except Exception:
                try:
                    proba = float(model.predict(model_input_df)[0])
                except Exception as e:
                    st.error(f"Model evaluation failed: {e}")

        if proba is not None:
            pct = int(round(proba * 100))
            if dataset_choice == "Survival":
                st.metric("Predicted 5-year survival (approx)", f"{pct}%")
                risk_label = "lower risk" if proba >= 0.5 else "intermediate-to-higher"
                st.write(f"Risk group (threshold 0.5): **{risk_label}**")
            else:
                st.metric("Predicted malignant probability", f"{pct}%")
                risk_label = "elevated" if proba >= 0.5 else "lower"
                st.write(f"Risk group (threshold 0.5): **{risk_label}**")

            shap_runs = sorted(
                [p for p in RESULTS_DIR.glob("shap_compare_*") if p.is_dir()]
            )
            shap_local = None
            if shap_runs:
                latest = shap_runs[-1]
                shap_model_name = (
                    "logistic_regression"
                    if "logistic" in str(model_path).lower()
                    else "random_forest"
                )
                shap_local = load_shap_local(latest, shap_model_name)

            top_drivers = []
            if shap_local is not None:
                try:
                    feat_names = pipeline_input_columns(model) or model_input_df.columns.tolist()
                    top_drivers = shap_local_to_top(shap_local, feat_names, top_n=5)
                except Exception:
                    top_drivers = []

            if not top_drivers and df is not None:
                background_df = input_columns_for_dataset(dataset_choice, df)
                live_shap, feat_names = compute_live_shap_local(
                    model, model_input_df, background_df
                )
                if live_shap is not None:
                    try:
                        top_drivers = shap_local_to_top(live_shap, feat_names, top_n=5)
                    except Exception:
                        top_drivers = []

            if top_drivers:
                st.subheader("Top contributing features (local SHAP)")
                for fn, val in top_drivers:
                    st.write(f"- **{fn}**: {val:+.3f}")
            else:
                st.info(
                    "SHAP could not be computed for this model and patient row."
                )

            st.subheader("Patient-facing explanation (suggested)")
            context = []
            for k in ["age", "tumor_size", "regional_node_positive", "radius_mean", "diagnosis"]:
                if k in patient_df.columns:
                    context.append(f"{k}={patient_df.iloc[0][k]}")

            driver_text = ", ".join(fn.replace("num__", "").replace("cat__", "") for fn, _ in top_drivers[:3])
            if not driver_text:
                driver_text = "clinical and tumor-related measurements"

            if dataset_choice == "Survival":
                rl = "intermediate-to-higher risk" if proba < 0.5 else "lower risk"
                expl = (
                    f"Based on the information provided, this model estimates about a {pct}% "
                    f"chance of being alive at 5 years, placing this case in a {rl} group. "
                    f"The factors that influenced this prediction most were: {driver_text}. "
                    "This estimate is probabilistic and not a definitive medical judgment."
                )
                prompt = {
                    "Predicted_probability_alive_5yr": f"{pct}%",
                    "Risk_group_label": rl,
                    "Top_drivers": [{fn: float(val)} for fn, val in top_drivers[:3]],
                    "Example_patient_context": ", ".join(context) or "n/a",
                }
            else:
                rl = "elevated malignant risk" if proba >= 0.5 else "lower malignant risk"
                expl = (
                    f"This model estimates a {pct}% probability of malignancy ({rl}). "
                    f"The strongest contributors were: {driver_text}. "
                    "This is a research tool and not a clinical diagnosis."
                )
                prompt = {
                    "Predicted_malignant_probability": f"{pct}%",
                    "Risk_group_label": rl,
                    "Top_drivers": [{fn: float(val)} for fn, val in top_drivers[:3]],
                    "Example_patient_context": ", ".join(context) or "n/a",
                }

            st.write(expl)
            st.subheader("LLM prompt (copy to your preferred LLM)")
            prompt["Model_limitations"] = (
                "Model provides probability estimates from historical data only."
            )
            prompt["Suggested_next_step"] = (
                "Review with oncology team; combine with imaging and pathology"
            )
            st.code(json.dumps(prompt, indent=2))

else:
    st.info("Provide a patient row or select an index to proceed.")
