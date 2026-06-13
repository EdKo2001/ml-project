#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure repo root on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

RANDOM_STATE = 0
EXAMPLE_INDEX = 1500


def make_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
    return ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)])


def compute_shap_for_pipeline(pipeline, X_source: pd.DataFrame, example_index: Optional[int], out_dir: Path, name: str):
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("SHAP library required") from exc

    preproc = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_bg = X_source.sample(n=min(200, len(X_source)), random_state=RANDOM_STATE)
    X_bg_trans = preproc.transform(X_bg)

    try:
        expl = shap.Explainer(model, X_bg_trans)
    except Exception:
        try:
            expl = shap.Explainer(model, X_bg_trans, check_additivity=False)
        except Exception:
            # last resort: use TreeExplainer with additivity check disabled
            try:
                expl = shap.TreeExplainer(model, data=X_bg_trans, check_additivity=False)
            except Exception:
                raise

    # global importance from a sample
    X_samp = X_source.sample(n=min(500, len(X_source)), random_state=RANDOM_STATE)
    X_samp_trans = preproc.transform(X_samp)
    try:
        shap_vals = expl(X_samp_trans)
        arr = np.abs(shap_vals.values).mean(axis=0)
    except Exception:
        # fallback to TreeExplainer with probability output and disabled additivity check
        try:
            tre = shap.TreeExplainer(model, data=X_bg_trans, model_output="probability")
            sv = tre.shap_values(X_samp_trans)
            # shap_values may be list (for multiclass) or array
            if isinstance(sv, list):
                sv = np.array(sv)
                # choose second class if binary
                if sv.shape[0] >= 2:
                    arr = np.abs(sv[1]).mean(axis=0)
                else:
                    arr = np.abs(sv[0]).mean(axis=0)
            else:
                arr = np.abs(sv).mean(axis=0)
        except Exception:
            # TreeExplainer failed — try KernelExplainer as a slower fallback
            try:
                print(f"TreeExplainer failed for {name}, falling back to KernelExplainer on pipeline (slower)")
                # Use KernelExplainer on the full pipeline with original DataFrame inputs
                bg_n = min(50, max(1, X_bg.shape[0]))
                bg_orig = X_bg.iloc[:bg_n]
                f_pipe = lambda z: pipeline.predict_proba(pd.DataFrame(z, columns=X_source.columns))[:, 1]
                ke = shap.KernelExplainer(f_pipe, bg_orig)
                samp_n = min(100, X_samp.shape[0])
                ke_sv = ke.shap_values(X_samp.iloc[:samp_n], nsamples=100)
                if isinstance(ke_sv, list):
                    ke_sv = np.array(ke_sv)
                    if ke_sv.ndim == 3 and ke_sv.shape[0] >= 2:
                        arr = np.abs(ke_sv[1]).mean(axis=0)
                    else:
                        arr = np.abs(ke_sv[0]).mean(axis=0)
                else:
                    arr = np.abs(ke_sv).mean(axis=0)
            except Exception as e:
                print("Pipeline KernelExplainer also failed, trying transformed KernelExplainer:", e)
                try:
                    # try KernelExplainer on transformed data as last resort
                    bg_n = min(50, max(1, X_bg_trans.shape[0]))
                    bg = X_bg_trans[:bg_n]
                    f = lambda z: model.predict_proba(z)[:, 1]
                    ke = shap.KernelExplainer(f, bg)
                    samp_n = min(100, X_samp_trans.shape[0])
                    ke_sv = ke.shap_values(X_samp_trans[:samp_n], nsamples=100)
                    if isinstance(ke_sv, list):
                        ke_sv = np.array(ke_sv)
                        if ke_sv.ndim == 3 and ke_sv.shape[0] >= 2:
                            arr = np.abs(ke_sv[1]).mean(axis=0)
                        else:
                            arr = np.abs(ke_sv[0]).mean(axis=0)
                    else:
                        arr = np.abs(ke_sv).mean(axis=0)
                except Exception as e2:
                    print("KernelExplainer also failed:", e2)
                    arr = np.zeros(len(preproc.get_feature_names_out(X_samp.columns.tolist())))
    try:
        feat_names = preproc.get_feature_names_out(X_source.columns.tolist())
    except Exception:
        feat_names = [f"f{i}" for i in range(len(arr))]

    def _to_scalar(v):
        a = np.asarray(v)
        if a.size == 1:
            return float(a)
        return float(a.mean())

    feat_imp = dict(sorted({fn: _to_scalar(val) for fn, val in zip(feat_names, arr)}.items(), key=lambda x: x[1], reverse=True))
    (out_dir / f"{name}_shap_feature_importance.json").write_text(json.dumps(feat_imp, indent=2))

    # plot top features
    top_n = min(25, len(feat_names))
    top_feats = list(feat_imp.keys())[:top_n]
    top_vals = [feat_imp[f] for f in top_feats]
    fig, ax = plt.subplots(figsize=(8, max(3, top_n*0.22)))
    ax.barh(top_feats[::-1], top_vals[::-1])
    ax.set_title(f"{name}: Mean |SHAP| feature importance")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_shap_feature_importance.png", dpi=150)
    plt.close(fig)

    # local explanation for example
    if example_index is None or example_index < 0 or example_index >= len(X_source):
        ex_idx = 0
    else:
        ex_idx = example_index
    ex_row = X_source.iloc[[ex_idx]]
    ex_trans = preproc.transform(ex_row)
    try:
        ex_shap = expl(ex_trans)
    except Exception:
        try:
            tre = shap.TreeExplainer(model, data=X_bg_trans, model_output="probability")
            ex_shap = tre.shap_values(ex_trans)
        except Exception:
            try:
                # try KernelExplainer for local explanation
                bg_n = min(50, max(1, X_bg_trans.shape[0]))
                bg = X_bg_trans[:bg_n]
                f = lambda z: model.predict_proba(z)[:, 1]
                ke = shap.KernelExplainer(f, bg)
                ex_ke = ke.shap_values(ex_trans, nsamples=100)
                ex_shap = ex_ke
            except Exception:
                # failed to compute local SHAP for this model
                ex_shap = None
    joblib.dump(ex_shap, out_dir / f"{name}_shap_example_local.joblib")
    try:
        shap.plots.bar(ex_shap[0], show=False)
        plt.gcf().tight_layout()
        plt.savefig(out_dir / f"{name}_shap_example_local.png", dpi=150)
        plt.close()
    except Exception:
        pass

    return feat_imp


def compare_and_plot(base_imp: dict, other_imp: dict, out_path: Path, title: str = "SHAP comparison", base_label: str = 'base', other_label: str = 'other'):
    # union of top features
    all_feats = list(dict.fromkeys(list(base_imp.keys()) + list(other_imp.keys())))
    top_n = 25
    feats = all_feats[:top_n]
    base_vals = [base_imp.get(f, 0.0) for f in feats]
    other_vals = [other_imp.get(f, 0.0) for f in feats]

    y = np.arange(len(feats))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, max(4, len(feats)*0.18)))
    # Explicit colors so legend matches models consistently
    base_color = 'orange'  # RandomForest
    other_color = 'blue'   # LogisticRegression
    ax.barh(y - height/2, base_vals, height, label=base_label, color=base_color)
    ax.barh(y + height/2, other_vals, height, label=other_label, color=other_color)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(example_index: Optional[int] = None):
    example_index = EXAMPLE_INDEX if example_index is None else example_index

    data_path = PROCESSED_DATA_DIR / "breast_cancer_survival_processed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing CSV at {data_path}")
    df = pd.read_csv(data_path)

    if "survival_months" in df.columns:
        df["survival_months"] = pd.to_numeric(df["survival_months"], errors="coerce")
    if "survived_5yr" not in df.columns:
        df["survived_5yr"] = (df["survival_months"] >= 60).astype(int)

    feature_cols = [c for c in df.columns if c not in ["survival_months", "status", "survived_5yr"]]
    X = df[feature_cols]
    y = df["survived_5yr"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    preprocessor = make_preprocessor(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    lr = LogisticRegression(max_iter=2000, solver='liblinear')

    pipe_rf = Pipeline([('preprocessor', preprocessor), ('model', rf)])
    pipe_lr = Pipeline([('preprocessor', preprocessor), ('model', lr)])

    print('Fitting RandomForest...')
    pipe_rf.fit(X_trainval, y_trainval)
    print('Fitting LogisticRegression...')
    pipe_lr.fit(X_trainval, y_trainval)

    run_id = datetime.now().strftime('shap_compare_%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # compute SHAP
    try:
        base_imp = compute_shap_for_pipeline(pipe_rf, X_trainval, example_index, run_dir, 'random_forest')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print('RandomForest SHAP failed:', repr(exc))
        base_imp = {}
    try:
        other_imp = compute_shap_for_pipeline(pipe_lr, X_trainval, example_index, run_dir, 'logistic_regression')
    except Exception as exc:
        print('Logistic SHAP failed:', exc)
        other_imp = {}

    # compare
    try:
        compare_and_plot(base_imp, other_imp, run_dir / 'shap_compare_models.png', title='RandomForest vs Logistic: Mean |SHAP|', base_label='RandomForest', other_label='LogisticRegression')
    except Exception as exc:
        print('Comparison plot failed:', exc)

    print('Saved SHAP comparison run to', run_dir)


if __name__ == '__main__':
    main()
