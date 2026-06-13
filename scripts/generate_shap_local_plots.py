#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS = ["random_forest", "logistic_regression"]


def find_latest_run(pattern: str) -> Path:
    runs = sorted([p for p in RESULTS_DIR.glob(pattern) if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return runs[-1]


def generate_local_plot(run_dir: Path, model_name: str, overwrite: bool) -> None:
    job = run_dir / f"{model_name}_shap_example_local.joblib"
    out = run_dir / f"{model_name}_shap_example_local.png"
    if out.exists() and not overwrite:
        print(out, "already exists")
        return
    if not job.exists():
        print("Missing joblib for", model_name)
        return

    obj = joblib.load(job)
    if obj is None:
        print("No SHAP object saved for", model_name)
        return

    try:
        # Handle shap.Explanation objects
        if hasattr(obj, "values"):
            vals_arr = np.asarray(obj.values)
            if vals_arr.ndim == 3:
                class_idx = 1 if vals_arr.shape[2] > 1 else 0
                feat_contrib = vals_arr[0, :, class_idx]
            elif vals_arr.ndim == 2:
                feat_contrib = vals_arr[0, :]
            else:
                feat_contrib = np.ravel(vals_arr)
            vals = np.abs(feat_contrib)
        else:
            arr = None
            if isinstance(obj, list):
                try:
                    a = np.array(obj)
                    if a.ndim == 3 and a.shape[0] >= 2:
                        arr = a[1]
                    else:
                        arr = a[0]
                except Exception:
                    arr = np.asarray(obj[0])
            else:
                arr = np.asarray(obj)

            if arr.ndim == 2:
                vals = np.abs(arr).mean(axis=0)
            elif arr.ndim == 1:
                vals = np.abs(arr)
            else:
                vals = np.ravel(np.abs(arr))

        feat_json = run_dir / f"{model_name}_shap_feature_importance.json"
        if feat_json.exists():
            with open(feat_json) as f:
                imp = json.load(f)
            feat_names = list(imp.keys())
        else:
            feat_names = [f"f{i}" for i in range(len(vals))]

        top_n = min(20, len(feat_names))
        top_idx = np.argsort(vals)[-top_n:][::-1]
        top_feats = [feat_names[i] for i in top_idx]
        top_vals = [float(vals[i]) for i in top_idx]
        fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.22)))
        ax.barh(top_feats[::-1], top_vals[::-1])
        ax.set_title(f"{model_name}: Local SHAP (example patient)")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("Wrote", out)
    except Exception as e:
        print("Failed to create plot for", model_name, e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local SHAP PNG plots from saved joblib artifacts.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Specific SHAP run folder.")
    parser.add_argument("--run-pattern", default="shap_compare_*", help="Pattern used when --run-dir is omitted.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir is not None else find_latest_run(args.run_pattern)
    for model in MODELS:
        generate_local_plot(run_dir, model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
