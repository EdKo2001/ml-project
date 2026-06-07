#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "breast_cancer_survival_processed.csv"
DEFAULT_PATTERN = "shap_compare_*"
DEFAULT_OUTPUT = "presentation_shap_story.pptx"
DEFAULT_EXAMPLE_INDEX = 1500

EXPLANATION_SLIDES: list[tuple[str, list[str]]] = [
    (
        "What SHAP Means",
        [
            "SHAP quantifies how each feature contributes to a single prediction.",
            "Positive contribution pushes toward higher 5-year survival probability; negative contribution pushes lower.",
            "SHAP explains model behavior, not clinical causality.",
            "We use SHAP to improve transparency and communication of predictions.",
        ],
    ),
    (
        "Global SHAP Interpretation",
        [
            "Global SHAP summarizes which features matter most across many patients.",
            "Both models emphasize disease-burden factors such as tumor size and nodal burden.",
            "Shared top drivers increase confidence that key signals are robust.",
            "Ranking differences are expected because RandomForest is nonlinear while LogisticRegression is linear.",
        ],
    ),
    (
        "RandomForest SHAP Interpretation",
        [
            "RandomForest captures nonlinear interactions between clinical features.",
            "High mean absolute SHAP indicates strong influence across the cohort.",
            "This view is useful for detecting complex interaction patterns.",
            "Use this explanation with calibration and clinical review for decisions.",
        ],
    ),
    (
        "LogisticRegression SHAP Interpretation",
        [
            "LogisticRegression SHAP provides a linear additive explanation.",
            "High mean absolute SHAP indicates consistent directional effect.",
            "This view is often easier to communicate to non-technical stakeholders.",
            "Use as a baseline interpretability reference against more flexible models.",
        ],
    ),
    (
        "Local SHAP for One Patient",
        [
            "Local SHAP decomposes one patient prediction into feature-level pushes.",
            "Bars show which factors increase or decrease predicted survival for this patient.",
            "This makes the model output explainable at case level.",
            "Interpret as model evidence rather than causal clinical evidence.",
        ],
    ),
    (
        "Model Comparison for One Patient",
        [
            "Agreement between models on top local drivers strengthens explanation confidence.",
            "Disagreement indicates model sensitivity and higher uncertainty.",
            "Model disagreement should trigger deeper expert review, not automatic override.",
            "Compare both model explanations before high-impact decisions.",
        ],
    ),
    (
        "Limitations and Guardrails",
        [
            "SHAP is associative and does not prove causality.",
            "Correlated features can split or shift attribution mass.",
            "Background sampling and preprocessing choices affect SHAP values.",
            "Validate explanation stability on external cohorts before deployment claims.",
        ],
    ),
]


def find_latest_run(results_dir: Path, pattern: str) -> Path:
    runs = sorted([p for p in results_dir.glob(pattern) if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run folders found for pattern: {pattern}")
    return runs[-1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def clean_feature_name(raw_name: str) -> str:
    name = raw_name.replace("num__", "").replace("cat__", "")
    if "_" not in name:
        return name
    head, tail = name.split("_", 1)
    # Numeric features usually remain as a single token after prefix removal.
    if head in {"age", "grade", "tumor", "regional"} and not tail.startswith(("T", "N", "I")):
        return name.replace("_", " ")
    return f"{head} = {tail}"


def top_items(d: dict[str, Any], n: int = 5) -> list[tuple[str, float]]:
    pairs = []
    for k, v in d.items():
        try:
            pairs.append((str(k), float(v)))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]


def load_patient_row(data_path: Path, example_index: int) -> dict[str, Any]:
    if not data_path.exists():
        return {}
    df = pd.read_csv(data_path)
    if df.empty:
        return {}
    idx = max(0, min(example_index, len(df) - 1))
    row = df.iloc[idx].to_dict()
    if "survived_5yr" not in row and "survival_months" in row:
        try:
            row["survived_5yr"] = int(float(row["survival_months"]) >= 60)
        except Exception:
            row["survived_5yr"] = "unknown"
    return row


def ensure_local_shap_image(
    run_dir: Path,
    model_name: str,
    fallback_feature_order: list[tuple[str, float]],
) -> Path:
    img_path = run_dir / f"{model_name}_shap_example_local.png"
    if img_path.exists():
        return img_path

    job_path = run_dir / f"{model_name}_shap_example_local.joblib"
    if not job_path.exists():
        return img_path

    try:
        obj = joblib.load(job_path)
    except Exception:
        return img_path

    if obj is None:
        return img_path

    # Prefer SHAP's own bar plot so visual style matches existing SHAP slides.
    try:
        import shap

        if hasattr(obj, "values"):
            shap.plots.bar(obj[0], show=False)
            plt.gcf().tight_layout()
            plt.savefig(img_path, dpi=150)
            plt.close()
            return img_path
    except Exception:
        pass

    try:
        if hasattr(obj, "values"):
            vals = np.asarray(obj.values)
            if vals.ndim == 3:
                class_idx = 1 if vals.shape[2] > 1 else 0
                contrib = vals[0, :, class_idx]
            elif vals.ndim == 2:
                contrib = vals[0, :]
            else:
                contrib = np.ravel(vals)
            magnitudes = np.abs(contrib)
        else:
            arr = np.asarray(obj)
            if arr.ndim == 3:
                contrib = arr[1, 0, :] if arr.shape[0] > 1 else arr[0, 0, :]
            elif arr.ndim == 2:
                contrib = arr[0, :]
            else:
                contrib = np.ravel(arr)
            magnitudes = np.abs(contrib)

        if magnitudes.size == 0:
            return img_path

        if fallback_feature_order:
            feature_names = [clean_feature_name(k) for k, _ in fallback_feature_order]
            if len(feature_names) < magnitudes.size:
                feature_names += [f"f{i}" for i in range(len(feature_names), magnitudes.size)]
            feature_names = feature_names[: magnitudes.size]
        else:
            feature_names = [f"f{i}" for i in range(magnitudes.size)]

        top_n = min(20, magnitudes.size)
        top_idx = np.argsort(magnitudes)[-top_n:][::-1]
        top_vals = [float(magnitudes[i]) for i in top_idx]
        top_feats = [feature_names[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.22)))
        ax.barh(top_feats[::-1], top_vals[::-1])
        ax.set_title(f"{model_name}: Local SHAP (example patient)")
        fig.tight_layout()
        fig.savefig(img_path, dpi=150)
        plt.close(fig)
    except Exception:
        return img_path

    return img_path


def summarize_conclusion(
    rf_top: list[tuple[str, float]],
    lr_top: list[tuple[str, float]],
    patient: dict[str, Any],
) -> list[str]:
    rf_names = [k for k, _ in rf_top]
    lr_names = [k for k, _ in lr_top]
    overlap = [k for k in rf_names if k in lr_names]

    bullets = []
    if overlap:
        readable = ", ".join(clean_feature_name(x) for x in overlap[:3])
        bullets.append(f"Both models agree on core drivers: {readable}.")
    else:
        bullets.append("The models focus on different drivers, so explanations are model-sensitive.")

    if rf_top and lr_top:
        rf_lead = clean_feature_name(rf_top[0][0])
        lr_lead = clean_feature_name(lr_top[0][0])
        bullets.append(f"Top RF driver: {rf_lead}; top LR driver: {lr_lead}.")

    status = patient.get("status", "unknown")
    age = patient.get("age", "unknown")
    tumor = patient.get("tumor_size", "unknown")
    nodes = patient.get("regional_node_positive", "unknown")
    bullets.append(
        f"Sample patient context: age {age}, tumor size {tumor}, positive nodes {nodes}, observed status {status}."
    )

    bullets.append(
        "Recommendation: keep SHAP as explanation support, and pair with calibration/performance before decisions."
    )
    return bullets


def add_title(prs: Presentation, run_dir: Path) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "SHAP Explainability Story"
    s.placeholders[1].text = (
        "RandomForest vs LogisticRegression\n"
        f"Run: {run_dir.name}  |  Generated: {datetime.now().isoformat(timespec='seconds')}"
    )


def add_bullets_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    tf = s.shapes.placeholders[1].text_frame
    tf.clear()
    first = True
    for item in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.text = item
        first = False


def add_image_slide(prs: Presentation, title: str, image_path: Path, width: float = 8.4) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = title
    if image_path.exists():
        s.shapes.add_picture(str(image_path), Inches(0.8), Inches(1.35), width=Inches(width))
    else:
        tb = s.shapes.add_textbox(Inches(0.9), Inches(2.5), Inches(8), Inches(1.5))
        tb.text_frame.text = f"Missing image: {image_path.name}"


def add_explanation_slides(prs: Presentation) -> None:
    for title, bullets in EXPLANATION_SLIDES:
        add_bullets_slide(prs, title, bullets)


def add_patient_model_slide(
    prs: Presentation,
    title: str,
    patient: dict[str, Any],
    model_top: list[tuple[str, float]],
    local_img: Path,
) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = title

    tb = s.shapes.add_textbox(Inches(0.6), Inches(1.2), Inches(4.7), Inches(5.8))
    tf = tb.text_frame
    tf.clear()

    p = tf.paragraphs[0]
    p.text = "Patient snapshot"
    shown = 0
    for key, value in patient.items():
        if key in {"survival_months", "status", "survived_5yr"}:
            continue
        line = tf.add_paragraph()
        line.level = 1
        line.text = f"{key}: {value}"
        shown += 1
        if shown >= 8:
            break

    tf.add_paragraph().text = "Top SHAP drivers"
    if model_top:
        for feat, val in model_top[:5]:
            line = tf.add_paragraph()
            line.level = 1
            line.text = f"{clean_feature_name(feat)} ({val:.3f})"
    else:
        line = tf.add_paragraph()
        line.level = 1
        line.text = "No SHAP feature importance file found."

    if local_img.exists():
        s.shapes.add_picture(str(local_img), Inches(5.4), Inches(1.4), width=Inches(4.1))
    else:
        missing = s.shapes.add_textbox(Inches(5.5), Inches(2.5), Inches(3.8), Inches(1.5))
        missing.text_frame.text = f"Missing local SHAP image: {local_img.name}"


def build_presentation(
    run_dir: Path,
    patient: dict[str, Any],
    rf_top: list[tuple[str, float]],
    lr_top: list[tuple[str, float]],
    output_name: str,
) -> Path:
    prs = Presentation()

    rf_local = ensure_local_shap_image(run_dir, "random_forest", rf_top)
    lr_local = ensure_local_shap_image(run_dir, "logistic_regression", lr_top)

    add_title(prs, run_dir)
    add_bullets_slide(
        prs,
        "Recommended SHAP Presentation Style",
        [
            "Use a narrative flow: Purpose -> Global comparison -> Model details -> Patient-level evidence -> Decision guidance.",
            "Keep one claim per slide and pair each claim with one visual artifact.",
            "Separate global and local explanation slides to avoid mixing audience levels.",
            "End with agreement/disagreement and action-oriented caveats.",
        ],
    )
    add_explanation_slides(prs)

    add_image_slide(prs, "Global SHAP Comparison", run_dir / "shap_compare_models.png")
    add_image_slide(prs, "RandomForest Global SHAP", run_dir / "random_forest_shap_feature_importance.png")
    add_image_slide(prs, "LogisticRegression Global SHAP", run_dir / "logistic_regression_shap_feature_importance.png")

    add_patient_model_slide(
        prs,
        "Patient Evidence - RandomForest",
        patient,
        rf_top,
        rf_local,
    )
    add_patient_model_slide(
        prs,
        "Patient Evidence - LogisticRegression",
        patient,
        lr_top,
        lr_local,
    )

    add_bullets_slide(prs, "Data-Driven Conclusion", summarize_conclusion(rf_top, lr_top, patient))

    out_path = run_dir / output_name
    prs.save(str(out_path))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a SHAP storytelling presentation deck.")
    parser.add_argument("--run-pattern", default=DEFAULT_PATTERN, help="Result folder glob pattern.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Processed CSV path for patient row.")
    parser.add_argument("--example-index", type=int, default=DEFAULT_EXAMPLE_INDEX, help="Sample patient row index.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT, help="Output PPTX filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = find_latest_run(RESULTS_DIR, args.run_pattern)

    rf_imp = load_json(run_dir / "random_forest_shap_feature_importance.json")
    lr_imp = load_json(run_dir / "logistic_regression_shap_feature_importance.json")
    rf_top = top_items(rf_imp, n=8)
    lr_top = top_items(lr_imp, n=8)
    patient = load_patient_row(Path(args.data_path), args.example_index)

    out = build_presentation(
        run_dir=run_dir,
        patient=patient,
        rf_top=rf_top,
        lr_top=lr_top,
        output_name=args.output_name,
    )
    print(f"Wrote presentation: {out}")


if __name__ == "__main__":
    main()

