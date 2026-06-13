#!/usr/bin/env python3
"""Build the CS582 final project presentation aligned with the course proposal."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = PROJECT_ROOT / "slides"
OUTPUT_NAME = "CS582_ML_Project_Presentation.pptx"

# Slide geometry (16:9)
SLIDE_W = Inches(10)
SLIDE_H = Inches(5.625)
MARGIN_L = Inches(0.6)
CONTENT_W = Inches(8.8)
CONTENT_TOP = Inches(1.25)
FOOTER_Y = Inches(5.15)

# Palette
NAVY = RGBColor(0x0D, 0x2B, 0x4E)
TEAL = RGBColor(0x00, 0x79, 0x6B)
LIGHT_TEAL = RGBColor(0xE0, 0xF2, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x21, 0x21, 0x21)
MUTED = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF4, 0xF6, 0xF8)

COURSE = "CS582 — Machine Learning"
PROJECT_TITLE = "Breast Cancer Prediction Using\nMachine Learning and Feature Selection"
TEAM_MEMBERS = [
    ("Eduard Koshkelyan", "618667"),
    ("Pham Hoang Quoc Viet Nguyen", "618690"),
    ("Quang Thanh Nguyen", "618992"),
    ("Thi Thanh Sen Doan", "618665"),
]


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self._slide_num = 0

    def _blank(self):
        self._slide_num += 1
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def _accent_bar(self, slide, color: RGBColor = TEAL) -> None:
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(0.08)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

    def _footer(self, slide, label: str = COURSE) -> None:
        box = slide.shapes.add_textbox(MARGIN_L, FOOTER_Y, CONTENT_W, Inches(0.35))
        tf = box.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = f"{label}  |  Slide {self._slide_num}"
        p.font.size = Pt(9)
        p.font.color.rgb = MUTED

    @staticmethod
    def _tf(box_or_cell, wrap: bool = True):
        tf = box_or_cell.text_frame if hasattr(box_or_cell, "text_frame") else box_or_cell
        tf.word_wrap = wrap
        tf.margin_left = Pt(6)
        tf.margin_right = Pt(6)
        tf.margin_top = Pt(3)
        tf.margin_bottom = Pt(3)
        return tf

    @staticmethod
    def _p(paragraph, text: str, *, size: int = 14, bold: bool = False, color: RGBColor = DARK, space: int = 5):
        paragraph.text = text
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(space)

    def title_slide(self) -> None:
        slide = self._blank()
        self._accent_bar(slide, NAVY)

        # Left panel
        panel = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.08), Inches(3.6), SLIDE_H
        )
        panel.fill.solid()
        panel.fill.fore_color.rgb = NAVY
        panel.line.fill.background()

        side = slide.shapes.add_textbox(Inches(0.45), Inches(1.6), Inches(2.8), Inches(3.0))
        tf = self._tf(side)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        self._p(tf.paragraphs[0], COURSE, size=13, bold=True, color=WHITE)
        self._p(tf.add_paragraph(), "Final Project", size=11, color=LIGHT_TEAL)
        self._p(tf.add_paragraph(), datetime.now().strftime("%B %Y"), size=11, color=LIGHT_TEAL)

        main = slide.shapes.add_textbox(Inches(4.0), Inches(1.15), Inches(5.5), Inches(3.5))
        tf = self._tf(main)
        self._p(tf.paragraphs[0], "Breast Cancer Prediction Using", size=28, bold=True, color=NAVY, space=4)
        self._p(tf.add_paragraph(), "Machine Learning and", size=28, bold=True, color=NAVY, space=4)
        self._p(tf.add_paragraph(), "Feature Selection Techniques", size=28, bold=True, color=TEAL, space=12)
        self._p(tf.add_paragraph(), "CS582 Project Team", size=14, bold=True, color=MUTED, space=4)
        for name, student_id in TEAM_MEMBERS:
            self._p(tf.add_paragraph(), f"{name} · {student_id}", size=11, color=MUTED, space=2)
        self._footer(slide)

    def section(self, title: str, subtitle: str = "") -> None:
        slide = self._blank()
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
        bg.fill.solid()
        bg.fill.fore_color.rgb = NAVY
        bg.line.fill.background()

        box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(2.0))
        tf = self._tf(box)
        self._p(tf.paragraphs[0], title, size=36, bold=True, color=WHITE, space=8)
        if subtitle:
            self._p(tf.add_paragraph(), subtitle, size=18, color=LIGHT_TEAL)
        self._footer(slide, COURSE)

    def bullets(self, title: str, items: list[str], *, size: int = 14, subtitle: str = "") -> None:
        slide = self._blank()
        self._accent_bar(slide)
        head = slide.shapes.add_textbox(MARGIN_L, Inches(0.25), CONTENT_W, Inches(0.75))
        self._p(self._tf(head).paragraphs[0], title, size=24, bold=True, color=NAVY)

        top = CONTENT_TOP
        if subtitle:
            sub = slide.shapes.add_textbox(MARGIN_L, Inches(0.95), CONTENT_W, Inches(0.45))
            self._p(self._tf(sub).paragraphs[0], subtitle, size=12, color=MUTED)
            top = Inches(1.35)

        body = slide.shapes.add_textbox(MARGIN_L, top, CONTENT_W, Inches(3.7))
        tf = self._tf(body)
        tf.vertical_anchor = MSO_ANCHOR.TOP
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            self._p(p, item, size=size)
        self._footer(slide)

    def two_col(self, title: str, left_h: str, left: list[str], right_h: str, right: list[str]) -> None:
        slide = self._blank()
        self._accent_bar(slide)
        head = slide.shapes.add_textbox(MARGIN_L, Inches(0.25), CONTENT_W, Inches(0.75))
        self._p(self._tf(head).paragraphs[0], title, size=24, bold=True, color=NAVY)

        col_w = Inches(4.15)
        gap = Inches(0.5)
        for x, heading, items in [
            (MARGIN_L, left_h, left),
            (MARGIN_L + col_w + gap, right_h, right),
        ]:
            box = slide.shapes.add_textbox(x, CONTENT_TOP, col_w, Inches(3.8))
            tf = self._tf(box)
            self._p(tf.paragraphs[0], heading, size=15, bold=True, color=TEAL, space=8)
            for item in items:
                self._p(tf.add_paragraph(), item, size=12)

        self._footer(slide)

    def table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
        *,
        col_widths: list[float] | None = None,
        footnote: str = "",
        font_size: int = 10,
        row_h: float = 0.58,
    ) -> None:
        slide = self._blank()
        self._accent_bar(slide)
        head = slide.shapes.add_textbox(MARGIN_L, Inches(0.25), CONTENT_W, Inches(0.75))
        self._p(self._tf(head).paragraphs[0], title, size=22, bold=True, color=NAVY)

        n_rows, n_cols = len(rows) + 1, len(headers)
        t_top = Inches(1.05)
        t_h = Inches(row_h * n_rows)
        shape = slide.shapes.add_table(n_rows, n_cols, MARGIN_L, t_top, CONTENT_W, t_h)
        table = shape.table

        if col_widths and len(col_widths) == n_cols:
            for i, w in enumerate(col_widths):
                table.columns[i].width = Inches(w)

        for c, h in enumerate(headers):
            cell = table.cell(0, c)
            cell.text = h
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in self._tf(cell).paragraphs:
                self._p(p, h, size=font_size, bold=True, color=WHITE, space=0)

        for r, row in enumerate(rows, 1):
            fill = LIGHT_GRAY if r % 2 == 0 else WHITE
            for c, val in enumerate(row):
                cell = table.cell(r, c)
                cell.text = val
                cell.fill.solid()
                cell.fill.fore_color.rgb = fill
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                for p in self._tf(cell).paragraphs:
                    self._p(p, val, size=font_size, space=0)

        if footnote:
            fn = slide.shapes.add_textbox(MARGIN_L, t_top + t_h + Inches(0.08), CONTENT_W, Inches(0.55))
            self._p(self._tf(fn).paragraphs[0], footnote, size=9, color=MUTED, space=0)
        self._footer(slide)

    def highlight_box(self, title: str, headline: str, bullets: list[str]) -> None:
        slide = self._blank()
        self._accent_bar(slide)
        head = slide.shapes.add_textbox(MARGIN_L, Inches(0.25), CONTENT_W, Inches(0.75))
        self._p(self._tf(head).paragraphs[0], title, size=24, bold=True, color=NAVY)

        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN_L, Inches(1.15), CONTENT_W, Inches(1.05)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_TEAL
        card.line.color.rgb = TEAL
        self._p(self._tf(card).paragraphs[0], headline, size=16, bold=True, color=NAVY, space=0)

        body = slide.shapes.add_textbox(MARGIN_L, Inches(2.35), CONTENT_W, Inches(2.8))
        tf = self._tf(body)
        for i, item in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            self._p(p, item, size=13)
        self._footer(slide)

    def agenda(self) -> None:
        items = [
            "1.  Background & motivation",
            "2.  Research objectives",
            "3.  Dataset & exploratory analysis",
            "4.  Methodology & feature selection",
            "5.  Model comparison & results",
            "6.  Explainability (SHAP) & demo",
            "7.  Conclusions, limitations & references",
        ]
        slide = self._blank()
        self._accent_bar(slide)
        head = slide.shapes.add_textbox(MARGIN_L, Inches(0.25), CONTENT_W, Inches(0.75))
        self._p(self._tf(head).paragraphs[0], "Agenda", size=24, bold=True, color=NAVY)

        y = 1.15
        for i, item in enumerate(items):
            row = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, MARGIN_L, Inches(y), CONTENT_W, Inches(0.48)
            )
            row.fill.solid()
            row.fill.fore_color.rgb = LIGHT_GRAY if i % 2 else WHITE
            row.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
            self._p(self._tf(row).paragraphs[0], item, size=13, color=DARK, space=0)
            y += 0.52
        self._footer(slide)

    def build(self) -> Path:
        self.title_slide()
        self.agenda()

        self.table(
            "Project Team",
            ["Name", "Student ID"],
            [[name, student_id] for name, student_id in TEAM_MEMBERS],
            col_widths=[6.2, 2.6],
            row_h=0.55,
            footnote="CS582 — Machine Learning final project group.",
        )

        self.bullets(
            "Clinical Motivation",
            [
                "Breast cancer is a leading cause of cancer-related deaths among women worldwide.",
                "Early and accurate diagnosis significantly improves treatment outcomes and survival.",
                "Machine learning can assist clinicians by classifying tumors as benign or malignant.",
                "A key challenge: redundant, correlated features that add noise and reduce generalization.",
            ],
            size=14,
        )

        self.bullets(
            "Research Problem",
            [
                "Many studies compare only a single algorithm under different conditions.",
                "Feature redundancy in medical datasets increases complexity without improving reliability.",
                "We need a fair, reproducible comparison of multiple classifiers on the same pipeline.",
                "Research question: Which model performs best, and does feature selection help?",
            ],
            size=14,
        )

        self.bullets(
            "Project Objectives",
            [
                "Classify Wisconsin Breast Cancer samples (Benign vs Malignant) using ML.",
                "Apply feature selection to reduce redundancy and improve model reliability.",
                "Compare Random Forest, Decision Tree, MLP, and Logistic Regression under identical conditions.",
                "Evaluate with Accuracy, Precision, Recall, F1-Score, ROC-AUC, and confusion matrices.",
                "Deliver reproducible code, visualizations, and explainability for clinical trust.",
            ],
            size=13,
        )

        self.section("Data & Exploration", "Wisconsin Breast Cancer Dataset (UCI / WBCD)")

        self.two_col(
            "Dataset Overview",
            "Source & structure",
            [
                "UCI Wisconsin Breast Cancer Dataset (WBCD)",
                "569 digitized fine-needle aspirate samples",
                "30 numeric features per sample",
                "Target: Benign (B) or Malignant (M)",
            ],
            "Feature groups (×3 each)",
            [
                "Mean — average cell nucleus measurement",
                "SE — standard error of the measurement",
                "Worst — largest (worst) observed value",
                "Attributes: radius, texture, perimeter, area,",
                "  smoothness, compactness, concavity, symmetry",
            ],
        )

        self.table(
            "Class Distribution & Split",
            ["Class", "Count", "Share", "Role"],
            [
                ["Benign (B)", "357", "62.7%", "Majority class"],
                ["Malignant (M)", "212", "37.3%", "Minority / positive class"],
                ["Train set", "455", "80%", "Model fitting"],
                ["Test set", "114", "20%", "Held-out evaluation"],
            ],
            col_widths=[2.2, 1.2, 1.2, 2.2],
            footnote="Stratified split (random_state=42). No missing values after preprocessing.",
        )

        self.section("Methodology", "Preprocessing, feature selection, and evaluation")

        self.bullets(
            "Experimental Pipeline",
            [
                "Load raw CSV → drop ID columns → encode diagnosis labels",
                "Stratified 80/20 train/test split on all experiments",
                "StandardScaler on all 30 numeric features",
                "Train classifiers on identical preprocessed splits",
                "Report metrics on held-out test set only",
                "Python stack: Pandas, NumPy, scikit-learn, Matplotlib, Jupyter",
            ],
            size=13,
        )

        self.two_col(
            "Feature Selection Strategy",
            "Analysis performed",
            [
                "Pearson correlation matrix across 30 features",
                "Identify highly correlated feature pairs (redundancy)",
                "Random Forest feature importance ranking",
                "SHAP mean |value| for global attribution",
            ],
            "Rationale",
            [
                "Redundant features increase noise and overfitting risk",
                "Selection improves interpretability for clinicians",
                "Top drivers: perimeter, concave points, radius, area",
                "Consistent with published WBCD literature",
            ],
        )

        self.bullets(
            "Models Compared",
            [
                "Random Forest — primary model in src/breast_cancer_pipeline.py",
                "Logistic Regression — interpretable linear baseline",
                "Decision Tree (max_depth=5) — fast, interpretable reference",
                "Multi-Layer Perceptron (64, 32) — neural network baseline",
                "All models evaluated with the same preprocessing and test split",
            ],
            size=13,
        )

        self.section("Results", "Held-out test set performance (114 samples)")

        self.table(
            "Model Performance Comparison",
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [
                ["Random Forest (deployed)", "0.974", "1.000", "0.929", "0.963", "0.996"],
                ["Decision Tree", "0.921", "0.946", "0.833", "0.886", "0.945"],
                ["MLP (64, 32)", "0.860", "0.964", "0.643", "0.771", "0.978"],
            ],
            col_widths=[2.2, 1.0, 1.1, 0.9, 0.9, 1.0],
            font_size=9,
            row_h=0.52,
            footnote="Positive class = Malignant (M). RF from src/breast_cancer_pipeline.py; others from notebooks/01_shared_setup.ipynb.",
        )

        self.highlight_box(
            "Selected Model: Random Forest",
            "Random Forest is used in the main pipeline, saved model, and Streamlit demo.",
            [
                "Strong F1 (0.963) with perfect malignant precision on the held-out test set",
                "Built-in feature importance supports our feature-selection analysis",
                "Integrates cleanly with SHAP TreeExplainer for explainability",
                "Saved artifact: results/metrics/breast_cancer_model.joblib",
            ],
        )

        self.two_col(
            "Explainability & Demo",
            "SHAP analysis",
            [
                "Global SHAP ranks most influential nucleus features",
                "Local SHAP explains individual patient predictions",
                "Compared linear vs nonlinear model attributions",
                "Supports clinical review — not a substitute for diagnosis",
            ],
            "Streamlit application",
            [
                "Command: streamlit run src/ui_streamlit.py",
                "Select patient → view prediction + probability",
                "Display top SHAP contributors per case",
                "Reproducible artifact export to results/",
            ],
        )

        self.bullets(
            "Limitations",
            [
                "Single dataset — no external validation cohort yet",
                "Small test set (114 samples) — metrics have sampling variance",
                "Digitized features only — no imaging deep-learning pipeline yet",
                "Research prototype — not validated for clinical deployment",
                "Class imbalance (37% malignant) requires careful metric selection",
            ],
            size=13,
        )

        self.bullets(
            "Future Work",
            [
                "Deep learning feature extraction (autoencoders, 1D-CNNs) per recent literature",
                "Stratified k-fold cross-validation and systematic hyperparameter search",
                "Expand to survival/outcome datasets with richer clinical staging",
                "External validation on independent breast cancer cohorts",
                "Calibration analysis and decision-curve evaluation for clinical utility",
            ],
            size=13,
        )

        self.bullets(
            "Key Conclusions",
            [
                "Multiple ML models achieve strong WBCD classification (AUC > 0.94).",
                "Random Forest is the selected model in our unified pipeline and demo.",
                "Feature selection reduces redundancy and aligns with clinical intuition.",
                "F1, precision, and recall are essential — accuracy alone is insufficient.",
                "SHAP and interactive demos improve transparency for stakeholders.",
            ],
            size=13,
        )

        self.bullets(
            "References",
            [
                "UCI ML Repository — Wisconsin Breast Cancer Dataset",
                "Carriero et al. (2024). Deep Learning in Breast Cancer Imaging. Diagnostics, 14(8), 848.",
                "Almansour et al. Neural Networks in Medical Diagnosis. MDPI.",
                "Recent ML/DL Review on Breast Cancer Detection (2025). Diagnostics, 16(4), 637.",
            ],
            size=11,
        )

        # Closing slide
        slide = self._blank()
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
        bg.fill.solid()
        bg.fill.fore_color.rgb = NAVY
        bg.line.fill.background()
        box = slide.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(7.0), Inches(2.0))
        tf = self._tf(box)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        self._p(tf.paragraphs[0], "Thank You", size=40, bold=True, color=WHITE, space=10)
        self._p(tf.add_paragraph(), "Questions & Discussion", size=20, color=LIGHT_TEAL, space=8)
        for name, student_id in TEAM_MEMBERS:
            self._p(tf.add_paragraph(), f"{name} ({student_id})", size=12, color=WHITE, space=2)
        self._footer(slide, COURSE)

        SLIDES_DIR.mkdir(parents=True, exist_ok=True)
        out = SLIDES_DIR / OUTPUT_NAME
        self.prs.save(str(out))
        return out


def main() -> None:
    out = Deck().build()
    print(f"Wrote presentation: {out}")


if __name__ == "__main__":
    main()
