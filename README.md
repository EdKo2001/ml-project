# ml-project

CS582 breast cancer machine learning project — diagnostic classification and survival modeling.

## Repository layout

- `data/` — raw and processed breast cancer datasets
- `notebooks/` — exploratory analysis and modeling notebooks
- `src/` — reusable modules, pipelines, and Streamlit UI
- `results/` — generated metrics, models, and visualizations
- `slides/` — presentation materials
- `scripts/` — training, SHAP, and utility scripts

## Datasets

| Dataset | File | Task |
|---|---|---|
| Wisconsin diagnostic | `data/raw/breast_cancer_dataset.csv` | Benign vs malignant (30 cell features) |
| Survival / staging | `data/raw/breast_cancer_dataset_2.csv` | 5-year survival and mortality status |

Run `python -m src.finalize_dataset` to build processed CSVs under `data/processed/`.

## Data splitting (train / test / demo holdout)

All modeling scripts use the same reproducible split settings unless noted otherwise:

| Split | Share | Purpose |
|---|---|---|
| **Training** | 80% | Fit preprocessing, resampling (SMOTE), and the model |
| **Test** | 20% | Final metrics only — never used for training or tuning |
| **Demo holdout** (optional) | ~5% | Fixed patient rows for Streamlit/UI demos |

**How the split is done**

1. Load a processed CSV from `data/processed/`.
2. Optionally remove demo rows first (see below).
3. Run a **stratified** `train_test_split` with `test_size=0.2` and `random_state=42` so class proportions stay balanced in both sets.

This is implemented in:

- `src/preprocessing.py` → `split_data()`
- `src/breast_cancer_pipeline.py` (diagnostic Random Forest)
- `scripts/train_survival_model.py` and notebooks `06` / `07` (survival task)

**Validation**

The repo does **not** keep a separate validation holdout by default. During experiments, treat the 80% training block as the development set. For hyperparameter search or model selection, use **stratified k-fold cross-validation on the training split only** (for example `GridSearchCV` with `cv=5`), then report the final chosen model once on the 20% test set.

Do not tune on the test set.

**Demo holdout (separate from train/test)**

For reproducible UI examples, create small stratified slices that are excluded from training when their UID list is present:

```bash
python scripts/create_heldout.py --frac 0.05 --random-state 42
```

This writes:

- `data/processed/heldout_diagnostic.csv`
- `data/processed/heldout_survival.csv`
- `results/metrics/heldout_*_uids.json`

`scripts/train_survival_model.py` drops any rows whose `uid` appears in `heldout_survival_uids.json` before the 80/20 split. The Streamlit app (`src/ui_streamlit.py`) can load these heldout CSVs for patient-level demos.

**Example sizes (diagnostic dataset, 569 rows)**

- Train: ~455 samples
- Test: ~114 samples
- Demo holdout (5%): ~28 samples (when generated)

**Rules of thumb**

- Fit preprocessing and resampling **inside** the training split only (pipelines in this repo do this via `Pipeline` + train-only SMOTE).
- Pick models using F1 / recall on imbalanced targets, not accuracy alone.
- Save JSON metric reports under `results/metrics/` for reproducibility.

## Quick start

```bash
pip install -r requirements.txt
python -m src.finalize_dataset
python -m src.breast_cancer_pipeline
streamlit run src/ui_streamlit.py
```

## Core components

- `src/prepare_breast_cancer_dataset.py` — clean and profile the diagnostic dataset
- `src/prepare_breast_cancer_survival_dataset.py` — clean and profile the survival dataset
- `src/data_pipeline.py` / `src/preprocessing.py` — shared preprocessing and splits
- `src/models.py` — Random Forest, Decision Tree, MLP factories
- `src/breast_cancer_pipeline.py` — train, evaluate, and save the diagnostic model
- `src/eval.py` / `src/imbalance.py` — metrics and imbalance handling
- `src/ui_streamlit.py` — patient-level predictions and SHAP explanations

## Evaluation

Report accuracy, precision, recall, F1, and ROC-AUC. Prefer F1 on imbalanced targets. Use stratified splits and store JSON reports under `results/metrics/`.

## Models

Primary deployed model: **Random Forest** (`results/metrics/breast_cancer_model.joblib`).

Compared baselines in notebooks: Decision Tree, MLP, Logistic Regression (survival task).
