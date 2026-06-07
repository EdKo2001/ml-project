# ml-project

Shared code and experiments for the CS582 predictive-maintenance and breast-cancer modeling tasks.

## Repository layout

- `data/` — raw and processed datasets
- `notebooks/` — exploratory analysis and modeling notebooks
- `src/` — reusable modules and scripts
- `results/` — generated metrics, models, and visualizations
- `slides/` — presentation materials

## Purpose

This repository provides a shared preprocessing pipeline, model helpers, dataset preparation scripts, and evaluation utilities intended for classroom experiments and reproducible analyses.

## Notes on setup

Use a recent Python 3.10+ runtime. Install project dependencies from `requirements.txt` in your preferred environment manager. The code has been tested with scikit-learn and the common scientific Python stack.

## Core components

- `src/data_pipeline.py`: shared entrypoint to load data, build preprocessing, and produce train/test splits.
- `src/preprocessing.py`: preprocessing helpers and a ColumnTransformer builder.
- `src/models.py`: factory for common classifiers used in experiments.
- `src/prepare_breast_cancer_dataset.py` and `src/prepare_breast_cancer_survival_dataset.py`: dataset-specific cleaning and profiling scripts.
- `src/eval.py` and `src/imbalance.py`: lightweight evaluation and imbalance-handling helpers.
- `src/breast_cancer_pipeline.py`: example pipeline that trains, evaluates (F1), and saves a breast-cancer model.

## Evaluation guidance

- Always report `precision`, `recall`, and `F1` in addition to accuracy and AUC when classes are imbalanced.
- Use stratified cross-validation and report per-class metrics.
- For imbalance handling, consider estimator `class_weight` or resampling (SMOTE/oversampling/undersampling) depending on the experiment.

## Explainability

The project includes `shap` as an optional dependency. Use SHAP to generate feature importance summaries and per-sample explanations for final models; save results to `results/metrics` for reproducibility.

## Recent work

- Added evaluation and imbalance helpers, and an example breast-cancer pipeline that prepares data, trains a classifier, computes F1, and stores model/metrics artifacts.

## Next steps

- Add SHAP visualizations for chosen models and export findings to `results/metrics`.
- Prepare a concise `RESULTS.md` summarizing evaluation numbers and recommendations for the final report.

## Artifacts & versioning

Avoid committing large model binaries to git. Store models and large artifacts in external artifact storage or keep them in `results/` locally and add to `.gitignore` for repository cleanliness.

---

If you want the README shortened further or tailored to a specific audience (instructors, teammates, or reviewers), tell me which audience and I'll refine it.

## Actionable next steps (proposal & professor feedback)

 
- Standardize evaluation: use stratified cross-validation and report accuracy, precision, recall, F1, and ROC-AUC for all models; store JSON reports in `results/metrics` (see `src/eval.py`).
- Baseline model: implement Logistic Regression baseline (proposal) and compare with Random Forest / MLP baselines; prefer F1 as primary metric.
- Explainability: run SHAP on final models to produce global and local explanations; export SHAP plots and short interpretation notes to `results/metrics`.
- Future work extensions: experiment with deep-learning feature extraction (autoencoders, 1D-CNNs or pretrained encoders) and evaluate their impact on downstream classifiers.
- Documentation & deliverables: produce `RESULTS.md` summarizing experiments, a short methods section, and final presentation slides; include the Lab 6 solution as referenced by the instructor.

These steps map directly to the proposal and to the professor's comments: prioritize balancing the dataset, use F1 for selection, and add SHAP-based explanations and a future-work plan involving deep-learning feature extraction.
