# ml-project

Shared setup for the CS582 predictive maintenance project.

## Structure

- `data/` for raw and cleaned data
- `notebooks/` for shared analysis and model work
- `src/` for reusable code
- `results/` for metrics, plots, and exports
- `slides/` for presentation files

## Next steps

1. Download the Kaggle dataset.
2. Load it in one shared notebook.
3. Check target balance and basic feature types.
4. Build one preprocessing pipeline that everyone uses.

## Recommended versions & setup

Use a recent stable Python 3 release. We recommend Python 3.10, 3.11, or 3.12 for compatibility with the libraries used in this project.

Use either the classic Jupyter Notebook or JupyterLab to run the notebooks. Recommended versions:
- Python: 3.10 - 3.12
- Jupyter Notebook: >=6.0 or JupyterLab: >=3.0

Quick setup (Windows / PowerShell):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Run the shared notebook:
```powershell
jupyter notebook notebooks/01_shared_setup.ipynb
```

If you prefer `conda`, create a conda env and install the dependencies from `requirements.txt` or manually install the packages listed there.

## Shared pipeline usage

Use the shared entrypoint so everyone loads data, builds preprocessing, and splits the same way:

```python
from src.data_pipeline import build_data_pipeline

artifacts = build_data_pipeline(
	"data/raw/predictive_maintenance.csv",
	target_column="Machine failure",
	test_size=0.2,
	random_state=42,
)

X_train = artifacts.X_train
y_train = artifacts.y_train
```

## MLP baseline results
Total records in dataset: 10,000
Test set (n=2000):
- Accuracy: 0.998
- ROC AUC: 0.987
- Confusion matrix: [[1931, 1], [4, 64]]
- Class 1 (failure) precision/recall: 0.985 / 0.941
Interpretation:
- Class 1 (failures) has precision 0.985 and recall 0.941.
- Out of 68 failures, it missed 4 (false negatives) and correctly caught 64.
- Class 0 has almost perfect precision/recall.
- Confusion matrix breakdown: 1931 true negatives, 1 false positive, 4 false negatives, 64 true positives.
- Confusion matrix meanings:
	- 1931 = actual 0, predicted 0 (true negative)
	- 1 = actual 0, predicted 1 (false positive)
	- 4 = actual 1, predicted 0 (false negative)
	- 64 = actual 1, predicted 1 (true positive)
