"""Small evaluation helpers for classification experiments.

Minimal functions to compute common metrics and save a JSON report.
"""

from __future__ import annotations

from typing import Dict, Iterable
import json
from pathlib import Path
import numpy as np


def _choose_average(y_true: Iterable) -> str:
    return "binary" if len(np.unique(list(y_true))) == 2 else "weighted"


def evaluate(
    y_true: Iterable, y_pred: Iterable, zero_division: int = 0
) -> Dict[str, float]:
    """Return a dict with accuracy, precision, recall, and f1."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    avg = _choose_average(y_true)
    pos_label = None
    if avg == "binary":
        uniq = np.unique(list(y_true))
        pos_label = uniq[-1]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, average=avg, zero_division=zero_division, pos_label=pos_label
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, average=avg, zero_division=zero_division, pos_label=pos_label
            )
        ),
        "f1": float(
            f1_score(y_true, y_pred, average=avg, zero_division=zero_division, pos_label=pos_label)
        ),
    }


def save_report(report: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")


def classification_summary(
    y_true: Iterable, y_pred: Iterable, *, labels: Iterable | None = None
) -> str:
    """Return a text classification report using sklearn's `classification_report`."""
    from sklearn.metrics import classification_report

    return classification_report(y_true, y_pred, labels=labels, zero_division=0)
