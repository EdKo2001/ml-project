"""Helpers for handling class imbalance during training.

Minimal helpers: compute class weights and perform simple resampling.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Optional
import numpy as np


def compute_class_weights(y: Iterable):
    """Return a dict mapping class_label -> weight usable in `class_weight`."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = sorted(set(y))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=list(y))
    return dict(zip(classes, map(float, weights)))


def resample_dataset(X, y, method: str = "smote", random_state: Optional[int] = 42) -> Tuple:
    """Resample the dataset and return (X_res, y_res).

    Supported methods: 'smote', 'oversample', 'undersample'.
    """
    method = method.lower()
    if method == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=random_state)
    elif method == "oversample":
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(random_state=random_state)
    elif method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError("Unsupported resampling method: choose 'smote', 'oversample', or 'undersample'.")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res
