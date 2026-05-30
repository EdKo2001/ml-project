"""Model helpers for the course project."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = {
    "svm": "Support Vector Machine",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def get_model(name: str, random_state: int = 42):
    """Return one of the shared baseline classifiers."""
    normalized_name = name.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized_name == "svm":
        return SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )

    if normalized_name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            random_state=random_state,
        )

    if normalized_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    valid_names = ", ".join(sorted(MODEL_NAMES))
    raise ValueError(f"Unknown model '{name}'. Choose one of: {valid_names}.")
