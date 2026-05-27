"""Model helpers for the course project."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_model(name: str, **kwargs):
    """Return a model instance by name.

    Supported names: 'decision_tree', 'random_forest', 'svm', 'mlp'
    """
    name = name.lower()
    if name in ("decision_tree", "dt"):
        params = {"random_state": 42, **kwargs}
        return DecisionTreeClassifier(**params)
    if name in ("random_forest", "rf"):
        params = {"random_state": 42, **kwargs}
        return RandomForestClassifier(**params)
    if name in ("svm", "svc"):
        params = {"probability": True, "random_state": 42, **kwargs}
        return SVC(**params)
    if name in ("mlp", "mlpclassifier", "neural_network"):
        # sensible defaults for quick training; callers can override via kwargs
        defaults = {"hidden_layer_sizes": (100,), "max_iter": 200, "random_state": 42}
        params = {**defaults, **kwargs}
        return MLPClassifier(**params)
    raise ValueError(f"Unknown model name: {name}")
