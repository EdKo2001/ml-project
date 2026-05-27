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
        return DecisionTreeClassifier(random_state=kwargs.get("random_state", 42), **{k: v for k, v in kwargs.items() if k != 'random_state'})
    if name in ("random_forest", "rf"):
        return RandomForestClassifier(random_state=kwargs.get("random_state", 42), **{k: v for k, v in kwargs.items() if k != 'random_state'})
    if name in ("svm", "svc"):
        return SVC(probability=True, random_state=kwargs.get("random_state", 42), **{k: v for k, v in kwargs.items() if k != 'random_state'})
    if name in ("mlp", "mlpclassifier", "neural_network"):
        # sensible defaults for quick training; callers can override via kwargs
        defaults = {"hidden_layer_sizes": (100,), "max_iter": 200}
        params = {**defaults, **{k: v for k, v in kwargs.items() if k != 'random_state'}}
        return MLPClassifier(random_state=kwargs.get("random_state", 42), **params)
    raise ValueError(f"Unknown model name: {name}")
