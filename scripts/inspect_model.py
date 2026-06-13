import joblib, json, sys
from pathlib import Path

p = Path("results/metrics/breast_cancer_model.joblib")
if not p.exists():
    print("MISSING")
    sys.exit(1)
model = joblib.load(p)
out = {}
out["type"] = str(type(model))
if hasattr(model, "named_steps"):
    out["steps"] = list(model.named_steps.keys())
    pre = model.named_steps.get("preprocessor")
    out["pre_type"] = str(type(pre))
    try:
        out["pre_feature_names"] = list(pre.get_feature_names_out())
    except Exception as e:
        out["pre_feature_names_error"] = str(e)
    fni = getattr(model, "feature_names_in_", None)
    try:
        # convert numpy arrays to lists
        if hasattr(fni, "tolist"):
            out["feature_names_in_"] = fni.tolist()
        else:
            out["feature_names_in_"] = fni
    except Exception:
        out["feature_names_in_"] = str(fni)
else:
    out["attrs"] = [k for k in model.__dict__.keys() if not k.startswith("__")]


def safe(obj):
    try:
        import numpy as _np

        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


print(json.dumps({k: safe(v) for k, v in out.items()}, indent=2))
