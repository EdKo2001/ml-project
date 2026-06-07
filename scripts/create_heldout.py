"""Create small stratified held-out sets and save UIDs for reproducible demos.

Saves:
- data/processed/heldout_survival.csv
- data/processed/heldout_diagnostic.csv
- results/metrics/heldout_survival_uids.json
- results/metrics/heldout_diagnostic_uids.json

Usage:
    $env:PYTHONPATH="."; python scripts/create_heldout.py --frac 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "survival": {
        "proc": ROOT / "data" / "processed" / "breast_cancer_survival_processed.csv",
        "heldout_csv": ROOT / "data" / "processed" / "heldout_survival.csv",
        "heldout_uids": ROOT / "results" / "metrics" / "heldout_survival_uids.json",
        "target": "survived_5yr",
        "derived_target": "survival_months",
        "derived_target_threshold": 60,
    },
    "diagnostic": {
        "proc": ROOT / "data" / "processed" / "breast_cancer_processed.csv",
        "heldout_csv": ROOT / "data" / "processed" / "heldout_diagnostic.csv",
        "heldout_uids": ROOT / "results" / "metrics" / "heldout_diagnostic_uids.json",
        "target": "diagnosis",
        "derived_target": None,
        "derived_target_threshold": None,
    },
}


def _prepare_df(proc_path: Path, target: str, derived_target: str | None, derived_target_threshold: int | None):
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {proc_path}")

    df = pd.read_csv(proc_path).reset_index(drop=True)

    if "uid" not in df.columns:
        df["uid"] = df.index.astype(str)

    if target not in df.columns:
        if derived_target and derived_target in df.columns and derived_target_threshold is not None:
            df[target] = (df[derived_target] >= derived_target_threshold).astype(int)
        else:
            raise ValueError(f"Target column '{target}' not found in processed CSV {proc_path}")

    return df


def _make_heldout_for_spec(name: str, frac: float, random_state: int):
    spec = SPECS[name]
    df = _prepare_df(spec["proc"], spec["target"], spec["derived_target"], spec["derived_target_threshold"])

    if frac <= 0 or frac >= 1:
        raise ValueError("frac must be between 0 and 1")

    heldout, _remainder = train_test_split(
        df, test_size=1 - frac, stratify=df[spec["target"]], random_state=random_state
    )

    spec["heldout_csv"].parent.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(spec["heldout_csv"], index=False)

    spec["heldout_uids"].parent.mkdir(parents=True, exist_ok=True)
    spec["heldout_uids"].write_text(json.dumps(list(heldout["uid"])))

    print(f"Wrote heldout CSV: {spec['heldout_csv']}")
    print(f"Wrote heldout UIDs: {spec['heldout_uids']}")


def make_heldout(frac: float = 0.05, random_state: int = 42, dataset: str = "all"):
    dataset = dataset.lower()
    if dataset == "all":
        for name in SPECS:
            _make_heldout_for_spec(name, frac, random_state)
        return
    if dataset not in SPECS:
        raise ValueError("dataset must be one of: all, survival, diagnostic")
    _make_heldout_for_spec(dataset, frac, random_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create stratified heldout sets for survival and/or diagnostic datasets"
    )
    parser.add_argument(
        "--frac", type=float, default=0.05, help="Fraction to hold out (0-1)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "survival", "diagnostic"],
        help="Which dataset to split",
    )
    args = parser.parse_args()
    make_heldout(frac=args.frac, random_state=args.random_state, dataset=args.dataset)
