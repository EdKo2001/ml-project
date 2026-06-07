"""Create a small stratified held-out set and save UIDs for reproducible demos.

Saves:
- data/processed/heldout_survival.csv  (rows from the processed survival CSV)
- results/metrics/heldout_survival_uids.json  (list of UIDs)

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
PROC = ROOT / "data" / "processed" / "breast_cancer_survival_processed.csv"
HELDOUT_CSV = ROOT / "data" / "processed" / "heldout_survival.csv"
HELDOUT_UIDS = ROOT / "results" / "metrics" / "heldout_survival_uids.json"


def make_heldout(frac: float = 0.05, random_state: int = 42):
    if not PROC.exists():
        raise FileNotFoundError(f"Processed survival CSV not found: {PROC}")

    df = pd.read_csv(PROC).reset_index(drop=True)

    # Add stable uid
    if "uid" not in df.columns:
        df["uid"] = df.index.astype(str)

    # Ensure binary target exists
    if "survived_5yr" not in df.columns:
        if "survival_months" in df.columns:
            df["survived_5yr"] = (df["survival_months"] >= 60).astype(int)
        else:
            raise ValueError("Neither 'survived_5yr' nor 'survival_months' found in processed CSV")

    # Determine heldout size
    if frac <= 0 or frac >= 1:
        raise ValueError("frac must be between 0 and 1")

    # Use stratified split to preserve label ratios
    heldout, remainder = train_test_split(
        df, test_size=1 - frac, stratify=df["survived_5yr"], random_state=random_state
    )

    # Save heldout rows (UID + columns for convenience)
    HELDOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(HELDOUT_CSV, index=False)

    # Save UIDs list in results/metrics
    HELDOUT_UIDS.parent.mkdir(parents=True, exist_ok=True)
    HELDOUT_UIDS.write_text(json.dumps(list(heldout["uid"])))

    print(f"Wrote heldout CSV: {HELDOUT_CSV}")
    print(f"Wrote heldout UIDs: {HELDOUT_UIDS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified heldout set for survival dataset")
    parser.add_argument("--frac", type=float, default=0.05, help="Fraction to hold out (0-1)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()
    make_heldout(frac=args.frac, random_state=args.random_state)
