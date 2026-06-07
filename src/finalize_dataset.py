"""Ensure processed datasets exist for the breast-cancer tasks.

This script runs the dataset preparation helpers to produce cleaned
CSV and profile JSON files under `data/processed/` when they are missing.
"""
from __future__ import annotations

from pathlib import Path
from src.prepare_breast_cancer_dataset import (
    DEFAULT_RAW_PATH as BC_RAW,
    DEFAULT_PROCESSED_PATH as BC_PROCESSED,
    DEFAULT_PROFILE_PATH as BC_PROFILE,
    prepare_dataset as prepare_bc,
)
from src.prepare_breast_cancer_survival_dataset import (
    DEFAULT_RAW_PATH as SURV_RAW,
    DEFAULT_PROCESSED_PATH as SURV_PROCESSED,
    DEFAULT_PROFILE_PATH as SURV_PROFILE,
    prepare_dataset as prepare_surv,
)


def ensure(path: Path) -> bool:
    if path.exists():
        return False
    return True


def main() -> None:
    # Breast cancer diagnostic
    if ensure(Path(BC_PROCESSED)):
        print(f"Preparing diagnostic dataset -> {BC_PROCESSED}")
        prepare_bc(Path(BC_RAW), Path(BC_PROCESSED), Path(BC_PROFILE))
    else:
        print(f"Diagnostic processed file already exists: {BC_PROCESSED}")

    # Survival / status dataset
    if ensure(Path(SURV_PROCESSED)):
        print(f"Preparing survival dataset -> {SURV_PROCESSED}")
        prepare_surv(Path(SURV_RAW), Path(SURV_PROCESSED), Path(SURV_PROFILE))
    else:
        print(f"Survival processed file already exists: {SURV_PROCESSED}")


if __name__ == "__main__":
    main()
