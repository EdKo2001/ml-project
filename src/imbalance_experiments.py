"""Run imbalance-handling experiments and save comparison report.

This script runs the existing `src.breast_cancer_pipeline.run` function using
different strategies and aggregates the resulting metrics into a single JSON
report under `results/metrics/imbalance_comparison.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.breast_cancer_pipeline import ensure_processed, run


def main():
    ensure_processed()

    results: list[Any] = []
    methods = [None, "smote", "oversample", "undersample"]

    for method in methods:
        name = "class_weight" if method is None else method
        outdir = Path("results/metrics") / f"imbalance_{name}"
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            res = run(output_dir=outdir, resample_method=method)
            results.append({"method": name, "ok": True, "result": res})
            print(f"Completed: {name}")
        except Exception as exc:  # pragma: no cover - runtime environment differences
            # Record the error and continue
            results.append({"method": name, "ok": False, "error": str(exc)})
            print(f"Skipped {name}: {exc}")

    comp_path = Path("results/metrics/imbalance_comparison.json")
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    with comp_path.open("w", encoding="utf8") as fh:
        json.dump({"experiments": results}, fh, indent=2)
        fh.write("\n")

    print(f"Wrote comparison report to {comp_path}")


if __name__ == "__main__":
    main()
