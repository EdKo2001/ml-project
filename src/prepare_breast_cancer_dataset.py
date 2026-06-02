"""Prepare the breast cancer diagnostic dataset for team modeling work."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer_dataset.csv"
DEFAULT_PROCESSED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "breast_cancer_processed.csv"
)
DEFAULT_PROFILE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "breast_cancer_profile.json"
)

TARGET_COLUMN = "diagnosis"
DROP_COLUMNS = {"id"}
DIAGNOSIS_MAPPING = {
    "B": "benign",
    "M": "malignant",
}
POSITIVE_CLASS = "M"


def project_relative(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


def normalize_column_name(column: str) -> str:
    """Convert source headers to simple snake_case names."""
    column = column.strip().lower()
    column = re.sub(r"\s+", "_", column)
    column = re.sub(r"[^a-z0-9_]", "", column)
    column = re.sub(r"_+", "_", column)
    return column.strip("_")


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = list(reader)
    return header, rows


def validate_numeric_features(
    header: list[str], rows: list[list[str]], target_column: str
) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column, value in zip(header, row):
            if column == target_column or value == "":
                continue
            try:
                float(value)
            except ValueError as exc:
                raise ValueError(
                    f"Expected numeric value in column '{column}' on row {row_number}, "
                    f"got {value!r}."
                ) from exc


def prepare_dataset(raw_path: Path, processed_path: Path, profile_path: Path) -> dict:
    raw_path = raw_path.resolve()
    processed_path = processed_path.resolve()
    profile_path = profile_path.resolve()

    raw_header, raw_rows = read_csv_rows(raw_path)
    normalized_header = [normalize_column_name(column) for column in raw_header]

    keep_indices = []
    dropped_columns = []
    for index, column in enumerate(normalized_header):
        if not column:
            dropped_columns.append("empty_trailing_header_column")
            continue
        if column in DROP_COLUMNS:
            dropped_columns.append(column)
            continue
        keep_indices.append(index)

    processed_header = [normalized_header[index] for index in keep_indices]

    if TARGET_COLUMN not in processed_header:
        raise ValueError(f"Could not find target column '{TARGET_COLUMN}'.")

    target_index = processed_header.index(TARGET_COLUMN)
    processed_rows: list[list[str]] = []
    missing_values = Counter()

    for row_number, raw_row in enumerate(raw_rows, start=2):
        processed_row = []
        for index, column in zip(keep_indices, processed_header):
            value = raw_row[index].strip() if index < len(raw_row) else ""
            if value == "":
                missing_values[column] += 1
            processed_row.append(value)

        diagnosis = processed_row[target_index]
        if diagnosis not in DIAGNOSIS_MAPPING:
            raise ValueError(
                f"Unexpected diagnosis label {diagnosis!r} on row {row_number}."
            )

        processed_rows.append(processed_row)

    validate_numeric_features(processed_header, processed_rows, TARGET_COLUMN)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(processed_header)
        writer.writerows(processed_rows)

    raw_id_index = normalized_header.index("id") if "id" in normalized_header else None
    raw_ids = [
        raw_row[raw_id_index]
        for raw_row in raw_rows
        if raw_id_index is not None and raw_id_index < len(raw_row)
    ]
    target_counts = Counter(row[target_index] for row in processed_rows)

    profile = {
        "source_file": project_relative(raw_path),
        "processed_file": project_relative(processed_path),
        "raw_shape": [len(raw_rows), len(raw_header)],
        "processed_shape": [len(processed_rows), len(processed_header)],
        "dropped_columns": dropped_columns,
        "target_column": TARGET_COLUMN,
        "target_counts": dict(target_counts),
        "target_mapping": DIAGNOSIS_MAPPING,
        "positive_class": POSITIVE_CLASS,
        "feature_columns": [
            column for column in processed_header if column != TARGET_COLUMN
        ],
        "missing_values": dict(missing_values),
        "duplicate_rows": len(processed_rows) - len({tuple(row) for row in processed_rows}),
        "duplicate_ids": len(raw_ids) - len(set(raw_ids)),
    }

    with profile_path.open("w") as profile_file:
        json.dump(profile, profile_file, indent=2)
        profile_file.write("\n")

    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the breast cancer diagnostic CSV into data/processed."
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--profile-path", type=Path, default=DEFAULT_PROFILE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = prepare_dataset(args.raw_path, args.processed_path, args.profile_path)
    print(f"Wrote {profile['processed_file']}")
    print(f"Wrote {project_relative(args.profile_path)}")
    print(f"Processed shape: {profile['processed_shape']}")
    print(f"Target counts: {profile['target_counts']}")


if __name__ == "__main__":
    main()
