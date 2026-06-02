"""Prepare the breast cancer survival/status dataset for team modeling work."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer_dataset_2.csv"
DEFAULT_PROCESSED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "breast_cancer_survival_processed.csv"
)
DEFAULT_PROFILE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "breast_cancer_survival_profile.json"
)

TARGET_COLUMN = "status"
OUTCOME_COLUMNS = ["survival_months", "status"]
POSITIVE_CLASS = "Dead"
STATUS_EVENT_MAPPING = {
    "Alive": 0,
    "Dead": 1,
}
HEADER_RENAMES = {
    "6th_stage": "sixth_stage",
    "differentiate": "differentiation",
    "reginol_node_positive": "regional_node_positive",
}
GRADE_MAPPING = {
    "1": "1",
    "2": "2",
    "3": "3",
    "anaplastic; grade iv": "4",
}
NUMERIC_COLUMNS = {
    "age",
    "grade",
    "tumor_size",
    "regional_node_examined",
    "regional_node_positive",
    "survival_months",
}


def normalize_column_name(column: str) -> str:
    """Convert source headers to simple snake_case names."""
    column = column.strip().lower()
    column = re.sub(r"\s+", "_", column)
    column = re.sub(r"[^a-z0-9_]", "", column)
    column = re.sub(r"_+", "_", column)
    column = column.strip("_")
    return HEADER_RENAMES.get(column, column)


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = list(reader)
    return header, rows


def normalize_grade(value: str, row_number: int) -> str:
    normalized_value = value.strip().lower()
    if normalized_value not in GRADE_MAPPING:
        raise ValueError(f"Unexpected grade value {value!r} on row {row_number}.")
    return GRADE_MAPPING[normalized_value]


def validate_numeric_value(column: str, value: str, row_number: int) -> None:
    if value == "":
        return
    try:
        float(value)
    except ValueError as exc:
        raise ValueError(
            f"Expected numeric value in column '{column}' on row {row_number}, "
            f"got {value!r}."
        ) from exc


def deduplicate_rows(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    seen = set()
    deduped_rows = []
    duplicate_count = 0

    for row in rows:
        row_key = tuple(row)
        if row_key in seen:
            duplicate_count += 1
            continue
        seen.add(row_key)
        deduped_rows.append(row)

    return deduped_rows, duplicate_count


def prepare_dataset(raw_path: Path, processed_path: Path, profile_path: Path) -> dict:
    raw_header, raw_rows = read_csv_rows(raw_path)
    processed_header = [normalize_column_name(column) for column in raw_header]

    duplicate_columns = [
        column
        for column, count in Counter(processed_header).items()
        if count > 1
    ]
    if duplicate_columns:
        raise ValueError(f"Duplicate processed columns found: {duplicate_columns}")
    if TARGET_COLUMN not in processed_header:
        raise ValueError(f"Could not find target column '{TARGET_COLUMN}'.")

    processed_rows = []
    missing_values = Counter()

    for row_number, raw_row in enumerate(raw_rows, start=2):
        if len(raw_row) != len(raw_header):
            raise ValueError(
                f"Expected {len(raw_header)} columns on row {row_number}, "
                f"got {len(raw_row)}."
            )

        processed_row = []
        for column, value in zip(processed_header, raw_row):
            cleaned_value = value.strip()
            if column == "grade":
                cleaned_value = normalize_grade(cleaned_value, row_number)
            if cleaned_value == "":
                missing_values[column] += 1
            if column in NUMERIC_COLUMNS:
                validate_numeric_value(column, cleaned_value, row_number)
            processed_row.append(cleaned_value)

        status = processed_row[processed_header.index(TARGET_COLUMN)]
        if status not in STATUS_EVENT_MAPPING:
            raise ValueError(f"Unexpected status value {status!r} on row {row_number}.")
        processed_rows.append(processed_row)

    processed_rows, duplicate_rows_removed = deduplicate_rows(processed_rows)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(processed_header)
        writer.writerows(processed_rows)

    target_index = processed_header.index(TARGET_COLUMN)
    target_counts = Counter(row[target_index] for row in processed_rows)
    categorical_columns = [
        column
        for column in processed_header
        if column not in NUMERIC_COLUMNS and column != TARGET_COLUMN
    ]
    categorical_levels = {
        column: sorted(
            {
                row[processed_header.index(column)]
                for row in processed_rows
            }
        )
        for column in categorical_columns + [TARGET_COLUMN]
    }

    profile = {
        "source_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "processed_file": str(processed_path.relative_to(PROJECT_ROOT)),
        "raw_shape": [len(raw_rows), len(raw_header)],
        "processed_shape": [len(processed_rows), len(processed_header)],
        "normalized_columns": dict(zip(raw_header, processed_header)),
        "target_column": TARGET_COLUMN,
        "target_counts": dict(target_counts),
        "positive_class": POSITIVE_CLASS,
        "status_event_mapping": STATUS_EVENT_MAPPING,
        "outcome_columns": OUTCOME_COLUMNS,
        "classification_feature_columns": [
            column
            for column in processed_header
            if column not in OUTCOME_COLUMNS
        ],
        "numeric_columns": sorted(NUMERIC_COLUMNS),
        "categorical_columns": categorical_columns,
        "categorical_levels": categorical_levels,
        "grade_mapping": {
            "1": "1",
            "2": "2",
            "3": "3",
            "anaplastic; Grade IV": "4",
        },
        "missing_values": dict(missing_values),
        "duplicate_rows_removed": duplicate_rows_removed,
        "modeling_note": (
            "For status classification, do not use survival_months as a feature; "
            "it is an outcome/time-to-event field."
        ),
    }

    with profile_path.open("w") as profile_file:
        json.dump(profile, profile_file, indent=2)
        profile_file.write("\n")

    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the breast cancer survival/status CSV into data/processed."
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--profile-path", type=Path, default=DEFAULT_PROFILE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = prepare_dataset(args.raw_path, args.processed_path, args.profile_path)
    print(f"Wrote {profile['processed_file']}")
    print(f"Wrote {args.profile_path.relative_to(PROJECT_ROOT)}")
    print(f"Processed shape: {profile['processed_shape']}")
    print(f"Target counts: {profile['target_counts']}")
    print(f"Duplicate rows removed: {profile['duplicate_rows_removed']}")


if __name__ == "__main__":
    main()
