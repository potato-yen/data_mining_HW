"""Prepare the Palmer Penguins classification data used in Homework 3.

The script keeps the original CSV unchanged. It creates a compact modeling
table and a machine-readable audit so later parts can reuse exactly the same
data decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RAW_FEATURES = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
]

RENAMED_FEATURES = {
    "Culmen Length (mm)": "culmen_length_mm",
    "Culmen Depth (mm)": "culmen_depth_mm",
    "Flipper Length (mm)": "flipper_length_mm",
    "Body Mass (g)": "body_mass_g",
}

SPECIES_MAP = {
    "Adelie Penguin (Pygoscelis adeliae)": "Adelie",
    "Chinstrap penguin (Pygoscelis antarctica)": "Chinstrap",
    "Gentoo penguin (Pygoscelis papua)": "Gentoo",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "penguins_raw.csv",
        help="Raw Palmer Penguins CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "penguins_classification.csv",
        help="Prepared classification CSV.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=project_root / "results" / "part1_data_audit.json",
        help="JSON file containing the Part 1 audit results.",
    )
    return parser.parse_args()


def iqr_audit(series: pd.Series) -> dict[str, float | int]:
    values = series.dropna()
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((values < lower) | (values > upper)).sum())
    return {
        "observed_min": float(values.min()),
        "observed_max": float(values.max()),
        "q1": q1,
        "q3": q3,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outlier_count,
    }


def prepare(input_path: Path, output_path: Path, audit_path: Path) -> None:
    # Read literal "NA" as missing while preserving all other source values.
    raw = pd.read_csv(input_path, keep_default_na=False).replace("NA", pd.NA)
    for column in RAW_FEATURES:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    unknown_species = sorted(set(raw["Species"].dropna()) - set(SPECIES_MAP))
    if unknown_species:
        raise ValueError(f"Unexpected Species labels: {unknown_species}")

    incomplete_mask = raw[RAW_FEATURES].isna().any(axis=1)
    incomplete_rows = raw.loc[
        incomplete_mask,
        ["studyName", "Sample Number", "Species", "Individual ID", *RAW_FEATURES],
    ].copy()
    incomplete_rows.insert(0, "source_row", incomplete_rows.index + 2)

    prepared = raw.loc[~incomplete_mask, ["Species", *RAW_FEATURES]].copy()
    prepared.insert(0, "source_row", prepared.index + 2)
    prepared["Species"] = prepared["Species"].map(SPECIES_MAP)
    prepared = prepared.rename(columns={"Species": "species", **RENAMED_FEATURES})
    prepared["flipper_length_mm"] = prepared["flipper_length_mm"].astype("int64")
    prepared["body_mass_g"] = prepared["body_mass_g"].astype("int64")

    feature_names = list(RENAMED_FEATURES.values())
    assert len(raw) == 344
    assert len(prepared) == 342
    assert prepared[feature_names + ["species"]].isna().sum().sum() == 0
    assert prepared["species"].nunique() == 3
    assert int(raw.duplicated().sum()) == 0

    missing_counts = {
        column: int(count)
        for column, count in raw.isna().sum().items()
    }
    class_counts_raw = {
        SPECIES_MAP[label]: int(count)
        for label, count in raw["Species"].value_counts().items()
    }
    class_counts_prepared = {
        label: int(count)
        for label, count in prepared["species"].value_counts().items()
    }
    incomplete_records = incomplete_rows.astype(object).where(
        pd.notna(incomplete_rows), None
    ).to_dict(orient="records")
    audit = {
        "source_file": input_path.name,
        "raw_shape": {"rows": int(raw.shape[0]), "columns": int(raw.shape[1])},
        "selected_features": feature_names,
        "label": "species",
        "missing_counts_raw": missing_counts,
        "incomplete_selected_feature_rows": incomplete_records,
        "full_row_duplicate_count": int(raw.duplicated().sum()),
        "duplicate_complete_feature_vector_count": int(
            raw.loc[~incomplete_mask].duplicated(RAW_FEATURES, keep=False).sum()
        ),
        "iqr_audit": {column: iqr_audit(raw[column]) for column in RAW_FEATURES},
        "class_counts_raw": class_counts_raw,
        "class_counts_prepared": class_counts_prepared,
        "prepared_shape": {
            "rows": int(prepared.shape[0]),
            "model_columns": 5,
            "trace_columns": 1,
        },
        "preprocessing_deferred": [
            "train_test_split",
            "feature_scaling fitted on training data only",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Raw rows: {len(raw)}")
    print(f"Removed incomplete rows: {int(incomplete_mask.sum())}")
    print(f"Prepared rows: {len(prepared)}")
    print(f"Prepared data: {output_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    arguments = parse_args()
    prepare(arguments.input, arguments.output, arguments.audit)
