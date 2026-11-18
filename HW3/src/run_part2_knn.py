"""Run the reproducible KNN experiments for Homework 3 Part 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
LABEL = "species"
CLASS_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
K = 5


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "penguins_classification.csv",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=project_root / "results" / "part2_knn_results.json",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=project_root / "results" / "train_test_split.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=project_root / "results" / "part2_test_predictions.csv",
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=project_root / "figures" / "part2_knn_confusion_matrix",
    )
    return parser.parse_args()


def make_models() -> dict[str, dict[str, object]]:
    return {
        "raw_euclidean": {
            "label": "Raw Euclidean",
            "scaling": "None",
            "metric": "Euclidean (Minkowski p=2)",
            "model": KNeighborsClassifier(
                n_neighbors=K, metric="minkowski", p=2, weights="uniform"
            ),
        },
        "zscore_euclidean": {
            "label": "Z-score Euclidean",
            "scaling": "StandardScaler",
            "metric": "Euclidean (Minkowski p=2)",
            "model": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "knn",
                        KNeighborsClassifier(
                            n_neighbors=K,
                            metric="minkowski",
                            p=2,
                            weights="uniform",
                        ),
                    ),
                ]
            ),
        },
        "zscore_manhattan": {
            "label": "Z-score Manhattan",
            "scaling": "StandardScaler",
            "metric": "Manhattan (Minkowski p=1)",
            "model": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "knn",
                        KNeighborsClassifier(
                            n_neighbors=K,
                            metric="minkowski",
                            p=1,
                            weights="uniform",
                        ),
                    ),
                ]
            ),
        },
    }


def class_counts(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts()
    return {label: int(counts.get(label, 0)) for label in CLASS_ORDER}


def plot_confusion_matrix(
    matrix: np.ndarray,
    accuracy: float,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.max()))
    ax.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_xlabel("Predicted species", labelpad=10)
    ax.set_ylabel("Actual species", labelpad=10)
    ax.set_title(
        "KNN confusion matrix: raw Euclidean distance\n"
        f"k={K}, test n={int(matrix.sum())}, accuracy={accuracy:.2%}",
        pad=16,
        weight="bold",
    )

    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if value > threshold else "#17213A",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    colorbar.set_label("Number of test observations")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(
    input_path: Path,
    results_path: Path,
    split_path: Path,
    predictions_path: Path,
    figure_prefix: Path,
) -> None:
    data = pd.read_csv(input_path)
    expected_columns = {"source_row", LABEL, *FEATURES}
    if set(data.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns. Expected {sorted(expected_columns)}, "
            f"received {sorted(data.columns)}"
        )
    if data[[*FEATURES, LABEL]].isna().sum().sum() != 0:
        raise ValueError("Prepared data still contains missing modeling values")
    if sorted(data[LABEL].unique()) != sorted(CLASS_ORDER):
        raise ValueError("Unexpected class labels")

    row_positions = np.arange(len(data))
    train_positions, test_positions = train_test_split(
        row_positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[LABEL],
    )
    train_positions = np.sort(train_positions)
    test_positions = np.sort(test_positions)

    train = data.iloc[train_positions]
    test = data.iloc[test_positions]
    X_train = train[FEATURES]
    y_train = train[LABEL]
    X_test = test[FEATURES]
    y_test = test[LABEL]

    split = data[["source_row", LABEL]].copy()
    split["split"] = "train"
    split.loc[test_positions, "split"] = "test"
    split = split[["source_row", LABEL, "split"]]

    predictions = test[["source_row", LABEL]].rename(columns={LABEL: "actual"}).copy()
    experiments: list[dict[str, object]] = []
    representative_matrix: np.ndarray | None = None
    representative_accuracy: float | None = None

    for setting_id, specification in make_models().items():
        model = specification["model"]
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, predicted))
        matrix = confusion_matrix(y_test, predicted, labels=CLASS_ORDER)
        predictions[setting_id] = predicted

        experiments.append(
            {
                "setting_id": setting_id,
                "label": specification["label"],
                "features": FEATURES,
                "k": K,
                "weights": "uniform",
                "scaling": specification["scaling"],
                "metric": specification["metric"],
                "correct_predictions": int(np.trace(matrix)),
                "incorrect_predictions": int(matrix.sum() - np.trace(matrix)),
                "accuracy": accuracy,
                "error_rate": 1.0 - accuracy,
                "confusion_matrix": matrix.tolist(),
                "class_order": CLASS_ORDER,
            }
        )

        if setting_id == "raw_euclidean":
            representative_matrix = matrix
            representative_accuracy = accuracy

    if representative_matrix is None or representative_accuracy is None:
        raise RuntimeError("Representative setting was not evaluated")

    results = {
        "data_file": input_path.name,
        "rows": int(len(data)),
        "features": FEATURES,
        "label": LABEL,
        "split": {
            "method": "stratified train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_class_counts": class_counts(y_train),
            "test_class_counts": class_counts(y_test),
        },
        "experiments": experiments,
        "representative_setting": "raw_euclidean",
        "representative_reason": (
            "Its errors make the effect of unequal raw feature scales visible."
        ),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    split.to_csv(split_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    plot_confusion_matrix(
        representative_matrix, representative_accuracy, figure_prefix
    )

    print(
        f"Split: train={len(train)}, test={len(test)}, "
        f"random_state={RANDOM_STATE}"
    )
    for experiment in experiments:
        print(
            f"{experiment['label']}: "
            f"accuracy={experiment['accuracy']:.4f}, "
            f"errors={experiment['incorrect_predictions']}"
        )
    print(f"Results: {results_path}")
    print(f"Split registry: {split_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Figure: {figure_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input,
        arguments.results,
        arguments.split_output,
        arguments.predictions_output,
        arguments.figure_prefix,
    )
