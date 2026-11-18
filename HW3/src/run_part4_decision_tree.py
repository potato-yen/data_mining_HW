"""Run the reproducible Decision Tree experiments for Homework 3 Part 4."""

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
from sklearn.tree import DecisionTreeClassifier, plot_tree


FEATURES = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
LABEL = "species"
CLASS_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
RANDOM_STATE = 42
MAIN_MAX_DEPTH = 3
COMPARISON_MAX_DEPTH = 2


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "penguins_classification.csv",
    )
    parser.add_argument(
        "--split-input",
        type=Path,
        default=project_root / "results" / "train_test_split.csv",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=project_root / "results" / "part4_decision_tree_results.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=project_root / "results" / "part4_test_predictions.csv",
    )
    parser.add_argument(
        "--tree-figure-prefix",
        type=Path,
        default=project_root / "figures" / "part4_decision_tree",
    )
    parser.add_argument(
        "--matrix-figure-prefix",
        type=Path,
        default=project_root / "figures" / "part4_decision_tree_confusion_matrix",
    )
    return parser.parse_args()


def load_fixed_split(data_path: Path, split_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(data_path)
    split = pd.read_csv(split_path)

    expected_data_columns = {"source_row", LABEL, *FEATURES}
    if set(data.columns) != expected_data_columns:
        raise ValueError("Prepared data columns do not match the fixed experiment contract")
    if set(split.columns) != {"source_row", LABEL, "split"}:
        raise ValueError("Split file has unexpected columns")
    if data["source_row"].duplicated().any() or split["source_row"].duplicated().any():
        raise ValueError("source_row must uniquely identify each observation")
    if set(split["split"]) != {"train", "test"}:
        raise ValueError("Split file must contain train and test rows")

    merged = data.merge(
        split,
        on=["source_row", LABEL],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(data) or merged["split"].isna().any():
        raise ValueError("Split file does not map every prepared observation exactly once")
    if merged[[*FEATURES, LABEL]].isna().sum().sum() != 0:
        raise ValueError("Modeling data contains missing values")
    if sorted(merged[LABEL].unique()) != sorted(CLASS_ORDER):
        raise ValueError("Unexpected class labels")

    train = merged.loc[merged["split"] == "train"].copy()
    test = merged.loc[merged["split"] == "test"].copy()
    if (len(train), len(test)) != (273, 69):
        raise ValueError("Fixed split must contain 273 train rows and 69 test rows")
    return train, test


def evaluate_model(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted, labels=CLASS_ORDER)
    accuracy = float(accuracy_score(y_test, predicted))
    return predicted, matrix, accuracy


def describe_path(
    model: DecisionTreeClassifier,
    sample: pd.DataFrame,
) -> tuple[list[dict[str, object]], int]:
    node_indicator = model.decision_path(sample)
    leaf_id = int(model.apply(sample)[0])
    node_ids = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]
    path: list[dict[str, object]] = []

    for node_id in node_ids:
        if int(node_id) == leaf_id:
            break
        feature_index = int(model.tree_.feature[node_id])
        threshold = float(model.tree_.threshold[node_id])
        feature = FEATURES[feature_index]
        value = float(sample.iloc[0, feature_index])
        operator = "<=" if value <= threshold else ">"
        path.append(
            {
                "node_id": int(node_id),
                "feature": feature,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "next_node": int(
                    model.tree_.children_left[node_id]
                    if value <= threshold
                    else model.tree_.children_right[node_id]
                ),
            }
        )
    return path, leaf_id


def plot_main_tree(model: DecisionTreeClassifier, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    display_features = [
        "culmen length",
        "culmen depth",
        "flipper length",
        "body mass",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    plot_tree(
        model,
        feature_names=display_features,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=2,
        fontsize=11,
        ax=ax,
    )
    ax.set_title(
        "Decision Tree classifier (max_depth=3, criterion=Gini)",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    accuracy: float,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=int(matrix.max()))
    ax.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_xlabel("Predicted species", labelpad=10)
    ax.set_ylabel("Actual species", labelpad=10)
    ax.set_title(
        "Decision Tree confusion matrix\n"
        f"max_depth={MAIN_MAX_DEPTH}, test n={int(matrix.sum())}, "
        f"accuracy={accuracy:.2%}",
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


def run(args: argparse.Namespace) -> None:
    train, test = load_fixed_split(args.input, args.split_input)
    X_train = train[FEATURES]
    y_train = train[LABEL]
    X_test = test[FEATURES]
    y_test = test[LABEL]

    main_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAIN_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    comparison_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=COMPARISON_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    main_prediction, main_matrix, main_accuracy = evaluate_model(
        main_model, X_train, y_train, X_test, y_test
    )
    comparison_prediction, comparison_matrix, comparison_accuracy = evaluate_model(
        comparison_model, X_train, y_train, X_test, y_test
    )

    sample = test.iloc[[0]]
    sample_features = sample[FEATURES]
    path, leaf_id = describe_path(main_model, sample_features)
    sample_prediction = str(main_model.predict(sample_features)[0])

    predictions = test[["source_row", LABEL]].rename(columns={LABEL: "actual"}).copy()
    predictions["tree_depth_3"] = main_prediction
    predictions["tree_depth_2"] = comparison_prediction

    root_feature_index = int(main_model.tree_.feature[0])
    results = {
        "data_file": args.input.name,
        "split_file": args.split_input.name,
        "features": FEATURES,
        "label": LABEL,
        "class_order": CLASS_ORDER,
        "split": {"train_rows": len(train), "test_rows": len(test)},
        "main_model": {
            "criterion": "gini",
            "max_depth": MAIN_MAX_DEPTH,
            "random_state": RANDOM_STATE,
            "tree_depth": int(main_model.get_depth()),
            "leaf_count": int(main_model.get_n_leaves()),
            "node_count": int(main_model.tree_.node_count),
            "root_feature": FEATURES[root_feature_index],
            "root_threshold": float(main_model.tree_.threshold[0]),
            "correct_predictions": int(np.trace(main_matrix)),
            "incorrect_predictions": int(main_matrix.sum() - np.trace(main_matrix)),
            "accuracy": main_accuracy,
            "error_rate": 1.0 - main_accuracy,
            "confusion_matrix": main_matrix.tolist(),
        },
        "comparison_model": {
            "criterion": "gini",
            "max_depth": COMPARISON_MAX_DEPTH,
            "random_state": RANDOM_STATE,
            "tree_depth": int(comparison_model.get_depth()),
            "leaf_count": int(comparison_model.get_n_leaves()),
            "node_count": int(comparison_model.tree_.node_count),
            "root_feature": FEATURES[int(comparison_model.tree_.feature[0])],
            "root_threshold": float(comparison_model.tree_.threshold[0]),
            "correct_predictions": int(np.trace(comparison_matrix)),
            "incorrect_predictions": int(
                comparison_matrix.sum() - np.trace(comparison_matrix)
            ),
            "accuracy": comparison_accuracy,
            "error_rate": 1.0 - comparison_accuracy,
            "confusion_matrix": comparison_matrix.tolist(),
        },
        "sample_prediction_path": {
            "source_row": int(sample.iloc[0]["source_row"]),
            "actual": str(sample.iloc[0][LABEL]),
            "features": {
                feature: float(sample.iloc[0][feature]) for feature in FEATURES
            },
            "steps": path,
            "leaf_id": leaf_id,
            "predicted": sample_prediction,
            "correct": sample_prediction == str(sample.iloc[0][LABEL]),
        },
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    predictions.to_csv(args.predictions_output, index=False)
    plot_main_tree(main_model, args.tree_figure_prefix)
    plot_confusion_matrix(main_matrix, main_accuracy, args.matrix_figure_prefix)


if __name__ == "__main__":
    run(parse_args())
