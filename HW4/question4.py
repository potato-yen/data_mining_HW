"""Homework 4, Question 4: Iris decision-tree experiments.

Run from the project root with:
    uv run python question4.py

The script prints every requested result and writes reproducible tables, tree
rules, and a summary figure for inclusion in the LaTeX report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Keep Matplotlib and font caches inside the project so the script also runs in
# restricted environments without writing to the user's home directory.
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


SPLIT_RANDOM_STATE = 42
TREE_RANDOM_STATE = 2
TEST_SIZE = 0.25
SUBSET_FRACTIONS = (0.20, 0.40, 0.60, 0.80, 1.00)
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

DISCUSSION = """Interpretation
Standardization leaves the two trees' feature partitions and all test
predictions unchanged; only the numerical thresholds are expressed on a new
scale. This differs from Euclidean neighbors because Euclidean distance mixes
differences across features, so feature scale changes the distance. The modest
gap between training and test accuracy indicates some loss of generalization;
a much larger gap would suggest overfitting. Across training sizes, petal width
remains the root feature, but its threshold changes from 0.70 to 0.80 and the
leaf count rises from four to five. Test accuracy peaks at the 60% subset and
then falls back, so more data does not guarantee a deeper or more accurate tree.
"""

SYNTHESIS = """Standardization changed the numerical thresholds but not the
selected features, partitions, predictions, or accuracy of the decision tree.
This is consistent with how axis-aligned trees operate: a monotonic rescaling
preserves the ordering of values within each feature. Euclidean-distance
neighbors behave differently because feature scales directly change the
combined distance. The original tree achieved 0.982 training accuracy and
0.921 test accuracy, a small generalization gap; a much larger gap would
suggest overfitting. Across stratified training subsets, petal width remained
the root feature, although its threshold and the number of leaves changed.
Test accuracy peaked with 60 percent of the training data and then returned
to its earlier value. Thus, more data did not monotonically increase depth or
accuracy. Sampling variation can change impurity-optimal splits, while the
fixed depth limit constrains model complexity."""


def new_tree() -> DecisionTreeClassifier:
    """Return a classifier with the assignment's fixed settings."""

    return DecisionTreeClassifier(max_depth=3, random_state=TREE_RANDOM_STATE)


def tree_summary(
    tree: DecisionTreeClassifier, feature_names: list[str]
) -> dict[str, object]:
    """Collect the tree properties requested in the assignment."""

    root_index = int(tree.tree_.feature[0])
    return {
        "root_feature": feature_names[root_index],
        "root_threshold": float(tree.tree_.threshold[0]),
        "tree_depth": int(tree.get_depth()),
        "leaf_nodes": int(tree.get_n_leaves()),
    }


def stratified_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic stratified fraction of the original training set."""

    if fraction == 1.0:
        return x_train.copy(), y_train.copy()

    x_subset, _, y_subset, _ = train_test_split(
        x_train,
        y_train,
        train_size=fraction,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y_train,
    )
    return x_subset, y_subset


def save_summary_figure(size_results: pd.DataFrame) -> None:
    """Plot test accuracy and structural complexity versus training-set size."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].plot(
        size_results["training_records"],
        size_results["test_accuracy"],
        marker="o",
        linewidth=2,
        color="#315a7d",
    )
    axes[0].set_xlabel("Number of training records")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_ylim(0.85, 1.0)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        size_results["training_records"],
        size_results["tree_depth"],
        marker="o",
        linewidth=2,
        label="Depth",
        color="#8c4c3f",
    )
    axes[1].plot(
        size_results["training_records"],
        size_results["leaf_nodes"],
        marker="s",
        linewidth=2,
        label="Leaves",
        color="#5e7b54",
    )
    axes[1].set_xlabel("Number of training records")
    axes[1].set_ylabel("Tree complexity")
    axes[1].set_yticks(range(0, 9))
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "q4_training_size.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    iris = load_iris()
    x = iris.data
    y = iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y,
    )

    # Tree A: original features.
    tree_original = new_tree()
    tree_original.fit(x_train, y_train)
    pred_original = tree_original.predict(x_test)
    original_summary = tree_summary(tree_original, feature_names)

    # Tree B: the scaler is fitted on training data only.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    tree_scaled = new_tree()
    tree_scaled.fit(x_train_scaled, y_train)
    pred_scaled = tree_scaled.predict(x_test_scaled)
    scaled_summary = tree_summary(tree_scaled, feature_names)

    tree_comparison = {
        "original": {
            **original_summary,
            "training_accuracy": float(tree_original.score(x_train, y_train)),
            "test_accuracy": float(accuracy_score(y_test, pred_original)),
            "test_predictions": pred_original.tolist(),
        },
        "standardized": {
            **scaled_summary,
            "training_accuracy": float(
                tree_scaled.score(x_train_scaled, y_train)
            ),
            "test_accuracy": float(accuracy_score(y_test, pred_scaled)),
            "test_predictions": pred_scaled.tolist(),
        },
        "predictions_identical": bool(np.array_equal(pred_original, pred_scaled)),
    }

    original_rules = export_text(tree_original, feature_names=feature_names)
    scaled_rules = export_text(tree_scaled, feature_names=feature_names)
    (RESULTS_DIR / "q4_tree_original.txt").write_text(
        original_rules, encoding="utf-8"
    )
    (RESULTS_DIR / "q4_tree_standardized.txt").write_text(
        scaled_rules, encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for fraction in SUBSET_FRACTIONS:
        x_subset, y_subset = stratified_subset(x_train, y_train, fraction)
        model = new_tree()
        model.fit(x_subset, y_subset)
        summary = tree_summary(model, feature_names)
        rows.append(
            {
                "training_fraction": fraction,
                "training_records": len(y_subset),
                **summary,
                "test_accuracy": float(model.score(x_test, y_test)),
            }
        )

    size_results = pd.DataFrame(rows)
    size_results.to_csv(RESULTS_DIR / "q4_training_size.csv", index=False)
    save_summary_figure(size_results)

    dataset_summary = {
        "records": int(x.shape[0]),
        "features": int(x.shape[1]),
        "class_names": class_names,
        "class_counts": np.bincount(y).tolist(),
        "split_random_state": SPLIT_RANDOM_STATE,
        "training_records": int(len(y_train)),
        "test_records": int(len(y_test)),
    }
    complete_summary = {
        "dataset": dataset_summary,
        "tree_comparison": tree_comparison,
    }
    (RESULTS_DIR / "q4_summary.json").write_text(
        json.dumps(complete_summary, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "q4_interpretation.txt").write_text(
        DISCUSSION + "\nSynthesis (131 words)\n" + SYNTHESIS + "\n",
        encoding="utf-8",
    )

    print("Dataset summary")
    print(json.dumps(dataset_summary, indent=2))
    print("\nOriginal versus standardized trees")
    print(json.dumps(tree_comparison, indent=2))
    print("\nTraining-size experiment")
    print(size_results.to_string(index=False))
    print("\nOriginal-feature tree rules")
    print(original_rules)
    print("Standardized-feature tree rules")
    print(scaled_rules)
    print(DISCUSSION)
    print("Synthesis (131 words)")
    print(SYNTHESIS)


if __name__ == "__main__":
    main()
