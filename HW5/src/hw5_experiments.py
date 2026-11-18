"""Reproducible experiments for HW5 classifier comparison."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42

CV_GRIDS: dict[str, list[Any]] = {
    "kNN": [1, 3, 5, 9, 15],
    "Decision Tree": [2, 3, 5, 10, None],
    "SVM": [0.1, 1.0, 10.0],
}

EXPECTED_RESULT_FILES = [
    "dataset_summaries.json",
    "splits_a.csv",
    "splits_b.csv",
    "part1_practice.csv",
    "part1_evaluation.json",
    "baseline_a.csv",
    "scaling_a.csv",
    "cv_search_a.csv",
    "cv_selected_a.json",
    "final_a.csv",
    "comparison_b.csv",
    "run_metadata.json",
]


@dataclass(frozen=True)
class SplitData:
    """One deterministic train/test split and its source-row indices."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    dataset_name: str


def load_dataset_a():
    """Load Dataset A, Breast Cancer Wisconsin."""

    return load_breast_cancer()


def load_dataset_b():
    """Load Dataset B, Digits."""

    return load_digits()


def make_fixed_split(X: np.ndarray, y: np.ndarray, dataset_name: str) -> SplitData:
    """Create the assignment's single deterministic stratified 80/20 split."""

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    indices = np.arange(len(y_array))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=y_array,
        random_state=RANDOM_STATE,
    )
    return SplitData(
        X_train=X_array[train_indices],
        X_test=X_array[test_indices],
        y_train=y_array[train_indices],
        y_test=y_array[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        dataset_name=dataset_name,
    )


def evaluate_model(
    model: ClassifierMixin,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit one classifier and return the two assignment metrics."""

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }


def baseline_models() -> dict[str, ClassifierMixin]:
    """Return the exact five raw baseline estimators required in Part 2."""

    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian NB": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=1.0),
    }


def build_scaled_model(model_name: str, parameter: float | int) -> Pipeline:
    """Build a leakage-safe scaled kNN or RBF-SVM pipeline."""

    if model_name == "knn":
        classifier = KNeighborsClassifier(n_neighbors=int(parameter))
    elif model_name == "svm":
        classifier = SVC(kernel="rbf", C=float(parameter))
    else:
        raise ValueError(f"Unsupported scaled model: {model_name}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def run_cv_search(X_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """Evaluate the exact Part 4 grid on fixed stratified folds."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    model_specs = [
        ("kNN", "n_neighbors", CV_GRIDS["kNN"]),
        ("Decision Tree", "max_depth", CV_GRIDS["Decision Tree"]),
        ("SVM", "C", CV_GRIDS["SVM"]),
    ]
    for classifier_name, parameter_name, values in model_specs:
        for value in values:
            if classifier_name == "kNN":
                model = build_scaled_model("knn", value)
            elif classifier_name == "SVM":
                model = build_scaled_model("svm", value)
            else:
                model = DecisionTreeClassifier(
                    max_depth=value,
                    random_state=RANDOM_STATE,
                )
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="f1_macro",
            )
            row: dict[str, Any] = {
                "classifier": classifier_name,
                "parameter": parameter_name,
                "value": value,
            }
            row.update({f"fold_{index}": float(score) for index, score in enumerate(scores, 1)})
            row["mean_macro_f1"] = float(np.mean(scores))
            row["std_macro_f1"] = float(np.std(scores, ddof=0))
            rows.append(row)
    return pd.DataFrame(rows)


def select_cv_setting(results: pd.DataFrame, classifier_name: str) -> dict[str, Any]:
    """Select one setting using CV mean, CV spread, then model simplicity."""

    candidates = results.loc[results["classifier"] == classifier_name].copy()
    if candidates.empty:
        raise ValueError(f"No CV candidates for {classifier_name}")

    def complexity_rank(value: Any) -> float:
        if classifier_name == "kNN":
            return -float(value)
        if classifier_name == "Decision Tree":
            return float("inf") if value is None or pd.isna(value) else float(value)
        return float(value)

    candidates["_complexity_rank"] = candidates["value"].map(complexity_rank)
    selected = candidates.sort_values(
        ["mean_macro_f1", "std_macro_f1", "_complexity_rank"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    return {
        key: _json_value(value)
        for key, value in selected.drop(labels=["_complexity_rank"]).to_dict().items()
    }


def _fit_metrics_predictions(
    model: ClassifierMixin,
    split: SplitData,
) -> tuple[dict[str, float], np.ndarray]:
    model.fit(split.X_train, split.y_train)
    predictions = model.predict(split.X_test)
    metrics = {
        "accuracy": float(accuracy_score(split.y_test, predictions)),
        "macro_f1": float(f1_score(split.y_test, predictions, average="macro")),
    }
    return metrics, predictions


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_json_value(data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_split(path: Path, split: SplitData) -> None:
    rows = [
        {"source_index": int(index), "split": "train", "class": int(label)}
        for index, label in zip(split.train_indices, split.y_train, strict=True)
    ]
    rows.extend(
        {"source_index": int(index), "split": "test", "class": int(label)}
        for index, label in zip(split.test_indices, split.y_test, strict=True)
    )
    pd.DataFrame(rows).sort_values("source_index").to_csv(path, index=False)


def _split_hash(split: SplitData) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(split.train_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(split.test_indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _rank_results(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["macro_f1", "accuracy", "classifier"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _run_part1(split: SplitData) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings: list[tuple[str, str, ClassifierMixin]] = [
        ("KNeighborsClassifier", "n_neighbors=3, StandardScaler", build_scaled_model("knn", 3)),
        ("KNeighborsClassifier", "n_neighbors=15, StandardScaler", build_scaled_model("knn", 15)),
        ("GaussianNB", "var_smoothing=1e-9", GaussianNB(var_smoothing=1e-9)),
        ("GaussianNB", "var_smoothing=1e-7", GaussianNB(var_smoothing=1e-7)),
        (
            "DecisionTreeClassifier",
            "max_depth=3, random_state=42",
            DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        ),
        (
            "DecisionTreeClassifier",
            "max_depth=None, random_state=42",
            DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE),
        ),
        ("SVC", "C=0.1, kernel=rbf, StandardScaler", build_scaled_model("svm", 0.1)),
        ("SVC", "C=10, kernel=rbf, StandardScaler", build_scaled_model("svm", 10.0)),
        (
            "RandomForestClassifier",
            "n_estimators=50, random_state=42",
            RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE),
        ),
        (
            "RandomForestClassifier",
            "n_estimators=200, random_state=42",
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        ),
        (
            "MLPClassifier",
            "hidden_layer_sizes=(50,), solver=lbfgs, random_state=42",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(50,),
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "MLPClassifier",
            "hidden_layer_sizes=(100, 50), solver=lbfgs, random_state=42",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(100, 50),
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for classifier_name, setting, model in settings:
        metrics, _ = _fit_metrics_predictions(model, split)
        rows.append({"classifier": classifier_name, "setting": setting, **metrics})

    evaluation_model = build_scaled_model("svm", 1.0)
    evaluation_model.fit(split.X_train, split.y_train)
    predictions = evaluation_model.predict(split.X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        build_scaled_model("svm", 1.0),
        split.X_train,
        split.y_train,
        cv=cv,
        scoring="f1_macro",
    )
    evaluation = {
        "model": "Pipeline(StandardScaler, SVC(kernel='rbf', C=1.0))",
        "confusion_matrix": confusion_matrix(split.y_test, predictions).tolist(),
        "classification_report": classification_report(
            split.y_test,
            predictions,
            target_names=["malignant", "benign"],
            output_dict=True,
            zero_division=0,
        ),
        "f1_score_macro": float(f1_score(split.y_test, predictions, average="macro")),
        "cross_validation": {
            "method": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "scoring": "f1_macro",
            "fold_scores": cv_scores.tolist(),
            "mean": float(np.mean(cv_scores)),
            "std_ddof_0": float(np.std(cv_scores, ddof=0)),
        },
    }
    return pd.DataFrame(rows), evaluation


def _run_baseline_a(split: SplitData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for classifier_name, model in baseline_models().items():
        metrics, _ = _fit_metrics_predictions(model, split)
        rows.append({"classifier": classifier_name, **metrics})
    return _rank_results(pd.DataFrame(rows))


def _scaled_comparison_model(classifier_name: str) -> Pipeline:
    if classifier_name == "kNN":
        classifier: ClassifierMixin = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == "SVM":
        classifier = SVC(kernel="rbf", C=1.0)
    elif classifier_name == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif classifier_name == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported scaling comparison: {classifier_name}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def _run_scaling_a(split: SplitData, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_index = baseline.set_index("classifier")
    for classifier_name in ["kNN", "SVM", "Decision Tree", "Random Forest"]:
        scaled_metrics, _ = _fit_metrics_predictions(
            _scaled_comparison_model(classifier_name),
            split,
        )
        rows.append(
            {
                "classifier": classifier_name,
                "raw_accuracy": float(baseline_index.loc[classifier_name, "accuracy"]),
                "scaled_accuracy": scaled_metrics["accuracy"],
                "raw_macro_f1": float(baseline_index.loc[classifier_name, "macro_f1"]),
                "scaled_macro_f1": scaled_metrics["macro_f1"],
                "accuracy_change": scaled_metrics["accuracy"]
                - float(baseline_index.loc[classifier_name, "accuracy"]),
                "macro_f1_change": scaled_metrics["macro_f1"]
                - float(baseline_index.loc[classifier_name, "macro_f1"]),
            }
        )
    return pd.DataFrame(rows)


def _model_from_selection(classifier_name: str, value: Any) -> ClassifierMixin:
    if classifier_name == "kNN":
        return build_scaled_model("knn", int(value))
    if classifier_name == "SVM":
        return build_scaled_model("svm", float(value))
    if classifier_name == "Decision Tree":
        depth = None if value is None or pd.isna(value) else int(value)
        return DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    raise ValueError(f"Unsupported selected classifier: {classifier_name}")


def _run_final_a(
    split: SplitData,
    baseline: pd.DataFrame,
    scaling: pd.DataFrame,
    cv_results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    baseline_index = baseline.set_index("classifier")
    scaling_index = scaling.set_index("classifier")
    for classifier_name in ["kNN", "Decision Tree", "SVM"]:
        choice = select_cv_setting(cv_results, classifier_name)
        selected[classifier_name] = choice
        metrics, _ = _fit_metrics_predictions(
            _model_from_selection(classifier_name, choice["value"]),
            split,
        )
        rows.append(
            {
                "classifier": classifier_name,
                "selected_parameter": choice["parameter"],
                "selected_value": choice["value"],
                "cv_mean_macro_f1": choice["mean_macro_f1"],
                "cv_std_macro_f1": choice["std_macro_f1"],
                "baseline_raw_accuracy": float(baseline_index.loc[classifier_name, "accuracy"]),
                "baseline_raw_macro_f1": float(baseline_index.loc[classifier_name, "macro_f1"]),
                "fixed_parameter_scaled_accuracy": float(
                    scaling_index.loc[classifier_name, "scaled_accuracy"]
                ),
                "fixed_parameter_scaled_macro_f1": float(
                    scaling_index.loc[classifier_name, "scaled_macro_f1"]
                ),
                "final_accuracy": metrics["accuracy"],
                "final_macro_f1": metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows), selected


def _run_comparison_b(split: SplitData) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = baseline_models()
    models["kNN"] = build_scaled_model("knn", 5)
    models["SVM"] = build_scaled_model("svm", 1.0)
    settings = {
        "Decision Tree": "raw; DecisionTreeClassifier(random_state=42)",
        "Random Forest": "raw; n_estimators=200, random_state=42",
        "kNN": "StandardScaler; n_neighbors=5",
        "Gaussian NB": "raw; GaussianNB()",
        "SVM": "StandardScaler; RBF, C=1.0",
    }
    rows: list[dict[str, Any]] = []
    predictions_by_name: dict[str, np.ndarray] = {}
    for classifier_name, model in models.items():
        metrics, predictions = _fit_metrics_predictions(model, split)
        predictions_by_name[classifier_name] = predictions
        rows.append(
            {
                "classifier": classifier_name,
                "setting": settings[classifier_name],
                **metrics,
            }
        )
    ranked = _rank_results(pd.DataFrame(rows))
    best_name = str(ranked.iloc[0]["classifier"])
    matrix = confusion_matrix(split.y_test, predictions_by_name[best_name], labels=np.arange(10))
    matrix_data = {
        "classifier": best_name,
        "labels": list(range(10)),
        "matrix": matrix.tolist(),
        "total": int(matrix.sum()),
    }
    return ranked, matrix_data


def _dataset_summaries(dataset_a: Any, dataset_b: Any) -> dict[str, Any]:
    ranges = []
    for index, name in enumerate(dataset_a.feature_names):
        values = dataset_a.data[:, index]
        ranges.append(
            {
                "feature": str(name),
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            }
        )
    return {
        "dataset_a": {
            "name": "Breast Cancer Wisconsin",
            "samples": int(dataset_a.data.shape[0]),
            "features": int(dataset_a.data.shape[1]),
            "classes": [str(name) for name in dataset_a.target_names],
            "class_counts": {
                str(name): int(np.sum(dataset_a.target == index))
                for index, name in enumerate(dataset_a.target_names)
            },
            "feature_ranges": ranges,
        },
        "dataset_b": {
            "name": "Digits",
            "samples": int(dataset_b.data.shape[0]),
            "features": int(dataset_b.data.shape[1]),
            "classes": [int(value) for value in np.unique(dataset_b.target)],
            "class_counts": {
                str(value): int(np.sum(dataset_b.target == value))
                for value in np.unique(dataset_b.target)
            },
            "image_shape": list(dataset_b.images.shape[1:]),
            "feature_min": float(dataset_b.data.min()),
            "feature_max": float(dataset_b.data.max()),
        },
    }


def generate_charts(
    result_dir: str | Path = "results",
    figure_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Generate all required report charts from exported result files."""

    results = Path(result_dir)
    figures = Path(figure_dir)
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    scaling = pd.read_csv(results / "scaling_a.csv")
    labels = scaling["classifier"].tolist()
    positions = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    metric_specs = [
        ("Accuracy", "raw_accuracy", "scaled_accuracy"),
        ("Macro-F1", "raw_macro_f1", "scaled_macro_f1"),
    ]
    for axis, (metric_label, raw_column, scaled_column) in zip(axes, metric_specs, strict=True):
        axis.bar(positions - width / 2, scaling[raw_column], width, label="Raw", color="#8DA0B3")
        axis.bar(
            positions + width / 2,
            scaling[scaled_column],
            width,
            label="Standardized",
            color="#315A7D",
        )
        axis.set_title(metric_label)
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.set_ylim(0.84, 1.0)
        axis.set_ylabel(metric_label)
        axis.legend(loc="lower right")
    fig.suptitle("Dataset A: Effect of Standardization")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            figures / f"standardization_comparison.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    cv_results = pd.read_csv(results / "cv_search_a.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=True)
    for axis, classifier_name in zip(axes, ["kNN", "Decision Tree", "SVM"], strict=True):
        subset = cv_results.loc[cv_results["classifier"] == classifier_name].copy()
        x_positions = np.arange(len(subset))
        axis.errorbar(
            x_positions,
            subset["mean_macro_f1"],
            yerr=subset["std_macro_f1"],
            marker="o",
            linewidth=2,
            capsize=4,
            color="#315A7D",
        )
        values = ["None" if pd.isna(value) else f"{value:g}" for value in subset["value"]]
        axis.set_xticks(x_positions, values)
        axis.set_xlabel(str(subset.iloc[0]["parameter"]))
        axis.set_title(classifier_name)
        axis.set_ylim(0.87, 1.0)
    axes[0].set_ylabel("Mean CV Macro-F1")
    fig.suptitle("Dataset A: Hyperparameter Selection with 5-Fold CV")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            figures / f"cv_hyperparameters.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    matrix_frame = pd.read_csv(results / "digits_confusion_matrix.csv", index_col=0)
    matrix = matrix_frame.to_numpy(dtype=int)
    comparison = pd.read_csv(results / "comparison_b.csv")
    best_classifier = str(comparison.sort_values("rank").iloc[0]["classifier"])
    fig, axis = plt.subplots(figsize=(7.2, 6.3))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Digits Confusion Matrix: {best_classifier}",
    )
    threshold = matrix.max() / 2
    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#17213A",
                fontsize=8,
            )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            figures / f"digits_confusion_matrix.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    return {
        "standardization": {
            "metric_labels": [item[0] for item in metric_specs],
            "max_scaled_macro_f1": float(scaling["scaled_macro_f1"].max()),
        },
        "cv": {
            "y_label": "Mean CV Macro-F1",
            "candidate_count": int(len(cv_results)),
        },
        "digits_confusion_matrix": {
            "classifier": best_classifier,
            "shape": list(matrix.shape),
            "total": int(matrix.sum()),
        },
    }


def run_all(result_dir: str | Path = "results") -> dict[str, Any]:
    """Run Parts 1--5 experiments and export all machine-readable results."""

    prior_path = Path("docs/prior_hypothesis.md")
    if not prior_path.exists():
        raise FileNotFoundError("Record docs/prior_hypothesis.md before running baselines")

    output = Path(result_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_a = load_dataset_a()
    dataset_b = load_dataset_b()
    split_a = make_fixed_split(dataset_a.data, dataset_a.target, "A")
    split_b = make_fixed_split(dataset_b.data, dataset_b.target, "B")

    _write_json(output / "dataset_summaries.json", _dataset_summaries(dataset_a, dataset_b))
    _save_split(output / "splits_a.csv", split_a)
    _save_split(output / "splits_b.csv", split_b)

    part1_frame, part1_evaluation = _run_part1(split_a)
    part1_frame.to_csv(output / "part1_practice.csv", index=False)
    _write_json(output / "part1_evaluation.json", part1_evaluation)

    baseline = _run_baseline_a(split_a)
    baseline.to_csv(output / "baseline_a.csv", index=False)
    scaling = _run_scaling_a(split_a, baseline)
    scaling.to_csv(output / "scaling_a.csv", index=False)

    cv_results = run_cv_search(split_a.X_train, split_a.y_train)
    cv_results.to_csv(output / "cv_search_a.csv", index=False)
    final_a, selected = _run_final_a(split_a, baseline, scaling, cv_results)
    _write_json(output / "cv_selected_a.json", selected)
    final_a.to_csv(output / "final_a.csv", index=False)

    comparison_b, matrix_b = _run_comparison_b(split_b)
    comparison_b.to_csv(output / "comparison_b.csv", index=False)
    pd.DataFrame(
        matrix_b["matrix"],
        index=matrix_b["labels"],
        columns=matrix_b["labels"],
    ).to_csv(output / "digits_confusion_matrix.csv", index_label="actual")

    metadata = {
        "python": platform.python_version(),
        "packages": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "matplotlib": version("matplotlib"),
        },
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "stratified": True,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "cv_scoring": "f1_macro",
        "cv_std_ddof": 0,
        "split_hashes": {"dataset_a": _split_hash(split_a), "dataset_b": _split_hash(split_b)},
        "dataset_b_parameter_rule": "Part 2 baseline parameters; scaling only for kNN and SVM",
        "digits_best_confusion_matrix": matrix_b,
    }
    _write_json(output / "run_metadata.json", metadata)
    return {
        "dataset_summaries": _dataset_summaries(dataset_a, dataset_b),
        "baseline_a": baseline,
        "scaling_a": scaling,
        "cv_search_a": cv_results,
        "final_a": final_a,
        "comparison_b": comparison_b,
        "metadata": metadata,
    }


def main() -> None:
    """Run the complete reproducible experiment export."""

    run_all()
    generate_charts()
    print("HW5 experiments completed; results and figures were regenerated")


if __name__ == "__main__":
    main()
