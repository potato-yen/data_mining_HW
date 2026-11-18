import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.hw5_experiments import (
    CV_GRIDS,
    EXPECTED_RESULT_FILES,
    baseline_models,
    build_scaled_model,
    evaluate_model,
    generate_charts,
    load_dataset_a,
    load_dataset_b,
    make_fixed_split,
    run_all,
    run_cv_search,
    select_cv_setting,
)


class FixedSplitTests(unittest.TestCase):
    def test_breast_cancer_split_has_expected_size_and_is_deterministic(self):
        dataset = load_dataset_a()
        first = make_fixed_split(dataset.data, dataset.target, "A")
        second = make_fixed_split(dataset.data, dataset.target, "A")

        self.assertEqual((len(first.train_indices), len(first.test_indices)), (455, 114))
        np.testing.assert_array_equal(first.train_indices, second.train_indices)
        np.testing.assert_array_equal(first.test_indices, second.test_indices)
        self.assertEqual(np.bincount(first.y_train).tolist(), [170, 285])
        self.assertEqual(np.bincount(first.y_test).tolist(), [42, 72])

    def test_digits_split_has_expected_size_and_class_coverage(self):
        dataset = load_dataset_b()
        split = make_fixed_split(dataset.data, dataset.target, "B")

        self.assertEqual((len(split.train_indices), len(split.test_indices)), (1437, 360))
        self.assertEqual(len(np.unique(split.y_train)), 10)
        self.assertEqual(len(np.unique(split.y_test)), 10)


class MetricAndModelTests(unittest.TestCase):
    def test_evaluate_model_returns_accuracy_and_macro_f1(self):
        dataset = load_dataset_a()
        split = make_fixed_split(dataset.data, dataset.target, "A")
        model = build_scaled_model("knn", 5)
        metrics = evaluate_model(
            model,
            split.X_train,
            split.X_test,
            split.y_train,
            split.y_test,
        )

        self.assertEqual(set(metrics), {"accuracy", "macro_f1"})
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)

    def test_scaled_distance_models_are_pipelines(self):
        self.assertIsInstance(build_scaled_model("knn", 5), Pipeline)
        self.assertIsInstance(build_scaled_model("svm", 1.0), Pipeline)


class CrossValidationContractTests(unittest.TestCase):
    def test_cv_grid_has_exact_required_candidates(self):
        self.assertEqual(CV_GRIDS["kNN"], [1, 3, 5, 9, 15])
        self.assertEqual(CV_GRIDS["Decision Tree"], [2, 3, 5, 10, None])
        self.assertEqual(CV_GRIDS["SVM"], [0.1, 1.0, 10.0])

    def test_cv_search_returns_13_candidates_and_five_fold_scores(self):
        dataset = load_dataset_a()
        split = make_fixed_split(dataset.data, dataset.target, "A")
        results = run_cv_search(split.X_train, split.y_train)

        self.assertEqual(len(results), 13)
        self.assertEqual(
            [column for column in results.columns if column.startswith("fold_")],
            ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
        )
        recalculated = results[[f"fold_{i}" for i in range(1, 6)]].std(axis=1, ddof=0)
        np.testing.assert_allclose(results["std_macro_f1"], recalculated)

    def test_selection_uses_cv_mean_then_std_without_test_metrics(self):
        candidates = pd.DataFrame(
            [
                {"classifier": "SVM", "value": 0.1, "mean_macro_f1": 0.91, "std_macro_f1": 0.03},
                {"classifier": "SVM", "value": 1.0, "mean_macro_f1": 0.92, "std_macro_f1": 0.04},
                {"classifier": "SVM", "value": 10.0, "mean_macro_f1": 0.92, "std_macro_f1": 0.02},
            ]
        )

        selected = select_cv_setting(candidates, "SVM")

        self.assertEqual(selected["value"], 10.0)
        self.assertNotIn("test_accuracy", selected)

    def test_fitted_scaler_uses_training_mean_not_full_dataset_mean(self):
        dataset = load_dataset_a()
        split = make_fixed_split(dataset.data, dataset.target, "A")
        model = build_scaled_model("knn", 5)
        model.fit(split.X_train, split.y_train)

        scaler_mean = model.named_steps["scaler"].mean_
        np.testing.assert_allclose(scaler_mean, split.X_train.mean(axis=0))
        self.assertGreater(np.max(np.abs(scaler_mean - dataset.data.mean(axis=0))), 1e-6)


class OutputContractTests(unittest.TestCase):
    def test_expected_result_filenames_are_declared(self):
        required = {
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
        }
        self.assertEqual(set(EXPECTED_RESULT_FILES), required)

    def test_baseline_models_match_assignment_parameters(self):
        models = baseline_models()

        self.assertEqual(models["Decision Tree"].get_params()["random_state"], 42)
        self.assertEqual(models["Random Forest"].get_params()["n_estimators"], 200)
        self.assertEqual(models["Random Forest"].get_params()["random_state"], 42)
        self.assertEqual(models["kNN"].get_params()["n_neighbors"], 5)
        self.assertEqual(models["SVM"].get_params()["kernel"], "rbf")
        self.assertEqual(models["SVM"].get_params()["C"], 1.0)

    def test_run_all_exports_results_and_reuses_saved_split_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results"
            run_all(result_dir=result_dir)

            self.assertEqual(
                {path.name for path in result_dir.iterdir() if path.name in EXPECTED_RESULT_FILES},
                set(EXPECTED_RESULT_FILES),
            )
            split_a = pd.read_csv(result_dir / "splits_a.csv")
            self.assertEqual(len(split_a), 569)
            self.assertEqual(split_a["source_index"].nunique(), 569)
            self.assertEqual(split_a["split"].value_counts().to_dict(), {"train": 455, "test": 114})

            baseline = pd.read_csv(result_dir / "baseline_a.csv")
            self.assertEqual(len(baseline), 5)
            self.assertEqual(set(baseline["classifier"]), set(baseline_models()))

            cv_results = pd.read_csv(result_dir / "cv_search_a.csv")
            self.assertEqual(len(cv_results), 13)
            self.assertFalse(cv_results[[f"fold_{i}" for i in range(1, 6)]].isna().any().any())

            part1 = pd.read_csv(result_dir / "part1_practice.csv")
            self.assertEqual(len(part1), 12)
            self.assertEqual(part1["classifier"].nunique(), 6)

            digits = pd.read_csv(result_dir / "comparison_b.csv")
            self.assertEqual(len(digits), 5)
            self.assertEqual(sorted(digits["rank"].tolist()), [1, 2, 3, 4, 5])


class ChartContractTests(unittest.TestCase):
    def test_generate_charts_creates_six_nonempty_files_from_exported_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results"
            figure_dir = root / "figures"
            run_all(result_dir=result_dir)

            metadata = generate_charts(result_dir=result_dir, figure_dir=figure_dir)

            expected = {
                "standardization_comparison.png",
                "standardization_comparison.pdf",
                "cv_hyperparameters.png",
                "cv_hyperparameters.pdf",
                "digits_confusion_matrix.png",
                "digits_confusion_matrix.pdf",
            }
            self.assertEqual({path.name for path in figure_dir.iterdir()}, expected)
            self.assertTrue(all(path.stat().st_size > 1_000 for path in figure_dir.iterdir()))
            self.assertIn("Accuracy", metadata["standardization"]["metric_labels"])
            self.assertIn("Macro-F1", metadata["standardization"]["metric_labels"])
            self.assertEqual(metadata["cv"]["y_label"], "Mean CV Macro-F1")

            scaling = pd.read_csv(result_dir / "scaling_a.csv")
            self.assertAlmostEqual(
                metadata["standardization"]["max_scaled_macro_f1"],
                scaling["scaled_macro_f1"].max(),
            )
            matrix = pd.read_csv(result_dir / "digits_confusion_matrix.csv", index_col=0)
            self.assertEqual(matrix.shape, (10, 10))
            self.assertEqual(int(matrix.to_numpy().sum()), 360)


class Part1DocumentContractTests(unittest.TestCase):
    def test_completed_record_contains_required_experiment_evidence(self):
        from src.build_part1_record import build_part1_record

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "completed.docx"
            build_part1_record(output_path=output_path)

            self.assertTrue(output_path.exists())
            with zipfile.ZipFile(output_path) as archive:
                document_xml = archive.read("word/document.xml")
            for namespace_prefix in (b"w15", b"w16se", b"w16cid", b"w16cex", b"w16sdtdh", b"wp14"):
                self.assertIn(b"xmlns:" + namespace_prefix + b"=", document_xml)
            root = ET.fromstring(document_xml)
            text = "\n".join(root.itertext())

            for classifier in (
                "KNeighborsClassifier",
                "GaussianNB",
                "DecisionTreeClassifier",
                "SVC",
                "MLPClassifier",
                "RandomForestClassifier",
            ):
                self.assertIn(classifier, text)
            for required_text in (
                "程式：",
                "設定 A",
                "設定 B",
                "Accuracy",
                "Macro-F1",
                "觀察",
                "confusion_matrix",
                "classification_report",
                "f1_score",
                "5-fold",
            ):
                self.assertIn(required_text, text)
            for code_fragment in (
                "for k in (3, 15):",
                "GaussianNB(var_smoothing=v)",
                "DecisionTreeClassifier(max_depth=d, random_state=42)",
                'SVC(kernel="rbf", C=c)',
                "RandomForestClassifier(n_estimators=n, random_state=42)",
                "MLPClassifier(hidden_layer_sizes=h, solver=\"lbfgs\"",
                "cross_val_score(",
                "confusion_matrix(y_test, pred)",
                "classification_report(y_test, pred",
                'f1_score(y_test, pred, average="macro")',
            ):
                self.assertIn(code_fragment, text)
            self.assertNotIn("詳見附錄", text)
            self.assertNotIn("HW5 Part 1 實驗附錄", text)
            self.assertNotIn("非本次必做", text)

            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            page_size = root.find(".//w:sectPr/w:pgSz", namespace)
            self.assertIsNotNone(page_size)
            self.assertEqual(
                page_size.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}orient"),
                "landscape",
            )
            optional_markers = (
                "KNighborsRegressor",
                "MultinomialNB",
                "DecisionTreeRegressor",
                "LinearSVC",
                "MLPRegressor",
                "GradientBoostingClassifier",
                "ShuffleSplit",
            )
            found_optional = set()
            required_rows_with_page_control = 0
            evaluation_rows_with_page_control = 0
            code_run_count = 0
            for row in root.findall(".//w:tr", namespace):
                cells = row.findall("w:tc", namespace)
                if len(cells) != 4:
                    continue
                function_text = "".join(cells[2].itertext())
                if function_text.strip() in {
                    "KNeighborsClassifier()",
                    "GaussianNB()",
                    "DecisionTreeClassifier()",
                    "SVC()",
                    "MLPClassifier()",
                    "RandomForestClassifier()",
                }:
                    self.assertIsNotNone(row.find("w:trPr/w:cantSplit", namespace))
                    code_runs = []
                    for run in cells[3].findall(".//w:r", namespace):
                        fonts = run.find("w:rPr/w:rFonts", namespace)
                        if fonts is not None and fonts.get(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii"
                        ) == "Consolas":
                            code_runs.append(run)
                    self.assertGreaterEqual(len(code_runs), 3)
                    code_run_count += len(code_runs)
                    required_rows_with_page_control += 1
                if function_text.strip() == "KFold()" or "confusion_matrix" in function_text:
                    self.assertIsNotNone(row.find("w:trPr/w:cantSplit", namespace))
                    evaluation_rows_with_page_control += 1
                for marker in optional_markers:
                    if marker in function_text:
                        found_optional.add(marker)
                        self.assertEqual("".join(cells[3].itertext()).strip(), "")
            self.assertEqual(found_optional, set(optional_markers))
            self.assertEqual(required_rows_with_page_control, 6)
            self.assertEqual(evaluation_rows_with_page_control, 2)
            self.assertGreaterEqual(code_run_count, 18)

            main_table = root.find(".//w:tbl", namespace)
            self.assertIsNotNone(main_table)
            header_row = main_table.find("w:tr", namespace)
            self.assertIsNotNone(header_row.find("w:trPr/w:tblHeader", namespace))


class DeliveryVerificationContractTests(unittest.TestCase):
    def test_report_covers_questions_and_q23_has_required_han_count(self):
        from src.verify_delivery import count_q23_han_characters, extract_question_numbers

        root = Path(__file__).resolve().parents[1]
        section_dir = root / "report" / "sections"

        self.assertEqual(extract_question_numbers(section_dir), list(range(4, 25)))
        self.assertEqual(count_q23_han_characters(section_dir / "part5.tex"), 225)

    def test_full_delivery_verification_checks_core_artifacts(self):
        from src.verify_delivery import verify_delivery

        root = Path(__file__).resolve().parents[1]
        summary = verify_delivery(root)

        self.assertEqual(summary["cv_candidate_count"], 13)
        self.assertEqual(summary["digits_confusion_matrix_total"], 360)
        self.assertEqual(summary["report_page_count"], 5)
        self.assertLessEqual(summary["report_page_count"], 10)
        self.assertEqual(summary["part1_required_classifier_count"], 6)
        self.assertTrue(summary["all_checks_passed"])


if __name__ == "__main__":
    unittest.main()
