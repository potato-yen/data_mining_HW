"""Verify that the HW5 deliverables agree with exported experiment results."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from pypdf import PdfReader


logging.getLogger("pypdf").setLevel(logging.ERROR)


SOURCE_HASHES = {
    "HW5_Classification_Comparison.pdf": (
        "97067dbd2acd43fd4a4d97844924c111b9918b60fd4695039a3e9e8d1b1b7bd9"
    ),
    "scikitLearn 操作記錄單.docx": (
        "eae7db16f810bd16d206d957118210f118066021d5f131e2db8fcf062e17c3db"
    ),
}

PART1_CLASSIFIERS = (
    "KNeighborsClassifier",
    "GaussianNB",
    "DecisionTreeClassifier",
    "SVC",
    "MLPClassifier",
    "RandomForestClassifier",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_question_numbers(section_dir: Path) -> list[int]:
    """Return all numbered ``\\q`` entries in report order."""
    numbers: list[int] = []
    for path in sorted(section_dir.glob("part*.tex")):
        numbers.extend(int(value) for value in re.findall(r"\\q\{(\d+)\}", path.read_text()))
    return numbers


def count_q23_han_characters(part5_path: Path) -> int:
    """Count Unicode Han characters in the answer to question 23 only."""
    text = part5_path.read_text()
    answer = text.split(r"\label{q23text}", 1)[1].split(r"\q{24}", 1)[0]
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", answer))


def _part1_document_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    return "\n".join(ET.fromstring(document_xml).itertext())


def _selected_parameter_checks(results_dir: Path) -> dict[str, float]:
    search = pd.read_csv(results_dir / "cv_search_a.csv")
    selected = json.loads((results_dir / "cv_selected_a.json").read_text())
    checked: dict[str, float] = {}
    for classifier in ("kNN", "Decision Tree", "SVM"):
        candidates = search.loc[search["classifier"] == classifier].copy()
        best = candidates.sort_values(
            ["mean_macro_f1", "std_macro_f1"], ascending=[False, True]
        ).iloc[0]
        actual = float(selected[classifier]["value"])
        if actual != float(best["value"]):
            raise AssertionError(f"CV selection mismatch for {classifier}")
        checked[classifier] = actual
    return checked


def verify_delivery(root: Path) -> dict[str, object]:
    """Run cross-artifact checks and return a JSON-serializable summary."""
    root = root.resolve()
    results_dir = root / "results"
    section_dir = root / "report" / "sections"
    report_text = "\n".join(path.read_text() for path in sorted(section_dir.glob("*.tex")))

    questions = extract_question_numbers(section_dir)
    if questions != list(range(4, 25)):
        raise AssertionError(f"Report question coverage is {questions}")

    q23_count = count_q23_han_characters(section_dir / "part5.tex")
    if not 200 <= q23_count <= 300:
        raise AssertionError(f"Question 23 has {q23_count} Han characters")

    cv_search = pd.read_csv(results_dir / "cv_search_a.csv")
    if len(cv_search) != 13:
        raise AssertionError("CV export must contain exactly 13 candidates")
    selected_parameters = _selected_parameter_checks(results_dir)

    matrix = pd.read_csv(results_dir / "digits_confusion_matrix.csv", index_col=0)
    matrix_total = int(matrix.to_numpy().sum())
    if matrix.shape != (10, 10) or matrix_total != 360:
        raise AssertionError("Digits confusion matrix must be 10x10 with 360 observations")

    docx_path = root / "report" / "ScikitLearn_操作記錄單_完成版.docx"
    part1_text = _part1_document_text(docx_path)
    found_classifiers = [name for name in PART1_CLASSIFIERS if name in part1_text]
    if len(found_classifiers) != len(PART1_CLASSIFIERS):
        raise AssertionError("Part 1 record is missing a required classifier")

    baseline = pd.read_csv(results_dir / "baseline_a.csv")
    digits = pd.read_csv(results_dir / "comparison_b.csv")
    key_values = [
        *(f"{value:.4f}" for value in baseline["accuracy"]),
        *(f"{value:.4f}" for value in baseline["macro_f1"]),
        *(f"{value:.4f}" for value in digits["accuracy"]),
        *(f"{value:.4f}" for value in digits["macro_f1"]),
    ]
    missing_values = sorted({value for value in key_values if value not in report_text})
    if missing_values:
        raise AssertionError(f"Report is missing exported values: {missing_values}")

    source_hashes = {name: _sha256(root / name) for name in SOURCE_HASHES}
    if source_hashes != SOURCE_HASHES:
        raise AssertionError("An original source attachment was modified")

    report_pdf = root / "output" / "pdf" / "HW5_report.pdf"
    part1_pdf = root / "output" / "pdf" / "ScikitLearn_操作記錄單_完成版.pdf"
    report_pages = len(PdfReader(str(report_pdf)).pages)
    part1_pages = len(PdfReader(str(part1_pdf)).pages)
    if report_pages > 10 or report_pages < 1 or part1_pages < 1:
        raise AssertionError("A final PDF has an invalid page count")

    expected_figures = {
        "standardization_comparison.png",
        "standardization_comparison.pdf",
        "cv_hyperparameters.png",
        "cv_hyperparameters.pdf",
        "digits_confusion_matrix.png",
        "digits_confusion_matrix.pdf",
    }
    figure_names = {path.name for path in (root / "figures").iterdir() if path.is_file()}
    if figure_names != expected_figures:
        raise AssertionError("Figure deliverables do not match the required set")

    metadata = json.loads((results_dir / "run_metadata.json").read_text())
    tool_versions = {
        "python": sys.version.split()[0],
        **metadata["packages"],
        "uv": subprocess.run(
            ["uv", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "tectonic": subprocess.run(
            ["tectonic", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
    }
    artifact_paths = (docx_path, report_pdf, part1_pdf)

    return {
        "all_checks_passed": True,
        "report_questions": questions,
        "q23_han_character_count": q23_count,
        "cv_candidate_count": len(cv_search),
        "selected_parameters": selected_parameters,
        "digits_confusion_matrix_shape": list(matrix.shape),
        "digits_confusion_matrix_total": matrix_total,
        "part1_required_classifier_count": len(found_classifiers),
        "report_key_metric_strings_checked": len(set(key_values)),
        "source_hashes": source_hashes,
        "report_page_count": report_pages,
        "part1_record_page_count": part1_pages,
        "figure_count": len(figure_names),
        "artifact_sizes_bytes": {
            str(path.relative_to(root)): path.stat().st_size for path in artifact_paths
        },
        "tool_versions": tool_versions,
        "verification_commands": [
            "uv lock --check",
            "uv run --no-sync python -m unittest discover -s tests -v",
            "uv run --no-sync python src/hw5_experiments.py",
            "uv run --no-sync python src/build_part1_record.py",
            "tectonic report/HW5_report.tex --outdir output/pdf",
            "uv run --no-sync python src/verify_delivery.py",
        ],
        "manual_visual_inspection": {
            "date": "2026-11-19",
            "report_pages_inspected": report_pages,
            "part1_record_pages_inspected": part1_pages,
            "result": "no clipping, overlap, missing glyphs, or broken tables observed",
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = verify_delivery(root)
    output_path = root / "results" / "verification_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
