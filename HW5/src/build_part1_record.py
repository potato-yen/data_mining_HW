"""Build the completed Part 1 operation record from the retained DOCX template.

The implementation intentionally uses only the Python standard library so the
artifact remains buildable in the assignment's offline sandbox.  It preserves
every OOXML package member and replaces only ``word/document.xml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOCX = ROOT / "scikitLearn 操作記錄單.docx"
RESULT_CSV = ROOT / "results" / "part1_practice.csv"
EVALUATION_JSON = ROOT / "results" / "part1_evaluation.json"
DEFAULT_OUTPUT = ROOT / "report" / "ScikitLearn_操作記錄單_完成版.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
IGNORABLE_NAMESPACES = {
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
    "w16se": "http://schemas.microsoft.com/office/word/2015/wordml/symex",
    "w16cid": "http://schemas.microsoft.com/office/word/2016/wordml/cid",
    "w16": "http://schemas.microsoft.com/office/word/2018/wordml",
    "w16cex": "http://schemas.microsoft.com/office/word/2018/wordml/cex",
    "w16sdtdh": "http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash",
    "wp14": "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
}
ET.register_namespace("w", W_NS)
ET.register_namespace("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("o", "urn:schemas-microsoft-com:office:office")
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("m", "http://schemas.openxmlformats.org/officeDocument/2006/math")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("w10", "urn:schemas-microsoft-com:office:word")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup")
ET.register_namespace("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk")
ET.register_namespace("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")


def q(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def _run(
    text: str,
    *,
    bold: bool = False,
    size: int = 20,
    color: str = "1F2937",
    font: str = "Times New Roman",
) -> ET.Element:
    run = ET.Element(q("r"))
    props = ET.SubElement(run, q("rPr"))
    fonts = ET.SubElement(props, q("rFonts"))
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(q(key), font if key != "eastAsia" else "PingFang TC")
    if bold:
        ET.SubElement(props, q("b"))
        ET.SubElement(props, q("bCs"))
    ET.SubElement(props, q("color")).set(q("val"), color)
    ET.SubElement(props, q("sz")).set(q("val"), str(size))
    ET.SubElement(props, q("szCs")).set(q("val"), str(size))
    value = ET.SubElement(run, q("t"))
    value.set(f"{{{XML_NS}}}space", "preserve")
    value.text = text
    return run


def _paragraph(
    text: str = "",
    *,
    bold: bool = False,
    size: int = 20,
    color: str = "1F2937",
    font: str = "Times New Roman",
    align: str | None = None,
    before: int = 0,
    after: int = 80,
    line: int = 260,
    page_break_before: bool = False,
) -> ET.Element:
    paragraph = ET.Element(q("p"))
    props = ET.SubElement(paragraph, q("pPr"))
    spacing = ET.SubElement(props, q("spacing"))
    spacing.set(q("before"), str(before))
    spacing.set(q("after"), str(after))
    spacing.set(q("line"), str(line))
    spacing.set(q("lineRule"), "auto")
    if align:
        ET.SubElement(props, q("jc")).set(q("val"), align)
    if page_break_before:
        ET.SubElement(props, q("pageBreakBefore"))
    paragraph.append(_run(text, bold=bold, size=size, color=color, font=font))
    return paragraph


def _cell(text: str, *, header: bool = False, width: int | None = None) -> ET.Element:
    cell = ET.Element(q("tc"))
    props = ET.SubElement(cell, q("tcPr"))
    if width is not None:
        width_element = ET.SubElement(props, q("tcW"))
        width_element.set(q("w"), str(width))
        width_element.set(q("type"), "dxa")
    if header:
        shading = ET.SubElement(props, q("shd"))
        shading.set(q("val"), "clear")
        shading.set(q("color"), "auto")
        shading.set(q("fill"), "245B78")
    vertical = ET.SubElement(props, q("vAlign"))
    vertical.set(q("val"), "center")
    cell.append(
        _paragraph(
            text,
            bold=header,
            size=19 if header else 18,
            color="FFFFFF" if header else "1F2937",
            align="center" if header else None,
            after=20,
        )
    )
    return cell


def _table(headers: list[str], rows: list[list[str]], widths: list[int]) -> ET.Element:
    table = ET.Element(q("tbl"))
    props = ET.SubElement(table, q("tblPr"))
    ET.SubElement(props, q("tblStyle")).set(q("val"), "TableGrid")
    ET.SubElement(props, q("tblW"), {q("w"): "0", q("type"): "auto"})
    borders = ET.SubElement(props, q("tblBorders"))
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(
            borders,
            q(side),
            {q("val"): "single", q("sz"): "6", q("space"): "0", q("color"): "B7C4CE"},
        )
    grid = ET.SubElement(table, q("tblGrid"))
    for width in widths:
        ET.SubElement(grid, q("gridCol")).set(q("w"), str(width))

    header_row = ET.SubElement(table, q("tr"))
    row_props = ET.SubElement(header_row, q("trPr"))
    ET.SubElement(row_props, q("tblHeader"))
    for header, width in zip(headers, widths, strict=True):
        header_row.append(_cell(header, header=True, width=width))
    for row_index, values in enumerate(rows):
        row = ET.SubElement(table, q("tr"))
        for value, width in zip(values, widths, strict=True):
            cell = _cell(value, width=width)
            if row_index % 2:
                cell.find(q("tcPr")).append(ET.Element(q("shd"), {q("fill"): "F3F7F9"}))
            row.append(cell)
    return table


def _cell_text(cell: ET.Element) -> str:
    return "".join(cell.itertext()).strip()


def _replace_cell_text(cell: ET.Element, text: str) -> None:
    props = cell.find(q("tcPr"))
    preserved = deepcopy(props) if props is not None else ET.Element(q("tcPr"))
    for child in list(cell):
        cell.remove(child)
    cell.append(preserved)
    for line in text.split("\n"):
        cell.append(_paragraph(line, size=18, after=15))


def _replace_practice_cell(
    cell: ET.Element,
    *,
    introduction: str,
    code_lines: list[str],
    result_lines: list[str],
    observation: str,
) -> None:
    props = cell.find(q("tcPr"))
    preserved = deepcopy(props) if props is not None else ET.Element(q("tcPr"))
    for child in list(cell):
        cell.remove(child)
    cell.append(preserved)
    cell.append(_paragraph(introduction, size=18, after=25))
    cell.append(_paragraph("程式：", bold=True, size=18, after=5))
    for line in code_lines:
        cell.append(
            _paragraph(
                line,
                size=17,
                color="374151",
                font="Consolas",
                after=0,
                line=220,
            )
        )
    cell.append(_paragraph("結果：", bold=True, size=18, before=20, after=5))
    for line in result_lines:
        cell.append(_paragraph(line, size=18, after=5))
    cell.append(_paragraph(f"觀察：{observation}", size=18, before=10, after=10))


def _ensure_row_property(row: ET.Element, property_name: str) -> None:
    properties = row.find(q("trPr"))
    if properties is None:
        properties = ET.Element(q("trPr"))
        row.insert(0, properties)
    if properties.find(q(property_name)) is None:
        ET.SubElement(properties, q(property_name))


def _load_results() -> tuple[dict[str, list[dict[str, str]]], dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with RESULT_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["classifier"]].append(row)
    with EVALUATION_JSON.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)
    return grouped, evaluation


def _serialize_document(root: ET.Element) -> bytes:
    """Serialize while retaining namespace declarations named by mc:Ignorable.

    ElementTree removes declarations that only occur as tokens inside the
    ``mc:Ignorable`` attribute.  Microsoft Word requires those tokens to stay
    bound, so restore the original Word extension declarations explicitly.
    """

    serialized = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    text = serialized.decode("utf-8")
    insertion = "".join(
        f' xmlns:{prefix}="{uri}"'
        for prefix, uri in IGNORABLE_NAMESPACES.items()
        if f"xmlns:{prefix}=" not in text.split(">", 1)[0]
    )
    return text.replace("<w:document ", f"<w:document{insertion} ", 1).encode("utf-8")


def _summary(model: str, model_rows: list[dict[str, str]]) -> dict[str, object]:
    descriptions = {
        "KNeighborsClassifier": "依鄰近樣本投票；n_neighbors 控制鄰居數（先標準化）。",
        "GaussianNB": "以類別的高斯分布估計機率；var_smoothing 增加穩定項。",
        "DecisionTreeClassifier": "依特徵閾值遞迴分割；max_depth 限制樹深。",
        "SVC": "以 RBF 核建立非線性邊界；C 控制錯分懲罰（先標準化）。",
        "RandomForestClassifier": "以多棵隨機樹投票；n_estimators 控制樹數。",
        "MLPClassifier": "以多層感知器學習非線性邊界；hidden_layer_sizes 控制結構。",
    }
    observations = {
        "KNeighborsClassifier": "k=3 較 k=15 的 Macro-F1 高 0.0097。",
        "GaussianNB": "var_smoothing=1e-9 較佳，Macro-F1 高 0.0209。",
        "DecisionTreeClassifier": "限制深度為 3 較佳，Macro-F1 高 0.0262。",
        "SVC": "C=10 較 C=0.1 佳，Macro-F1 高 0.0290。",
        "RandomForestClassifier": "50 與 200 棵樹在本次測試集的兩項分數相同。",
        "MLPClassifier": "兩種隱藏層結構在本次測試集的兩項分數相同。",
    }
    code_lines = {
        "KNeighborsClassifier": [
            "for k in (3, 15):",
            "    model = make_pipeline(StandardScaler(),",
            "        KNeighborsClassifier(n_neighbors=k))",
            "    pred = model.fit(X_train, y_train).predict(X_test)",
            "    print(k, accuracy_score(y_test, pred),",
            "        f1_score(y_test, pred, average=\"macro\"))",
        ],
        "GaussianNB": [
            "for v in (1e-9, 1e-7):",
            "    model = GaussianNB(var_smoothing=v)",
            "    pred = model.fit(X_train, y_train).predict(X_test)",
            "    print(v, accuracy_score(y_test, pred),",
            "        f1_score(y_test, pred, average=\"macro\"))",
        ],
        "DecisionTreeClassifier": [
            "for d in (3, None):",
            "    model = DecisionTreeClassifier(max_depth=d, random_state=42)",
            "    pred = model.fit(X_train, y_train).predict(X_test)",
            "    print(d, accuracy_score(y_test, pred),",
            "        f1_score(y_test, pred, average=\"macro\"))",
        ],
        "SVC": [
            "for c in (0.1, 10):",
            "    model = make_pipeline(StandardScaler(),",
            "        SVC(kernel=\"rbf\", C=c))",
            "    pred = model.fit(X_train, y_train).predict(X_test)",
            "    print(c, accuracy_score(y_test, pred),",
            "        f1_score(y_test, pred, average=\"macro\"))",
        ],
        "RandomForestClassifier": [
            "for n in (50, 200):",
            "    model = RandomForestClassifier(n_estimators=n, random_state=42)",
            "    pred = model.fit(X_train, y_train).predict(X_test)",
            "    print(n, accuracy_score(y_test, pred),",
            "        f1_score(y_test, pred, average=\"macro\"))",
        ],
        "MLPClassifier": [
            "for h in ((50,), (100, 50)):",
            "    model = make_pipeline(StandardScaler(),",
            "        MLPClassifier(hidden_layer_sizes=h, solver=\"lbfgs\",",
            "            max_iter=2000, random_state=42))",
            "    pred = model.fit(X_train, y_train).predict(X_test)",
            "    print(h, accuracy_score(y_test, pred),",
            "        f1_score(y_test, pred, average=\"macro\"))",
        ],
    }
    result_lines = []
    for label, row in zip(("A", "B"), model_rows, strict=True):
        setting = row["setting"]
        for fixed_parameter in (", random_state=42", ", solver=lbfgs"):
            setting = setting.replace(fixed_parameter, "")
        setting = setting.replace(", ", "; ")
        result_lines.append(
            f"設定 {label}: {setting}; Accuracy={float(row['accuracy']):.4f}; "
            f"Macro-F1={float(row['macro_f1']):.4f}"
        )
    return {
        "introduction": f"功能：{descriptions[model]}",
        "code_lines": code_lines[model],
        "result_lines": result_lines,
        "observation": observations[model],
    }


def _update_template_table(root: ET.Element, results: dict[str, list[dict[str, str]]], evaluation: dict) -> None:
    required = {
        "KNighborsClassifier": "KNeighborsClassifier",
        "Gaussian Naive Bayes": "GaussianNB",
        "DecisionTreeClassifier": "DecisionTreeClassifier",
        "SVC()": "SVC",
        "MLPClassifier": "MLPClassifier",
        "RandomForestClassifier": "RandomForestClassifier",
    }
    non_required = (
        "KNighborsRegressor",
        "MultinomialNB",
        "DecisionTreeRegressor",
        "LinearSVC",
        "MLPRegressor",
        "GradientBoostingClassifier",
    )
    for row in root.findall(".//" + q("tr")):
        cells = row.findall(q("tc"))
        if len(cells) != 4:
            continue
        function_text = _cell_text(cells[2])
        if function_text == "Function":
            _ensure_row_property(row, "tblHeader")
        if any(marker in function_text for marker in non_required):
            _replace_cell_text(cells[3], "")
            continue
        for marker, result_key in required.items():
            if marker in function_text:
                _ensure_row_property(row, "cantSplit")
                _replace_cell_text(cells[2], f"{result_key}()")
                _replace_practice_cell(
                    cells[3],
                    **_summary(result_key, results[result_key]),
                )
                break
        else:
            if "KFold" in function_text:
                _ensure_row_property(row, "cantSplit")
                cv = evaluation["cross_validation"]
                scores = ", ".join(f"{score:.4f}" for score in cv["fold_scores"])
                _replace_practice_cell(
                    cells[3],
                    introduction="用途：以五折交叉驗證估計模型在未見資料上的穩定性。",
                    code_lines=[
                        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
                        "scores = cross_val_score(",
                        "    make_pipeline(StandardScaler(), SVC(kernel=\"rbf\", C=1.0)),",
                        "    X_train, y_train, cv=cv, scoring=\"f1_macro\")",
                        "print(scores, scores.mean(), scores.std())",
                    ],
                    result_lines=[
                        f"5-fold Macro-F1: [{scores}]",
                        f"平均={cv['mean']:.4f}；標準差={cv['std_ddof_0']:.4f}",
                    ],
                    observation="五折分數介於 0.9402--0.9882，整體表現穩定。",
                )
            elif "ShuffleSplit" in function_text:
                _replace_cell_text(cells[3], "")
            elif "confusion_matrix" in function_text:
                _ensure_row_property(row, "cantSplit")
                matrix = evaluation["confusion_matrix"]
                report = evaluation["classification_report"]
                _replace_practice_cell(
                    cells[3],
                    introduction="用途：檢查分類錯誤、各類 precision/recall/F1 與整體 Macro-F1。",
                    code_lines=[
                        "pred = model.fit(X_train, y_train).predict(X_test)",
                        "print(confusion_matrix(y_test, pred))",
                        "print(classification_report(y_test, pred,",
                        "    target_names=[\"malignant\", \"benign\"]))",
                        "print(f1_score(y_test, pred, average=\"macro\"))",
                    ],
                    result_lines=[
                        f"confusion_matrix={matrix}",
                        "classification_report: malignant F1="
                        f"{report['malignant']['f1-score']:.4f}；benign F1="
                        f"{report['benign']['f1-score']:.4f}；Accuracy={report['accuracy']:.4f}",
                        f"f1_score(average=\"macro\")={evaluation['f1_score_macro']:.4f}",
                    ],
                    observation="兩類各錯分 1 筆；Macro-F1=0.9812，類別間表現接近。",
                )


def build_part1_record(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    source_path: Path = SOURCE_DOCX,
) -> Path:
    """Create a completed copy and return its path."""

    results, evaluation = _load_results()
    with zipfile.ZipFile(source_path, "r") as source_archive:
        document_xml = source_archive.read("word/document.xml")
        root = ET.fromstring(document_xml)
        _update_template_table(root, results, evaluation)
        updated_xml = _serialize_document(root)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w") as output_archive:
            for member in source_archive.infolist():
                payload = updated_xml if member.filename == "word/document.xml" else source_archive.read(member)
                output_archive.writestr(member, payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_part1_record(output_path=args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
