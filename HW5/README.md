# HW5 分類器效能比較

本專案以固定的資料切分與隨機種子完成 HW5 Part 1--5 的實驗、圖表、操作記錄單與中文 LaTeX 報告。

## 環境

一般可連線環境：

```bash
export UV_CACHE_DIR="$PWD/tmp/uv-cache"
uv sync --locked
source .venv/bin/activate
```

目前受限工作區無法連線 PyPI，因此本次以 `uv` 建立專案內 `.venv`，並離線沿用同一門課既有專案中已鎖定的 Python 3.12 科學運算套件；沒有全域安裝。`pyproject.toml` 與 `uv.lock` 保存一般可連線環境的可重現規格。

## 執行

```bash
source .venv/bin/activate
export UV_CACHE_DIR="$PWD/tmp/uv-cache"
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python src/hw5_experiments.py
uv run --no-sync python src/build_part1_record.py
tectonic report/HW5_report.tex --outdir output/pdf
uv run --no-sync python src/verify_delivery.py
```

主程式會重建 `results/` 與 `figures/` 中的實驗產物。`build_part1_record.py` 產生可編輯的完成版 DOCX；靜態 PDF 由本機 Microsoft Word 轉出。Tectonic 的穩定入口 `report/HW5_report.tex` 會直接產生 `output/pdf/HW5_report.pdf`。

## 主要交付物

- `src/hw5_experiments.py`：Part 1--5 從資料載入到圖表輸出的完整程式。
- `report/ScikitLearn_操作記錄單_完成版.docx` 與 `output/pdf/ScikitLearn_操作記錄單_完成版.pdf`：Part 1 記錄單。
- `report/main.tex`、`report/sections/` 與 `output/pdf/HW5_report.pdf`：Part 2--5 中文報告及其 LaTeX 內容。
- `results/verification_summary.json`：題號、字數、CV、混淆矩陣、來源雜湊、頁數與檔案大小的交叉核對結果。
