# HW6 AdaBoost 學習單

採題目指定的三份固定抽樣清單，比較 A–F 六種 Decision Stump；選分類器使用抽樣集，計算 ε、α 及更新權重使用原始十筆資料。原始 `AdaBoost_學習單.docx` 保持唯讀。

## 交付檔案

- `output/pdf/worksheet.pdf`：完成版學習單，可供繳交，包含三輪表格、計算示範、完整樣本權重、預測及三道思考題。
- `output/pdf/main.pdf`：中文計算說明，另含原始資料、公式、重現方法及驗證說明。
- `src/adaboost.py`：通用候選比較、加權計算及輸出主程式，無第三方 Python 依賴。
- `results/`：18 個候選錯誤筆數、三輪摘要、兩輪共 20 筆權重更新、四筆預測及完整 JSON。
- `report/`：分節 LaTeX；`sections/` 由程式從計算結果產生，請勿手動修改。
- `docs/verification.md`：驗收對照與實際驗證紀錄。

## 從頭重建

以下命令皆由 `HW6/` 開始，Python 明確綁定本專案環境。使用區域快取，避免寫入使用者層級 uv 快取。

```sh
uv --cache-dir .uv-cache sync --offline --frozen
uv --cache-dir .uv-cache run --offline python -m unittest discover -s tests -v
uv --cache-dir .uv-cache run --offline python src/adaboost.py
mkdir -p output/pdf
cd report
tectonic --keep-logs --outdir ../output/pdf worksheet.tex
tectonic --keep-logs --outdir ../output/pdf main.tex
```

若電腦尚未安裝 Python 3.12，第一次 `uv sync` 需允許 uv 下載專案 Python；移除 `--offline` 即可。Tectonic 首次編譯可能需要下載 LaTeX 套件。這兩者均不要求全域安裝 Python 套件或新字體。

## PDF 版面檢視

本機沒有 Poppler，使用 macOS 內建 PDFKit，無需新增依賴。回到 `HW6/` 執行：

```sh
mkdir -p tmp/swift-cache
swift -module-cache-path tmp/swift-cache src/inspect_pdf.swift output/pdf/worksheet.pdf tmp/qa/worksheet
swift -module-cache-path tmp/swift-cache src/inspect_pdf.swift output/pdf/main.pdf tmp/qa/main
uv --cache-dir .uv-cache run --offline python src/verify_delivery.py
```

每頁會產生 PNG，並記錄 PDF 頁數及擷取文字至 `inspection.json`；圖片仍需逐頁視覺檢查。

## 結果與慣例

三輪選擇依序為 A、E、D；ε 分別為 1/5、3/16、5/26。新資料 2.5、5.5、8.5、9.8 的預測依序為 −1、+1、−1、+1。
中間值保留雙精度，報告顯示六位小數；第三輪不更新樣本權重。候選同分依 A–F 順序取第一個，零投票回傳 0，本題均未觸發。學習單採 PDF 交付，沒有改寫或新增完成版 DOCX，也沒有自行增加額外的隨機實驗。
