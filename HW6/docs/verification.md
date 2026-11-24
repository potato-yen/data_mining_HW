# HW6 驗收紀錄

驗證日期：2026-11-24。範圍僅 HW6；原始附件 `AdaBoost_學習單.docx` 保留，SHA-256 為 `0a59dd16bdacae86604cc1e5db8f24de575b26198dc769c977d16b25112ecdff`。
已先閱讀根目錄 AGENTS.md 及 job.md，再以標準函式庫擷取原始學習單全文，確認資料、固定抽樣清單及繳交要求一致。

## 自動化驗證

以下命令實際執行成功；除註明外皆在 HW6 執行。

```sh
uv --cache-dir .uv-cache sync --offline --frozen
uv --cache-dir .uv-cache run --offline python -m unittest discover -s tests -v
uv --cache-dir .uv-cache run --offline python src/adaboost.py
```

- Python 3.12.13，無第三方 Python 依賴；pyproject.toml、.python-version、uv.lock 及 .venv 已建立。
- 5 項 unittest 全數通過，涵蓋重複抽樣次數、三輪分類器、原始資料加權錯誤率、α、錯分 ID、完整權重與正規化、切分點等號、四筆投票與 LaTeX 控制字元。
- 數值期望值以手算分數及對數給定，不從受測函式產生。初始測試先因尚無計算模組而失敗，再實作使其通過。
- 產生器曾出現非 raw 字串將 LaTeX 反斜線解讀為控制字元的問題，新增測試重現失敗後修正；最終測試及產生器均無該警告。
- 從頭重跑主程式已重建四份 CSV、一份完整 JSON、console 摘要及五份 LaTeX 分節，不需修改中間檔。

在 `HW6/report/` 實際執行：

```sh
tectonic --keep-logs --outdir ../output/pdf worksheet.tex
tectonic --keep-logs --outdir ../output/pdf main.tex
```

兩份最終 PDF 編譯成功。初次在受限環境缺少 size10.clo 等快取，經核准網路存取後使用既有 tectonic 下載所需 TeX 檔案並完成編譯；未全域安裝新套件或字體。主報告首段曾有 overfull 警告，分段後重新編譯已消除。

接著在 HW6 執行：

```sh
swift -module-cache-path tmp/swift-cache src/inspect_pdf.swift output/pdf/worksheet.pdf tmp/qa/worksheet
swift -module-cache-path tmp/swift-cache src/inspect_pdf.swift output/pdf/main.pdf tmp/qa/main
uv --cache-dir .uv-cache run --offline python src/verify_delivery.py
```

本機沒有 Poppler，改用既有 macOS PDFKit，兩份 PDF 均成功開啟並逐頁輸出圖片。交付檢查確認：

- 學習單 4 頁、96,788 bytes；主報告 5 頁、141,550 bytes。
- 18 筆候選錯誤、20 筆更新權重、4 筆預測符合獨立期望值；兩輪權重和均為 1（浮點容差 1e-12）。
- PDF 擷取文字包含三輪 ε、α、兩輪所有顯示權重、四筆投票分數與 Q1–Q3。
- 最終編譯 log 無 Overfull、Missing character 或 undefined references；擷取文字無 `??`。
- 檢查含壓縮物件串流的 PDF 字體描述，確認存在內嵌字體資料；逐頁視覺檢查另確認中文與數學符號呈現。
- 結果及 PDF 雜湊記錄於 `results/verification_summary.json`。

## 人工視覺檢查與獨立審查

已開啟並檢查學習單全部 4 張與主報告全部 5 張 PDFKit 渲染圖片：字型可讀、沒有缺字、表格未截斷、數學式不重疊、題號與頁碼正確、表格標題及主報告交叉引用可辨識。學習單沿用原題三至七節題號；一、二節的資料與公式另列於主報告。

獨立唯讀審查以精確有理數覆核所有數值及規格覆蓋，確認 A/E/D、三個 ε、α、兩輪權重與四筆投票一致，未提出需修正事項。此審查與自動化測試分開進行；版面以實際圖片檢視判定。

## job.md 驗收對照

| 驗收項目 | 證據 | 狀態 |
|---|---|---|
| 18 個候選錯誤筆數、三個最佳分類器 | candidate_errors.csv、round_summary.csv、兩份 PDF 各輪候選表 | 完成 |
| 三輪 ε、α | round_summary.csv、各輪公式 | 完成 |
| Round 1、2 各一筆更新示範及完整十筆權重 | weights.csv、各輪更新表與公式 | 完成 |
| Round 1 錯分 ID 與兩種新權重 | ID 8、9；分對 1/16、分錯 1/4 | 完成 |
| Round 2 更新後權重和 | 5/26 + 3/6 + 4/13 = 1；程式及測試檢查 | 完成 |
| Round 3 對原始資料的錯誤筆數 | ID 1、2、3、4、10，共 5 筆 | 完成 |
| 四筆完整預測表及投票示範 | predictions.csv、兩份 PDF 第六節 | 完成 |
| 三道思考題各 1–2 句 | 兩份 PDF 第七節，每題兩句繁體中文 | 完成 |

## 限制與仍需使用者處理的事項

- 交付學習單採重新排版的完成版 PDF，未回填原 DOCX，也未新增完成版 DOCX。原題允許完成學習單及附計算表格，沒有指定必須使用 DOCX。
- 中文 LaTeX 使用本機既有 Songti TC／Heiti TC；PDF 已內嵌字體，但在其他作業系統重新編譯 LaTeX 時，需自行指定可用中文字體，未驗證跨系統編譯。
- 程式只處理題目固定案例，不是一般資料集訓練套件；同分取 A–F 順序第一個、零投票回傳 0，這兩種慣例均未被本題資料觸發。
- PDF 未加入姓名或學號，題目與工作區沒有提供此資料；若課程繳交需要識別資訊，請使用者依繳交方式補上。尚未代為上傳課程平台。
