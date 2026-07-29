# CLAUDE.md — Breeze Elf 工作準則

本機優先的即時語音轉錄與音樂分析工具（FastAPI + faster-whisper/CT2 後端、無框架前端），
繁體中文為第一語言。本檔是跨對話的長期記憶：只放**不隨單一功能過期的原則與硬規則**。

## 文件地圖

| 檔案 | 角色 |
|---|---|
| `CLAUDE.md`（本檔） | 原則與硬規則，開工前必讀 |
| `計畫.md` | 現行待辦——下一個對話的起點，做完就更新 |
| `TRAINING_PLAN.md` | 訓練計畫與執行紀錄（被程式碼註解引用，勿刪勿改編號） |
| `README.md` | 使用者文件；新功能、新設定都要同步進去 |

（歷史：`ROADMAP.md`、`OPTIMIZATION_PLAN.md` 已全數完成並移除，經驗併入本檔；細節在 git 歷史。）

## 核心原則

1. **全本地、離線優先。** 不偷連網下載模型——模型一律指向本地目錄，缺目錄就明確報錯或
   no-op，離線靠程式強制而非文件宣示。隱私敏感資料（音訊、聲紋、逐字稿）永不出機。
2. **重功能一律 opt-in extra，缺依賴則 no-op。** 仿 `enhance.py` 模式：protocol + Null 實作 +
   `build_*(settings)` 工廠，依賴或模型缺失時退回 Null，核心功能不受影響。
3. **推論側 torch-free。** 只用 ctranslate2 + onnxruntime + numpy；訓練需要 torch 時開
   **完全隔離的獨立環境**，絕不污染推論 venv（cuDNN/DLL 衝突踩過多次）。
4. **授權是紅線。** NC（非商用）資料集與模型（NLLB、MIR-1K…）不預設啟用、README 明標；
   商用線只用 Apache/CC-BY/自建資料。
5. **沒有 baseline 不開訓、不優化。** 先量測、再改動、每項單獨量增益；免訓練的旋鈕掃描
   順位永遠在微調之前——若旋鈕增益大於 LoRA，直接做成產品設定。結果不佳時先懷疑資料
   而非超參（英文線 16 首歌過擬的教訓）。
6. **單一事實來源。** 幻覺判定只在 `breeze_elf/hallucination.py`、文字正規化只在
   `tools/text_norm.py`；評測量到的必須是產品在跑的同一份程式，不准在工具裡複寫副本。
7. **量測程式與數字分離。** 純函式指標寫成免 GPU 可單測；評估正規化規則決定結論
   （標點、台羅註解），改規則用 `tools/rescore.py` 回溯重算文字距離——但幻覺率、
   對齊誤差、RTF 需要音檔與時間戳，一律重跑 `tools/eval_asr.py`。

## 環境與指令

- Python **>=3.11**（onnxruntime 1.24 起無 cp310 wheel）。
- 開發驗證照 CI 順序跑：`uv sync --extra dev` → `ruff check .` →
  `BREEZE_ASR_PROVIDER=mock` + `unittest discover` → `uv build --wheel`。
- **訓練／評估一律 `.venv/Scripts/python.exe` 直呼，禁止 `uv run`**——它會把
  `tokenizers` sync 回 0.23.1 炸掉 transformers（訓練堆疊釘 0.22.2），也可能把手動裝的
  CUDA torch 換成 CPU 版。
- 新測試寫成 `unittest.TestCase`：CI 用 `unittest discover`，pytest 風格的 module-level
  function 在 CI 是**沒有跑到的**。

## 硬性規則（違反必炸）

1. **永不覆蓋 `models/breeze-asr-25-ct2`**——原始與微調版都要留著才能 A/B。
2. **preset 有兩條註冊路徑**：env-var 閘控的內建 preset 在 process 啟動時讀（要重啟）；
   `tools/deploy_model.py` 寫進 `models/presets.json` 的動態註冊每 request 重讀（免重啟）。
   要讓手機立刻看到就走後者。
3. **改 `web/app.js` 必 bump `?v=` + service worker cache**，否則使用者拿到舊版。
4. **transcript block 加新欄位要同步三處**：`serializeBlocksForSave`（白名單）、
   `normalizeTranscriptBlockForRestore`、session persist——少一處就不會存／還原。
5. **speaker／translation 等衍生資訊只掛 `final` 事件**，不掛低延遲的 `partial`。
6. **模型轉檔（HF → CT2）後必跑 round-trip 一致性測試**——轉檔的靜默錯誤最難查
   （NLLB flores 語碼 token 不在 sentencepiece 的教訓）。
7. **共用的推論物件在 `asr_concurrency > 1` 下要有 inference lock**（仿 `DeepFilterEnhancer._lock`）。
8. **微調不能弄壞 word timestamps**——簡譜、歌詞對齊、音準評分全掛在它上面；
   Breeze 線每次訓練必跑說話回歸集，相對退步 ≥5% 就回滾。
