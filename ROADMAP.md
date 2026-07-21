# Breeze Elf — 未完功能路線圖

> 2026-07-18。以下三份設計都經過 research workflow（repo grounding + 網路查證）+ adversarial
> critique,critique 的修正已併入各節「注意」。完整設計細節見對話中的 `plans_digest.md`
> 與 memory `[[cross-transcript-search]]` / `[[silero-vad-and-capture-profiles]]`。

## 狀態摘要

| 功能 | 狀態 |
|---|---|
| #1 Silero VAD | ✅ 已完成、已 review、已驗證 |
| #2 收音設定檔 (EC/NS on、AGC off) | ✅ 已完成、已 review、已驗證 |
| #5a 跨稿搜尋 (FTS5 trigram) | ✅ 已完成、已 review、已驗證 |
| #5b 會後摘要 (extractive + 本地 Ollama) | ✅ 已完成、已 review、已驗證(對抗式 review 已補;修好 non-object JSON 回應會破契約的 bug) |
| **#3 語者分離 (diarization)** | ✅ 已完成、已 review、已驗證(torch-free ONNX embedder + 純 numpy 線上分群) |
| **#4 翻譯 (translation)** | ✅ 已完成、已 review、已驗證(NLLB→繁中,ctranslate2、免 torch) |

**全部完成**。2026-07-18 一次做完 #4 + #3,並補完 #5b 的對抗式 review。
決策紀錄:#4 走 NLLB 完整版(非 Whisper MVP)、#3 用 torch-free ONNX embedder、commit 分兩包
(#5b 一包;#4+#3 因在 config/main/app.js 逐行交織而合一包)。對抗式 multi-agent review 只挑出 3 個
low(2 個 README 缺列、1 個 Ollama non-object JSON 契約 bug),皆已修;核心(NLLB recipe、分群、
main 接線、前端)0 confirmed。兩功能都是 opt-in extra、缺依賴/模型則 no-op,核心邏輯全單元測試。

---

## #3 語者分離 (Diarization)

**一句話**:每段 VAD 語句貼上匿名、session 內的說話者標籤(說話者 1／2…),做法是把 utterance
audio 做 speaker embedding + 純 numpy 線上分群,**全本地**,以 opt-in `[diarize]` extra 開啟,缺依賴則 no-op。

- **Effort**:medium。
- **MVP**:`breeze_elf/diarize.py`(仿 `enhance.py` 結構)= `SpeakerEmbedder` protocol + `NullDiarizer` +
  一個真 embedder + 純 numpy `OnlineSpeakerClusterer`(L2-normalized running centroids、cosine 指派、
  低於門檻就開新說話者、`BREEZE_DIARIZE_MAX_SPEAKERS` 上限)+ `build_diarizer(settings)`(缺依賴/載入失敗 → NullDiarizer)。
- **關鍵決定 — embedder**:
  - `Resemblyzer`(Apache-2.0,權重內建於 wheel、零下載、16kHz 原生、CPU 可跑)— **但拉 torch**,且
    Windows 上 `webrtcvad` C-build 有摩擦、可能覆蓋你手動裝的 CUDA torch。
  - `SpeechBrain ECAPA`(192-dim,較強)— 首次從 HF 下載 ~20MB(免 token),依賴樹較重。
  - **torch-free ONNX embedder(最貼合本專案 ethos)**— 用已在的 onnxruntime,但要找權限允許的 ONNX
    模型 + 自寫 numpy fbank 前端(較多程式)。
- **依賴**:`onnxruntime`/`numpy` 已在(torch-free 路線零新依賴);torch 路線走既有 `[enhance]` 的 torch,
  embedder 建議跑 CPU 避免與 Whisper 搶 VRAM 及 cudnn64_9.dll 載入順序問題。
- **整合點**:
  - `breeze_elf/audio.py`:`AudioUtteranceBuffer` 已給出完整 per-utterance `window.samples`,**不用改 segmenter**。
  - `breeze_elf/main.py`:module-scope `diarizer`;每連線 `StreamState` 一個新 `clusterer`(標籤 per-session 重置);
    `_process_windows` 內對「有文字」的 window 算 embedding、指派 speaker,只加到 **`final`** 事件;
    `TranscriptBlock += speaker`;`/health += diarize`。
  - `web/app.js`:entry-meta 加彩色「說話者 N」chip。
- **⚠ critique 修正(務必照做)**:
  1. **embedding 要吃 raw `window.samples`,不是 `asr_samples`** — 後者經 DeepFilterNet 去噪去殘響,會抹掉聲紋特徵。
  2. 前端 `serializeBlocksForSave`(白名單)+ `normalizeTranscriptBlockForRestore` + session persist 三處都要帶 `speaker`,否則不會存/還原。
  3. speaker 只掛 `final`,不掛低延遲的 `partial`。
  4. 共用 module-scope diarizer 在 `asr_concurrency>1` 下需 inference lock(仿 DeepFilterEnhancer 的 `_lock`)。
  5. 極短/低能量語句先用 min-duration/RMS gate 擋掉,且 clusterer 要拒絕 non-finite embedding(否則污染 centroid)。
  6. 標籤 off-by-one:clusterer 回 0-based,UI 顯示 `N = index + 1`。
  7. 改 app.js 記得 bump `?v=` + service-worker cache。
- **隱私**:聲紋比逐字稿更敏感,一律本地;MVP 用匿名 session 標籤(不做跨 session 聲紋連結)。
  最高準度的 file-mode `pyannote/speaker-diarization-3.1` 需 HF token + 接受兩份 gated 條款(帳號綁定、非離線)→ 嚴格 opt-in、明確揭露,非預設。
- **未來**:file-mode 全域 re-cluster 提升準度;選配 pyannote-3.1 處理重疊語音;選配「具名說話者」重用既有 `voices/` 儲存
  (但 diarization embedding 要與 OpenVoice tone-color embedding **分開存**)。

---

## #4 翻譯 (Translation)

**一句話**:對繁中使用者最有價值的方向是「任意語言 → 繁體中文」,用 NLLB-200(跑在 faster-whisper 已自帶的
ctranslate2 runtime 上、**免 torch**),每句轉錄後再翻一次,雙語稿呈現。

- **Effort**:English-only MVP **small**;NLLB 完整版 **large**。
- **兩條路**:
  - **MVP(小)**:Whisper `task="translate"` — 零新模型零依賴,但**只能翻成英文**,且每句要**再解碼一次 Whisper**
    (單模型序列化 → 每句延遲約翻倍)。對繁中使用者價值有限,建議只當里程碑、別當成品。
  - **完整版(large,建議直接做)**:`breeze_elf/translate.py` 的 `NllbTranslator` 包 `ctranslate2.Translator` +
    `sentencepiece`,語碼映射到 flores(zh→`zho_Hant`、en→`eng_Latn`…),lazy load、缺模型/OOM → NullTranslator。
    跑在已在的 ctranslate2,**不加新的重依賴、不新增 cuDNN/DLL 衝突**(相對任何 torch-based MT 的優勢)。
- **依賴**:`ctranslate2`/`huggingface-hub` 已是 faster-whisper 的 runtime 依賴;只新增輕量 `sentencepiece`(`[translate]` extra)。
  NLLB-200-distilled-600M CT2 權重 ~0.6GB(int8)/~1.2GB(fp16),一次性下載,runtime ~1–2GB VRAM,和 Whisper medium 一起吃得下 3060 的 12GB。
- **整合點**:
  - `breeze_elf/asr.py`:把 `task` 參數穿過 `transcribe`。
  - `breeze_elf/asr_queue.py`:`task` 穿過 `_ASRJob → transcribe → _worker`。
  - `breeze_elf/main.py`:翻譯**只在轉錄通過 `_should_drop_asr_result` 之後**跑(別在每個 window 前跑、浪費解碼);
    只掛 `final`(非 `partial`);把 `translate`/`translateTarget` 也穿進 `_process_windows` 的簽章 + `create_task`;
    capability 要**不載入權重**就能判斷(config + 模型目錄存在與否)。
  - `breeze_elf/protocol.py`:`StartMessage += translate/translateTarget`。
  - `web/app.js`:block 加 `translation` 欄 + 雙語堆疊行(重用簡譜 dual-line CSS)+「翻譯」toggle;
    `serializeBlocksForSave`/restore/session persist 三處都要帶 `translation`;bump `?v=` + SW cache。
  - `breeze_elf/config.py`:`BREEZE_TRANSLATE`(off|whisper|nllb)、`_TARGET`、`_MODEL`、`_DEVICE`、`_COMPUTE_TYPE`、`_BEAM`。
- **⚠ critique 修正**:
  1. Whisper translate 那一 pass 要**關掉繁中 prompt / 詞庫 / OpenCC**(否則 zh 偏誤灌進英文輸出、OpenCC 套在英文上語意錯亂)。
  2. 整段英文翻譯無法對齊逐字 `novel_text` 去重 → 用重疊 window 時翻譯行會重複;翻譯**掛在整段而非 novel_text 子字串**要另處理。
  3. NLLB sentencepiece-only recipe 的地雷:flores 語碼 token(`zho_Hant`…)**不在** `sentencepiece.bpe.model`,是 CT2 共享 vocab 的 added special tokens;要餵 **token 字串**(`sp.encode(text, out_type=str)` 前綴語碼 + `</s>`),不是 piece id。務必寫一個 round-trip 測試鎖住。
  4. 模型預設指向本地目錄、**不在請求時偷偷 snapshot_download**(離線優先靠程式強制,不只文件寫寫)。
- **隱私/授權**:兩條路都全本地、無雲端 API。但 **NLLB-200 是 CC-BY-NC-4.0(非商用)**,若未來商用發佈不能預設啟用 → README 標註 + opt-in。
- **已查證事實**:Whisper `task=translate` 僅英文輸出;`medium` 支援 translate(turbo 不支援);ctranslate2 4.x/huggingface-hub 已在;
  NLLB CT2 免 torch、支援 `zho_Hant`;NLLB-600M ~0.6–1.2GB。

---

## #5 摘要 — 已完成 + 後續

- **已完成**:`breeze_elf/summarize.py`(extractive 預設、本地 Ollama opt-in、無雲端)、`POST /api/summary`、
  前端摘要 dialog、`BREEZE_SUMMARY_*` 設定。程式與 UI 已驗證。
- **待補**:對抗式 multi-agent review(因平台 classifier 暫時中斷未跑完)+ 最終全套測試複跑。
- **後續(選配)**:
  - Ollama tier 已就緒 — 使用者裝 [ollama](https://ollama.com) + `ollama pull qwen3:4b-instruct`,設 `BREEZE_SUMMARY_PROVIDER=ollama` 即得抽象式摘要,資料不出機。
  - extractive 目前用字頻均值挑句;可升級成 bigram / TextRank 提升品質。
  - 可加「對搜尋結果的稿子直接摘要」(目前是對現場逐字稿摘要)。

---

## 尚待決定(給你挑)

1. **下一個做哪個** — #4 翻譯(NLLB→繁中,價值高但 large)vs #3 語者分離(medium,要先選 embedder)。
2. **#3 embedder** — torch-free ONNX(貼合 ethos、較多程式)vs Resemblyzer(快但 Windows/torch 摩擦)。
3. **#4 是否要 English-only MVP 先墊** — 還是直接上 NLLB。
4. **commit 策略** — 目前 #1/#2/#5a/#5b 都還沒 commit。
