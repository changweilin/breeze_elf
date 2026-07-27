# Breeze Elf — 唱歌辨識（ALT）訓練計畫

> 2026-07-27。目標：**英文歌 → Whisper 線**、**台灣中文歌 → Breeze 線**，兩條各自微調成
> 可切換的 preset。本文件延續 `ROADMAP.md` / `OPTIMIZATION_PLAN.md` 的格式：每項都要有
> 驗證方式、退場條件、以及踩過的雷（⚠）。

---

## 0. 前提：先承認 repo 已經做完的事

過去的「微調」其實大多是**不訓練的適配**，而且都還在跑：

| 既有機制 | 位置 | 對唱歌的意義 |
|---|---|---|
| Demucs htdemucs 人聲分離（`🎵 音樂` 情境） | `enhance.py` / `POST /api/enhance/separate` | 已證實是 ALT 最大單一增益來源，且**訓練資料必須沿用同一條路徑** |
| 幻覺閘門（credit 字串 + `no_speech_prob` + RMS） | `main.py:_should_drop_asr_result` | 註解裡已寫明「這是 loud-歌詞 case」——唱歌本來就是幻覺重災區 |
| 慣用詞庫（編輯 diff → `原本→改成` + prompt biasing） | `web/app.js` + `_apply_glossary` | 現成的**人工標註介面**，是資料飛輪的入口 |
| 多語限制 / 自由偵測 / 繁中 prompt / OpenCC | `asr.py` | 訓練標註的正規化規則必須跟它一致，否則後處理互打 |
| 逐字時間戳 → 每字簡譜／基頻 | `main.py:_character_payloads` | 微調不能弄壞 word timestamp，否則簡譜功能連帶壞掉 |
| 模型熱切換 preset（本地 CT2 目錄、缺目錄就報錯不下載） | `asr_models.py` | 微調產物的**唯一上線通道**，不需要新架構 |
| 缺依賴/缺模型一律 no-op、opt-in extra、README 標授權 | 全專案 | NLLB 的 NC 授權處理方式，唱歌資料集要照抄 |

**結論：訓練只從「這些機制都調到極限之後仍然不夠」的地方開始。** 沒有量化的 baseline 就開訓，
等於重演 NLLB 那次「語碼 token 不在 sentencepiece」的教訓——錯了也不知道錯在哪。

### 目標 / 非目標

- **目標**：有伴奏流行歌的歌詞辨識 CER/WER 顯著下降；間奏與純樂器段不吐字；word timing 不退化。
- **非目標**：不做歌聲合成、不做音高預測（音高走既有 DSP，別讓模型碰）、不追求逐字 100%
  （ALT 的 SOTA 在乾淨清唱也常在 WER 20–30%，對外別承諾「精準歌詞」）。

---

## 0.5 目標順序修訂（2026-07-27）

盤點程式之後，微調**降級為 Plan B**。理由是產品的核心輸出是簡譜，而簡譜的正確性卡在
比歌詞錯字更前面的地方：

| 優先 | 目標 | 成本 | 狀態 |
|---|---|---|---|
| 1 | **主音改成調性偵測** — 原本 `tonic = median(f0)`，但旋律中位數通常落在三度或五度，不是主音；簡譜整體平移、冒出大量升降記號 | ~2 天，純 numpy、零依賴 | ✅ 已完成（`audio.py:estimate_tonic`） |
| 2 | **歌詞強制對齊取代歌詞辨識** — 使用者貼上已知歌詞，問題從「認出字」變成「對時間」；幻覺歸零、對齊誤差遠低於自由解碼、**零訓練資料**，且產出的對齊語料正好就是微調要的資料 | ~1–2 週 | 待辦 |
| 3 | **節奏／拍點量化** — 目前只有時長沒有拍點，簡譜缺了節奏就只是音高數字序列 | ~1 週 | 待辦 |
| 4 | **音準評分**（走音段落標示、cents 統計）— 靠 1 的正確主音即可做，零 ASR 依賴 | ~3 天 | 待辦 |
| 5 | 微調（本文件第 1–8 節） | 10 週 | **只有在 1–4 做完仍有明確缺口時才啟動** |

**⚠ #2 的待驗證點**：最理想的實作是重用 ctranslate2 的 cross-attention DTW（faster-whisper
的 word timestamps 就是靠它），零新依賴——但**尚未驗證其 `align` API**，實作前要先確認。
備案：teacher-forcing 走既有 `word_timestamps=True` 路徑逐段對齊；或 ONNX CTC + 純 numpy
CTC-segmentation（與 diarization 同一套 torch-free 做法）。

### 若啟動微調，本計畫要先改四處

1. **順序反過來：先 ZH、後 EN**。原排程 EN 先（因為公開資料多），但使用者是台灣人，EN 線的
   產品價值低 → **EN 線先找現成的公開微調權重跑分，不要自己訓**，自建資料的力氣全投 ZH。
2. **先在 `medium` 上跑通整條 pipeline**（資料 → LoRA → merge → CT2 → preset → 評測）再上
   large-v3。轉檔與接線的坑要在便宜的模型上踩。
3. **評測改用配對比較 + bootstrap 信賴區間**。每層 30–40 段算出的絕對 CER 信賴區間很寬，
   容易把雜訊當成增益。
4. **10 週串行改成 W1–W2 的 kill-switch**：評測集 + 免訓練旋鈕掃描做完後，若增益已達需求，
   整個訓練計畫直接取消。

---

## 1. 階段 0：評測集（W1，**沒有這個就不要往下做**）

### 1.1 資料切分

自建 `eval/` 三層，**全部凍結，永不進訓練集**：

| 分層 | 內容 | 每層目標量 |
|---|---|---|
| A 清唱 | 無伴奏／KTV 導唱關掉 | EN 30 段、ZH 30 段 |
| B 有伴奏 | 原曲混音（走 Demucs 前後各測一次） | EN 40 段、ZH 40 段 |
| C 負樣本 | 純間奏、前奏、純伴奏、環境噪音 | 各 30 段，**標註為空字串** |

再交叉標記維度：男/女聲、慢歌/快歌、中英混唱（Breeze 線必測）、有無假音/氣音。
每段 8–20 秒，對應現行 VAD 切窗尺度。

### 1.2 指標（一次全跑，缺一不可）

1. **WER（EN）／CER（ZH）**：ZH 一律先 OpenCC 轉繁體再比，跟產品輸出對齊。
2. **幻覺率**：C 層輸出非空的比例 + credit 字串命中率（直接量 `_should_drop_asr_result` 前後）。
3. **對齊誤差**：word/char 起訖時間與人工標註的中位絕對誤差（ms）——簡譜的命脈。
4. **RTF**：RTX 3060 上的即時率，跟現行 preset 比。
5. **說話回歸**：一份純說話評測（會議錄音 20 段），防災難性遺忘，**Breeze 線必跑**。

### 1.3 落地

- `training/eval/manifest.jsonl`：`{audio, text, lang, layer, singer_gender, tempo, has_accompaniment}`。
- `training/run_eval.py`：吃 manifest + preset 名稱，輸出上述 5 個數字成 CSV，可重複執行。
- 驗收：`uv run python -m training.run_eval --preset breeze --preset large-v3` 產出 baseline 表。

---

## 2. 階段 1：免訓練的上限（W2）

依序掃描，**每項單獨量測增益**，把最佳組合當成新的 baseline：

| 旋鈕 | 現值 | 要試的 | 預期 |
|---|---|---|---|
| Demucs 分離 | 檔案模式 opt-in | 強制開；比較 `htdemucs` vs `htdemucs_ft` | 已有論文證實不微調也降 WER（見來源） |
| `beam_size` | 串流 1 | 檔案模式 5（ALT 論文的設定） | 檔案模式值得，串流不動 |
| `temperature` fallback / `compression_ratio_threshold` | 未設 | 開啟 fallback | 抑制重複歌詞迴圈 |
| `condition_on_previous_text` | False | 檔案模式試 True | 歌詞有強句法連續性，但幻覺風險↑，需 C 層把關 |
| `initial_prompt` | 繁中 prompt + 詞庫 | 加「以下是歌詞」風格 prompt；已知歌名時餵歌詞片段 | 低成本、可能有效 |
| `no_speech_prob` / RMS 門檻 | 0.6 / 0.02 | 針對分離後人聲重新掃（分離會改變能量分佈） | **現行門檻是對說話調的，唱歌一定要重掃** |
| int8 量化 | CT2 int8 | 比較 int8 vs int8_float16 vs float16 | 唱歌對量化可能比說話敏感 |

⚠ 這一階段常常就吃掉一半的預期增益。若某旋鈕給的增益 > 後續 LoRA，就先把它做成產品設定，
別把成果算在訓練頭上。

---

## 3. 階段 2：資料（W3–W4，最貴、也最決定成敗）

### 3.1 公開資料

| 資料集 | 語言 | 規模 | 授權/取得 | 用途 |
|---|---|---|---|---|
| **DSing** (Sing!300x30x2) | EN | ~149 h、4.3k 首、3205 位歌手、KTV 清唱 | 需向 Smule 申請 DAMP 授權 | EN 線主力訓練 |
| **DALI v2** | 多語（EN 佔 8 成） | 7756 首、弱對齊 | 標註 CC，**音訊要自行取得**（版權風險） | EN 線補量，弱標註 |
| **Jamendo Lyrics** | EN 等 | 小 | CC 授權音樂 | **可公開發表的評測集** |
| **MPop600** | 中文（台灣製作，2 男 2 女、600 首） | 詞/譜/音對齊、清唱 | 學術用途，需洽作者 | ZH 線的對齊種子 |
| MIR-1K / Opencpop / M4Singer | 中文（多為簡體/普通話） | 中小 | 多為 **NC（非商用）** | 僅作研究對照，**不進商用訓練集** |

⚠ **授權紅線**（照 NLLB 那次的處理方式）：NC 資料集只能進「研究用 preset」，README 明標、
永不預設啟用；商用線只用 Apache/CC-BY/自建資料。DALI 的音訊自行抓取屬版權灰區，僅本地實驗。

### 3.2 自建資料（**台灣中文歌只能靠這條**）

公開中文歌唱資料幾乎都是簡體語境的合成用清唱，跟「台灣流行歌 + 有伴奏 + 中英混唱」的
目標分佈差很遠。流程：

```
原曲/自錄 → Demucs 分離人聲 → 既有 VAD 切句 → 現行 preset 產生弱標註
        → 已知歌詞做 forced alignment（whisper word timestamps + DTW，或 CTC aligner）
        → 人工在既有「編輯逐字稿」介面修正 → 匯出 manifest
```

- 目標量：**EN 20–50 h 弱標註 + 5 h 人工校正**；**ZH 10–20 h 弱標註 + 3–5 h 人工校正**。
  LoRA 在 5–20 h 高品質資料就會有效，別一開始追 100 h。
- 人工校正優先投在：中英混唱、假音/氣音、快嘴段、台灣特有用詞與人名。
- 標註正規化（**先寫死成規則檔，否則訓練標籤和產品輸出永遠對不齊**）：
  一律繁體（OpenCC 轉換後再人校）、標點規則比照現行輸出、英文詞保留原拼寫大小寫、
  數字寫法統一、`（間奏）` 之類的非歌詞標記一律改成空字串進 C 層。

### 3.3 資料增強（決定魯棒性）

- **訓練資料要是「Demucs 分離後的人聲」而非乾淨清唱** —— 推論時吃的就是分離殘留 artifact 的音訊，
  訓練分佈必須一致。乾淨清唱訓出來的模型在產品裡會退化。
- 對清唱資料：混入伴奏（SNR −5/0/+5/+10 dB）→ 再過 Demucs → 得到「有 artifact 的人聲」。
- pitch shift ±2 半音、time stretch ±10%、輕度 reverb；**不要**做會破壞 word timing 的增強。
- **負樣本佔比 5–10%**：純伴奏/間奏 → 目標文字為空。這是直接對著現行幻覺痛點訓練。

---

## 4. 階段 3：訓練（W5–W8）

環境：**獨立 venv／甚至租 GPU**，絕不與推論環境共用（既有 torch/cuDNN DLL 衝突的教訓）。
訓練用 HF `transformers` + `peft`；產品側維持 CT2、torch-free。

### 4.1 EN 線（Whisper）

- 底模：`openai/whisper-large-v3`（MIT）。若要對齊產品的 medium preset，另訓一份 medium 作低配。
- 方法：**LoRA**（r=16–32，target `q_proj`/`v_proj`，可加 `k_proj`/`out_proj`），bf16 +
  gradient checkpointing + 8-bit optimizer → 3060 12 GB 可跑 large-v3；full fine-tune 不可行。
- 超參起點：lr 1e-3（LoRA）/ warmup 10% / 2–4 epoch / effective batch 32（grad accum 補）/
  SpecAugment 開 / eval 每 500 步 / 以 dev WER early stop。
- **不要凍結 encoder**：唱歌 vs 說話的差異主要在聲學（音高範圍、母音延長、顫音），
  encoder 才是要動的地方；decoder 動的是歌詞句法。若顯存吃緊，寧可降 r 也別凍 encoder。

### 4.2 ZH 線（Breeze-ASR-25）

- 底模：`MediaTek-Research/Breeze-ASR-25`（1.54 B、whisper-large-v2 微調、**Apache-2.0**、zh/en 中英混講）。
  授權對商用友善，這是選它當底的關鍵理由之一。
- 它**已經是一次微調的產物**，再訓最大風險是**災難性遺忘**（把說話能力訓壞）：
  - lr 比 EN 線低一階（LoRA 2e-4 起跳）、epoch 更少（1–3）。
  - **Replay**：訓練集混入 15–25% 的一般台灣中文說話資料（可用自己的既有逐字稿），
    每次 eval 必跑階段 0 的「說話回歸」項，退步超過相對 5% 就回滾。
  - 中英混唱樣本要刻意加權——這是 Breeze 相對通用 Whisper 的優勢，別在唱歌微調中弄丟。
- 兩階段可選：先只訓 encoder LoRA（聲學適配）→ 再解凍 decoder LoRA（歌詞語言）；
  分兩段可看出增益來自哪一半。

### 4.3 產出與轉檔

```bash
# 1) merge LoRA 回底模
python -m training.merge_lora --base <base> --adapter <ckpt> --out models/hf/<name>
# 2) 轉 CT2（與 breeze / NLLB 同一套做法，離線優先）
ct2-transformers-converter --model models/hf/<name> \
  --output_dir models/<name>-ct2 --quantization int8_float16
```

⚠ **轉檔後必跑 round-trip 一致性測試**（HF 版與 CT2 版在同 20 段音訊上的輸出差異），
理由同 NLLB 的 flores 語碼事件：轉檔階段的靜默錯誤最難查。量化前後的 WER 也要各量一次。

---

## 5. 階段 4：接回產品（W9）

改動面刻意壓到最小，完全走既有機制：

- `asr_models.py`：新增 preset `sing-zh`（`models/breeze-asr-25-sing-ct2`）、`sing-en`
  （`models/whisper-sing-en-ct2`），沿用熱切換與「缺目錄就明確報錯、不偷連 HF」。
- `config.py`：`BREEZE_ASR_SING_ZH_MODEL` / `BREEZE_ASR_SING_EN_MODEL`（比照 `asr_breeze_model`）。
- `web/app.js`：`🎵 音樂` 情境時，把歌唱 preset 排在切換清單前面（或提示可切換）；改 app.js 記得
  bump `?v=` + service worker cache。
- `doctor.py`：模型目錄存在性檢查加進去。
- `README.md`：新增章節 + 設定表 + **資料集授權聲明**（NC 資料訓出的權重只能研究用）。
- `tests/`：manifest schema 測試、preset 清單測試、CT2 round-trip 測試。

---

## 6. 上線門檻（達不到就只當 opt-in preset，不改預設）

| 條件 | 門檻 |
|---|---|
| 目標分層（B 有伴奏）CER/WER | 相對階段 1 baseline **降 ≥ 15%** |
| C 層幻覺率 | **不上升**（理想：下降） |
| word/char timing 中位誤差 | 不劣於 baseline（簡譜不能壞） |
| 說話回歸（Breeze 線） | 相對退步 **< 5%** |
| RTF on 3060 | ≤ baseline × 1.2 |

A/B 方式：直接用現成的模型切換器讓實際使用比較，不另做框架。

---

## 7. 時程

| 週 | 產出 |
|---|---|
| W1 | 評測集 + `run_eval.py` + baseline 數字 |
| W2 | 免訓練旋鈕掃描完成，新 baseline 定案 |
| W3–W4 | 資料流水線 + 首批人工校正（EN 5 h / ZH 3 h） |
| W5–W6 | EN LoRA 第一輪 + 評測 |
| W7–W8 | ZH LoRA（含 replay）+ 評測 |
| W9 | 轉 CT2、接進 preset、回歸測試、README |
| W10 | 依評測缺口補資料再訓一輪，決定是否升為預設 |

---

## 8. 風險與退場

1. **授權** — NC 資料集、DALI 音訊、原曲版權。→ 商用線只用 Apache/CC-BY/自建；權重分「研究/商用」兩包。
2. **災難性遺忘** — Breeze 線最大風險。→ replay + 說話回歸為硬性 gate，可隨時回滾 preset。
3. **幻覺換了形狀** — 訓練後可能不再吐 credit 字串，改吐「像歌詞的胡話」，現行文字比對閘門會失效。
   → C 層負樣本是主要防線，`_is_credit_hallucination` 的字串表需重新盤點。
4. **環境衝突** — 訓練引入 torch/CUDA 版本，可能污染推論環境。→ 訓練環境完全隔離。
5. **投入產出不成比例** — 若階段 1 就吃掉大部分增益，**應該停在階段 1**，把時間投到簡譜/對齊體驗。
   ALT 的天花板本來就低，產品價值在「音高 + 大致歌詞」而非逐字精準。
6. **台語歌** — 需另一條線（漢字 vs 台羅的標註系統要先決定、資料另尋），本計畫不含，別混進中文線訓練。

---

## 9. 尚待決定

1. EN 底模用 `large-v3` 還是對齊產品的 `medium`（或兩者都訓）。
2. DSing 是否申請 DAMP 授權（決定 EN 線是靠公開資料還是全自建）。
3. ZH 人工校正的時數上限（3 h vs 10 h，直接決定 W7–W8 的成敗）。
4. 訓練硬體：本機 3060 12 GB（慢但免費）vs 租 A100（快、可跑 full FT）。
5. 是否把「歌詞 forced alignment」也做成產品功能（使用者貼歌詞 → 自動對時間軸），
   它同時是資料工具與產品功能，投報率可能高於微調本身。

---

## 參考來源

- [Exploiting Music Source Separation for Automatic Lyrics Transcription with Whisper (arXiv 2506.15514)](https://arxiv.org/abs/2506.15514) — 分離人聲在**不微調**下即可降 WER；使用 faster-whisper large-v2 + beam size 5。
- [LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription (arXiv 2306.17103)](https://arxiv.org/html/2306.17103v4)
- [PDAugment: Data Augmentation by Pitch and Duration Adjustments for ALT (arXiv 2109.07940)](https://arxiv.org/pdf/2109.07940)
- [VietLyrics: A Large-Scale Dataset and Models for Vietnamese ALT (arXiv 2510.22295)](https://arxiv.org/html/2510.22295) — Whisper-large-v2 微調後 WER 24.61%（大小寫敏感）的參考量級。
- [MPop600: A Mandarin Popular Song Database with Aligned Audio, Lyrics, and Musical Scores](http://www.apsipa.org/proceedings/2020/pdfs/0001647.pdf)
- [MediaTek-Research/Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) — whisper-large-v2 微調、1.54 B、Apache-2.0、zh/en。
