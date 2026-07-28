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
| 2 | **歌詞強制對齊取代歌詞辨識** — 使用者貼上已知歌詞，問題從「認出字」變成「對時間」；幻覺歸零、**零訓練資料**，且產出的對齊語料正好就是微調要的資料 | ~1–2 週 | ✅ 已完成（`lyrics.py` + `POST /api/transcript/lyrics` + 前端「歌詞對齊」） |
| 3 | **節奏／拍點量化** — 目前只有時長沒有拍點，簡譜缺了節奏就只是音高數字序列 | ~1 週 | ✅ 已完成（`audio.py:estimate_tempo` + `quantize_beats`，後處理回報 BPM 與每字音符時值） |
| 4 | **音準評分**（走音段落標示、cents 統計）— 靠 1 的正確主音即可做，零 ASR 依賴 | ~3 天 | ✅ 已完成（`audio.py:score_intonation`，整體/逐句分數 + 每字 準／略偏／走音） |
| 5 | 微調（本文件第 1–8 節） | 10 週 | **1–4 已全部完成 → 現在才輪到它，但先跑第 1 節的評測集** |

**#2 的實作決定（2026-07-27）**：ctranslate2 4.7.2 的 `Whisper.align` 確認存在、faster-whisper
也已公開 `find_alignment`，但**第一版沒有用它**——既有的 word timestamps 已經是同一套
cross-attention DTW 的產物，所以先做「把已知歌詞對到既有時間軸」的純文字對齊：零模型、
零下載、可完全單元測試，而且辨識全錯的字也能救回來。改用 `align` 直接逐 token 強制對齊
是**下一步的精修**，值得做的地方只有一個：辨識整句漏掉時，目前只能在缺口內平均內插
（回傳的 `anchoredRatio` 會標出來），強制對齊則能量出真正的時間。

對齊用的是最佳編輯距離（Needleman-Wunsch，numpy 逐列向量化），不是 `difflib`：difflib 最大化
最長共同子序列，遇到重複字會配錯——`小星星` 裡把 `星` 聽成 `心`，它會拿第二個 `星` 去配第一個，
之後每個音節都早一拍、最後一個字落到音檔之外。歌詞裡重複字到處都是（一閃一閃、慢慢、星星）。

**1–4 全部完成（2026-07-27）**。簡譜這條線現在是：正確的調性 → 正確的歌詞與時間軸 →
音符時值 → 音準分數。下一步照第 1 節做評測集，量出「免訓練的上限」之後，再決定微調要不要啟動。

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

原規劃另開 `training/` 目錄，實作時併回既有的 `tools/`——manifest 與報告格式都已經在那裡，
再開一套只會讓兩邊的正規化規則各走各的（這正是 §P0.3 的教訓）。

- `dataset/manifests/{train,dev,test}.jsonl`：`{id, audio, text, lang, source, split, layer}`。
  `layer` 是 `lyric` / `negative`；人工標好時間軸的 clip 再加一個 `words`
  （`[{word, start, end}]`），對齊誤差就會把那些 clip 算進去。
- `tools/eval_asr.py`：吃 manifest + 模型路徑，**一次輸出五個數字**成
  `dataset/eval_reports/<tag>.json`，並列出這一輪沒量到的指標與原因。
  純函式（CER/MER、對齊誤差、幻覺率、RTF）就在 `tools/eval_asr.py` 內，
  不需要 GPU / 模型即可單元測試（`tests/test_eval_asr.py`）。
- 幻覺率量的是 `breeze_elf/hallucination.py` 這個**產品實際在跑的閘門**（`main.py`
  只是把 window 能量與 settings 綁上去），不是評測工具自己複寫的一份。
- ✅ 已完成（2026-07-28）：五個指標的量測程式與 C 層負樣本管線。**實際數字還沒跑**——
  需要 GPU 機器、`dataset/`、以及 `speech_` 前綴的 20 段會議錄音。
- 驗收：`.venv/Scripts/python.exe tools/eval_asr.py --model models/breeze-asr-25-ct2
  --sources nan,mir1k,negative,speech --tag breeze_baseline` 產出 baseline 表。

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
  ✅ 已實作（2026-07-28）：`dataset_builder.instrumental_chunks` 把歌詞句之間的空檔
  （前奏／間奏／尾奏）切出來、以空字串匯出，比例由 `--negative_ratio`（預設 0.08）控制，
  只在**有伴奏**的錄音上做——清唱與朗讀的空檔是靜音，而靜音早就被 RMS 閘門擋掉了，
  訓練分佈缺的是**大聲的非語音**。太短的空檔（`--min_negative`，預設 3 秒）是換氣不是間奏，
  會被丟掉；靜音的空檔也會被 RMS 門檻擋下。
  `tools/make_manifests.py` 不再把空文字當 `dropped["empty_text"]`，而是標成
  `layer=negative` 收進 manifest，並在報告裡印出每個 split 的負樣本比例。

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

---

# 附錄:台語/歌唱 ASR LoRA 後訓練執行紀錄(2026-07-22 ~ 07-23)

> 上節是「唱歌辨識(ALT)」的 go-forward 策略(EN 歌 / ZH 歌,微調降為 Plan B);本附錄是另一條
> **已執行完成**的台語(nan)朗讀 + zh-TW 歌唱 LoRA 後訓練紀錄,含 P0–P3 實際成績與 v2 改良,
> 可作為上節「轉檔 / 接線 / 評測 / 熱切換」步驟的實作先例。

> 給執行 agent 的自足文件:讀完本檔即可動工,不需回溯對話。
> 建立:2026-07-22。資料管線見 `build_lyric_dataset.py` / `breeze_elf/dataset_builder.py`。

## 目標

1. **A 線(主力)**:以 LoRA 後訓練 **Breeze-ASR-25**,補強台語(nan)辨識 + zh-TW 歌唱域適應。
2. **B 線**:以 LoRA 後訓練 **whisper-medium**,做英文歌唱域適應。
3. 產出 CT2 模型接回 breeze_elf 既有模型熱切換機制做 A/B。

**範圍外**:ja/ko(無資料)、台語「歌唱」本體(現有 nan 資料是朗讀語音;台語歌等 YouTube+LRC 素材第二輪)。

## 硬約束(先讀,違反必炸)

- **GPU:RTX 3060 12GB** → large-v2 只能 LoRA(fp16 + gradient checkpointing + 8-bit optimizer,batch 1–2 × grad-accum 16,LoRA r=16 掛 q_proj/v_proj)。OOM 退路:r=8 → batch 1 → 凍結 encoder 只 LoRA decoder。
- **一律 `uv run` / `uv pip install`,禁止 `uv sync`**(會把手動裝的 CUDA torch 2.6.0+cu124 換成 CPU 版)。
- 系統 Python 無 torch;所有指令在 repo 根目錄用 `.venv`。
- 本機 `models/breeze-asr-25-ct2` 是 **CTranslate2 推論格式,不可訓練**;訓練要另抓 HF 原始權重 `MediaTek-Research/Breeze-ASR-25`(~3GB,HF 帳號 winniexchang 已登入)。
- Whisper 沒有 nan 語言 token → nan 資料一律用 `<|zh|>` + task=transcribe,文字目標照 metadata 現狀(教育部漢字為主,純台羅句照舊)。

## 資料現況

`dataset/metadata.csv`(HF audiofolder:`file_name, transcription, duration, language, source_dataset_or_song_id`),chunks 皆 16kHz mono PCM16 wav,共 31,454 筆 / 24.6 h:

| 來源 | id 前綴 | 語言 | 量 | 性質 |
|---|---|---|---|---|
| Common Voice nan-tw validated | `cv_` | nan | 29,608 句 / 21.6 h | 朗讀語音 |
| MIR-1K | `mir1k_` | zh-TW | 1,000 clips / 2.2 h | 業餘歌聲(乾人聲=原檔右聲道) |
| JamendoLyrics | `jamendo_` | en | 846 chunks / 0.8 h | htdemucs_ft 分離人聲 |

增強用原始素材(都在本機):
- `dataset/MIR-1K/Wavfile/*.wav`:左聲道=伴奏(摻回用)、右聲道=人聲。
- `dataset/_cache/jamendolyrics/mp3/`:原始混音(與分離人聲同 chunk 時間戳可對切)。
- `dataset/nan-tw/`:CV 官方 tsv(train/dev/test 說話人不相交)。

## 切分規則(防洩漏,不可用隨機 chunk 級切分)

| 來源 | 規則 | 具體做法 |
|---|---|---|
| nan | 沿用 CV 官方分割 | 以 `dataset/nan-tw/{train,dev,test}.tsv` 的 path stem 對映 `cv_<stem>` |
| MIR-1K | 按歌手 | 檔名前綴(`abjones`, `amy`, …共 19 位):17 訓 / 1 dev / 1 test |
| Jamendo | 按歌 | 20 首:16 訓 / 2 dev / 2 test |

## 執行步驟

### P0 — 環境與基線(先做,半天)

1. `uv pip install transformers peft accelerate jiwer bitsandbytes soundfile`(留意:**不要** `uv sync`)。
2. 產生 train/dev/test manifest(建議寫 `tools/make_manifests.py` 或放 `breeze_elf/` 下,輸出 jsonl:`{audio, text, lang, source}`)。
3. **Zero-shot 基線**(沒有基線不准開訓):
   - Breeze-ASR-25(可先用現成 CT2 + faster-whisper 跑,快)對 nan test、MIR-1K test → CER。
   - whisper-medium 對 Jamendo test → WER。
   - 評估前文字正規化:去標點、全半形統一、去空白(CJK)再算分;混台羅句用 MER。

### P1 — Breeze 聯合 LoRA(過夜,約 10–15 h)

- 訓練集:nan 語音 + MIR-1K 歌唱**混訓**,取樣比 ~10:1(單一 adapter,不拆兩個;若 dev 顯示歌唱把語音拉退步再拆)。
- 資料增強(歌唱側):隨機 SNR 0–15dB 摻回伴奏、pitch ±2 semitone、tempo 0.9–1.1、SpecAugment。
- 超參起點:lr 1e-4(LoRA)、2–3 epochs、warmup 500 steps、以 dev CER 早停。
- 每 epoch 存 checkpoint;訓後跑 zh-TW 保留驗證(任意國語錄音對照原模型,確認無明顯遺忘)。

### P2 — Whisper-medium en LoRA(1–2 h)

- 同樣增強;46 分鐘資料,期望值放低(相對 WER 降 ≥15% 即達標),早停防過擬合。

### P3 — 評估與部署(半天)

1. test 集評分,與 P0 基線對照。驗收線:nan CER 相對降 ≥30%;MIR-1K / Jamendo 相對降 ≥15%。
2. merge LoRA → `ct2-transformers-converter --model <merged_dir> --output_dir models/breeze-asr-25-nan-ct2 --quantization float16`。
3. 接回 app:走 `breeze_elf/asr_models.py` 的模型熱切換(breeze/whisper 切換器)註冊新路徑做 A/B;**不要**直接覆蓋 `models/breeze-asr-25-ct2`。

## 改善 v2(2026-07-23,誤差解剖後)

誤差解剖(`tools/rescore.py` 從既有 preds 重算,免 GPU)推翻「殘留來自吐空」的猜測,找到兩大根因並對症:

1. **指標/標籤 bug**:CV nan ref 含台羅發音註解 `漢字(Kong-kuán|…)`,佔訓練標籤字元 **48%**、且把「完全正確」判成大量刪除。`tools/text_norm.py::strip_reference_gloss`,`eval_asr.py`(僅 ref)+ `make_manifests.py`(訓練標籤)都套用。修正後 **v1 真實成績 = nan CER 0.4806(對 base 1.2149 = −60.4%)**,非原報 −39.7%。
2. **CV 官方切分病態**:274 說話人裡 256 在 test、13 在 dev、**train 只剩 5 人**。改成 test 不動、非 test 全收進 train(留 3 中型當 dev)→ nan train 15 人 21654 句。
3. **Utterance packing**(`tools/pack_manifest.py`):同 speaker 串 ~25s 窗補「短句 CER 高」弱點;`train_lora.py --manifest train_packed`、nan 用 resample speed-perturb(模擬 speaker)、Ampere 自動 bf16。

**v2 配方**:`--r 32 --lora_targets q,k,v,out_proj --batch 2 --grad_accum 16 --epochs 3 --warmup 80`(batch 4 OOM;warmup 要 ~10% steps)。7h,eval_loss 0.747→0.612。

**v2 結果(去註解計分)**:

| test | base | v1 | **v2** |
|---|---|---|---|
| nan CER | 1.2149 | 0.4806 | **0.4356**(−64.1% vs base、−9.4% vs v1) |
| MIR-1K CER | 0.0646 | 0.0620 | **0.0620**(零退步) |

部署:`tools/deploy_model.py --id breeze-nan-v2 --ct2-dir models/breeze-asr-25-nan-v2-ct2`(→ `models/presets.json`)。switcher 現有 原始 / v1 / v2 三檔。

## 已知風險 / 陷阱

- bitsandbytes 在 Windows 需 ≥0.43;若裝不起來,8-bit optimizer 改用 `adafactor` 也可(稍慢)。
- MIR-1K 歌詞無標點、Breeze 輸出有標點 → 不做評估正規化會高估 CER。
- Jamendo 45 分鐘英文資料是 B 線天花板;結果不佳是資料問題,別在超參上耗。
- 訓練腳本內載入音檔請直接讀 `dataset/chunks/*.wav`(已 16k mono),不要重跑分離。
- 磁碟:HF 權重 3GB + merged 3GB + CT2 1.5GB,開工前確認空間。

## 待辦擴充(非本輪,但影響設計)

- CV zh-TW(Mozilla Data Collective 下載後 `--dataset_name common_voice --input <dir> --target_lang zh-TW` 即可入庫)→ 混 5% 防遺忘。
- 台語歌 YouTube+LRC(`--source_type youtube --lyrics *.lrc --target_lang nan`)→ 第二輪台語歌唱 LoRA。
- TAT-Vol1 授權核可後入庫。

## 進度追蹤

- [x] P0 依賴安裝 — `tokenizers` 釘 0.22.2;訓練/評估用 `.venv/Scripts/python.exe` 直呼(**勿** `uv run`,會 sync 回 0.23.1 炸 transformers)。
- [x] P0 manifests — `tools/make_manifests.py` → `dataset/manifests/{train,dev,test}.jsonl`(train 17697 / dev 6121 / test 6594;nan 防洩漏丟 1042)。
- [x] P0 zero-shot 基線 — `tools/eval_asr.py`,報告在 `dataset/eval_reports/`:
  - **Breeze nan CER = 1.0963**(n=6430;台語幾乎全錯)→ 驗收線 ≤0.767。
  - **Breeze MIR-1K CER = 0.0646**(n=65;已很好,混訓勿退步)→ 驗收線 ≤0.0549。
  - **whisper-medium Jamendo WER = 0.788**(n=99)→ 驗收線 ≤0.670。
- [x] P1 Breeze 聯合 LoRA 訓練 — `tools/train_lora.py`(8-bit 底+LoRA r16 q/v+grad-ckpt+adamw_bnb_8bit,batch4×ga8 eff32、2 epoch、lr1e-4/warmup500、nan+MIR 10:1、MIR 伴奏摻回+pitch/tempo+SpecAugment)。9.6h,eval_loss 0.942→0.843,adapter `models/lora/breeze-nan/adapter_best`。
- [x] P1 zh-TW 防遺忘驗證 — 用 MIR-1K zh-TW 當保留集:CER 0.0646→**0.0620**(微升,無退步)→ 無明顯遺忘,聯合 adapter 不需拆。
- [x] P2 whisper-medium en LoRA — `train_lora.py --model_id openai/whisper-medium --language english --sources jamendo`(batch8×ga2、6 epoch、early-stop 取 epoch3)。dev eval_loss 1.60→0.724,但 **test WER 0.788→0.798(持平/略退,未達 −15%)**:16 首訓練歌過擬、無法泛化到 held-out 歌 —— 符合計畫「46 分鐘是 B 線天花板、結果不佳是資料問題」的預期。**不部署**(不優於 app 既有 whisper-medium);adapter 與 `whisper_lora.json` 留檔,未註冊 preset。第二輪需更多英文歌唱資料。
- [x] P3 test 評分對照 — `tools/eval_asr.py`,`dataset/eval_reports/breeze_lora.json`:**nan CER 1.0963→0.6614(相對降 39.7% ✓≥30%)**;MIR-1K 0.0646→0.0620(降 4.0%,未達 15% 但因基線已 6.5% 近天花板且無退步)。
- [x] P3 CT2 轉換 — `tools/merge_and_convert.py` → `models/breeze-asr-25-nan-ct2`(float16,faster-whisper 實測可載)。
- [x] P3 app A/B 接入 — `config.py asr_breeze_nan_model`(env `BREEZE_ASR_BREEZE_NAN_MODEL`)+ `asr_models.py` dir-gated preset「Breeze ASR 台語強化」;預設仍原 breeze,新模型 opt-in;`tests/test_asr_models.py` 已更新(12/12 過)。**未覆蓋原 `models/breeze-asr-25-ct2`**。
- [x] P3 熱切換修正與實機 A/B(2026-07-23)— `_run_asr_switch` 原本對所有 breeze-kind preset 都解析 `settings.asr_breeze_model`,切到 `breeze-nan` 會**載入原版模型卻標示 nan 路徑**(假 A/B)。`resolve_breeze_model_dir(settings, model)` 加路徑參數、switch 傳 `option.model`;回歸測試 `test_switch_to_lora_preset_loads_its_own_dir`。實機同音檔對照(CV nan test):原版「外地之前要說事情做什麼」vs LoRA「話底真情愛講代誌做啥」(ref「月底進前愛共代誌做煞」)。前端動態渲染 preset,手機端不需改版;但因本輪走的是 `BREEZE_ASR_BREEZE_NAN_MODEL` **env-var 閘控**的內建 preset(process 啟動時讀取),**手機連的 server process 需重啟**新 preset 才會出現。改走 `tools/deploy_model.py`→`presets.json` 的動態註冊路徑則每 request 重讀、免重啟(見 README「Deploying a post-trained model」)。
