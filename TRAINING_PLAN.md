# 歌唱/台語 ASR 後訓練計畫(TRAINING_PLAN)

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

- [ ] P0 依賴安裝
- [ ] P0 manifests
- [ ] P0 zero-shot 基線(Breeze nan / Breeze MIR-1K / whisper-medium Jamendo)
- [ ] P1 Breeze 聯合 LoRA 訓練
- [ ] P1 zh-TW 防遺忘驗證
- [ ] P2 whisper-medium en LoRA
- [ ] P3 test 評分對照
- [ ] P3 CT2 轉換
- [ ] P3 app A/B 接入
