# Breeze Elf — Automatic Lyrics Transcription (ALT) training plan

> Goal: English songs on the Whisper line, Taiwanese-Mandarin songs on the Breeze line, each
> fine-tuned into a switchable preset. Every item carries a verification method, an exit
> condition, and the traps already hit (⚠). Cross-cutting principles are in `CLAUDE.md`;
> open work is in `計畫.md`. **Section numbers here are referenced from code comments — do
> not renumber.**

---

## 0. What the repo already does

Most past "fine-tuning" was really training-free adaptation, and all of it is still running:

| Mechanism | Location | Meaning for singing |
|---|---|---|
| Demucs `htdemucs` vocal separation (🎵 music scenario) | `enhance.py` / `POST /api/enhance/separate` | Proven to be the single largest ALT gain, and **training data must go through the same path** |
| Hallucination gate (credit strings + `no_speech_prob` + RMS) | `breeze_elf/hallucination.py` | Singing is a hallucination hotspot by nature |
| Glossary (edit diff → `from→to` + prompt biasing) | `web/app.js` + `_apply_glossary` | A ready-made annotation interface — the entry point of the data flywheel |
| Language restriction / auto-detect / zh prompt / OpenCC | `asr.py` | Training-label normalisation must match it or the post-processing fights itself |
| Word timestamps → per-character jianpu and f0 | `main.py:_character_payloads` | Fine-tuning must not break word timestamps or jianpu breaks with them |
| Model hot-switch presets (local CT2 dir, clear error instead of downloading) | `asr_models.py` | The **only** shipping channel for a fine-tuned model; no new architecture needed |
| No-op on missing deps/models, opt-in extras, licences in README | project-wide | Copy the NLLB NC-licence handling for singing datasets |

**Training starts only where these mechanisms, pushed to their limit, are still not enough.**
Training without a quantified baseline repeats the NLLB lesson (flores language codes absent
from sentencepiece): wrong, with no way to see where.

### Goals / non-goals

- **Goal:** clearly lower CER/WER on accompanied pop songs; no output over interludes and
  instrumental passages; no word-timing regression.
- **Non-goal:** no singing synthesis, no pitch prediction (pitch stays with the existing DSP
  — do not let the model touch it), no chase for perfect transcription. ALT state of the art
  is often 20–30 % WER even on clean a cappella; never promise "accurate lyrics".

---

## 0.5 Priority revision (2026-07-27): fine-tuning is Plan B

The product's core output is the jianpu score, and its correctness is bottlenecked well
before lyric character errors. Items 1–4 are all complete, so fine-tuning's turn has now
arrived — but only after §1's evaluation set.

| # | Goal | Status |
|---|---|---|
| 1 | **Key detection as the tonic** — `tonic = median(f0)` usually lands on the third or fifth, shifting the whole score and spraying accidentals | ✅ `audio.py:estimate_tonic` |
| 2 | **Forced lyric alignment instead of lyric recognition** — the user pastes known lyrics, turning "recognise the characters" into "align the time"; hallucination goes to zero, needs zero training data, and the aligned output *is* the fine-tuning corpus | ✅ `lyrics.py` + `POST /api/transcript/lyrics` |
| 3 | **Beat/rhythm quantisation** — without beats a jianpu score is just a sequence of pitch digits | ✅ `audio.py:estimate_tempo` + `quantize_beats` |
| 4 | **Intonation scoring** (off-key passages, cents statistics) — needs only a correct tonic, zero ASR dependency | ✅ `audio.py:score_intonation` |
| 5 | Fine-tuning (§1–§8 below) | now in scope |

**On #2's implementation:** ctranslate2 4.7.2 does expose `Whisper.align` and faster-whisper
exposes `find_alignment`, but the first version deliberately does not use them — the existing
word timestamps come from the same cross-attention DTW, so a pure-text alignment of known
lyrics onto the existing timeline is zero-model, zero-download, fully unit-testable, and
rescues characters that were transcribed entirely wrong. Per-token forced alignment is the
next refinement, worth it for exactly one case: when a whole line is missed, the current code
can only interpolate evenly across the gap (`anchoredRatio` marks it).

Alignment uses optimal edit distance (Needleman-Wunsch, row-vectorised numpy), **not
`difflib`**: difflib maximises the longest common subsequence and mis-pairs repeated
characters — hear the 星 in 小星星 as 心 and it matches the second 星 to the first, after
which every syllable is early and the last falls outside the audio. Repeated characters are
everywhere in lyrics.

### Four changes to make before starting fine-tuning

1. **Reverse the order: ZH first, EN second.** The original schedule ran EN first because
   public data is plentiful, but the user is in Taiwan and the EN line has low product value
   → **find existing public fine-tuned weights for EN and benchmark them rather than training
   them**; spend the self-built-data effort on ZH.
2. **Get the whole pipeline working on `medium` first** (data → LoRA → merge → CT2 → preset →
   evaluation) before large-v3. Hit the conversion and wiring traps on the cheap model.
3. **Evaluate with paired comparison + bootstrap confidence intervals.** Absolute CER from
   30–40 clips per layer has wide intervals and turns noise into apparent gains.
4. **Replace the serial 10-week plan with a W1–W2 kill switch.** If the evaluation set and
   the training-free sweep already reach the target, cancel the training plan outright.

---

## 1. Stage 0: the evaluation set (W1 — do not proceed without it)

### 1.1 Splits

A self-built three-layer `eval/`, **frozen, never entering training**:

| Layer | Content | Target size |
|---|---|---|
| A a cappella | no accompaniment / karaoke guide vocal off | EN 30, ZH 30 |
| B accompanied | original mix (measured before and after Demucs) | EN 40, ZH 40 |
| C negatives | interludes, intros, backing only, ambient noise | 30 each, **labelled as empty string** |

Cross-tag by dimension: male/female, slow/fast, mixed zh-en singing (mandatory for the Breeze
line), falsetto/breathy or not. Clips of 8–20 s, matching the current VAD window scale.

### 1.2 Metrics (all in one pass, none optional)

1. **WER (EN) / CER (ZH)** — ZH always converted to Traditional with OpenCC first, matching
   the product output.
2. **Hallucination rate** — the fraction of layer C outputs that are non-empty, plus the
   credit-string hit rate, measured before and after the gate.
3. **Alignment error** — median absolute error in ms of word/character start and end times
   against hand annotation. This is the lifeline of the jianpu feature.
4. **RTF** — realtime factor on the RTX 3060, compared against the current preset.
5. **Speech regression** (§1.2.5) — a pure-speech evaluation set (20 meeting recordings) to
   catch catastrophic forgetting. **Mandatory for the Breeze line.**

### 1.3 Implementation

The original plan called for a separate `training/` directory; in practice everything went
into the existing `tools/`, because the manifest and report formats already live there and a
second set would let the two normalisation paths drift apart (the §P0.3 lesson).

- `dataset/manifests/{train,dev,test}.jsonl`: `{id, audio, text, lang, source, split, layer}`
  where `layer` is `lyric` or `negative`. Clips with hand-annotated timing add
  `words: [{word, start, end}]` and are then counted in the alignment error.
- `tools/eval_asr.py` takes a manifest and a model path and **emits all five numbers at once**
  into `dataset/eval_reports/<tag>.json`, listing which metrics it could not measure and why.
  The pure functions (CER/MER, alignment error, hallucination rate, RTF) live in the same file
  and are unit-testable without a GPU or a model (`tests/test_eval_asr.py`).
- Hallucination rate measures `breeze_elf/hallucination.py`, **the gate the product actually
  runs** (`main.py` only binds window energy and settings to it) — not a copy inside the tool.
- ✅ Done 2026-07-28: the harness and the C-layer negative pipeline. **Three of five metrics
  have real numbers** (see `dataset/eval_reports/breeze_baseline_5m.json`); hallucination
  rate, alignment error and speech regression still need `neg_` audio, `align_gold.jsonl`, and
  the `speech_`-prefixed recordings.
- ⚠ `jiwer.mer` degenerates to whole-utterance exact match on whitespace-stripped CJK — use
  `jiwer.process_characters`. Both `eval_asr.py` and `tools/rescore.py` are fixed.

---

## 2. Stage 1: the training-free ceiling (W2)

Sweep in order, **measuring each knob's gain separately**, and adopt the best combination as
the new baseline. Results below are measured on mir1k + jamendo with the baseline model.

| Knob | Was | Tried | Result |
|---|---|---|---|
| `beam_size` | 1 (streaming) | 5 in file mode | ✅ **The only real gain**: mir1k CER 0.0646→0.0543 (−16 %), jamendo WER 0.778→0.768, 2–3× RTF but still 5–13× realtime. Shipped as `BREEZE_ASR_FILE_BEAM` (file mode only) |
| `temperature` fallback / `compression_ratio_threshold` | unset | enable fallback | No movement at all on clean sung clips |
| `condition_on_previous_text` | False | True in file mode | No movement |
| `initial_prompt` | zh prompt + glossary | lyrics-style prompt | Helps Mandarin (−0.004), hurts English (+0.04) — language-bound, not a default |
| `vad_filter` | off | on | **Shreds singing** (jamendo WER +0.13). Confirms the thresholds are speech-tuned. Stays off |
| int8 quantisation | CT2 int8 | int8_float16 vs float16 | Accuracy loss not worth the gain |
| Demucs separation | file-mode opt-in | force on; `htdemucs` vs `htdemucs_ft` | **Not yet swept** (needs a `transcribe_one` change) |
| `no_speech_prob` / RMS thresholds | 0.6 / 0.02 | re-sweep against separated vocals | **Not yet swept** — the current values are speech-tuned, and separation changes the energy distribution |

⚠ beam 5 was validated on single-pass `WhisperModel.transcribe`, but the file endpoint uses
`BatchedInferencePipeline` with its own internal VAD — and VAD is what hurt singing. The batch-
path A/B has not been run.

⚠ This stage often takes half the expected gain. If a knob beats the later LoRA, make it a
product setting and do not credit the result to training.

---

## 3. Stage 2: data (W3–W4 — the most expensive part, and the one that decides the outcome)

### 3.1 Public datasets

| Dataset | Lang | Size | Licence / access | Use |
|---|---|---|---|---|
| **DSing** (Sing!300x30x2) | EN | ~149 h, 4.3k songs, 3205 singers, karaoke a cappella | Requires a DAMP licence from Smule | EN main training set |
| **DALI v2** | multi (~80 % EN) | 7756 songs, weak alignment | Annotations CC, **audio must be sourced yourself** (copyright risk) | EN volume, weak labels |
| **Jamendo Lyrics** | EN etc. | small | CC-licensed music | **A publishable evaluation set** |
| **MPop600** | zh (Taiwan, 2M/2F, 600 songs) | lyric/score/audio aligned, a cappella | Academic use, contact the authors | ZH alignment seed |
| MIR-1K / Opencpop / M4Singer | zh (mostly simplified/Mandarin) | small–medium | mostly **NC** | Research comparison only, **never in a commercial training set** |

⚠ **Licence red line** (handled the same way as NLLB): NC datasets may only feed a "research"
preset, are labelled in README, and are never enabled by default. Self-sourced DALI audio is a
copyright grey area — local experiments only.

Availability as verified 2026-07: Common Voice left HF in 2025-10 (Mozilla Data Collective
only; feed the extracted tar with `--input`); mirlab.org is down so MIR-1K comes from Kaggle
(`elemento/mir-1k`); TAT-Vol1 needs a signed licence; DALI v2's YouTube ids are mostly dead;
JamendoLyrics clones automatically (en/fr/de/es only).

### 3.2 Self-built data (the only route for Taiwanese Mandarin songs)

Public Chinese singing corpora are almost all simplified-context a cappella built for
synthesis, far from the target distribution of "Taiwanese pop + accompaniment + mixed zh-en".

```
original/self-recorded → Demucs vocal separation → existing VAD segmentation
    → current preset produces weak labels → forced alignment against known lyrics
    → human correction in the existing transcript-editing UI → export manifest
```

- Target: **EN 20–50 h weak labels + 5 h corrected**; **ZH 10–20 h weak + 3–5 h corrected**.
  LoRA works from 5–20 h of good data; do not chase 100 h up front.
- Prioritise human correction on mixed zh-en singing, falsetto/breath, fast passages, and
  Taiwan-specific vocabulary and names.
- Label normalisation must be written down as rules first, or training labels and product
  output will never line up: always Traditional (OpenCC first, then human review), punctuation
  matching current output, English words keeping their original spelling and case, uniform
  number formatting, and non-lyric markers like `（間奏）` becoming empty strings in layer C.

### 3.3 Augmentation (this decides robustness)

- **Train on Demucs-separated vocals, not clean a cappella.** Inference sees audio carrying
  separation artefacts, so the training distribution must match. A model trained on clean
  a cappella degrades in the product.
- For a cappella sources: mix in accompaniment (SNR −5/0/+5/+10 dB) → run Demucs → obtain
  vocals with artefacts.
- Pitch shift ±2 semitones, time stretch ±10 %, light reverb. **Do not** use augmentations
  that damage word timing.
- **Negatives at 5–10 %**: backing only / interludes with empty target text — training
  directly against the current hallucination pain point.
  ✅ Implemented 2026-07-28: `dataset_builder.instrumental_chunks` cuts the gaps between lyric
  lines (intro/interlude/outro) and exports them with empty text, ratio controlled by
  `--negative_ratio` (default 0.08), **only on mixed recordings** — gaps in a cappella and
  read speech are silence, and silence is already caught by the RMS gate; what the training
  distribution lacks is *loud* non-speech. Gaps shorter than `--min_negative` (default 3 s) are
  breaths, not interludes, and are dropped. `tools/make_manifests.py` no longer counts empty
  text as `dropped["empty_text"]` but tags it `layer=negative` and reports the negative ratio
  per split.

---

## 4. Stage 3: training (W5–W8)

Environment: a **separate venv, or a rented GPU** — never shared with inference (the existing
torch/cuDNN DLL conflict lesson). Training uses HF `transformers` + `peft`; the product side
stays CT2 and torch-free.

### 4.1 EN line (Whisper)

- Base: `openai/whisper-large-v3` (MIT). Train a `medium` variant too if it must match the
  product's medium preset.
- Method: **LoRA** (r=16–32, targeting `q_proj`/`v_proj`, optionally `k_proj`/`out_proj`),
  bf16 + gradient checkpointing + 8-bit optimizer → large-v3 fits a 3060 12 GB; full
  fine-tuning does not.
- Starting hyperparameters: lr 1e-3 (LoRA), warmup 10 %, 2–4 epochs, effective batch 32 via
  gradient accumulation, SpecAugment on, eval every 500 steps, early stop on dev WER.
- **Do not freeze the encoder.** Singing differs from speech mainly acoustically (pitch range,
  sustained vowels, vibrato), so the encoder is what must move; the decoder handles lyric
  syntax. If VRAM is tight, lower r rather than freezing the encoder.

### 4.2 ZH line (Breeze-ASR-25)

- Base: `MediaTek-Research/Breeze-ASR-25` (1.54 B, whisper-large-v2 fine-tuned, **Apache-2.0**,
  mixed zh/en speech). The commercially friendly licence is a key reason for choosing it.
- It is **already a fine-tune**, so the biggest risk of training it again is catastrophic
  forgetting of speech:
  - lr one order below the EN line (LoRA from 2e-4), fewer epochs (1–3).
  - **Replay**: mix 15–25 % ordinary Taiwanese Mandarin speech into the training set (existing
    transcripts work), run the §1.2.5 speech regression at every eval, and roll back on a
    relative regression above 5 %.
  - Deliberately upweight mixed zh-en samples — that is Breeze's advantage over generic
    Whisper and must not be lost in a singing fine-tune.
- Optional two-stage: encoder LoRA first (acoustic adaptation), then decoder LoRA (lyric
  language), which shows which half produced the gain.

### 4.3 Output and conversion

Both steps are `tools/merge_and_convert.py`: merge the adapter back into the base model in
fp16, then `ct2-transformers-converter --model <merged_dir> --output_dir models/<name>-ct2
--quantization float16` (same recipe as breeze / NLLB, offline first). It always writes a new
directory and never overwrites `models/breeze-asr-25-ct2`.

⚠ **Always run a round-trip consistency test after conversion** (HF vs CT2 output on the same
20 clips), for the same reason as the NLLB flores incident: silent conversion errors are the
hardest to find. Measure WER before and after quantisation too.

---

## 5. Stage 4: wiring back into the product (W9)

✅ Solved generically. `tools/deploy_model.py` → `models/presets.json` registers a preset with
no code change and no server restart (re-read every request); it verifies the CT2 files and
smoke-transcribes a clip **before** registering, so a broken conversion never reaches the
phone. `doctor.py` lists every deployed preset. See README, "Deploying a post-trained model".

Still required per model: a README entry and, for weights trained on NC data, a **dataset
licence statement** marking them research-only.

---

## 6. Shipping thresholds (below these it stays an opt-in preset and the default is unchanged)

| Condition | Threshold |
|---|---|
| Target layer (B, accompanied) CER/WER | **≥15 % relative reduction** against the stage-1 baseline |
| Layer C hallucination rate | **Must not rise** (ideally falls) |
| word/char timing median error | No worse than baseline (jianpu must not break) |
| Speech regression (Breeze line) | Relative regression **< 5 %** |
| RTF on the 3060 | ≤ baseline × 1.2 |

A/B by letting real usage compare through the existing model switcher; do not build a
framework for it.

---

## 7. Schedule

Superseded. The serial W1–W10 schedule was replaced by the §0.5 kill switch, and the pipeline
(data → LoRA → merge → CT2 → preset → evaluation) has since been walked end to end in the
appendix. The W-labels above survive only as stage names. Current open work is tracked in
`計畫.md`.

---

## 8. Risks and exits

1. **Licensing** — NC datasets, DALI audio, original recordings. → Commercial line uses only
   Apache/CC-BY/self-built; ship "research" and "commercial" weight bundles separately.
2. **Catastrophic forgetting** — the Breeze line's biggest risk. → Replay plus the speech
   regression as a hard gate; a preset can be rolled back at any time.
3. **Hallucination changes shape** — after training the model may stop emitting credit strings
   and start emitting lyric-shaped nonsense, defeating the string-matching gate. → Layer C
   negatives are the main defence; re-audit `_is_credit_hallucination`'s string table.
4. **Environment conflicts** — training's torch/CUDA versions can contaminate inference. →
   Fully isolated training environment.
5. **Disproportionate effort** — if stage 1 captures most of the gain, **stop at stage 1** and
   spend the time on the jianpu/alignment experience. The ALT ceiling is low; the product value
   is "pitch plus roughly correct lyrics", not per-character accuracy.
6. **Taiwanese songs** — a separate line (the Han-character vs Tâi-lô annotation system must be
   decided first, and the data sourced separately). Not part of this plan; do not mix it into
   the Chinese line.

---

## 9. Open decisions

1. ~~EN base model~~ — settled: `whisper-medium` (matches the product preset); it overfit on
   46 minutes of data and was not deployed, so a larger base is pointless until there is more
   English singing data.
2. Whether to apply for the DSing DAMP licence — still open, and it is the deciding factor for
   whether the EN line ever gets enough data.
3. The cap on ZH human correction hours (3 h vs 10 h).
4. ~~Training hardware~~ — settled: local RTX 3060 12 GB, LoRA only.
5. ~~Ship forced lyric alignment as a product feature~~ — settled: yes, shipped
   (`lyrics.py` + `POST /api/transcript/lyrics`), and it turned out higher-return than
   fine-tuning, as suspected.

---

## References

- [Exploiting Music Source Separation for Automatic Lyrics Transcription with Whisper (arXiv 2506.15514)](https://arxiv.org/abs/2506.15514) — separation lowers WER **without** fine-tuning; uses faster-whisper large-v2 + beam size 5.
- [LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription (arXiv 2306.17103)](https://arxiv.org/html/2306.17103v4)
- [PDAugment: Data Augmentation by Pitch and Duration Adjustments for ALT (arXiv 2109.07940)](https://arxiv.org/pdf/2109.07940)
- [VietLyrics: A Large-Scale Dataset and Models for Vietnamese ALT (arXiv 2510.22295)](https://arxiv.org/html/2510.22295) — Whisper-large-v2 fine-tuned to 24.61 % WER (case sensitive), a useful order of magnitude.
- [MPop600: A Mandarin Popular Song Database with Aligned Audio, Lyrics, and Musical Scores](http://www.apsipa.org/proceedings/2020/pdfs/0001647.pdf)
- [MediaTek-Research/Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) — whisper-large-v2 fine-tuned, 1.54 B, Apache-2.0, zh/en.

---

# Appendix: Taiwanese / singing LoRA post-training execution log (2026-07-22 – 07-23)

> The sections above are the go-forward ALT strategy (fine-tuning demoted to Plan B); this
> appendix records a **completed** post-training run on Taiwanese (nan) read speech + zh-TW
> singing, including P0–P3 results and the v2 improvements. It is the worked precedent for the
> conversion / wiring / evaluation / hot-switch steps above, and is self-contained: reading it
> is enough to start work.

## Goals

1. **Line A (main):** LoRA post-train **Breeze-ASR-25** for Taiwanese (nan) plus zh-TW singing
   domain adaptation.
2. **Line B:** LoRA post-train **whisper-medium** for English singing domain adaptation.
3. Ship the resulting CT2 models through the existing model hot-switch for A/B.

**Out of scope:** ja/ko (no data), and Taiwanese *singing* itself (the nan data on hand is read
speech; Taiwanese songs wait for YouTube + LRC material in a second round).

## Hard constraints (read first)

- **GPU: RTX 3060 12 GB** → large-v2 is LoRA-only (fp16/bf16 + gradient checkpointing + 8-bit
  optimizer, batch 1–2 × grad-accum 16, LoRA r=16 on q_proj/v_proj). OOM fallbacks: r=8 →
  batch 1 → freeze the encoder and LoRA the decoder only.
- **Use `.venv/Scripts/python.exe` directly; never `uv run`, never `uv sync`** — `uv run`
  syncs `tokenizers` from the pinned 0.22.2 back to 0.23.1 and breaks transformers, and
  `uv sync` replaces the manually installed CUDA torch 2.6.0+cu124 with the CPU build.
- The system Python has no torch; run everything from the repo root against `.venv`.
- Local `models/breeze-asr-25-ct2` is **CTranslate2 inference format and cannot be trained** —
  training pulls the original HF weights `MediaTek-Research/Breeze-ASR-25` (~3 GB).
- Whisper has no `nan` language token → nan data uses `<|zh|>` + task=transcribe, with target
  text as it stands in the metadata (mostly Ministry-of-Education Han characters; pure Tâi-lô
  sentences left as they are).

## Data on hand

`dataset/metadata.csv` (HF audiofolder:
`file_name, transcription, duration, language, source_dataset_or_song_id`), all chunks 16 kHz
mono PCM16 — 31,454 rows / 24.6 h:

| Source | id prefix | Lang | Size | Nature |
|---|---|---|---|---|
| Common Voice nan-tw validated | `cv_` | nan | 29,608 / 21.6 h | read speech |
| MIR-1K | `mir1k_` | zh-TW | 1,000 clips / 2.2 h | amateur singing (dry vocal = right channel of the original) |
| JamendoLyrics | `jamendo_` | en | 846 chunks / 0.8 h | htdemucs_ft separated vocals |

Augmentation material, all local: `dataset/MIR-1K/Wavfile/*.wav` (left = accompaniment for
remixing, right = vocal); `dataset/_cache/jamendolyrics/mp3/` (original mixes, chunk timestamps
line up with the separated vocals); `dataset/nan-tw/` (Common Voice official tsv, speaker
disjoint).

## Split rules (§切分規則) — leakage-safe, never split randomly at chunk level

| Source | Rule | Implementation |
|---|---|---|
| nan | Common Voice official split | map the path stem in `dataset/nan-tw/{train,dev,test}.tsv` to `cv_<stem>` |
| MIR-1K | by singer | filename prefix (`abjones`, `amy`, … 19 total): 17 train / 1 dev / 1 test |
| Jamendo | by song | 20 songs: 16 train / 2 dev / 2 test |

⚠ The Common Voice official split is pathological: of 274 speakers, 256 are in test and 13 in
dev, leaving **5 in train**. Corrected by keeping test untouched and moving every non-test
speaker into train (3 mid-sized ones held out as dev) → nan train 15 speakers / 21,654
utterances.

## Steps

### P0 — environment and baseline (half a day)

1. `uv pip install transformers peft accelerate jiwer bitsandbytes soundfile` (⚠ **not**
   `uv sync`), and pin `uv pip install 'tokenizers>=0.22,<0.23'`.
2. Generate manifests (`tools/make_manifests.py` → `dataset/manifests/{train,dev,test}.jsonl`;
   train 17,697 / dev 6,121 / test 6,594, with 1,042 nan rows dropped for leakage).
3. **Zero-shot baseline (no baseline, no training)** via `tools/eval_asr.py`, reports in
   `dataset/eval_reports/`. Normalise before scoring: strip punctuation, unify full/half width,
   strip whitespace for CJK; use MER for mixed Tâi-lô sentences.
   - Breeze nan CER **1.0963** (n=6430) → acceptance ≤0.767
   - Breeze MIR-1K CER **0.0646** (n=65; already good, must not regress) → acceptance ≤0.0549
   - whisper-medium Jamendo WER **0.788** (n=99) → acceptance ≤0.670

### P1 — Breeze joint LoRA (overnight, ~10–15 h)

Train nan speech and MIR-1K singing **jointly** at roughly 10:1 (one adapter, not two; split
only if dev shows singing dragging speech down). Singing-side augmentation: random SNR 0–15 dB
accompaniment remix, pitch ±2 semitones, tempo 0.9–1.1, SpecAugment. Start at lr 1e-4, 2–3
epochs, warmup 500 steps, early stop on dev CER. Checkpoint each epoch and run a zh-TW held-out
check afterwards.

✅ `tools/train_lora.py` (8-bit base + LoRA r16 q/v + grad-ckpt + adamw_bnb_8bit, batch 4 × ga 8
= eff 32, 2 epochs). 9.6 h, eval_loss 0.942→0.843, adapter
`models/lora/breeze-nan/adapter_best`. Anti-forgetting check on MIR-1K zh-TW:
CER 0.0646→0.0620 — no regression, so the joint adapter does not need splitting.

### P2 — whisper-medium en LoRA (1–2 h)

Same augmentation; with 46 minutes of data, expectations are low (≥15 % relative WER reduction
would pass) and early stopping guards against overfitting.

✅ `train_lora.py --model_id openai/whisper-medium --language english --sources jamendo`
(batch 8 × ga 2, 6 epochs, early stop at epoch 3). dev eval_loss 1.60→0.724 but **test WER
0.788→0.798 (flat to slightly worse)** — 16 training songs overfit and do not generalise to
held-out songs, exactly as the plan predicted ("46 minutes is line B's ceiling; a bad result is
a data problem"). **Not deployed** (no better than the app's existing whisper-medium); the
adapter and `whisper_lora.json` are kept but no preset is registered.

### P3 — evaluation and deployment (half a day)

1. Score the test set against the P0 baseline. Acceptance: nan CER ≥30 % relative reduction;
   MIR-1K / Jamendo ≥15 %.
2. Merge the LoRA →
   `ct2-transformers-converter --model <merged_dir> --output_dir models/breeze-asr-25-nan-ct2
   --quantization float16` (`tools/merge_and_convert.py`).
3. Wire back through `breeze_elf/asr_models.py` hot switching for A/B; **never overwrite**
   `models/breeze-asr-25-ct2`.

✅ v1 scored nan CER 1.0963→0.6614 (−39.7 %) and MIR-1K 0.0646→0.0620 (−4.0 %; short of 15 %
but the baseline was already near the ceiling and there is no regression). Wired in via
`config.py asr_breeze_nan_model` (env `BREEZE_ASR_BREEZE_NAN_MODEL`) and a dir-gated
`asr_models.py` preset; the default remains the stock breeze model.

⚠ **The fake-A/B bug (fixed):** `_run_asr_switch` resolved `settings.asr_breeze_model` for
*every* `kind=="breeze"` preset, so switching to `breeze-nan` loaded the **stock** model while
reporting the nan path. Fix: `resolve_breeze_model_dir(settings, model)` takes the path, and
the switch passes `option.model`; regression test
`test_switch_to_lora_preset_loads_its_own_dir`. Real-device A/B on the same nan test clip:
stock「外地之前要說事情做什麼」vs LoRA「話底真情愛講代誌做啥」(ref「月底進前愛共代誌做煞」).

⚠ Because this round used the **env-var-gated builtin** preset (read at process start), the
server process the phone connects to must be **restarted** before the preset appears. Going
through `tools/deploy_model.py` → `presets.json` instead is re-read every request and needs no
restart.

## v2 improvements (2026-07-23, after error autopsy)

Error autopsy with `tools/rescore.py` (recomputed from stored predictions, no GPU) refuted the
"the residual is empty outputs" guess and found two root causes:

1. **A metric/label bug.** Common Voice nan references embed Tâi-lô pronunciation glosses,
   `漢字(Kong-kuán|…)` — **48 % of the training label characters** — and they scored perfectly
   correct hypotheses as mass deletions. `tools/text_norm.py::strip_reference_gloss` now runs
   in `eval_asr.py` (references only) and `make_manifests.py` (training labels). Corrected,
   v1's real score is **nan CER 0.4806 against a base of 1.2149 = −60.4 %**, not the −39.7 %
   originally reported. Pre-fix and post-fix numbers are not comparable.
2. **The pathological CV split** (see §切分規則 above).
3. **Utterance packing** (`tools/pack_manifest.py`): concatenate same-speaker clips into ~25 s
   windows (2,571 windows, 15 % left single) to attack the "short utterances have high CER"
   weakness. `train_lora.py --manifest train_packed` consumes records with an `audios` list;
   nan additionally gets resample speed perturbation (±10 %, simulating new speakers, capped
   below 29 s); Ampere auto-selects bf16.

**v2 recipe:** `--manifest train_packed --r 32 --lora_targets q,k,v,out_proj --batch 2
--grad_accum 16 --epochs 3 --warmup 80`. 7 h, eval_loss 0.747→0.612.
⚠ `--batch 4` OOMs (r32 across four modules enlarges the autograd graph). ⚠ Warmup must be
~10 % of steps — warmup 500 over a 627-step run kept lr near zero throughout and the loss
stalled at 36.

**v2 results (scored with glosses stripped):**

| test | base | v1 | **v2** |
|---|---|---|---|
| nan CER | 1.2149 | 0.4806 | **0.4356** (−64.1 % vs base, −9.4 % vs v1) |
| MIR-1K CER | 0.0646 | 0.0620 | **0.0620** (no regression) |

Qualitatively: v2 fixed 醫療紀念館 / 不在此限 and eliminated v1's repetition hallucinations
(海仔仔海仔仔). eval_loss was still falling at epoch 3, so more epochs or more Taiwanese data
should help further.

Deployed with
`tools/deploy_model.py --id breeze-nan-v2 --ct2-dir models/breeze-asr-25-nan-v2-ct2`
(→ `models/presets.json`); the switcher now offers stock / v1 / v2. Line B stays untouched —
its ceiling is data, not hyperparameters.

Later re-measured with the char-level MER fix as `breeze_nan_v2_5m`: nan CER 1.212→0.438,
MER 0.883→0.434, mir1k 0.065→0.062, **jamendo WER 0.778→0.849** — a mild English regression,
which is exactly why it ships as a switchable preset rather than the default.

## Known risks and traps

- bitsandbytes needs ≥0.43 on Windows; if it will not install, `adafactor` replaces the 8-bit
  optimizer (slightly slower).
- MIR-1K lyrics have no punctuation while Breeze emits punctuation — skipping evaluation
  normalisation overstates CER.
- Jamendo's 45 minutes of English is line B's ceiling; a bad result is a data problem, not a
  hyperparameter problem.
- Load audio directly from `dataset/chunks/*.wav` (already 16 k mono); do not re-run separation.
- Disk: HF weights 3 GB + merged 3 GB + CT2 1.5 GB. Check free space first.

## Follow-on work (not this round, but it shapes the design)

- Ingest CV zh-TW (`--dataset_name common_voice --input <dir> --target_lang zh-TW`) and mix 5 %
  as anti-forgetting replay.
- Taiwanese songs via YouTube + LRC (`--source_type youtube --lyrics *.lrc --target_lang nan`)
  → a second Taiwanese singing LoRA. That round also produces the C-layer negatives §1.2 needs.
- TAT-Vol1 once the licence is approved.
