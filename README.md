# Breeze Elf

Breeze Elf streams a phone microphone to this computer over a private Tailscale tailnet and transcribes it locally with `faster-whisper`.

## Quick Start

```powershell
uv sync
uv run python -m breeze_elf
```

The local server listens on `http://127.0.0.1:8788`.

Expose it privately to your tailnet with HTTPS:

```powershell
tailscale serve --bg 8788
tailscale serve status
```

Open the HTTPS Tailscale Serve URL on your phone, allow microphone access, then tap `開始`.

If the Tailscale command reports access denied on Windows, run the command from an Administrator PowerShell.

## Diagnostics (`breeze doctor`)

Before chasing a slow / CPU-only run, check the environment in one shot:

```powershell
uv run python -m breeze_elf doctor
```

It prints a PASS/WARN/FAIL line for the ASR runtime, GPU visibility (ctranslate2 for
Whisper, torch for the optional enhance/voice stages), the Windows cuDNN DLL, the opt-in
extras, and whether each configured local model is present — each with the exact fix. It
never loads a model, so it is safe to run on a broken setup. Add `--load` to also load
Whisper and confirm the resolved device / compute type. Exit code is non-zero if any
check fails.

## Remote Transcript Saving

The web app can save the current transcript back to the Breeze Elf host. Tap `遠端儲存`
after text appears, and the server writes a UTF-8 `.txt` file under
`remote_transcripts/` by default.

Set `BREEZE_REMOTE_STORAGE_DIR` to choose another host-side directory.

## Recognition Languages

Tap `🌐 語言` to choose which languages live recognition is restricted to. The default is
Traditional Chinese + English, and you can multi-select **up to 4** languages. When more
than one is selected the server detects each utterance's language but forces the
highest-probability language **within your chosen set**, so a stray foreign segment is
recognised as your primary (first) language instead of escaping into an unselected tongue.
Switch on **自由偵測** to let Whisper auto-detect freely with no restriction. Traditional
Chinese also adds a Taiwan-usage prompt; a pure non-Chinese selection skips the
simplified→traditional conversion. The selection is remembered on the device and sent in the
WebSocket `start` message. (Loaded audio files always use free detection — see below.)

## 慣用詞庫 (Custom Glossary)

Tap the ✎ on any finalized transcript block to edit its text in place. The diff between the
original and your edit is learned as a `原本 → 改成` correction and stored on the device
(`📖 慣用詞庫` lists and manages them). On the next recognition the glossary is used two ways:
the preferred spellings are added to Whisper's `initial_prompt` to bias decoding, and the
recognised text is auto-corrected with the same `原本 → 改成` substitutions. This makes
recurring names and terms progressively more accurate. The glossary stays on your device and
rides along in the `start` message; it is never uploaded to any cloud service.

## Pitch Mode

Tap `音高` to show each finalized transcript block with its matched audio time range and a
per-character 簡譜 (jianpu) line. Pitch is estimated locally from the same VAD/audio window
that produced the text, so copied, downloaded, and remotely saved transcripts stay as plain
text. A character whose pitch slides within its duration (滑音 / portamento) is shown as a
glide, e.g. `3↗5` (rising) or `5↘1` (falling), instead of a single number.

Click any finalized sentence to expand a detail panel listing each character's time range,
frequency (or the slide's start→end), scale tuning error in cents, and intensity trend
(漸強 / 漸弱 / 持平). This works in both plain and pitch modes.

## Loading Audio Files

Tap `載入音檔` to analyze a recording instead of the live microphone. Loaded files request
automatic language detection rather than forcing Traditional Chinese, so music or
non-Chinese audio is no longer transcribed as Chinese gibberish. Note that a speech model
still hallucinates lyrics on purely instrumental tracks — for melodies the pitch / 簡譜
output is the meaningful result.

### Switching the ASR model

The 模型與演算法 dialog (tap the backend line in the footer) has a model switcher at the
top. It hot-swaps the running engine between **Breeze ASR** (`breeze`), **Whisper medium**,
and **Whisper large-v3** without restarting the server — the new model loads in the
background (a progress bar tracks it) and the old one is released once the swap completes.
Switching is blocked while a recording is streaming, and the currently loaded model is
always shown as the active option. The Whisper sizes are downloaded/cached by
faster-whisper on first use; `breeze` is a local CTranslate2 directory (see
`BREEZE_ASR_BREEZE_MODEL`) and reports a clear error if the directory is missing rather
than reaching out to Hugging Face. Endpoints: `GET /api/asr/models`, `POST /api/asr/model`,
`GET /api/asr/model/status`.

For long recordings, set `BREEZE_ASR_FILE_BATCH_SIZE` (e.g. `16`) to transcribe the whole
file in one batched `BatchedInferencePipeline` pass (3–4× faster) via
`POST /api/transcribe/file`, instead of streaming it utterance-by-utterance. The batched
pass returns all blocks at once (no live partials); run `校準音準` afterwards for the
per-character 基頻/簡譜, which the offline pass measures against a single global 主音.

## Speech Enhancement & Source Separation (optional)

Two neural models can sit in front of Whisper on a GPU (tuned for a 12 GB RTX 3060).
Both are optional — the base install stays torch-free and falls back to the DSP-only
pipeline. Install the extra with the CUDA build matching the box:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install deepfilternet demucs
```

Then run with the live enhancer on. PowerShell (Windows):

```powershell
$env:BREEZE_ENHANCE_LIVE = "deepfilter"; uv run python -m breeze_elf
```

bash / zsh:

```bash
BREEZE_ENHANCE_LIVE=deepfilter uv run python -m breeze_elf
```

> Install torch from the CUDA wheel index as shown — **not** with `uv sync --extra enhance`,
> which resolves the CPU-only `torch` pinned in `uv.lock` and silently disables the GPU.
> `uv run` keeps the CUDA build already present in `.venv`; it does not strip it.

- **Real-time denoise + dereverb (DeepFilterNet3).** Set `BREEZE_ENHANCE_LIVE=deepfilter`
  (and/or `BREEZE_ENHANCE_FILE=deepfilter`) to run DeepFilterNet3 per utterance before
  Whisper. It is ~70–95× real time once warm (~15–60 ms per utterance), so it clears the
  far-mic / reverb / low-level phone pickup that otherwise degrades recognition. Any
  load/inference failure silently falls back to the raw audio.
- **Music source separation (Demucs htdemucs).** Pick `🎵 音樂(分離人聲)` in the 情境
  selector next to `載入音檔`. The whole file is sent to `/api/enhance/separate` once, the
  `vocals` stem is isolated, and recognition + 基頻/簡譜 then run on the clean vocal instead
  of hallucinating over the full mix. `🗣 一般人聲` skips separation. The choice persists
  in `localStorage` and needs no env var; the endpoint returns `503` when the extra is not
  installed.

`GET /health` reports the active stage via `enhanceLive` / `enhanceFile` / `enhanceDevice`
/ `separatorAvailable`, and each stream's `ready` event carries the active `enhance` name.

## 翻譯 (Translation, optional)

Translate any recognised language into Traditional Chinese (or another target), one
segment at a time, shown as a bilingual stacked line under the original. It runs on the
**ctranslate2** runtime faster-whisper already ships — **no torch, no new CUDA/DLL** — and
only adds the lightweight `sentencepiece` tokenizer (`[translate]` extra). Any missing
model / dependency degrades to a no-op, so recognition is never affected.

> **License:** NLLB-200 is **CC-BY-NC-4.0 (non-commercial)**, so this is strictly opt-in
> and never enabled by default. Do not ship it enabled in a commercial product.

1. Convert an NLLB-200 model to CT2 into a **local** directory (offline-first — the model
   is never downloaded on demand), keeping its sentencepiece file alongside:

   ```bash
   ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
     --output_dir models/nllb-200-distilled-600M-ct2 --quantization int8
   # copy sentencepiece.bpe.model into that dir (or point BREEZE_TRANSLATE_SPM at it)
   ```

2. Enable it: `BREEZE_TRANSLATE=nllb` (target defaults to 繁中 via `BREEZE_TRANSLATE_TARGET=zh`).
3. In the transcribe page, toggle the **翻譯** button before 開始; each committed segment then
   carries a translated line (only on `final`, never the low-latency partial). The choice
   persists in `localStorage`.

`GET /health` reports `translateProvider` / `translateAvailable` / `translateTarget`. The
~0.6 GB (int8) model plus Whisper `medium` fit together on a 12 GB card.

## 語者分離 (Speaker Diarization, optional)

Tag each utterance with an anonymous, **session-local** speaker label (`說話者 1／2…`) shown
as a coloured chip on the segment. It is **fully local and torch-free**: a pure-numpy
log-mel front-end feeds an ONNX speaker-embedding model over `onnxruntime` (the runtime
faster-whisper already bundles for Silero VAD), and a pure-numpy online clusterer assigns
labels. Supply a permissively-licensed ONNX speaker-embedding model at
`models/speaker_embedding.onnx` (or set `BREEZE_DIARIZE_MODEL`); without it the feature is a
no-op.

Enable with `BREEZE_DIARIZE=on`. **Privacy:** embeddings are computed from the *raw*
utterance audio (not the denoised ASR audio), are never stored, and the clusterer resets
every connection — there is no cross-session voiceprint linkage. `GET /health` reports
`diarizeEnabled` / `diarizeAvailable`.

## 變聲工作室 (Voice Studio)

The bottom icon tabs switch between the transcribe page (`逐字稿`) and a voice page
(`變聲`). The voice page does three things against the local server:

- **Save a voice A** — record or upload a clip of speaker A, name it, and store its
  voice features server-side under `voices/`. Toggle the ★ to keep it in 我的最愛;
  rename by tapping the name; delete with 🗑.
- **B → A** — record or upload speaker B, pick a saved A, and re-voice B as A.
- **文字 → A** — type text and synthesize it in A's voice.

A progress bar tracks the voice model warming up when you open the page (and again
if you retry after an error), so the model-switch wait is visible rather than a
silent hang.

By default `BREEZE_VOICE_PROVIDER=mock` runs a dependency-free DSP transform
(phase-vocoder pitch shift toward A's pitch, plus a simple text-to-buzz) so the whole
flow works without any downloads.

### Real cloning (OpenVoice v2)

Set `BREEZE_VOICE_PROVIDER=openvoice`. The engine needs **torch with CUDA** in the
same interpreter that runs the server. OpenVoice pins ancient `numpy`/`faster-whisper`
in its metadata, so install it **without deps** and add only the light text helpers it
imports at load time:

```powershell
pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
pip install inflect unidecode eng_to_ipa pypinyin cn2an jieba
```

Then download the v2 **converter** checkpoint into `checkpoints_v2/converter/`:

```powershell
# checkpoints_v2/converter/{config.json,checkpoint.pth}  (~131 MB)
curl -L -o checkpoints_v2/converter/config.json   https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json
curl -L -o checkpoints_v2/converter/checkpoint.pth https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth
```

That is enough for **B → A** re-voicing and saving A's voice features: Breeze Elf
bypasses `openvoice.se_extractor` (which would drag in `faster_whisper` +
`whisper_timestamped`) and stubs out `wavmark`, so only the converter is required.

**文字 → A (text-to-speech)** additionally needs [MeloTTS](https://github.com/myshell-ai/MeloTTS)
plus the base-speaker embeddings under `checkpoints_v2/base_speakers/ses/`. Without it,
B → A still works and synthesis returns a clear "needs MeloTTS" error.

Saved voices store an engine-defined embedding blob, the reference `.wav`, and a JSON
metadata sidecar. The mock and OpenVoice embeddings are not interchangeable, so a voice
saved under one provider is only usable by that provider.

## GitHub Actions Web Demo

The `Web Demo` workflow publishes the static `web/` folder to GitHub Pages. On
`*.github.io`, `file://`, or when opened with `?demo=1`, the page runs in demo
mode: microphone capture, WebSocket streaming, and remote transcript saving are
frozen, and the Start button plays a short sample transcript instead.

## Configuration

Environment variables:

| Name | Default | Purpose |
| --- | --- | --- |
| `BREEZE_HOST` | `127.0.0.1` | Local bind host. |
| `BREEZE_PORT` | `8788` | Local bind port. |
| `BREEZE_SAMPLE_RATE` | `16000` | Expected audio sample rate. |
| `BREEZE_WINDOW_SECONDS` | `2.0` | ASR window duration. |
| `BREEZE_OVERLAP_SECONDS` | `0.5` | Overlap between ASR windows. |
| `BREEZE_AUDIO_PREPROCESS` | `natural` | ASR audio preparation: `off`, `natural`, or stronger `speech`. |
| `BREEZE_ENHANCE_LIVE` | `off` | Neural denoise+dereverb on the live-mic path: `off` or `deepfilter` (needs the `[enhance]` extra). |
| `BREEZE_ENHANCE_FILE` | `off` | Neural denoise+dereverb on the loaded-file path: `off` or `deepfilter`. |
| `BREEZE_ENHANCE_DEVICE` | `auto` | Enhancement/separation device: `auto`, `cuda`, or `cpu`. |
| `BREEZE_MAX_QUEUE_WINDOWS` | `4` | Maximum pending ASR windows per client. |
| `BREEZE_SEGMENTER` | `vad` | `vad` for utterance segments, `window` for fixed windows, or `silero` (alias for `vad` + `BREEZE_VAD_DETECTOR=silero`). |
| `BREEZE_VAD_DETECTOR` | `rms` | Speech-onset gate for the `vad` segmenter: `rms` (energy threshold) or `silero` (neural voice detector via the ONNX model bundled with faster-whisper — no new dependency; silently falls back to `rms` if the model/onnxruntime is missing). |
| `BREEZE_VAD_SPEECH_THRESHOLD` | `0.5` | Silero speech probability at/above which a frame is speech. |
| `BREEZE_VAD_NEG_THRESHOLD` | `0.35` | Silero silence probability; between this and the speech threshold the previous decision is held (hysteresis). |
| `BREEZE_VAD_RMS_RELEASE_RATIO` | `0.5` | RMS gate attack/release hysteresis: an utterance ends only once RMS drops below `BREEZE_RMS_THRESHOLD × this` (onset still uses the full threshold), so a naturally decaying 句尾 syllable isn't clipped. `1.0` restores the old single-threshold gate. |
| `BREEZE_VAD_SILERO_MODEL` | *(bundled)* | Override path to a `silero_vad*.onnx`; defaults to the one shipped inside faster-whisper. |
| `BREEZE_VAD_FRAME_MS` | `100` | VAD frame size (RMS gate; Silero re-chunks internally to 32 ms). |
| `BREEZE_VAD_PRE_ROLL_MS` | `300` | Audio kept before detected speech. |
| `BREEZE_VAD_END_SILENCE_MS` | `700` | Silence required to finish an utterance. |
| `BREEZE_VAD_MAX_SEGMENT_SECONDS` | `18.0` | Max utterance length before a forced split. The split lands at the quietest recent frame (a syllable gap) so a long sung phrase is never cut mid-note. |
| `BREEZE_CHAR_VOICELESS_MARGIN` | `1.6` | Post-processing only: grow each 字's window outward through audio above `noise_floor × margin` (an unvoiced consonant/breath) the live VAD clipped. |
| `BREEZE_F0_CLEAN` | `true` | 基頻分析 post-processing. **(1) Octave/harmonic snap:** a bin mis-tracked a whole octave (or ×3/×4) off — the wrong-octave "stable" plateau that shows up as 劇烈變化 between two consistent notes — is folded back onto its local pitch; only integer harmonic ratios are tried within a tight tolerance, so a real fourth/fifth/sixth leap is never moved. **(2) Unstable-run fill:** a 音高劇烈變化 run that spikes past the flanking notes (up-then-down / down-then-up 反方向) or zig-zags widely is **filled from the surrounding 平穩音高** — interpolated between both neighbouring notes, held from the one adjacent note, or (only when silence sits on both sides) cleared — while a one-way 滑音 and vibrato are kept. **(3) Attack/release fill:** each note is grown into its 頭尾 — the quiet onset/tail the pitch detector drops — by holding the edge pitch through adjacent bins whose intensity stays above a noise-relative gate (bounded, so a breath/hiss at the room floor is never picked up). `0` keeps the raw per-bin f0 track. |
| `BREEZE_ASR_MODEL` | `breeze` | Recognition model loaded at startup. `breeze` is a preset that loads the local Breeze CT2 dir (`BREEZE_ASR_BREEZE_MODEL`), falling back to Whisper `medium` if that dir is absent; any other value (a Whisper size, HF id, or CT2 path) is passed straight to faster-whisper. Also switchable at runtime from 模型與演算法 (see below). |
| `BREEZE_ASR_BREEZE_MODEL` | `models/breeze-asr-25-ct2` | Local **CTranslate2** directory the `breeze` preset in the model switcher resolves to. Offline-first — faster-whisper only loads a CT2 model, so this is a converted Breeze ASR dir on disk, not a HF id. A relative path is resolved against the project root. |
| `BREEZE_ASR_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `BREEZE_ASR_COMPUTE_TYPE` | `int8` | Compute type for custom ASR device values. |
| `BREEZE_ASR_CONCURRENCY` | `1` | Maximum concurrent ASR transcriptions. |
| `BREEZE_ASR_PROVIDER` | `faster-whisper` | Set `mock` for development without ASR dependencies. |
| `BREEZE_ASR_LOAD_ON_STARTUP` | `1` | Load Whisper during server startup. |
| `BREEZE_ASR_NO_SPEECH_PROB_THRESHOLD` | `0.6` | Drop low-energy ASR results above this Whisper no-speech probability. |
| `BREEZE_ASR_HALLUCINATION_RMS_THRESHOLD` | `0.02` | RMS ceiling used when filtering likely silence hallucinations. |
| `BREEZE_ASR_CONTEXT_CHARS` | `0` | Cross-segment context: trailing characters of the committed transcript fed into the next utterance's `initial_prompt` (with the 慣用詞庫) for proper-noun consistency. `0` disables it. Opt-in and bounded (max 2000) — seeding recent text can make Whisper echo it on a short/quiet utterance, and it only applies when a language is fixed (free-detect drops the prompt). |
| `BREEZE_ASR_FILE_BATCH_SIZE` | `0` | Whole-file batched transcription via faster-whisper's `BatchedInferencePipeline` (3–4× throughput on long recordings). `0` keeps the per-utterance streaming file path; a value `1`–`32` enables the `POST /api/transcribe/file` endpoint and is the batch size. Live-mic streaming is never affected. |
| `BREEZE_MAX_AUDIO_UPLOAD_BYTES` | `256000000` | Upper bound (decoded bytes) on a base64 PCM upload to the whole-file endpoints (`/api/transcribe/file`, `/api/enhance/separate`), checked before decoding so an oversized body returns `413` instead of exhausting host memory. ~256 MB ≈ 2.2 h of 16 kHz mono; `0` disables the cap. |
| `BREEZE_STOP_DRAIN_TIMEOUT_SECONDS` | `60.0` | Time allowed to transcribe the final flushed utterance after stop. |
| `BREEZE_RMS_THRESHOLD` | `0.008` | Silence gate threshold. |
| `BREEZE_REMOTE_STORAGE_DIR` | `remote_transcripts` | Host-side directory for remotely saved transcript `.txt` files. |
| `BREEZE_SEARCH_ENABLED` | `true` | Cross-transcript full-text search (SQLite FTS5 trigram, no extra dependency). Auto-disables if the runtime SQLite lacks FTS5/trigram. |
| `BREEZE_SEARCH_MAX_RESULTS` | `50` | Maximum search results returned per query. |
| `BREEZE_SUMMARY_PROVIDER` | `extractive` | Post-meeting summary: `extractive` (stdlib, no model/VRAM/network), `ollama` (local Ollama daemon, degrades to extractive on failure), or `off`. No cloud path — transcripts never leave the machine. |
| `BREEZE_SUMMARY_MODEL` | `qwen3:4b-instruct` | Ollama model tag used when `BREEZE_SUMMARY_PROVIDER=ollama`. |
| `BREEZE_SUMMARY_OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint. Keep it loopback to preserve the privacy-first, on-device guarantee. |
| `BREEZE_SUMMARY_TIMEOUT_SECONDS` | `60.0` | Request timeout for the local Ollama call; raise it for a slow local model. |
| `BREEZE_SUMMARY_MAX_CHARS` | `8000` | Transcript is truncated to this many characters before summarizing. |
| `BREEZE_SUMMARY_MAX_SENTENCES` | `5` | Default number of points/sentences in a summary. |
| `BREEZE_TRANSLATE` | `off` | Post-recognition translation: `off` or `nllb` (NLLB-200 on the ctranslate2 runtime, no torch; needs the `[translate]` extra + a local CT2 model). CC-BY-NC-4.0 — opt-in, non-commercial. |
| `BREEZE_TRANSLATE_TARGET` | `zh` | Target language for translation (a language/flores code; `zh` → 繁中). |
| `BREEZE_TRANSLATE_MODEL` | `models/nllb-200-distilled-600M-ct2` | Local CT2 NLLB model directory. Never downloaded on demand (offline-first). |
| `BREEZE_TRANSLATE_SPM` | *(auto)* | Override path to the sentencepiece model; defaults to the `*.model` file inside the model dir. |
| `BREEZE_TRANSLATE_DEVICE` | `auto` | Translation device: `auto`, `cuda`, or `cpu`. |
| `BREEZE_TRANSLATE_COMPUTE_TYPE` | `auto` | ctranslate2 compute type (`auto` → `default`; e.g. `int8`, `float16`). |
| `BREEZE_TRANSLATE_BEAM` | `1` | Translation beam size. |
| `BREEZE_DIARIZE` | `off` | Anonymous in-session speaker diarization: `off` or `on` (needs a local ONNX speaker-embedding model + onnxruntime; torch-free). Degrades to a no-op when unavailable. |
| `BREEZE_DIARIZE_MODEL` | `models/speaker_embedding.onnx` | Local ONNX speaker-embedding model file. |
| `BREEZE_DIARIZE_MAX_SPEAKERS` | `6` | Maximum distinct speakers per session before new voices fold into the nearest. |
| `BREEZE_DIARIZE_THRESHOLD` | `0.75` | Cosine similarity below which an utterance starts a new speaker. |
| `BREEZE_DIARIZE_MIN_DURATION` | `0.4` | Utterances shorter than this (seconds) are not labelled (too noisy to cluster). |
| `BREEZE_DIARIZE_DEVICE` | `cpu` | Embedder device: `cpu` or `cuda`. |
| `BREEZE_DIARIZE_N_MELS` | `80` | Log-mel bands fed to the ONNX speaker model; match your model's expected feature dimension. |
| `BREEZE_VOICE_PROVIDER` | `mock` | Voice studio engine: `mock` (DSP, no downloads) or `openvoice` (OpenVoice v2 + MeloTTS). |
| `BREEZE_VOICE_STORAGE_DIR` | `voices` | Host-side directory for saved voice profiles (embedding + reference `.wav` + metadata). |
| `BREEZE_VOICE_SAMPLE_RATE` | `16000` | Sample rate used for voice capture and mock synthesis. |
| `BREEZE_VOICE_LANGUAGE` | `zh` | Default language for text-to-speech synthesis. |
| `BREEZE_VOICE_CHECKPOINTS_DIR` | `checkpoints_v2` | Location of the OpenVoice v2 checkpoints. |
| `BREEZE_VOICE_MOCK_WARMUP_SECONDS` | `0.9` | Simulated mock model-load time so the progress bar is visible. |

On an RTX 3060, `auto` tries CUDA with `float16`, then CUDA with `int8_float16`, then CPU with `int8`.

## Tests

The unit tests cover the streaming buffer, silence gate, WebSocket message validation, and mock ASR path.

```powershell
uv run python -m unittest discover
```

## Benchmarks

Run the local pipeline benchmark with mock ASR:

```powershell
npm.cmd run bench:mock
```

Run only audio segmentation, using the current environment configuration:

```powershell
npm.cmd run bench
```

Compare fixed windows against VAD utterance segments:

```powershell
npm.cmd run bench -- --segmenter window
npm.cmd run bench -- --segmenter vad
```
