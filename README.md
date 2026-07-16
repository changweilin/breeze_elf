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
| `BREEZE_SEGMENTER` | `vad` | `vad` for utterance segments, or `window` for fixed windows. |
| `BREEZE_VAD_FRAME_MS` | `100` | RMS VAD frame size. |
| `BREEZE_VAD_PRE_ROLL_MS` | `300` | Audio kept before detected speech. |
| `BREEZE_VAD_END_SILENCE_MS` | `700` | Silence required to finish an utterance. |
| `BREEZE_VAD_MAX_SEGMENT_SECONDS` | `18.0` | Max utterance length before a forced split. The split lands at the quietest recent frame (a syllable gap) so a long sung phrase is never cut mid-note. |
| `BREEZE_CHAR_VOICELESS_MARGIN` | `1.6` | Post-processing only: grow each 字's window outward through audio above `noise_floor × margin` (an unvoiced consonant/breath) the live VAD clipped. |
| `BREEZE_ASR_MODEL` | `medium` | Whisper model name. |
| `BREEZE_ASR_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `BREEZE_ASR_COMPUTE_TYPE` | `int8` | Compute type for custom ASR device values. |
| `BREEZE_ASR_CONCURRENCY` | `1` | Maximum concurrent ASR transcriptions. |
| `BREEZE_ASR_PROVIDER` | `faster-whisper` | Set `mock` for development without ASR dependencies. |
| `BREEZE_ASR_LOAD_ON_STARTUP` | `1` | Load Whisper during server startup. |
| `BREEZE_ASR_NO_SPEECH_PROB_THRESHOLD` | `0.6` | Drop low-energy ASR results above this Whisper no-speech probability. |
| `BREEZE_ASR_HALLUCINATION_RMS_THRESHOLD` | `0.02` | RMS ceiling used when filtering likely silence hallucinations. |
| `BREEZE_STOP_DRAIN_TIMEOUT_SECONDS` | `60.0` | Time allowed to transcribe the final flushed utterance after stop. |
| `BREEZE_RMS_THRESHOLD` | `0.008` | Silence gate threshold. |
| `BREEZE_REMOTE_STORAGE_DIR` | `remote_transcripts` | Host-side directory for remotely saved transcript `.txt` files. |
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
