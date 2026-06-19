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
| `BREEZE_MAX_QUEUE_WINDOWS` | `4` | Maximum pending ASR windows per client. |
| `BREEZE_SEGMENTER` | `vad` | `vad` for utterance segments, or `window` for fixed windows. |
| `BREEZE_VAD_FRAME_MS` | `100` | RMS VAD frame size. |
| `BREEZE_VAD_PRE_ROLL_MS` | `300` | Audio kept before detected speech. |
| `BREEZE_VAD_END_SILENCE_MS` | `700` | Silence required to finish an utterance. |
| `BREEZE_VAD_MAX_SEGMENT_SECONDS` | `12.0` | Maximum utterance length before forced flush. |
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
