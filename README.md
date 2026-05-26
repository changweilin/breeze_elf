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

## Configuration

Environment variables:

| Name | Default | Purpose |
| --- | --- | --- |
| `BREEZE_HOST` | `127.0.0.1` | Local bind host. |
| `BREEZE_PORT` | `8788` | Local bind port. |
| `BREEZE_SAMPLE_RATE` | `16000` | Expected audio sample rate. |
| `BREEZE_WINDOW_SECONDS` | `2.0` | ASR window duration. |
| `BREEZE_OVERLAP_SECONDS` | `0.5` | Overlap between ASR windows. |
| `BREEZE_MAX_QUEUE_WINDOWS` | `4` | Maximum pending ASR windows per client. |
| `BREEZE_ASR_MODEL` | `medium` | Whisper model name. |
| `BREEZE_ASR_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `BREEZE_ASR_COMPUTE_TYPE` | `int8` | Compute type for custom ASR device values. |
| `BREEZE_ASR_CONCURRENCY` | `1` | Maximum concurrent ASR transcriptions. |
| `BREEZE_ASR_PROVIDER` | `faster-whisper` | Set `mock` for development without ASR dependencies. |
| `BREEZE_ASR_LOAD_ON_STARTUP` | `1` | Load Whisper during server startup. |
| `BREEZE_RMS_THRESHOLD` | `0.008` | Silence gate threshold. |

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
