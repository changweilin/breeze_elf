# Breeze Elf Data Contracts

## Environment Settings

Settings live in `breeze_elf/config.py` as the `Settings` dataclass and `get_settings()` env reader. Public settings should be listed in `README.md` and covered by `tests/test_config.py`.

Important parameter groups:

- Server: `BREEZE_HOST`, `BREEZE_PORT`
- Audio: `BREEZE_SAMPLE_RATE`, `BREEZE_WINDOW_SECONDS`, `BREEZE_OVERLAP_SECONDS`, `BREEZE_RMS_THRESHOLD`
- Segmentation: `BREEZE_SEGMENTER`, `BREEZE_VAD_*`
- ASR: `BREEZE_ASR_*`, including model, device, provider, loading, concurrency, hallucination filtering
- Storage: `BREEZE_REMOTE_STORAGE_DIR`

## Protocol

Client messages are parsed in `breeze_elf/protocol.py`: `start`, `stop`, and `ping`. Server events are dictionaries created by `server_event()` and emitted from `breeze_elf/main.py`.

## Persistence

Transcript files are created by `save_transcript()` in `breeze_elf/storage.py`. Preserve UTF-8 output, unique filenames, safe slugs, and empty-text rejection.
