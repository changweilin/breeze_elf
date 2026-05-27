# Breeze Elf Text Surface Map

- `web/index.html`: first paint language, visible labels, `aria-label`, `title`, static layout text.
- `web/app.js`: runtime status, stats, demo transcript events, button/action feedback, theme labels, remote save errors.
- `breeze_elf/asr.py`: Whisper `initial_prompt`, OpenCC conversion, mock ASR text.
- `breeze_elf/protocol.py`: `language` field parsing and validation.
- `breeze_elf/main.py`: server event text, transcript de-duplication, hallucination fragments.
- `breeze_elf/storage.py`: UTF-8 transcript persistence and filename slugging.
- `README.md`: user-facing instructions and configuration descriptions.
- `tests/`: fixtures for language, text preservation, and hallucination filtering.

When adding language support, keep catalog keys stable and translate both visible and assistive strings.
