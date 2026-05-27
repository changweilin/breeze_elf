---
name: breeze-i18n-localization
description: Localize and protect Breeze Elf multilingual text surfaces. Use when changing Traditional Chinese or other UI copy, ASR language handling, transcript display/storage text, accessibility labels, README copy, demo strings, or encoding-sensitive test fixtures.
---

# Breeze I18n Localization

## Workflow

1. Map every text surface before editing:
   - Browser UI: `web/index.html`, `web/app.js`
   - ASR language behavior: `breeze_elf/asr.py`, `breeze_elf/protocol.py`, `breeze_elf/config.py`
   - Transcript persistence and API text: `breeze_elf/storage.py`, `breeze_elf/main.py`
   - User docs and fixtures: `README.md`, `tests/`
2. Preserve UTF-8 Traditional Chinese text. If text appears as mojibake in terminal output, verify the file with a UTF-8-aware editor or byte-safe read before rewriting strings.
3. Prefer a catalog when adding multiple languages. Keep status text, button text, `aria-label`, `title`, demo strings, and stats labels together rather than scattering conditionals.
4. Keep ASR language codes explicit. The browser currently starts with `language: "zh"` and backend defaults include `BREEZE_LANGUAGE=zh`; update protocol tests when language behavior changes.
5. Keep accessibility text localized with visible labels. Do not translate only the visual button text while leaving stale `aria-label` or `title` values.

## Guardrails

- Keep transcript content as user data: do not auto-translate saved transcripts unless the requested feature explicitly asks for translated output.
- Keep OpenCC conversion behavior in `FasterWhisperASR` visible when changing Chinese language handling.
- Add or update tests for language parsing, transcript text preservation, and any fallback behavior.

## Validation

```powershell
npm.cmd run check:web
npm.cmd run test
```

For visible UI language changes, also smoke the app in demo mode with `?demo=1`.

## Reference

Read `references/surface-map.md` when the task touches more than one text surface.
