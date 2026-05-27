---
name: breeze-parameter-data-governance
description: Govern Breeze Elf configuration, parameter records, protocol data, and storage contracts. Use when adding or changing BREEZE_* env vars, runtime defaults, validation/clamping, WebSocket message fields, transcript persistence, future parameter databases, README tables, or related tests.
---

# Breeze Parameter Data Governance

## Workflow

1. Identify the contract type:
   - Environment setting: `breeze_elf/config.py`, `README.md`, `tests/test_config.py`
   - WebSocket/API payload: `breeze_elf/protocol.py`, `breeze_elf/main.py`, protocol/main tests
   - Stored transcript data: `breeze_elf/storage.py`, storage/main tests
   - Future parameter database/profile: add a small typed boundary before adding persistence.
2. Keep source-of-truth fields synchronized. A new setting normally needs a dataclass field, env reader, README row, and tests for default plus override behavior.
3. Validate coercion and clamping. Numeric runtime parameters should reject or clamp unsafe values intentionally.
4. Keep persistence boring and local. Do not introduce a database for simple env parameters; use a DB only when profiles, history, querying, or multi-record management is required.
5. Avoid breaking existing protocol field names without an explicit migration plan.

## Guardrails

- Remote transcript storage must continue to sanitize filenames and write UTF-8 text.
- Settings read from environment at call time via `get_settings`; avoid module-level dynamic config except where the app intentionally captures startup settings.
- When adding a parameter database, include schema, migration/reset story, backup/export path, and tests that do not require network or a live ASR model.

## Validation

```powershell
npm.cmd run test
uv run python -m unittest tests.test_config tests.test_protocol tests.test_storage tests.test_main
```

## Reference

Read `references/data-contracts.md` before changing settings, API fields, or persistence.
