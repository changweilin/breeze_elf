# CLAUDE.md — Breeze Elf working rules

Local-first realtime speech transcription and music analysis (FastAPI + faster-whisper/CT2
backend, frameless frontend). Traditional Chinese is the product's first language; **these
agent-facing docs are English**. This file is long-term memory across conversations: it
holds only principles and hard rules that do not expire with a single feature.

## Document map

| File | Role |
|---|---|
| `CLAUDE.md` (this file) | Principles and hard rules — read before starting |
| `計畫.md` | Current open work — the starting point for the next conversation; update it when something lands |
| `TRAINING_PLAN.md` | Training plan and execution log (referenced by code comments — do not delete, do not renumber sections) |
| `README.md` | User documentation in Traditional Chinese; new features and settings must be mirrored there |

(History: `ROADMAP.md` and `OPTIMIZATION_PLAN.md` were completed and removed; their lessons
are folded in here, details in git history.)

## Core principles

1. **Fully local, offline first.** Never sneak in a model download — models always point at
   a local directory, and a missing directory must raise a clear error or no-op. Offline is
   enforced by code, not declared in docs. Privacy-sensitive data (audio, voice prints,
   transcripts) never leaves the machine.
2. **Heavy features are opt-in extras that no-op when dependencies are missing.** Follow the
   `enhance.py` pattern: protocol + Null implementation + `build_*(settings)` factory. When
   a dependency or model is absent, fall back to Null; core functionality is unaffected.
3. **The inference side stays torch-free** — ctranslate2 + onnxruntime + numpy only. When
   training needs torch, use a **fully isolated environment**; never contaminate the
   inference venv (cuDNN/DLL conflicts have bitten repeatedly).
4. **Licensing is a red line.** NC (non-commercial) datasets and models (NLLB, MIR-1K, …)
   are never enabled by default and are labelled in README; the commercial line uses only
   Apache/CC-BY/self-built data.
5. **No baseline, no training and no optimisation.** Measure first, then change, and measure
   each change's gain separately. Training-free knob sweeps always outrank fine-tuning — if
   a knob beats LoRA, ship the knob as a product setting. When results are poor, suspect the
   data before the hyperparameters (the English line overfitting on 16 songs).
6. **Single source of truth.** Hallucination decisions live only in
   `breeze_elf/hallucination.py`; text normalisation only in `tools/text_norm.py`. What the
   evaluation measures must be the same code the product runs — never write a second copy
   inside a tool.
7. **Separate measurement code from numbers.** Pure metric functions must be unit-testable
   without a GPU. Evaluation normalisation rules determine the conclusions (punctuation,
   Tâi-lô glosses); after changing a rule, recompute text distances from stored predictions
   with `tools/rescore.py`. Hallucination rate, alignment error and RTF need audio and
   timestamps, so those always require re-running `tools/eval_asr.py`.

## Environment and commands

- Python **>=3.11** (onnxruntime ships no cp310 wheel from 1.24 on). CI runs the matrix
  3.11 / 3.12 / 3.13.
- **`npm run verify`** runs the same sequence CI does: `uv sync --extra dev` →
  `uv lock --check` → `npm run check:web` → `ruff check .` → `unittest discover`.
  `.github/workflows/verify.yml` is the single definition of that sequence; `ci.yml` and
  `release.yml` both call it so release checks cannot drift from PR checks.
- **Training and evaluation call `.venv/Scripts/python.exe` directly; never `uv run`** — it
  syncs `tokenizers` back to 0.23.1 and breaks transformers (the training stack pins
  0.22.2), and it can replace the manually installed CUDA torch with the CPU build.
- Write new tests as `unittest.TestCase`. CI runs `unittest discover`, so pytest-style
  module-level functions **never execute in CI**.

## Hard rules (violating these breaks things)

1. **Never overwrite `models/breeze-asr-25-ct2`** — the stock and fine-tuned models must
   both exist for A/B.
2. **Presets have two registration paths.** Env-var-gated builtin presets are read at
   process start (restart required); dynamic entries written to `models/presets.json` by
   `tools/deploy_model.py` are re-read on every request (no restart). Use the latter to make
   something appear on the phone immediately.
3. **Changing `web/app.js` requires bumping `?v=` and the service worker cache**, or users
   get the stale version. If a shared module's *exports* change, bump the `?v=` on the
   import specifier too. Enforced two ways: `tests/test_web_assets.py` checks the versions
   agree across index.html / import specifiers / service-worker ASSETS, and the CI job
   `web-cache-bump` (`tools/check_web_bump.py`) fails a PR that edits a cached asset
   without moving `CACHE_NAME` and that file's `?v=`.
4. **Adding a field to a transcript block means editing three places**:
   `serializeBlocksForSave` (whitelist), `normalizeTranscriptBlockForRestore`, and session
   persist. Miss one and it will not save or will not restore.
5. **Derived data (speaker, translation, …) attaches to `final` events only**, never to the
   low-latency `partial`.
6. **Run a round-trip consistency test after any model conversion (HF → CT2)** — silent
   conversion errors are the hardest to find (the NLLB flores language-code tokens missing
   from sentencepiece).
7. **Shared inference objects need an inference lock when `asr_concurrency > 1`**
   (mirror `DeepFilterEnhancer._lock`).
8. **Fine-tuning must not break word timestamps** — jianpu, lyric alignment and intonation
   scoring all depend on them. Every Breeze training run must execute the speech regression
   set; roll back on a relative regression of ≥5 %.
