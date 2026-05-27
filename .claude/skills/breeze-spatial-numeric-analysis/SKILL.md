---
name: breeze-spatial-numeric-analysis
description: Analyze Breeze Elf spatial, audio, timing, and numerical behavior. Use when tuning PCM conversion, RMS thresholds, VAD utterance segmentation, overlapping windows, queue timing, benchmarks, sample-rate math, future spatial/audio geometry, or numeric regression tests.
---

# Breeze Spatial Numeric Analysis

## Workflow

1. State units first: samples, seconds, milliseconds, RMS, bytes, queue depth, and realtime factor.
2. Keep pure numeric behavior isolated in `breeze_elf/audio.py` or `breeze_elf/benchmark.py` before wiring it into `breeze_elf/main.py`.
3. Preserve audio invariants:
   - PCM16LE bytes convert to clipped float32 samples.
   - Fixed windows advance by `window_samples - overlap_samples`.
   - Utterance mode preserves pre-roll, trailing silence, max segment duration, and final flush behavior.
4. Use synthetic input for repeatable tests and benchmark comparisons. Avoid requiring a live microphone or Whisper model for numeric validation.
5. For future spatial analysis, define coordinate frame, units, sampling geometry, and error metrics before implementation.

## Guardrails

- Do not tune thresholds from one anecdotal clip only; record the expected failure mode and compare before/after metrics.
- Keep ASR hallucination filtering tied to both text and energy/no-speech probability.
- Keep queue timing changes coordinated with ASR queue telemetry.

## Validation

```powershell
uv run python -m unittest tests.test_audio tests.test_asr_queue tests.test_main
npm.cmd run bench -- --segmenter vad
npm.cmd run bench -- --segmenter window
npm.cmd run bench:mock
```

## Reference

Read `references/numeric-contracts.md` for formulas, invariants, and benchmark expectations.
