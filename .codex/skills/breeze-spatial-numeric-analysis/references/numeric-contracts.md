# Breeze Elf Numeric Contracts

## Audio Math

- PCM input is little-endian signed 16-bit audio. Odd trailing bytes are dropped.
- Float samples are `pcm / 32768.0`, clipped to `[-1.0, 1.0]`.
- RMS is `sqrt(mean(samples * samples))` over float samples.
- Fixed windows emit when buffered samples reach `window_samples`; each emission drops `step_samples`.
- Utterance segmentation processes frames, starts on RMS threshold, includes pre-roll, ends after trailing silence, and flushes active speech on stop.

## Timing

- Browser chunks default to `AUDIO_CHUNK_MS = 250`.
- Backend sample rate is expected to be `16000`.
- Benchmarks report segmentation time and realtime factor.

## Regression Signals

- Window count, start/end seconds, buffered seconds, RMS, `is_speech`, and `kind`.
- Queue wait milliseconds, ASR queue depth, dropped windows, and filtered silence results.
- For spatial extensions, define coordinate frame, microphone geometry, and error units before coding.
