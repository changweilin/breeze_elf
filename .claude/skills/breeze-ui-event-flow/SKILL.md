---
name: breeze-ui-event-flow
description: Maintain Breeze Elf frontend UI logic and browser runtime events. Use when changing microphone capture, AudioWorklet streaming, WebSocket lifecycle, button states, theme persistence, transcript actions, demo mode, service worker behavior, or browser smoke verification.
---

# Breeze UI Event Flow

## Workflow

1. Trace the state machine in `web/app.js` before editing:
   - `start`, `stop`, and `cleanupAudio`
   - `state.ws`, `state.stream`, `state.audioContext`, `state.worklet`
   - transcript actions: copy, download, save, clear
   - demo timers and `DEMO_MODE`
2. Keep UI state transitions paired. Any path that starts audio, timers, demo playback, or WebSocket traffic must have a cleanup path.
3. Preserve secure-context behavior. Real microphone capture should stay blocked outside HTTPS, localhost, or `127.0.0.1`.
4. When changing WebSocket event shapes, update `breeze_elf/protocol.py`, `breeze_elf/main.py`, and tests together.
5. When static assets change, consider the service worker cache name in `web/service-worker.js`.

## Guardrails

- Do not let demo mode call microphone, WebSocket, or remote transcript save APIs.
- Do not leave buttons enabled while a stop/drain operation is in progress.
- Keep `AudioWorkletNode` chunking aligned with `AUDIO_CHUNK_MS` and backend sample-rate expectations.
- Keep text changes coordinated with `$breeze-i18n-localization`.

## Validation

```powershell
npm.cmd run check:web
npm.cmd run test
```

For runtime changes, run `npm.cmd run dev:mock` and smoke `/`, `/?demo=1`, and start/stop/clear/copy/download/save transitions.

## Reference

Read `references/event-map.md` for the current UI event and cleanup map.
