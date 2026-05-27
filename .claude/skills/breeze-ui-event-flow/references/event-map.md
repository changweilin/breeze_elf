# Breeze Elf UI Event Map

- `start()`: chooses demo or live mode, checks secure context and media support, opens WebSocket, requests microphone, starts `AudioContext`, loads `audio-worklet.js`, sends `start`.
- `stop()`: stops demo or live audio, sends `stop`, disables controls while backend drains.
- `cleanupAudio()`: disconnects worklet/source/silence, stops tracks, closes audio context, resets level meter.
- `handleServerMessage()`: handles `ready`, `partial`, `final`, `stats`, and `error`.
- Transcript actions: `clear`, `copy`, `download`, and remote `save` all depend on `state.transcript` and demo mode.
- Theme flow: `preferredTheme()`, `applyTheme()`, `toggleTheme()`, and system change listener keep DOM, metadata, storage, and labels in sync.
- Demo mode: selected by `?demo`, `mode=demo`, `file:`, or `*.github.io`; must not call microphone, WebSocket, or remote save.

Runtime edits should be checked with syntax validation plus at least one real browser smoke path.
