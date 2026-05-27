# Breeze Elf Market Science Framework

## Current Product Facts

Breeze Elf streams a phone microphone over a private Tailscale tailnet to a local FastAPI host and transcribes locally with `faster-whisper`. It includes a static demo mode for GitHub Pages and remote transcript saving back to the host.

## Segments To Test

- Local-first transcription users who do not want cloud microphone upload.
- Developers or researchers who need private ad hoc speech capture.
- Bilingual Traditional Chinese users who care about local ASR and text cleanup.
- Teams already using Tailscale for private device access.

## Metric Templates

- Activation: user reaches first final transcript.
- Quality: useful final segments per minute, duplicate overlap rate, hallucination-filter rate.
- Reliability: connection failures, dropped client chunks, dropped backend windows, queue depth.
- Performance: ASR milliseconds per segment and realtime factor.
- Retention: repeat sessions and saved transcripts.

## Privacy Rule

Prefer local-only metrics and opt-in aggregate export. Do not collect transcript content by default.
