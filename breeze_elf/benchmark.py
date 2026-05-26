from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

import numpy as np

from .asr import build_asr_from_env
from .audio import AudioUtteranceBuffer, AudioWindowBuffer
from .config import get_settings


def _synthetic_pcm16(sample_rate: int, seconds: float, frequency: float, amplitude: float) -> bytes:
    total_samples = max(1, round(sample_rate * seconds))
    timeline = np.arange(total_samples, dtype=np.float32) / sample_rate
    samples = amplitude * np.sin(2 * math.pi * frequency * timeline)
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    return pcm.tobytes()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Breeze Elf local audio pipeline.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Synthetic audio duration.")
    parser.add_argument("--frequency", type=float, default=440.0, help="Synthetic tone frequency.")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.2,
        help="Synthetic tone amplitude from 0 to 1.",
    )
    parser.add_argument("--segmenter", choices=("vad", "window"), help="Override audio segmenter.")
    parser.add_argument(
        "--provider",
        choices=("mock", "faster-whisper"),
        help="Override ASR provider.",
    )
    parser.add_argument("--transcribe", action="store_true", help="Also run ASR on speech windows.")
    parser.add_argument(
        "--max-asr-windows",
        type=int,
        default=3,
        help="Maximum windows to transcribe.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    if args.provider:
        settings = replace(settings, asr_provider=args.provider)
    if args.segmenter:
        settings = replace(settings, segmenter=args.segmenter)

    payload = _synthetic_pcm16(settings.sample_rate, args.seconds, args.frequency, args.amplitude)
    if settings.segmenter == "window":
        buffer = AudioWindowBuffer(
            sample_rate=settings.sample_rate,
            window_seconds=settings.window_seconds,
            overlap_seconds=settings.overlap_seconds,
            rms_threshold=settings.rms_threshold,
        )
    else:
        buffer = AudioUtteranceBuffer(
            sample_rate=settings.sample_rate,
            frame_ms=settings.vad_frame_ms,
            pre_roll_ms=settings.vad_pre_roll_ms,
            end_silence_ms=settings.vad_end_silence_ms,
            max_segment_seconds=settings.vad_max_segment_seconds,
            rms_threshold=settings.rms_threshold,
        )

    started = time.perf_counter()
    windows = buffer.append_pcm16(payload)
    if hasattr(buffer, "flush"):
        windows.extend(buffer.flush())
    segment_ms = round((time.perf_counter() - started) * 1000, 2)
    speech_windows = [window for window in windows if window.is_speech]

    report: dict[str, object] = {
        "sampleRate": settings.sample_rate,
        "durationSeconds": args.seconds,
        "payloadBytes": len(payload),
        "segmenter": settings.segmenter,
        "windowSeconds": settings.window_seconds,
        "overlapSeconds": settings.overlap_seconds,
        "vadFrameMs": settings.vad_frame_ms,
        "vadEndSilenceMs": settings.vad_end_silence_ms,
        "rmsThreshold": settings.rms_threshold,
        "windows": len(windows),
        "speechWindows": len(speech_windows),
        "segmentMs": segment_ms,
        "segmentRealtimeFactor": round(segment_ms / max(args.seconds * 1000, 1), 4),
    }

    if args.transcribe:
        engine = build_asr_from_env(settings)
        engine.load()
        asr_started = time.perf_counter()
        transcribed = 0
        texts: list[str] = []
        for window in speech_windows[: max(0, args.max_asr_windows)]:
            result = engine.transcribe(window.samples, settings.sample_rate, settings.language)
            transcribed += 1
            if result.text:
                texts.append(result.text)
        asr_ms = round((time.perf_counter() - asr_started) * 1000, 2)
        report.update(
            {
                "asrProvider": engine.backend,
                "asrDevice": engine.device,
                "asrWindows": transcribed,
                "asrMs": asr_ms,
                "asrMsPerWindow": round(asr_ms / transcribed, 2) if transcribed else 0,
                "previewText": " ".join(texts)[:160],
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
