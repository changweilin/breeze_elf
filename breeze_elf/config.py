from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _choice_env(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in choices else default


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8788
    sample_rate: int = 16_000
    window_seconds: float = 2.0
    overlap_seconds: float = 0.5
    rms_threshold: float = 0.008
    audio_preprocess: str = "natural"
    max_queue_windows: int = 4
    segmenter: str = "vad"
    vad_frame_ms: int = 100
    vad_pre_roll_ms: int = 300
    vad_end_silence_ms: int = 700
    vad_max_segment_seconds: float = 12.0
    language: str = "zh"
    asr_model: str = "medium"
    asr_device: str = "auto"
    asr_provider: str = "faster-whisper"
    asr_load_on_startup: bool = True
    asr_concurrency: int = 1
    asr_no_speech_prob_threshold: float = 0.6
    asr_hallucination_rms_threshold: float = 0.02
    stop_drain_timeout_seconds: float = 60.0
    remote_storage_dir: str = "remote_transcripts"


def get_settings() -> Settings:
    return Settings(
        host=os.getenv("BREEZE_HOST", "127.0.0.1"),
        port=_int_env("BREEZE_PORT", 8788),
        sample_rate=_int_env("BREEZE_SAMPLE_RATE", 16_000),
        window_seconds=_float_env("BREEZE_WINDOW_SECONDS", 2.0),
        overlap_seconds=_float_env("BREEZE_OVERLAP_SECONDS", 0.5),
        rms_threshold=_float_env("BREEZE_RMS_THRESHOLD", 0.008),
        audio_preprocess=_choice_env(
            "BREEZE_AUDIO_PREPROCESS",
            "natural",
            {"off", "natural", "speech"},
        ),
        max_queue_windows=max(1, _int_env("BREEZE_MAX_QUEUE_WINDOWS", 4)),
        segmenter=os.getenv("BREEZE_SEGMENTER", "vad").strip().lower(),
        vad_frame_ms=max(1, _int_env("BREEZE_VAD_FRAME_MS", 100)),
        vad_pre_roll_ms=max(0, _int_env("BREEZE_VAD_PRE_ROLL_MS", 300)),
        vad_end_silence_ms=max(1, _int_env("BREEZE_VAD_END_SILENCE_MS", 700)),
        vad_max_segment_seconds=max(0.1, _float_env("BREEZE_VAD_MAX_SEGMENT_SECONDS", 12.0)),
        language=os.getenv("BREEZE_LANGUAGE", "zh"),
        asr_model=os.getenv("BREEZE_ASR_MODEL", "medium"),
        asr_device=os.getenv("BREEZE_ASR_DEVICE", "auto"),
        asr_provider=os.getenv("BREEZE_ASR_PROVIDER", "faster-whisper"),
        asr_load_on_startup=_bool_env("BREEZE_ASR_LOAD_ON_STARTUP", True),
        asr_concurrency=max(1, _int_env("BREEZE_ASR_CONCURRENCY", 1)),
        asr_no_speech_prob_threshold=_float_env("BREEZE_ASR_NO_SPEECH_PROB_THRESHOLD", 0.6),
        asr_hallucination_rms_threshold=_float_env("BREEZE_ASR_HALLUCINATION_RMS_THRESHOLD", 0.02),
        stop_drain_timeout_seconds=max(0.1, _float_env("BREEZE_STOP_DRAIN_TIMEOUT_SECONDS", 60.0)),
        remote_storage_dir=os.getenv("BREEZE_REMOTE_STORAGE_DIR", "remote_transcripts"),
    )
