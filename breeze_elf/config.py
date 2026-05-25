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


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("BREEZE_HOST", "127.0.0.1")
    port: int = _int_env("BREEZE_PORT", 8788)
    sample_rate: int = _int_env("BREEZE_SAMPLE_RATE", 16_000)
    window_seconds: float = _float_env("BREEZE_WINDOW_SECONDS", 2.0)
    overlap_seconds: float = _float_env("BREEZE_OVERLAP_SECONDS", 0.5)
    rms_threshold: float = _float_env("BREEZE_RMS_THRESHOLD", 0.008)
    max_queue_windows: int = _int_env("BREEZE_MAX_QUEUE_WINDOWS", 4)
    language: str = os.getenv("BREEZE_LANGUAGE", "zh")
    asr_model: str = os.getenv("BREEZE_ASR_MODEL", "medium")
    asr_device: str = os.getenv("BREEZE_ASR_DEVICE", "auto")
    asr_provider: str = os.getenv("BREEZE_ASR_PROVIDER", "faster-whisper")
    asr_load_on_startup: bool = _bool_env("BREEZE_ASR_LOAD_ON_STARTUP", True)


def get_settings() -> Settings:
    return Settings()

