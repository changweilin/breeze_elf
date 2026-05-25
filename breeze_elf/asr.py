from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Settings, get_settings


TRADITIONAL_CHINESE_PROMPT = (
    "以下是台灣繁體中文語音轉寫，內容可能包含國語、台灣用語、英文詞彙與標點。"
)


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str
    duration_ms: int
    backend: str
    device: str


class ASREngine(Protocol):
    backend: str
    device: str

    def load(self) -> None:
        ...

    def transcribe(self, samples: np.ndarray, sample_rate: int, language: str) -> ASRResult:
        ...


class MockASR:
    backend = "mock"
    device = "none"

    def __init__(self) -> None:
        self._count = 0

    def load(self) -> None:
        return None

    def transcribe(self, samples: np.ndarray, sample_rate: int, language: str) -> ASRResult:
        self._count += 1
        seconds = samples.size / sample_rate if sample_rate else 0
        return ASRResult(
            text=f"測試字幕 {self._count} ({seconds:.1f}s)",
            language=language,
            duration_ms=0,
            backend=self.backend,
            device=self.device,
        )


class FasterWhisperASR:
    backend = "faster-whisper"

    def __init__(self, model_name: str = "medium", device_preference: str = "auto") -> None:
        self.model_name = model_name
        self.device_preference = device_preference
        self.device = "unloaded"
        self.compute_type = "unloaded"
        self._model = None
        self._converter = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                from faster_whisper import WhisperModel
            except Exception as exc:  # pragma: no cover - depends on installed extras
                raise RuntimeError(
                    "faster-whisper is not installed. Run `uv sync`, or set "
                    "BREEZE_ASR_PROVIDER=mock for UI-only development."
                ) from exc

            errors: list[str] = []
            for device, compute_type in self._device_candidates():
                try:
                    self._model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute_type,
                    )
                    self.device = device
                    self.compute_type = compute_type
                    self._converter = _make_opencc_converter()
                    return
                except Exception as exc:  # pragma: no cover - hardware dependent
                    errors.append(f"{device}/{compute_type}: {exc}")

            detail = "; ".join(errors) if errors else "no candidates tried"
            raise RuntimeError(f"failed to load Whisper model {self.model_name!r}: {detail}")

    def transcribe(self, samples: np.ndarray, sample_rate: int, language: str) -> ASRResult:
        self.load()
        assert self._model is not None

        started = time.perf_counter()
        segments, info = self._model.transcribe(
            samples.astype(np.float32, copy=False),
            language=language,
            task="transcribe",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=TRADITIONAL_CHINESE_PROMPT,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        text = _to_traditional(text, self._converter)
        detected_language = getattr(info, "language", language) or language
        return ASRResult(
            text=text,
            language=detected_language,
            duration_ms=round((time.perf_counter() - started) * 1000),
            backend=self.backend,
            device=f"{self.device}/{self.compute_type}",
        )

    def _device_candidates(self) -> list[tuple[str, str]]:
        preference = self.device_preference.strip().lower()
        if preference == "auto":
            return [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]
        if preference == "cuda":
            return [("cuda", "float16"), ("cuda", "int8_float16")]
        if preference == "cpu":
            return [("cpu", "int8")]
        return [(preference, os.getenv("BREEZE_ASR_COMPUTE_TYPE", "int8"))]


def build_asr_from_env(settings: Settings | None = None) -> ASREngine:
    settings = settings or get_settings()
    if settings.asr_provider.strip().lower() == "mock":
        return MockASR()
    return FasterWhisperASR(settings.asr_model, settings.asr_device)


def _make_opencc_converter():
    try:
        from opencc import OpenCC
    except Exception:
        return None

    for config_name in ("s2twp", "s2tw", "s2t"):
        try:
            return OpenCC(config_name)
        except Exception:
            continue
    return None


def _to_traditional(text: str, converter) -> str:
    if not text or converter is None:
        return text
    try:
        return converter.convert(text)
    except Exception:
        return text

