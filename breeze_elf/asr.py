from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings, get_settings

LOGGER = logging.getLogger(__name__)

# Repo root (…/breeze_elf/asr.py → repo). Mirrors main.py / doctor.py so a relative
# ``asr_breeze_model`` resolves CWD-independently to the on-disk CTranslate2 dir.
ROOT_DIR = Path(__file__).resolve().parent.parent

TRADITIONAL_CHINESE_PROMPT = (
    "以下是台灣繁體中文語音轉寫，內容可能包含國語、台灣用語、英文詞彙與標點。"
)
# Sentinel languages that mean "let Whisper detect freely" (no restriction).
_AUTO_LANGUAGES = {"", "auto"}


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str
    duration_ms: int
    backend: str
    device: str
    no_speech_prob: float | None = None
    words: tuple[WordTiming, ...] = ()


@dataclass(frozen=True)
class FileSegment:
    """One segment of a whole-file (batched) transcription. ``start``/``end`` and
    the per-word timings are absolute seconds from the start of the file."""

    text: str
    start: float
    end: float
    words: tuple[WordTiming, ...] = ()


@dataclass(frozen=True)
class FileTranscription:
    segments: tuple[FileSegment, ...]
    language: str
    duration_ms: int
    backend: str
    device: str


class ASREngine(Protocol):
    backend: str
    device: str
    model_name: str
    compute_type: str

    def load(self) -> None:
        ...

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str,
        *,
        languages: Sequence[str] | None = None,
        prompt_terms: Sequence[str] | None = None,
        context: str | None = None,
    ) -> ASRResult:
        ...


class MockASR:
    backend = "mock"
    device = "none"
    model_name = "mock"
    compute_type = "none"

    def __init__(self) -> None:
        self._count = 0

    def load(self) -> None:
        return None

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str,
        *,
        languages: Sequence[str] | None = None,
        prompt_terms: Sequence[str] | None = None,
        context: str | None = None,
    ) -> ASRResult:
        del languages, prompt_terms, context  # mock ignores restriction / 詞庫 / 脈絡
        self._count += 1
        seconds = samples.size / sample_rate if sample_rate else 0
        text = f"測試字幕 {self._count} ({seconds:.1f}s)"
        return ASRResult(
            text=text,
            language=language,
            duration_ms=0,
            backend=self.backend,
            device=self.device,
            words=_even_word_timings(text, float(seconds)),
        )

    def transcribe_file(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str,
        *,
        languages: Sequence[str] | None = None,
        prompt_terms: Sequence[str] | None = None,
        context: str | None = None,
        batch_size: int = 16,
    ) -> FileTranscription:
        del languages, prompt_terms, context, batch_size
        seconds = samples.size / sample_rate if sample_rate else 0
        text = f"批次測試 ({seconds:.1f}s)"
        segment = FileSegment(
            text=text, start=0.0, end=float(seconds), words=_even_word_timings(text, float(seconds))
        )
        return FileTranscription(
            segments=(segment,),
            language=language,
            duration_ms=0,
            backend=self.backend,
            device=self.device,
        )


class FasterWhisperASR:
    backend = "faster-whisper"

    def __init__(
        self,
        model_name: str = "medium",
        device_preference: str = "auto",
        *,
        load_ref: str | None = None,
    ) -> None:
        # ``model_name`` is the display/identity string the 模型與演算法 switcher matches
        # against its presets (e.g. the *relative* breeze dir). ``load_ref`` is what
        # faster-whisper actually loads — an absolute CT2 path for offline-first breeze,
        # so a relative-but-present dir never falls through to a HuggingFace 404.
        self.model_name = model_name
        self._load_ref = load_ref or model_name
        self.device_preference = device_preference
        self.device = "unloaded"
        self.compute_type = "unloaded"
        self._model = None
        self._batched = None
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
                        self._load_ref,
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

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str,
        *,
        languages: Sequence[str] | None = None,
        prompt_terms: Sequence[str] | None = None,
        context: str | None = None,
    ) -> ASRResult:
        self.load()
        assert self._model is not None

        del sample_rate  # the model uses the fixed 16 kHz it was warmed for
        audio = samples.astype(np.float32, copy=False)
        # The allowed set: explicit ``languages`` (the 多選語言 picker) wins,
        # else the single back-compat ``language``. ``auto``/empty means free
        # detection — loaded files (music / non-Chinese) ask for it because
        # forcing Traditional Chinese on a melody produces hallucinated gibberish.
        allowed = _normalize_languages(languages, language)
        auto_detect = not allowed or any(code in _AUTO_LANGUAGES for code in allowed)

        if auto_detect:
            whisper_language: str | None = None
        elif len(allowed) == 1:
            whisper_language = allowed[0]
        else:
            # Strict restriction: detect, then force the highest-probability
            # language that is actually in the allowed set so a stray foreign
            # segment is recognised as the primary language, not a 4th tongue.
            whisper_language = self._resolve_language(audio, allowed)

        # In free-detect mode we drop the prompt entirely so it cannot bias the
        # detection (back toward zh, or toward 詞庫 terms / recent 脈絡 on a melody).
        initial_prompt = None if auto_detect else _build_prompt(allowed, prompt_terms, context)
        # OpenCC simplified→traditional only matters for Chinese output; skip it
        # for a pure non-zh restriction so shared CJK kanji aren't mangled.
        converter = self._converter if (auto_detect or "zh" in allowed) else None

        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=whisper_language,
            task="transcribe",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=True,
        )
        segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        text = _to_traditional(text, converter)
        detected_language = (
            getattr(info, "language", None)
            or whisper_language
            or (allowed[0] if allowed else "auto")
        )
        return ASRResult(
            text=text,
            language=detected_language,
            duration_ms=round((time.perf_counter() - started) * 1000),
            backend=self.backend,
            device=f"{self.device}/{self.compute_type}",
            no_speech_prob=_max_segment_float(segment_list, "no_speech_prob"),
            words=self._collect_words(segment_list, converter),
        )

    def transcribe_file(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str,
        *,
        languages: Sequence[str] | None = None,
        prompt_terms: Sequence[str] | None = None,
        context: str | None = None,
        batch_size: int = 16,
    ) -> FileTranscription:
        """Transcribe a whole recording in one batched pass (3-4x throughput on long
        files). ``BatchedInferencePipeline`` runs its own internal VAD and batches the
        chunks, so — unlike the live per-utterance path — it needs the entire audio at
        once; word timings come back absolute to the file start. Language / prompt /
        OpenCC handling mirrors :meth:`transcribe`."""
        self.load()
        assert self._model is not None

        del sample_rate
        audio = samples.astype(np.float32, copy=False)
        allowed = _normalize_languages(languages, language)
        auto_detect = not allowed or any(code in _AUTO_LANGUAGES for code in allowed)
        if auto_detect:
            whisper_language: str | None = None
        elif len(allowed) == 1:
            whisper_language = allowed[0]
        else:
            whisper_language = self._resolve_language(audio, allowed)
        initial_prompt = None if auto_detect else _build_prompt(allowed, prompt_terms, context)
        converter = self._converter if (auto_detect or "zh" in allowed) else None

        started = time.perf_counter()
        segments, info = self._ensure_batched().transcribe(
            audio,
            batch_size=max(1, batch_size),
            language=whisper_language,
            task="transcribe",
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )
        file_segments: list[FileSegment] = []
        for segment in segments:
            text = _to_traditional(segment.text.strip(), converter)
            if not text:
                continue
            start = float(getattr(segment, "start", 0.0) or 0.0)
            end = float(getattr(segment, "end", start) or start)
            file_segments.append(
                FileSegment(
                    text=text,
                    start=start,
                    end=max(start, end),
                    words=self._collect_words([segment], converter),
                )
            )
        detected_language = (
            getattr(info, "language", None)
            or whisper_language
            or (allowed[0] if allowed else "auto")
        )
        return FileTranscription(
            segments=tuple(file_segments),
            language=detected_language,
            duration_ms=round((time.perf_counter() - started) * 1000),
            backend=self.backend,
            device=f"{self.device}/{self.compute_type}",
        )

    def _ensure_batched(self):
        if self._batched is None:
            from faster_whisper import BatchedInferencePipeline

            self._batched = BatchedInferencePipeline(self._model)
        return self._batched

    def _resolve_language(self, audio: np.ndarray, allowed: tuple[str, ...]) -> str:
        """Pick the most likely language *within* ``allowed``; fall back to the
        primary (first) language when detection is unavailable or finds nothing
        in the set, so recognition never escapes the user's chosen languages."""
        detect = getattr(self._model, "detect_language", None)
        if detect is None:
            return allowed[0]
        try:
            _, _, all_probs = detect(audio)
        except Exception:  # pragma: no cover - depends on model internals
            return allowed[0]
        allowed_set = set(allowed)
        for code, _prob in all_probs or []:
            if code in allowed_set:
                return code
        return allowed[0]

    def _collect_words(self, segments, converter) -> tuple[WordTiming, ...]:
        words: list[WordTiming] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                text = _to_traditional((getattr(word, "word", "") or "").strip(), converter)
                if not text:
                    continue
                try:
                    start = float(word.start)
                    end = float(word.end)
                except (TypeError, ValueError):
                    continue
                words.append(WordTiming(word=text, start=start, end=max(start, end)))
        return tuple(words)

    def _device_candidates(self) -> list[tuple[str, str]]:
        preference = self.device_preference.strip().lower()
        if preference == "auto":
            return [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]
        if preference == "cuda":
            return [("cuda", "float16"), ("cuda", "int8_float16")]
        if preference == "cpu":
            return [("cpu", "int8")]
        return [(preference, os.getenv("BREEZE_ASR_COMPUTE_TYPE", "int8"))]


def _normalize_languages(
    languages: Sequence[str] | None, fallback: str
) -> tuple[str, ...]:
    """The de-duplicated, lower-cased allowed languages; falls back to the
    single ``fallback`` code so callers using the old positional ``language``
    argument keep working unchanged."""
    items: list[str] = []
    for value in languages or ():
        code = (value or "").strip().lower()
        if code and code not in items:
            items.append(code)
    if items:
        return tuple(items)
    code = (fallback or "").strip().lower()
    return (code,) if code else ()


def _build_prompt(
    allowed: tuple[str, ...],
    prompt_terms: Sequence[str] | None,
    context: str | None = None,
) -> str | None:
    """The Whisper ``initial_prompt``: the recent transcript ``context`` (so proper
    nouns stay consistent across segments), then the Traditional-Chinese base prompt
    when zh is in scope, then the 慣用詞庫 terms biasing the user's preferred
    spellings. ``None`` when there is nothing to bias with.

    Context is placed *first* on purpose: faster-whisper keeps only the tail (~223
    tokens) of an over-long ``initial_prompt``, so the higher-value base prompt +
    glossary must sit last to survive; the softer context is what gets evicted."""
    parts: list[str] = []
    tail = " ".join((context or "").split())
    if tail:
        parts.append(tail)
    if "zh" in allowed:
        parts.append(TRADITIONAL_CHINESE_PROMPT)
    terms = [term.strip() for term in (prompt_terms or ()) if term and term.strip()]
    if terms:
        parts.append("、".join(dict.fromkeys(terms)))
    return " ".join(parts) if parts else None


def resolve_breeze_model_dir(settings: Settings) -> Path:
    """Absolute path to the local Breeze CTranslate2 directory (offline-first).

    faster-whisper can only load a size string, a HF id, or a *local* CT2 dir, so the
    ``breeze`` preset is a directory on disk — resolved here CWD-independently."""
    path = Path(settings.asr_breeze_model).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def build_asr_from_env(settings: Settings | None = None) -> ASREngine:
    settings = settings or get_settings()
    if settings.asr_provider.strip().lower() == "mock":
        return MockASR()
    if settings.asr_model.strip() == "breeze":
        # Preset sentinel → local Breeze CT2 dir. Load via the absolute path but keep
        # the relative preset value as identity so the 模型與演算法 switcher highlights
        # "breeze". Missing dir → fall back to Whisper medium so a fresh clone (without
        # the 2.9 GB model) still boots instead of crashing / hitting HuggingFace.
        breeze_dir = resolve_breeze_model_dir(settings)
        if breeze_dir.exists():
            return FasterWhisperASR(
                settings.asr_breeze_model,
                settings.asr_device,
                load_ref=str(breeze_dir),
            )
        LOGGER.warning(
            "Breeze model dir not found (%s); falling back to Whisper medium. "
            "Place a CTranslate2 model there or set BREEZE_ASR_BREEZE_MODEL.",
            breeze_dir,
        )
        return FasterWhisperASR("medium", settings.asr_device)
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


def _even_word_timings(text: str, seconds: float) -> tuple[WordTiming, ...]:
    chars = [char for char in text if not char.isspace()]
    if not chars or seconds <= 0:
        return ()
    step = seconds / len(chars)
    return tuple(
        WordTiming(
            word=char,
            start=round(index * step, 3),
            end=round((index + 1) * step, 3),
        )
        for index, char in enumerate(chars)
    )


def _max_segment_float(segments, attr: str) -> float | None:
    values: list[float] = []
    for segment in segments:
        value = getattr(segment, attr, None)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None
