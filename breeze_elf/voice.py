"""Voice conversion + cloning engines for the 變聲 page.

This mirrors the ASR engine layout in :mod:`breeze_elf.asr`: a small
``VoiceEngine`` protocol, a dependency-free ``MockVoiceEngine`` that performs a
genuine (if simple) DSP transform so the UI works out of the box, and a lazily
loaded ``OpenVoiceEngine`` that uses OpenVoice v2 + MeloTTS for real speaker
cloning. The provider is chosen by ``BREEZE_VOICE_PROVIDER``; the default stays
``mock`` so the server runs without the multi-GB checkpoints installed.

Every engine speaks the same currency: mono ``float32`` samples in ``[-1, 1]``.
``extract_embedding`` turns A's reference audio into an opaque ``bytes`` blob
that the storage layer persists; ``convert`` re-voices B's speech as A and
``synthesize`` reads text in A's voice. ``load`` accepts a progress callback so
the frontend can show a real progress bar while a heavy model warms up.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import summarize_pitch
from .config import Settings, get_settings

# (fraction in [0, 1], human-readable stage label)
ProgressCallback = Callable[[float, str], None]

_DEFAULT_VOICE_HZ = 165.0
_MIN_VOICE_HZ = 70.0
_MAX_VOICE_HZ = 400.0
_MAX_SHIFT_SEMITONES = 12.0
_STFT_SIZE = 1024
_STFT_HOP = 256

# Playback speed multiplier bounds for 文字轉換 / 簡譜唱歌. The change is applied
# as a pitch-preserving time-stretch so the voice keeps its pitch and timbre and
# only the tempo changes (speed > 1 → faster/shorter, < 1 → slower/longer).
_MIN_SPEED = 0.5
_MAX_SPEED = 2.0

# Sung-note duration bounds. We keep each note as close to its requested length as
# possible so the sung clip lines up in time with the original recording (only a
# tiny floor to avoid zero-length notes and a generous ceiling for held notes);
# earlier a 0.3 s floor + 2.5 s cap stretched/clipped notes and drifted the timing.
_MIN_NOTE_SECONDS = 0.04
_MAX_NOTE_SECONDS = 8.0

# 簡譜 glide markers (rising / falling portamento between two degrees, e.g. 3↗5).
# Mirrors the constants in :mod:`breeze_elf.audio`; defined here so the singing
# engine can split a combined glide token without importing the private name.
_JIANPU_GLIDE_UP = "↗"
_JIANPU_GLIDE_DOWN = "↘"

# Spectral-envelope (timbre) transfer for the mock engine. The reference voice's
# average log-magnitude spectrum is summarized into a handful of log-spaced bands
# and stored in the embedding; on convert/synthesize the produced audio's own
# envelope is reshaped toward it so the clone takes on the reference's timbre
# (formant / brightness structure), not just its median pitch and loudness. This
# is what makes the mock 聲紋 noticeably closer to the original voice.
_ENVELOPE_BANDS = 28
_ENVELOPE_MIN_HZ = 90.0
_ENVELOPE_MAX_HZ = 7600.0
_ENVELOPE_MAX_GAIN_DB = 9.0


@dataclass(frozen=True)
class VoiceAudio:
    """Mono ``float32`` audio in ``[-1, 1]`` plus its sample rate."""

    samples: np.ndarray
    sample_rate: int


class VoiceEngine(Protocol):
    backend: str
    device: str
    model_name: str

    def load(self, progress: ProgressCallback | None = None) -> None:
        ...

    def extract_embedding(self, samples: np.ndarray, sample_rate: int) -> bytes:
        ...

    def convert(self, samples: np.ndarray, sample_rate: int, embedding: bytes) -> VoiceAudio:
        ...

    def synthesize(
        self,
        text: str,
        language: str,
        embedding: bytes,
        base_hz: float | None = None,
        speed: float = 1.0,
    ) -> VoiceAudio:
        ...


# --------------------------------------------------------------------------- #
# Mock engine (no external models)
# --------------------------------------------------------------------------- #


class MockVoiceEngine:
    """Dependency-free engine: a real but simple pitch/level transform.

    ``extract_embedding`` records A's median pitch and loudness; ``convert``
    pitch-shifts B toward A's pitch with a phase vocoder; ``synthesize`` voices
    text as a formant-ish buzz at A's pitch. Good enough to exercise the whole
    pipeline and demo the feature without downloading any checkpoints.
    """

    backend = "mock"
    device = "cpu"
    model_name = "dsp-mock"

    def __init__(
        self,
        sample_rate: int = 16_000,
        warmup_seconds: float = 0.0,
        use_os_tts: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.warmup_seconds = max(0.0, warmup_seconds)
        self.use_os_tts = use_os_tts
        self._loaded = False

    def load(self, progress: ProgressCallback | None = None) -> None:
        # The mock has nothing to download, but we still walk through staged
        # progress (optionally pacing it) so the model-load progress bar is
        # demonstrable end to end.
        stages = (
            (0.25, "初始化變聲引擎"),
            (0.6, "載入音色轉換器"),
            (0.9, "載入語音合成"),
            (1.0, "完成"),
        )
        slice_seconds = self.warmup_seconds / len(stages) if self.warmup_seconds else 0.0
        for fraction, label in stages:
            if slice_seconds:
                time.sleep(slice_seconds)
            _report(progress, fraction, label)
        self._loaded = True

    def extract_embedding(self, samples: np.ndarray, sample_rate: int) -> bytes:
        mono = _as_float32_mono(samples)
        median_hz = _estimate_median_hz(mono, sample_rate)
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        payload = {
            "version": 2,
            "kind": "mock",
            "medianHz": round(median_hz, 2),
            "rms": round(rms, 5),
            "centroidHz": round(_spectral_centroid(mono, sample_rate), 2),
            # Compact timbre fingerprint (log-magnitude envelope, mean-removed).
            "bands": _spectral_envelope_bands(mono, sample_rate),
        }
        return json.dumps(payload).encode("utf-8")

    def convert(self, samples: np.ndarray, sample_rate: int, embedding: bytes) -> VoiceAudio:
        mono = _as_float32_mono(samples)
        if mono.size == 0:
            return VoiceAudio(samples=mono, sample_rate=sample_rate)

        profile = _load_mock_embedding(embedding)
        source_hz = _estimate_median_hz(mono, sample_rate)
        target_hz = float(profile.get("medianHz") or _DEFAULT_VOICE_HZ)
        semitones = _shift_semitones(source_hz, target_hz)
        shifted = _pitch_shift(mono, semitones)

        # Reshape B's timbre toward A's so the conversion sounds like A, not just
        # B re-pitched. (No-op for old embeddings that have no stored envelope.)
        shifted = _apply_envelope_match(shifted, sample_rate, profile.get("bands"))

        target_rms = float(profile.get("rms") or 0.0)
        shifted = _match_loudness(shifted, target_rms)
        return VoiceAudio(samples=shifted.astype(np.float32, copy=False), sample_rate=sample_rate)

    def synthesize(
        self,
        text: str,
        language: str,
        embedding: bytes,
        base_hz: float | None = None,
        speed: float = 1.0,
    ) -> VoiceAudio:
        del language  # the OS voice picks its own language; the fallback is fixed
        profile = _load_mock_embedding(embedding)
        # An explicit 基音 Hz from the UI overrides the voice's measured pitch and
        # is allowed a wider retune so the requested register is actually reached.
        explicit = bool(base_hz and base_hz > 0)
        measured_hz = float(profile.get("medianHz") or _DEFAULT_VOICE_HZ)
        target_hz = float(base_hz) if explicit else measured_hz
        target_hz = float(min(_MAX_VOICE_HZ, max(_MIN_VOICE_HZ, target_hz)))
        max_shift = 12.0 if explicit else 7.0
        target_rms = float(profile.get("rms") or 0.0)
        bands = profile.get("bands")

        # Prefer the real OS text-to-speech voice (e.g. Microsoft Hanhan, zh-TW)
        # so the result is actual words, then nudge its pitch + timbre toward A.
        spoken = _system_tts(text) if self.use_os_tts else None
        if spoken is not None:
            samples, rate = spoken
            samples = _retune_to_target(samples, rate, target_hz, max_semitones=max_shift)
            samples = _apply_envelope_match(samples, rate, bands)
            samples = _match_loudness(samples, target_rms, default_peak=0.92)
            samples = _apply_speed(samples, speed)
            return VoiceAudio(samples=samples, sample_rate=rate)

        # Fallback: a synthetic vowel voice (no OS TTS available).
        samples = _synthesize_voice(text, self.sample_rate, target_hz)
        samples = _apply_envelope_match(samples, self.sample_rate, bands)
        samples = _match_loudness(samples, target_rms, default_peak=0.6)
        samples = _apply_speed(samples, speed)
        return VoiceAudio(samples=samples, sample_rate=self.sample_rate)

    def synthesize_song(
        self,
        notes: list[dict],
        tonic_hz: float,
        embedding: bytes,
        use_measured_hz: bool = False,
        speed: float = 1.0,
        target_median_hz: float = 0.0,
    ) -> VoiceAudio:
        profile = _load_mock_embedding(embedding)
        # The original median is the requested 主音 (the recording's pitch); fall
        # back to the voice's own pitch only for resolving relative 簡譜 degrees.
        original_median = float(tonic_hz) if tonic_hz and tonic_hz > 0 else 0.0
        resolve_tonic = original_median or float(profile.get("medianHz") or _DEFAULT_VOICE_HZ)
        resolved = _trim_edge_rests(_resolve_song_notes(notes, resolve_tonic, use_measured_hz))
        if not original_median:
            original_median = _median_voiced_freq(resolved)  # supply it from the notes
        target_median = float(target_median_hz) or float(profile.get("medianHz") or 0.0)
        resolved = _correct_song_register(resolved, original_median, target_median)
        target_rms = float(profile.get("rms") or 0.0)
        # Real words in the OS voice (the same voice 文字轉換 uses), pitched to
        # the melody; falls back to the synthetic vowel voice when unavailable.
        samples, rate, is_real = _song_base(resolved, self.use_os_tts, self.sample_rate)
        samples = _match_loudness(samples, target_rms, default_peak=0.92 if is_real else 0.85)
        samples = _apply_speed(samples, speed)
        return VoiceAudio(samples=samples, sample_rate=rate)


# --------------------------------------------------------------------------- #
# OpenVoice v2 engine (lazy, optional)
# --------------------------------------------------------------------------- #


class OpenVoiceEngine:
    """Real speaker cloning via OpenVoice v2 tone-color conversion.

    The tone-color converter (``torch`` + the v2 ``converter`` checkpoint) does
    the B→A re-voicing and A's embedding extraction. To keep the dependency
    footprint sane we bypass ``openvoice.se_extractor`` (which hard-imports
    ``faster_whisper`` + ``whisper_timestamped``) and call ``extract_se``
    directly, and we stub out ``wavmark`` so no output watermark model is
    needed. Text→A uses MeloTTS, loaded lazily on first use; if MeloTTS is not
    installed, conversion still works and synthesis raises a clear error.
    """

    backend = "openvoice"

    def __init__(
        self,
        checkpoints_dir: str | Path,
        language: str = "zh",
        device_preference: str = "auto",
        use_os_tts: bool = True,
    ) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.language = language
        self.device_preference = device_preference
        self.use_os_tts = use_os_tts
        self.device = "unloaded"
        self.model_name = "openvoice-v2"
        self._converter = None
        self._tts = None
        self._source_se = None
        self._speaker_id = None
        self._torch = None
        self._sf = None
        self._lock = threading.Lock()

    def load(self, progress: ProgressCallback | None = None) -> None:
        if self._converter is not None:
            _report(progress, 1.0, "完成")
            return

        with self._lock:
            if self._converter is not None:
                _report(progress, 1.0, "完成")
                return

            _report(progress, 0.05, "載入相依套件")
            try:
                import soundfile as sf
                import torch
            except Exception as exc:  # pragma: no cover - depends on optional extras
                raise RuntimeError(
                    "OpenVoice 需要 torch 與 soundfile,請先安裝,或保持 "
                    "BREEZE_VOICE_PROVIDER=mock。"
                ) from exc

            _install_wavmark_stub()
            try:
                from openvoice.api import ToneColorConverter
            except Exception as exc:  # pragma: no cover - depends on optional extras
                raise RuntimeError(
                    "OpenVoice 未安裝。安裝方式:`pip install --no-deps "
                    "git+https://github.com/myshell-ai/OpenVoice.git` 並安裝 "
                    "inflect unidecode eng_to_ipa pypinyin cn2an jieba,"
                    "或保持 BREEZE_VOICE_PROVIDER=mock。"
                ) from exc

            self._torch = torch
            self._sf = sf
            device = self._resolve_device(torch)
            self.device = device

            converter_dir = self.checkpoints_dir / "converter"
            config_path = converter_dir / "config.json"
            checkpoint_path = converter_dir / "checkpoint.pth"
            if not config_path.exists() or not checkpoint_path.exists():
                raise RuntimeError(
                    f"找不到 OpenVoice converter checkpoint ({converter_dir})。"
                    "請下載 v2 checkpoints(見 README)。"
                )

            _report(progress, 0.3, "載入音色轉換器")
            converter = ToneColorConverter(str(config_path), device=device)
            # No watermarking: we stubbed wavmark, so drop the model entirely.
            converter.watermark_model = None
            _report(progress, 0.6, "載入模型權重")
            converter.load_ckpt(str(checkpoint_path))
            self._converter = converter
            _report(progress, 1.0, "完成")

    def extract_embedding(self, samples: np.ndarray, sample_rate: int) -> bytes:
        self.load()
        with _temp_wav(samples, sample_rate) as src_path:
            target_se = self._converter.extract_se([str(src_path)])
        buffer = io.BytesIO()
        self._torch.save(target_se, buffer)
        return buffer.getvalue()

    def convert(self, samples: np.ndarray, sample_rate: int, embedding: bytes) -> VoiceAudio:
        self.load()
        target_se = self._torch.load(io.BytesIO(embedding), map_location=self.device)
        with _temp_wav(samples, sample_rate) as src_path, _temp_wav_path() as out_path:
            source_se = self._converter.extract_se([str(src_path)])
            self._converter.convert(
                audio_src_path=str(src_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(out_path),
                message="@BreezeElf",
            )
            data, out_rate = self._sf.read(str(out_path), dtype="float32", always_2d=False)
        return VoiceAudio(samples=_as_float32_mono(data), sample_rate=int(out_rate))

    def synthesize(
        self,
        text: str,
        language: str,
        embedding: bytes,
        base_hz: float | None = None,
        speed: float = 1.0,
    ) -> VoiceAudio:
        self.load()
        target_se = self._torch.load(io.BytesIO(embedding), map_location=self.device)

        # Preferred path: render real speech with the OS voice (Microsoft Hanhan,
        # zh-TW) and clone it to the target with the tone-color converter. This
        # needs no MeloTTS — only the converter checkpoint that B→A already uses.
        spoken = _system_tts(text) if self.use_os_tts else None
        if spoken is not None:
            base_samples, base_rate = spoken
            with _temp_wav(base_samples, base_rate) as base_path, _temp_wav_path() as out_path:
                source_se = self._converter.extract_se([str(base_path)])
                self._converter.convert(
                    audio_src_path=str(base_path),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=str(out_path),
                    message="@BreezeElf",
                )
                data, out_rate = self._sf.read(str(out_path), dtype="float32", always_2d=False)
        else:
            # Fallback (e.g. non-Windows host): MeloTTS base speaker + convert.
            self._load_tts()
            with _temp_wav_path() as base_path, _temp_wav_path() as out_path:
                self._tts.tts_to_file(text, self._speaker_id, str(base_path), speed=1.0)
                self._converter.convert(
                    audio_src_path=str(base_path),
                    src_se=self._source_se,
                    tgt_se=target_se,
                    output_path=str(out_path),
                    message="@BreezeElf",
                )
                data, out_rate = self._sf.read(str(out_path), dtype="float32", always_2d=False)

        samples = _as_float32_mono(data)
        out_rate = int(out_rate)
        if base_hz and base_hz > 0:
            samples = _retune_to_target(samples, out_rate, float(base_hz), max_semitones=12.0)
        samples = _apply_speed(samples, speed)
        return VoiceAudio(samples=samples, sample_rate=out_rate)

    def synthesize_song(
        self,
        notes: list[dict],
        tonic_hz: float,
        embedding: bytes,
        use_measured_hz: bool = False,
        speed: float = 1.0,
        target_median_hz: float = 0.0,
    ) -> VoiceAudio:
        self.load()
        original_median = float(tonic_hz) if tonic_hz and tonic_hz > 0 else 0.0
        resolve_tonic = original_median or _DEFAULT_VOICE_HZ
        resolved = _trim_edge_rests(_resolve_song_notes(notes, resolve_tonic, use_measured_hz))
        if not original_median:
            original_median = _median_voiced_freq(resolved)
        resolved = _correct_song_register(resolved, original_median, float(target_median_hz))
        # Sing a real-word base in the OS voice so the tone-color converter has
        # actual speech to re-voice (far more natural than converting a synth
        # buzz); falls back to the synthetic voice when no OS voice is available.
        base, base_rate, _ = _song_base(resolved, self.use_os_tts, 22_050)
        target_se = self._torch.load(io.BytesIO(embedding), map_location=self.device)
        with _temp_wav(base, base_rate) as base_path, _temp_wav_path() as out_path:
            source_se = self._converter.extract_se([str(base_path)])
            self._converter.convert(
                audio_src_path=str(base_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(out_path),
                message="@BreezeElf",
            )
            data, out_rate = self._sf.read(str(out_path), dtype="float32", always_2d=False)
        samples = _apply_speed(_as_float32_mono(data), speed)
        return VoiceAudio(samples=samples, sample_rate=int(out_rate))

    def _load_tts(self) -> None:
        if self._tts is not None:
            return
        try:
            from melo.api import TTS
        except Exception as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "文字轉換需安裝 MeloTTS(`pip install git+https://github.com/myshell-ai/"
                "MeloTTS.git` 並下載 base_speakers/ses)。B→A 變聲不需要它。"
            ) from exc

        melo_language = _melo_language(self.language)
        tts = TTS(language=melo_language, device=self.device)
        speaker_ids = tts.hps.data.spk2id
        self._speaker_id = next(iter(speaker_ids.values()))
        speaker_key = next(iter(speaker_ids.keys())).lower().replace("_", "-")
        source_se_path = self.checkpoints_dir / "base_speakers" / "ses" / f"{speaker_key}.pth"
        if not source_se_path.exists():
            source_se_path = (
                self.checkpoints_dir / "base_speakers" / "ses" / f"{melo_language.lower()}.pth"
            )
        if not source_se_path.exists():
            raise RuntimeError(
                f"找不到 MeloTTS 基礎音色 ({source_se_path})。請下載 base_speakers/ses。"
            )
        self._source_se = self._torch.load(str(source_se_path), map_location=self.device)
        self._tts = tts

    def _resolve_device(self, torch) -> str:
        preference = self.device_preference.strip().lower()
        if preference in {"cuda", "cpu", "mps"}:
            return preference
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def build_voice_from_env(settings: Settings | None = None) -> VoiceEngine:
    settings = settings or get_settings()
    provider = settings.voice_provider.strip().lower()
    if provider == "openvoice":
        return OpenVoiceEngine(
            checkpoints_dir=settings.voice_checkpoints_dir,
            language=settings.voice_language,
            device_preference=settings.asr_device,
            use_os_tts=settings.voice_os_tts,
        )
    return MockVoiceEngine(
        sample_rate=settings.voice_sample_rate,
        warmup_seconds=settings.voice_mock_warmup_seconds,
        use_os_tts=settings.voice_os_tts,
    )


# --------------------------------------------------------------------------- #
# DSP helpers
# --------------------------------------------------------------------------- #


def _report(progress: ProgressCallback | None, fraction: float, label: str) -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, fraction)), label)


def _install_wavmark_stub() -> None:
    """Inject a no-op ``wavmark`` module.

    ``ToneColorConverter.__init__`` imports ``wavmark`` and loads a watermark
    model unless told otherwise, but it forwards ``enable_watermark`` to a base
    ``__init__`` that rejects it. Stubbing the module lets the converter build;
    we then set ``watermark_model = None`` so output is never watermarked.
    """
    import sys
    import types

    if "wavmark" in sys.modules:
        return

    class _NoWatermark:
        def to(self, *args, **kwargs):
            return self

    stub = types.ModuleType("wavmark")
    stub.load_model = lambda *args, **kwargs: _NoWatermark()
    sys.modules["wavmark"] = stub


def _as_float32_mono(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=tuple(range(1, array.ndim)))
    return np.ascontiguousarray(array, dtype=np.float32)


def _load_mock_embedding(embedding: bytes) -> dict:
    try:
        data = json.loads(embedding.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return {}
    return data if isinstance(data, dict) else {}


def _estimate_median_hz(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size == 0:
        return _DEFAULT_VOICE_HZ
    summary = summarize_pitch(samples, sample_rate)
    median = summary.median_hz
    if not median or not math.isfinite(median):
        return _DEFAULT_VOICE_HZ
    return float(min(_MAX_VOICE_HZ, max(_MIN_VOICE_HZ, median)))


def _spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size == 0:
        return 0.0
    windowed = samples * np.hanning(samples.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed))
    total = float(spectrum.sum())
    if total <= 1e-8:
        return 0.0
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    return float((freqs * spectrum).sum() / total)


def _shift_semitones(source_hz: float, target_hz: float) -> float:
    if source_hz <= 0 or target_hz <= 0:
        return 0.0
    semitones = 12.0 * math.log2(target_hz / source_hz)
    return float(max(-_MAX_SHIFT_SEMITONES, min(_MAX_SHIFT_SEMITONES, semitones)))


def _match_loudness(
    samples: np.ndarray, target_rms: float, default_peak: float = 0.9
) -> np.ndarray:
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-6:
        return samples
    current_rms = float(np.sqrt(np.mean(np.square(samples))))
    if target_rms > 1e-5 and current_rms > 1e-6:
        gain = target_rms / current_rms
    else:
        gain = default_peak / peak
    scaled = samples * gain
    scaled_peak = float(np.max(np.abs(scaled)))
    if scaled_peak > 0.99:
        scaled = scaled * (0.99 / scaled_peak)
    return scaled.astype(np.float32, copy=False)


def _pitch_shift(samples: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-2 or samples.size < _STFT_SIZE:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    # Lengthen in time without changing pitch, then resample back to the
    # original length: the net effect raises (or lowers) the pitch by ``factor``
    # while preserving duration.
    stretched = _time_stretch(samples, 1.0 / factor)
    return _resample_to_length(stretched, samples.size)


def _time_stretch(samples: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0 or abs(rate - 1.0) < 1e-3:
        return samples
    stft = _stft(samples, _STFT_SIZE, _STFT_HOP)
    stretched = _phase_vocoder(stft, rate, _STFT_HOP)
    return _istft(stretched, _STFT_SIZE, _STFT_HOP)


def _stft(samples: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    padded = np.pad(samples, pad, mode="reflect")
    n_frames = 1 + max(0, (len(padded) - n_fft) // hop)
    frames = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for index in range(n_frames):
        start = index * hop
        segment = padded[start : start + n_fft] * window
        frames[:, index] = np.fft.rfft(segment)
    return frames


def _istft(stft: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = stft.shape[1]
    length = n_fft + hop * (n_frames - 1)
    out = np.zeros(length, dtype=np.float32)
    weight = np.zeros(length, dtype=np.float32)
    for index in range(n_frames):
        start = index * hop
        segment = np.fft.irfft(stft[:, index], n=n_fft).astype(np.float32)
        out[start : start + n_fft] += segment * window
        weight[start : start + n_fft] += window * window
    nonzero = weight > 1e-8
    out[nonzero] /= weight[nonzero]
    pad = n_fft // 2
    return out[pad : len(out) - pad] if pad > 0 else out


def _phase_vocoder(stft: np.ndarray, rate: float, hop: int) -> np.ndarray:
    n_bins, n_frames = stft.shape
    time_steps = np.arange(0, n_frames, rate)
    output = np.zeros((n_bins, len(time_steps)), dtype=np.complex64)
    phase_advance = 2.0 * np.pi * hop * np.arange(n_bins) / (2 * (n_bins - 1))
    padded = np.concatenate([stft, np.zeros((n_bins, 2), dtype=stft.dtype)], axis=1)
    phase = np.angle(padded[:, 0])
    two_pi = 2.0 * np.pi
    for out_index, step in enumerate(time_steps):
        left = int(np.floor(step))
        frac = step - left
        magnitude = (1.0 - frac) * np.abs(padded[:, left]) + frac * np.abs(padded[:, left + 1])
        output[:, out_index] = magnitude * np.exp(1j * phase)
        delta = np.angle(padded[:, left + 1]) - np.angle(padded[:, left]) - phase_advance
        delta -= two_pi * np.round(delta / two_pi)
        phase += phase_advance + delta
    return output


def _resample_to_length(samples: np.ndarray, target_length: int) -> np.ndarray:
    if samples.size == 0 or target_length <= 0:
        return np.zeros(max(0, target_length), dtype=np.float32)
    if samples.size == target_length:
        return samples.astype(np.float32, copy=False)
    source_index = np.linspace(0.0, samples.size - 1, num=target_length)
    resampled = np.interp(source_index, np.arange(samples.size), samples)
    return resampled.astype(np.float32, copy=False)


# A few canonical vowel formants (F1, F2, F3 in Hz). They drive a simple
# source-filter voice: a glottal-ish harmonic source shaped by these resonances.
# It is not real speech, but it sings vowels at A's pitch instead of beeping like
# a dial tone — a much better fallback when no real TTS model is installed.
_VOWEL_FORMANTS = (
    (730.0, 1090.0, 2440.0),  # "a"
    (530.0, 1840.0, 2480.0),  # "e"
    (270.0, 2290.0, 3010.0),  # "i"
    (570.0, 840.0, 2410.0),  # "o"
    (300.0, 870.0, 2240.0),  # "u"
)
_FORMANT_GAINS = (1.0, 0.55, 0.28)
_PAUSE_CHARS = set("，,。.!?！？、；;：:…—")


def _formant_response(freqs: np.ndarray, formants: tuple[float, float, float]) -> np.ndarray:
    """Lorentzian magnitude response of the three formant resonators."""
    response = np.full(freqs.shape, 0.04, dtype=np.float64)
    for center, gain in zip(formants, _FORMANT_GAINS):
        bandwidth = 0.10 * center + 60.0
        response += gain / (1.0 + ((freqs - center) / (bandwidth * 0.5)) ** 2)
    return response


def _syllable_envelope(count: int, sample_rate: int) -> np.ndarray:
    """Fast attack, slower release — voiced, not a symmetric "wah"."""
    env = np.ones(count, dtype=np.float32)
    attack = min(count, max(1, int(sample_rate * 0.02)))
    release = min(max(0, count - attack), max(1, int(sample_rate * 0.06)))
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if release:
        env[count - release :] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return env


def _synthesize_voice(text: str, sample_rate: int, base_hz: float) -> np.ndarray:
    chars = [char for char in text if not char.isspace()] or [" "]
    chars = chars[:200]
    base_hz = float(min(_MAX_VOICE_HZ, max(_MIN_VOICE_HZ, base_hz)))
    seconds_per_char = 0.22
    gap = np.zeros(max(1, int(sample_rate * 0.04)), dtype=np.float32)
    nyquist = sample_rate / 2.0
    max_harmonic = max(1, min(48, int((nyquist * 0.92) / base_hz)))
    harmonics = np.arange(1, max_harmonic + 1, dtype=np.float64)

    pieces: list[np.ndarray] = []
    for position, char in enumerate(chars):
        if char in _PAUSE_CHARS:
            pieces.append(np.zeros(max(1, int(sample_rate * 0.14)), dtype=np.float32))
            continue

        count = max(1, int(sample_rate * seconds_per_char))
        axis = np.arange(count, dtype=np.float64) / sample_rate
        # A gentle, character-keyed pitch contour (vibrato + slight declination)
        # so consecutive syllables differ instead of droning on one note.
        wobble = 0.045 * np.sin(2 * np.pi * (0.7 + (ord(char) % 5) * 0.18) * axis)
        decline = -0.03 * (axis / seconds_per_char)
        f0 = base_hz * (1.0 + wobble + decline)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate

        formants = _VOWEL_FORMANTS[ord(char) % len(_VOWEL_FORMANTS)]
        amplitudes = _formant_response(harmonics * base_hz, formants) / harmonics
        tone = np.zeros(count, dtype=np.float64)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            tone += amplitude * np.sin(harmonic * phase)

        peak = float(np.max(np.abs(tone)))
        if peak > 1e-9:
            tone /= peak
        pieces.append((tone * _syllable_envelope(count, sample_rate)).astype(np.float32))
        if position != len(chars) - 1:
            pieces.append(gap)

    return np.concatenate(pieces).astype(np.float32)


def _clamp_voice_hz(value: float) -> float:
    return float(min(_MAX_VOICE_HZ, max(_MIN_VOICE_HZ, value)))


def _glide_semitones(note: dict) -> tuple[float, float] | None:
    """Read a 簡譜 glide as ``(start, end)`` semitones, or ``None`` if it is not a
    glide. Accepts explicit ``jianpuStart``/``jianpuEnd`` or a combined token like
    ``3↗5`` / ``5↘1``."""
    from .audio import jianpu_to_semitones

    start_token = note.get("jianpuStart")
    end_token = note.get("jianpuEnd")
    token = str(note.get("jianpu") or "")
    for arrow in (_JIANPU_GLIDE_UP, _JIANPU_GLIDE_DOWN):
        if arrow in token:
            left, right = token.split(arrow, 1)
            start_token = start_token or left
            end_token = end_token or right
            break
    start = jianpu_to_semitones(start_token) if start_token else None
    end = jianpu_to_semitones(end_token) if end_token else None
    if start is not None and end is not None and abs(start - end) > 1e-6:
        return start, end
    return None


def _glide_position(note: dict) -> float:
    """The glide's slide position (0–1) the 逐字稿 measured, or 0.5 (middle)."""
    value = note.get("glideMid")
    try:
        position = float(value)
    except (TypeError, ValueError):
        return 0.5
    return position if 0.0 < position < 1.0 else 0.5


def _glide_contour(start_hz: float, end_hz: float, position: float = 0.5) -> list[float]:
    """Portamento curve: hold ``start_hz``, slide, then hold ``end_hz``, with the
    slide centred at ``position`` (a 0–1 fraction of the note).

    The 逐字稿 finds the slide's real position from the pitch's rate of change, so
    placing the move there reproduces the glide faithfully instead of always
    sliding through the middle."""
    p = position if (position and 0.0 < position < 1.0) else 0.5
    p = min(0.85, max(0.15, p))
    width = 0.18  # half-width of the slide window, as a fraction of the note
    points = []
    steps = 9
    for k in range(steps):
        frac = k / (steps - 1)
        if frac <= p - width:
            points.append(start_hz)
        elif frac >= p + width:
            points.append(end_hz)
        else:
            local = (frac - (p - width)) / (2.0 * width)
            points.append(start_hz + (end_hz - start_hz) * local)
    return points


def _resolve_contour(note: dict, tonic_hz: float, use_measured_hz: bool) -> list[float] | None:
    """Resolve a note's pitch *curve* in Hz (≥1 point), or ``None`` for a rest.

    Measured singing follows the recorded contour (``contour`` list, or
    ``startHz``/``endHz``, or a single ``hz``) so the syllable keeps its
    抑揚頓挫; 簡譜 singing follows the degree(s), rendering a glide (``3↗5``) as a
    portamento that holds the start, slides, then settles, and an ordinary degree
    as a single sustained pitch.
    """
    # Frequencies are returned raw (un-clamped): the singable-range handling and
    # any octave register correction happen once in _correct_song_register, so a
    # note an octave out of range can be folded back in tune instead of pinned
    # (clamped) to a wrong pitch.
    measured = note.get("hz")
    measured = float(measured) if measured else 0.0

    if use_measured_hz:
        contour = note.get("contour")
        if isinstance(contour, (list, tuple)):
            points = [float(hz) for hz in contour if hz and float(hz) > 0]
            if points:
                return points
        start, end = note.get("startHz"), note.get("endHz")
        if start and end and float(start) > 0 and float(end) > 0:
            return _glide_contour(float(start), float(end), _glide_position(note))
        if measured > 0:
            return [measured]
        return None

    glide = _glide_semitones(note)
    if glide is not None and tonic_hz > 0:
        return _glide_contour(
            tonic_hz * 2.0 ** (glide[0] / 12.0),
            tonic_hz * 2.0 ** (glide[1] / 12.0),
            _glide_position(note),
        )
    from .audio import jianpu_to_semitones

    semitones = jianpu_to_semitones(note.get("jianpu"))
    if semitones is not None and tonic_hz > 0:
        return [tonic_hz * 2.0 ** (semitones / 12.0)]
    if measured > 0:
        return [measured]
    return None


def _resolve_song_notes(
    notes: list[dict], tonic_hz: float, use_measured_hz: bool
) -> list[dict]:
    """Turn UI notes into resolved ``{char, freq, contour, duration, kind,
    intensity}`` notes.

    ``kind`` is ``voiced`` (pitched, follows ``contour``), ``breath`` (an
    unvoiced 氣音 with energy but no pitch), or ``rest`` (silence). ``contour`` is
    the pitch curve in Hz; ``freq`` is its first point for callers that want a
    single pitch.
    """
    resolved: list[dict] = []
    for note in notes:
        char = str(note.get("char") or "").strip()
        duration = note.get("durationSeconds")
        kind = str(note.get("kind") or "").strip().lower()

        if kind == "breath":
            resolved.append(
                {
                    "char": char or "h",
                    "freq": None,
                    "contour": None,
                    "duration": duration,
                    "kind": "breath",
                    "intensity": float(note.get("intensity") or 0.0),
                }
            )
            continue

        contour = _resolve_contour(note, tonic_hz, use_measured_hz)
        resolved.append(
            {
                "char": char or "a",
                "freq": contour[0] if contour else None,
                "contour": contour,
                "duration": duration,
                "kind": "voiced" if contour else "rest",
                "intensity": float(note.get("intensity") or 0.0),
            }
        )
    return resolved


def _median_voiced_freq(resolved: list[dict]) -> float:
    """Median of every voiced note's contour — used as the song's own pitch when
    the JSON/CSV gave no 主音 (so the register correction still has a reference)."""
    points = [
        hz
        for note in resolved
        if isinstance(note.get("contour"), (list, tuple))
        for hz in note["contour"]
        if hz and hz > 0
    ]
    return float(np.median(points)) if points else 0.0


def _fold_hz_into_range(hz: float) -> float:
    """Bring ``hz`` inside the singable range by whole octaves — keeps the note in
    tune (same pitch class) instead of clamping it to a wrong pitch."""
    if hz <= 0:
        return hz
    while hz > _MAX_VOICE_HZ:
        hz /= 2.0
    while hz < _MIN_VOICE_HZ:
        hz *= 2.0
    return hz


def _correct_song_register(
    resolved: list[dict], original_median: float, target_median: float
) -> list[dict]:
    """Calibrate the output pitch against the original and target median pitches.

    The whole melody is shifted by the nearest **whole octave** that moves the
    original median toward the target voice's median (so the song sits in the
    target's register without detuning or changing the 高低落差), then any note
    still outside the singable range is folded back by octaves. Shifting only by
    whole octaves keeps every interval intact, so it never makes the song sound
    sharp/flat — only an octave higher or lower when the voices are far apart."""
    shift = 0
    if original_median > 0 and target_median > 0:
        shift = int(round(math.log2(target_median / original_median)))
    factor = 2.0 ** shift
    if factor == 1.0:
        # No register shift, but still fold any out-of-range note into the octave.
        needs_fold = any(
            isinstance(n.get("contour"), (list, tuple))
            and any(hz > _MAX_VOICE_HZ or hz < _MIN_VOICE_HZ for hz in n["contour"])
            for n in resolved
        )
        if not needs_fold:
            return resolved
    for note in resolved:
        contour = note.get("contour")
        if isinstance(contour, (list, tuple)) and contour:
            folded = [_fold_hz_into_range(hz * factor) for hz in contour]
            note["contour"] = folded
            note["freq"] = folded[0]
    return resolved


def _note_contour(note: dict) -> list[float] | None:
    contour = note.get("contour")
    if isinstance(contour, (list, tuple)) and contour:
        points = [float(hz) for hz in contour if hz and float(hz) > 0]
        if points:
            return points
    freq = note.get("freq")
    return [float(freq)] if freq and float(freq) > 0 else None


def _contour_freqs(contour: list[float], count: int) -> np.ndarray:
    """Per-sample target frequency interpolated across the note's contour."""
    points = np.asarray([_clamp_voice_hz(hz) for hz in contour], dtype=np.float64)
    if points.size <= 1 or count <= 1:
        return np.full(max(1, count), points[0] if points.size else _DEFAULT_VOICE_HZ)
    source = np.linspace(0.0, 1.0, points.size)
    target = np.linspace(0.0, 1.0, count)
    return np.interp(target, source, points)


def _contour_steps(contour: list[float], k: int) -> list[float]:
    points = np.asarray(contour, dtype=np.float64)
    if k <= 1 or points.size <= 1:
        return [float(points[0])] * max(1, k)
    centers = (np.arange(k) + 0.5) / k
    source = np.linspace(0.0, 1.0, points.size)
    return [float(value) for value in np.interp(centers, source, points)]


def _window_ramp(count: int, ramp: int) -> np.ndarray:
    window = np.ones(count, dtype=np.float32)
    edge = min(ramp, count // 2)
    if edge > 0:
        up = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        window[:edge] = up
        window[count - edge :] = up[::-1]
    return window


def _breath_sound(count: int, intensity: float, sample_rate: int) -> np.ndarray:
    """Unvoiced 氣音: shaped noise at a level set by the measured intensity.

    Used for time points that carry energy but no pitch (breaths, fricatives),
    so the re-sung voice keeps that texture instead of going silent."""
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    noise = np.random.standard_normal(count).astype(np.float32)
    if count >= 3:  # gentle low-pass so it is breathy, not harsh white noise
        noise = np.convolve(noise, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
    shaped = noise * _syllable_envelope(count, sample_rate)
    peak = float(np.max(np.abs(shaped)))
    if peak > 1e-9:
        shaped = shaped / peak
    # Keep 氣音 subtle and proportional to the measured energy: a soft texture
    # between sung notes, not a loud noise burst. (Earlier the floor was 0.15 and
    # the slope 2.0, so even faint breaths blasted in as 奇怪短促的聲音.)
    level = float(min(0.3, max(0.04, 1.2 * max(0.0, intensity))))
    return (shaped * level).astype(np.float32)


def _clamp_note_seconds(seconds: float | None, default: float = 0.45) -> float:
    value = float(seconds) if seconds else default
    return max(_MIN_NOTE_SECONDS, min(_MAX_NOTE_SECONDS, value))


def _note_seconds(note: dict, default: float = 0.45) -> float:
    return _clamp_note_seconds(note.get("duration"), default)


def _note_count(rate: int, seconds: float | None, default: float = 0.45) -> int:
    """Samples for a note of ``seconds`` length — kept faithful so the sung clip
    matches the original timing (only clamped to a tiny floor / generous cap)."""
    return max(1, int(round(rate * _clamp_note_seconds(seconds, default))))


def _trim_edge_rests(notes: list[dict]) -> list[dict]:
    """Drop leading/trailing rest (silence) notes so the sung clip starts and ends
    on sound — only head/tail 靜默 is removed; the inner timing is untouched."""
    start = 0
    end = len(notes)
    while start < end and notes[start].get("kind") == "rest":
        start += 1
    while end > start and notes[end - 1].get("kind") == "rest":
        end -= 1
    return notes[start:end]


def _synthesize_song(notes: list[dict], sample_rate: int) -> np.ndarray:
    """Sing notes as vibrato'd vowels following each note's pitch contour, with
    unvoiced 氣音 for breath notes. Notes are concatenated back-to-back (no inserted
    gaps) so the total length stays faithful to the requested note durations."""
    nyquist = sample_rate / 2.0
    pieces: list[np.ndarray] = []
    for note in notes[:400]:
        count = _note_count(sample_rate, note.get("duration"))

        if note.get("kind") == "breath":
            pieces.append(_breath_sound(count, float(note.get("intensity") or 0.0), sample_rate))
            continue

        contour = _note_contour(note)
        if contour is None:
            pieces.append(np.zeros(count, dtype=np.float32))
            continue

        freqs = _contour_freqs(contour, count)
        center = float(np.median(freqs))
        axis = np.arange(count, dtype=np.float64) / sample_rate
        # Vibrato that eases in over the first ~180 ms — sounds sung, not synthy.
        vibrato = 0.012 * np.sin(2 * np.pi * 5.2 * axis) * np.clip(axis / 0.18, 0.0, 1.0)
        phase = 2 * np.pi * np.cumsum(freqs * (1.0 + vibrato)) / sample_rate

        max_harmonic = max(1, min(48, int((nyquist * 0.92) / center)))
        harmonics = np.arange(1, max_harmonic + 1, dtype=np.float64)
        char = note.get("char") or "a"
        formants = _VOWEL_FORMANTS[ord(char[0]) % len(_VOWEL_FORMANTS)]
        amplitudes = _formant_response(harmonics * center, formants) / harmonics
        tone = np.zeros(count, dtype=np.float64)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            tone += amplitude * np.sin(harmonic * phase)
        peak = float(np.max(np.abs(tone)))
        if peak > 1e-9:
            tone /= peak
        pieces.append((tone * _syllable_envelope(count, sample_rate)).astype(np.float32))

    if not pieces:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


# Melodies can sit well above/below the OS voice's natural pitch; allow a wide
# (but bounded) shift so the tune is faithful without runaway artifacts.
_MAX_SING_SHIFT_SEMITONES = 24.0
# Cap how many distinct syllables we shell out to the OS voice for, so a long
# lyric with many unique characters can't spawn an unbounded number of renders.
_MAX_SUNG_SYLLABLES = 120


def _clamp_sing_shift(semitones: float) -> float:
    return float(max(-_MAX_SING_SHIFT_SEMITONES, min(_MAX_SING_SHIFT_SEMITONES, semitones)))


def _sing_contour_os(
    syllable: np.ndarray,
    source_hz: float,
    contour: list[float],
    count: int,
    sample_rate: int,
) -> np.ndarray:
    """Pitch a spoken syllable onto a target pitch *contour*.

    The syllable is time-fitted to the note, then overlapping windows are each
    pitch-shifted to the contour and blended back (granular-style). Crucially each
    window is retuned using *its own* measured pitch, not one median for the whole
    syllable: that **flattens the spoken syllable's lexical tone** so a steady 簡譜
    degree comes out steady (a Chinese tone otherwise rode on top, inflating the
    高低落差), and a measured 基頻 contour is followed faithfully (抑揚頓挫)."""
    fitted = _fit_duration(syllable, count)
    if source_hz <= 0 or not contour:
        return fitted

    # Use several windows even for a steady note so the lexical tone gets levelled
    # out window by window; keep windows long enough (~90 ms) to measure pitch.
    min_segment = max(1, int(sample_rate * 0.09))
    max_steps = max(1, count // min_segment)
    k = int(min(max(len(contour), 4), max_steps, 8))
    if k <= 1:
        target = contour[0]
        if target <= 0:
            return fitted
        return _pitch_shift(fitted, _clamp_sing_shift(12.0 * math.log2(target / source_hz)))

    steps = _contour_steps(contour, k)
    bounds = np.linspace(0, count, k + 1).astype(int)
    overlap = max(1, int(sample_rate * 0.015))
    out = np.zeros(count, dtype=np.float32)
    weight = np.zeros(count, dtype=np.float32)
    for index in range(k):
        start = max(0, int(bounds[index]) - overlap)
        end = min(count, int(bounds[index + 1]) + overlap)
        segment = fitted[start:end]
        if segment.size == 0:
            continue
        target = steps[index]
        # Measure this window's own pitch so the shift lands it on the target;
        # fall back to the whole-syllable median for windows too short to track.
        seg_source = source_hz
        if segment.size >= _STFT_SIZE:
            measured = _estimate_median_hz(segment, sample_rate)
            if measured > 0:
                seg_source = measured
        semitones = _clamp_sing_shift(12.0 * math.log2(target / seg_source)) if target > 0 else 0.0
        shifted = _resample_to_length(_pitch_shift(segment, semitones), end - start)
        window = _window_ramp(end - start, overlap)
        out[start:end] += shifted * window
        weight[start:end] += window
    voiced = weight > 1e-6
    out[voiced] /= weight[voiced]
    return out.astype(np.float32)


def _trim_silence(samples: np.ndarray) -> np.ndarray:
    """Drop the leading/trailing near-silence the OS voice pads around a single
    spoken syllable.

    Without this the padding gets stretched into the note, so the note holds
    mostly silence and the voiced part sounds short and abrupt with big gaps
    between words. Trimming lets the voiced sound sustain the whole note.
    """
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-6:
        return samples
    threshold = max(0.015, 0.06 * peak)
    voiced = np.where(np.abs(samples) > threshold)[0]
    if voiced.size == 0:
        return samples
    return samples[voiced[0] : voiced[-1] + 1]


def _resample_rate(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample to a new sample rate, preserving duration and pitch."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    target_len = max(1, round(samples.size * dst_rate / src_rate))
    return _resample_to_length(samples, target_len)


def _fit_duration(samples: np.ndarray, target_len: int) -> np.ndarray:
    """Stretch / compress to ``target_len`` samples without changing pitch."""
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if abs(samples.size - target_len) <= 2:
        return _resample_to_length(samples, target_len)
    stretched = _time_stretch(samples, samples.size / target_len)
    return _resample_to_length(stretched, target_len)


def _apply_edge_fade(
    samples: np.ndarray, sample_rate: int, fade_seconds: float = 0.01
) -> np.ndarray:
    """Short fade in/out so concatenated sung notes don't click."""
    count = samples.size
    if count == 0:
        return samples
    fade = min(count // 2, max(1, int(sample_rate * fade_seconds)))
    if fade <= 0:
        return samples
    envelope = np.ones(count, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    envelope[:fade] = ramp
    envelope[count - fade :] = ramp[::-1]
    return (samples * envelope).astype(np.float32)


def _render_syllable(text: str, rate: int) -> tuple[np.ndarray, int, float] | None:
    """Speak one syllable with the OS voice, trimmed to its voiced part.

    Returns ``(samples, rate, median_hz)`` resampled to ``rate`` (or the OS rate
    when ``rate`` is 0), or ``None`` if nothing could be spoken.
    """
    spoken = _system_tts(text)
    if spoken is None:
        return None
    samples, srate = spoken
    out_rate = rate or srate
    samples = _trim_silence(_resample_rate(samples, srate, out_rate))
    if samples.size == 0:
        return None
    return samples, out_rate, _estimate_median_hz(samples, out_rate)


def _sing_with_os_voice(notes: list[dict]) -> tuple[np.ndarray, int] | None:
    """Sing ``{char, freq, duration}`` notes as real OS-TTS syllables.

    Each lyric character is spoken once by the OS voice (cached, since songs
    reuse characters), trimmed to its voiced part, then for every note pitch-
    shifted onto the melody note and time-stretched to fill the note's duration
    so it sustains like singing instead of a clipped syllable with a big gap.
    Notes run legato (no inserted gap); rests render as silence. A character the
    OS voice cannot speak is sung with a neutral vowel ("啦") so it stays a real
    voice rather than dropping to the synth buzz. Returns ``(samples, rate)``, or
    ``None`` when no syllable could be rendered (no OS voice) so the caller can
    fall back to the synthetic voice.
    """
    notes = notes[:400]
    cache: dict[str, tuple[np.ndarray, float] | None] = {}
    rate = 0
    for note in notes:
        char = str(note.get("char") or "").strip()
        freq = note.get("freq")
        if not char or not freq or freq <= 0 or char in cache:
            continue
        if len(cache) >= _MAX_SUNG_SYLLABLES:
            break
        rendered = _render_syllable(char, rate)
        if rendered is None:
            cache[char] = None
            continue
        samples, rate, median = rendered
        cache[char] = (samples, median)

    if rate == 0:
        return None  # nothing rendered — let the caller use the synth voice

    # A neutral sung vowel for any character the OS voice could not speak, so a
    # missing syllable stays a real voice instead of the synth buzz (喇叭聲).
    fallback: tuple[np.ndarray, float] | None = None
    if any(value is None for value in cache.values()):
        rendered = _render_syllable("啦", rate)
        if rendered is not None:
            fallback = (rendered[0], rendered[2])
    if fallback is None:
        fallback = next((value for value in cache.values() if value is not None), None)

    pieces: list[np.ndarray] = []
    for note in notes:
        kind = note.get("kind")
        # Keep every note at its requested length so the sung clip matches the
        # original timing (syllables are time-fitted, not stretched to a floor).
        count = _note_count(rate, note.get("duration"))
        if kind == "breath":
            pieces.append(_breath_sound(count, float(note.get("intensity") or 0.0), rate))
            continue

        contour = _note_contour(note)
        char = str(note.get("char") or "").strip()
        if contour is None:
            pieces.append(np.zeros(count, dtype=np.float32))  # rest
            continue
        entry = cache.get(char) or fallback
        if entry is None:
            # No real voice for this note at all — keep the melody with a tone.
            pieces.append(_synthesize_song([note], rate))
            continue
        syllable, source_hz = entry
        sung = _sing_contour_os(syllable, source_hz, contour, count, rate)
        pieces.append(_apply_edge_fade(sung, rate))

    if not pieces:
        return None
    return np.concatenate(pieces).astype(np.float32), rate


def _song_base(
    notes: list[dict], use_os_tts: bool, synth_rate: int
) -> tuple[np.ndarray, int, bool]:
    """Pick the singing base track shared by both engines.

    Returns ``(samples, sample_rate, is_real)``: a real-word OS-voice rendering
    when one is available (``is_real=True``), otherwise the synthetic vowel voice
    at ``synth_rate``. The mock plays this directly; OpenVoice re-voices it with
    the tone-color converter, which sounds far better given real speech.
    """
    if use_os_tts:
        sung = _sing_with_os_voice(notes)
        if sung is not None:
            samples, rate = sung
            return samples, rate, True
    return _synthesize_song(notes, synth_rate), synth_rate, False


_OS_TTS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$zh = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo } |"
    " Where-Object { $_.Culture.Name -like 'zh*' } | Select-Object -First 1;"
    "if ($zh) { $s.SelectVoice($zh.Name) };"
    "$s.SetOutputToWaveFile($env:BREEZE_TTS_OUT);"
    "$s.Speak($env:BREEZE_TTS_TEXT);"
    "$s.Dispose()"
)


def _system_tts(text: str) -> tuple[np.ndarray, int] | None:
    """Render real speech with the OS text-to-speech voice.

    On Windows this drives the built-in SAPI5 synthesizer (e.g. Microsoft
    Hanhan, zh-TW) through PowerShell and returns mono ``float32`` samples plus
    their sample rate. The text travels via environment variables and is never
    interpolated into the command, so arbitrary input cannot break out of the
    script. Returns ``None`` whenever no OS voice is usable (non-Windows host,
    missing PowerShell, empty text, or a failed render) so callers can fall back
    to the synthetic voice.
    """
    clean = (text or "").strip()
    if not clean or sys.platform != "win32":
        return None
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        out_path = Path(handle.name)
    env = dict(os.environ)
    env["BREEZE_TTS_TEXT"] = clean[:2000]
    env["BREEZE_TTS_OUT"] = str(out_path)
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", _OS_TTS_SCRIPT],
            env=env,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        data = out_path.read_bytes()
        if len(data) <= 44:
            return None
        from .voice_storage import decode_wav

        samples, rate = decode_wav(data)
        if samples.size == 0:
            return None
        return _as_float32_mono(samples), int(rate)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    finally:
        out_path.unlink(missing_ok=True)


def _retune_to_target(
    samples: np.ndarray,
    sample_rate: int,
    target_hz: float,
    max_semitones: float = 7.0,
) -> np.ndarray:
    """Shift real speech toward the target voice's pitch (kept modest/natural)."""
    if samples.size < _STFT_SIZE:
        return samples
    source_hz = _estimate_median_hz(samples, sample_rate)
    semitones = _shift_semitones(source_hz, target_hz)
    limit = float(max(0.0, max_semitones))
    semitones = float(max(-limit, min(limit, semitones)))
    if abs(semitones) < 0.5:
        return samples
    return _pitch_shift(samples, semitones)


def _clamp_speed(speed: float | None) -> float:
    try:
        value = float(speed)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 0:
        return 1.0
    return float(min(_MAX_SPEED, max(_MIN_SPEED, value)))


def _apply_speed(samples: np.ndarray, speed: float | None) -> np.ndarray:
    """Apply a pitch-preserving playback-speed multiplier (tempo only)."""
    rate = _clamp_speed(speed)
    if abs(rate - 1.0) < 1e-3 or samples.size < _STFT_SIZE:
        return samples.astype(np.float32, copy=False)
    # rate > 1 → fewer output frames → shorter / faster; rate < 1 → slower.
    return _time_stretch(samples, rate).astype(np.float32, copy=False)


def _envelope_band_edges(sample_rate: int) -> np.ndarray:
    top = min(_ENVELOPE_MAX_HZ, (sample_rate / 2.0) * 0.97)
    top = max(top, _ENVELOPE_MIN_HZ * 2.0)
    return np.geomspace(_ENVELOPE_MIN_HZ, top, _ENVELOPE_BANDS + 1)


def _spectral_envelope_bands(samples: np.ndarray, sample_rate: int) -> list[float] | None:
    """Summarize the average voiced log-magnitude spectrum into log-spaced bands.

    The result is mean-removed (so it captures spectral *shape* / timbre, not
    overall level — loudness is matched separately) and short / silent input
    yields ``None`` so callers can skip the transfer cleanly.
    """
    if samples.size < _STFT_SIZE:
        return None
    magnitude = np.abs(_stft(samples, _STFT_SIZE, _STFT_HOP))
    if magnitude.size == 0:
        return None
    frame_energy = magnitude.sum(axis=0)
    voiced = frame_energy > 0
    if np.any(voiced):
        threshold = 0.25 * float(np.median(frame_energy[voiced]))
        keep = frame_energy > threshold
        if not np.any(keep):
            keep = voiced
    else:
        return None
    average = magnitude[:, keep].mean(axis=1)
    freqs = np.fft.rfftfreq(_STFT_SIZE, d=1.0 / sample_rate)
    edges = _envelope_band_edges(sample_rate)
    bands = np.empty(_ENVELOPE_BANDS, dtype=np.float64)
    for index in range(_ENVELOPE_BANDS):
        selection = (freqs >= edges[index]) & (freqs < edges[index + 1])
        if np.any(selection):
            bands[index] = float(average[selection].mean())
        else:
            nearest = int(np.argmin(np.abs(freqs - math.sqrt(edges[index] * edges[index + 1]))))
            bands[index] = float(average[nearest])
    log_bands = np.log(np.maximum(bands, 1e-7))
    log_bands -= float(log_bands.mean())
    return [round(float(value), 4) for value in log_bands]


def _smooth_curve(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size <= 2:
        return values
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
    return np.convolve(values, kernel, mode="same")


def _apply_envelope_match(
    samples: np.ndarray, sample_rate: int, target_bands: list[float] | None
) -> np.ndarray:
    """Reshape ``samples``' spectral envelope toward ``target_bands``' timbre.

    A gentle, clamped, smoothed matching EQ (per-band log-gain = target − source)
    is interpolated across the spectrum and applied in the STFT domain, keeping
    phase and length. No-op when the target envelope is missing (old embeddings)
    or the audio is too short.
    """
    if not target_bands or samples.size < _STFT_SIZE:
        return samples.astype(np.float32, copy=False)
    target = np.asarray(target_bands, dtype=np.float64)
    if target.size != _ENVELOPE_BANDS:
        return samples.astype(np.float32, copy=False)
    source_bands = _spectral_envelope_bands(samples, sample_rate)
    if source_bands is None:
        return samples.astype(np.float32, copy=False)
    source = np.asarray(source_bands, dtype=np.float64)

    gain_db = (target - source) * (20.0 / math.log(10.0))
    gain_db = _smooth_curve(gain_db, 2)
    gain_db = np.clip(gain_db, -_ENVELOPE_MAX_GAIN_DB, _ENVELOPE_MAX_GAIN_DB)
    linear = 10.0 ** (gain_db / 20.0)

    edges = _envelope_band_edges(sample_rate)
    centers = np.sqrt(edges[:-1] * edges[1:])
    freqs = np.fft.rfftfreq(_STFT_SIZE, d=1.0 / sample_rate)
    bin_gain = np.interp(
        np.log(np.maximum(freqs, 1.0)),
        np.log(centers),
        linear,
        left=float(linear[0]),
        right=float(linear[-1]),
    ).astype(np.complex64)

    stft = _stft(samples, _STFT_SIZE, _STFT_HOP)
    stft *= bin_gain[:, None]
    shaped = _istft(stft, _STFT_SIZE, _STFT_HOP)
    return _resample_to_length(shaped, samples.size)


def _melo_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    mapping = {
        "zh": "ZH",
        "zh-tw": "ZH",
        "zh-cn": "ZH",
        "en": "EN",
        "ja": "JP",
        "jp": "JP",
        "ko": "KR",
        "kr": "KR",
        "es": "ES",
        "fr": "FR",
    }
    return mapping.get(normalized, "ZH")


def _temp_wav(samples: np.ndarray, sample_rate: int):
    from contextlib import contextmanager

    @contextmanager
    def _writer():
        import tempfile
        import wave

        mono = _as_float32_mono(samples)
        clipped = np.clip(mono, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        try:
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm.tobytes())
            yield path
        finally:
            path.unlink(missing_ok=True)

    return _writer()


def _temp_wav_path():
    from contextlib import contextmanager

    @contextmanager
    def _holder():
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)

    return _holder()
