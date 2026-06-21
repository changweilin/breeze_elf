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

    def synthesize(self, text: str, language: str, embedding: bytes) -> VoiceAudio:
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
            "version": 1,
            "kind": "mock",
            "medianHz": round(median_hz, 2),
            "rms": round(rms, 5),
            "centroidHz": round(_spectral_centroid(mono, sample_rate), 2),
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

        target_rms = float(profile.get("rms") or 0.0)
        shifted = _match_loudness(shifted, target_rms)
        return VoiceAudio(samples=shifted.astype(np.float32, copy=False), sample_rate=sample_rate)

    def synthesize(self, text: str, language: str, embedding: bytes) -> VoiceAudio:
        del language  # the OS voice picks its own language; the fallback is fixed
        profile = _load_mock_embedding(embedding)
        base_hz = float(profile.get("medianHz") or _DEFAULT_VOICE_HZ)
        target_rms = float(profile.get("rms") or 0.0)

        # Prefer the real OS text-to-speech voice (e.g. Microsoft Hanhan, zh-TW)
        # so the result is actual words, then nudge its pitch toward the target.
        spoken = _system_tts(text) if self.use_os_tts else None
        if spoken is not None:
            samples, rate = spoken
            samples = _retune_to_target(samples, rate, base_hz)
            samples = _match_loudness(samples, target_rms, default_peak=0.92)
            return VoiceAudio(samples=samples, sample_rate=rate)

        # Fallback: a synthetic vowel voice (no OS TTS available).
        samples = _synthesize_voice(text, self.sample_rate, base_hz)
        samples = _match_loudness(samples, target_rms, default_peak=0.6)
        return VoiceAudio(samples=samples, sample_rate=self.sample_rate)

    def synthesize_song(
        self,
        notes: list[dict],
        tonic_hz: float,
        embedding: bytes,
        use_measured_hz: bool = False,
    ) -> VoiceAudio:
        profile = _load_mock_embedding(embedding)
        if not tonic_hz or tonic_hz <= 0:
            tonic_hz = float(profile.get("medianHz") or _DEFAULT_VOICE_HZ)
        resolved = _resolve_song_notes(notes, tonic_hz, use_measured_hz)
        target_rms = float(profile.get("rms") or 0.0)
        # Real words in the OS voice (the same voice 文字轉語音 uses), pitched to
        # the melody; falls back to the synthetic vowel voice when unavailable.
        samples, rate, is_real = _song_base(resolved, self.use_os_tts, self.sample_rate)
        samples = _match_loudness(samples, target_rms, default_peak=0.92 if is_real else 0.85)
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

    def synthesize(self, text: str, language: str, embedding: bytes) -> VoiceAudio:
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
            return VoiceAudio(samples=_as_float32_mono(data), sample_rate=int(out_rate))

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
        return VoiceAudio(samples=_as_float32_mono(data), sample_rate=int(out_rate))

    def synthesize_song(
        self,
        notes: list[dict],
        tonic_hz: float,
        embedding: bytes,
        use_measured_hz: bool = False,
    ) -> VoiceAudio:
        self.load()
        if not tonic_hz or tonic_hz <= 0:
            tonic_hz = _DEFAULT_VOICE_HZ
        resolved = _resolve_song_notes(notes, tonic_hz, use_measured_hz)
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
        return VoiceAudio(samples=_as_float32_mono(data), sample_rate=int(out_rate))

    def _load_tts(self) -> None:
        if self._tts is not None:
            return
        try:
            from melo.api import TTS
        except Exception as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "文字轉語音需安裝 MeloTTS(`pip install git+https://github.com/myshell-ai/"
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


def _resolve_song_notes(
    notes: list[dict], tonic_hz: float, use_measured_hz: bool
) -> list[dict]:
    """Turn ``{char, jianpu, hz, durationSeconds}`` notes into ``{char, freq,
    duration}`` ones.

    Pitch comes from the 簡譜 degree relative to ``tonic_hz`` (quantized,
    in-tune singing); when ``use_measured_hz`` is set, or the 簡譜 is missing,
    the measured ``hz`` is used instead. Unparseable / rest notes get
    ``freq=None`` so the synthesizer renders silence for that beat.
    """
    from .audio import jianpu_to_semitones

    resolved: list[dict] = []
    for note in notes:
        char = str(note.get("char") or "").strip()
        hz = note.get("hz")
        measured = float(hz) if hz else 0.0
        freq: float | None = None
        if use_measured_hz and measured > 0:
            freq = measured
        else:
            semitones = jianpu_to_semitones(note.get("jianpu"))
            if semitones is not None and tonic_hz > 0:
                freq = tonic_hz * (2.0 ** (semitones / 12.0))
            elif measured > 0:
                freq = measured
        resolved.append(
            {"char": char or "a", "freq": freq, "duration": note.get("durationSeconds")}
        )
    return resolved


def _synthesize_song(notes: list[dict], sample_rate: int) -> np.ndarray:
    """Sing a sequence of notes as sustained, vibrato'd vowels (one per note)."""
    nyquist = sample_rate / 2.0
    gap = np.zeros(max(1, int(sample_rate * 0.02)), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for position, note in enumerate(notes[:400]):
        duration = note.get("duration")
        seconds = float(duration) if duration else 0.45
        seconds = max(0.12, min(2.0, seconds))
        count = max(1, int(sample_rate * seconds))

        freq = note.get("freq")
        if not freq or freq <= 0:
            pieces.append(np.zeros(count, dtype=np.float32))
            continue
        freq = float(min(_MAX_VOICE_HZ, max(_MIN_VOICE_HZ, freq)))

        axis = np.arange(count, dtype=np.float64) / sample_rate
        # Vibrato that eases in over the first ~180 ms — sounds sung, not synthy.
        vibrato = 0.012 * np.sin(2 * np.pi * 5.2 * axis) * np.clip(axis / 0.18, 0.0, 1.0)
        phase = 2 * np.pi * np.cumsum(freq * (1.0 + vibrato)) / sample_rate

        max_harmonic = max(1, min(48, int((nyquist * 0.92) / freq)))
        harmonics = np.arange(1, max_harmonic + 1, dtype=np.float64)
        char = note.get("char") or "a"
        formants = _VOWEL_FORMANTS[ord(char[0]) % len(_VOWEL_FORMANTS)]
        amplitudes = _formant_response(harmonics * freq, formants) / harmonics
        tone = np.zeros(count, dtype=np.float64)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            tone += amplitude * np.sin(harmonic * phase)
        peak = float(np.max(np.abs(tone)))
        if peak > 1e-9:
            tone /= peak
        pieces.append((tone * _syllable_envelope(count, sample_rate)).astype(np.float32))
        if position != len(notes) - 1:
            pieces.append(gap)

    if not pieces:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


# Melodies can sit well above/below the OS voice's natural pitch; allow a wide
# (but bounded) shift so the tune is faithful without runaway artifacts.
_MAX_SING_SHIFT_SEMITONES = 24.0
# Cap how many distinct syllables we shell out to the OS voice for, so a long
# lyric with many unique characters can't spawn an unbounded number of renders.
_MAX_SUNG_SYLLABLES = 120


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


def _sing_with_os_voice(notes: list[dict]) -> tuple[np.ndarray, int] | None:
    """Sing ``{char, freq, duration}`` notes as real OS-TTS syllables.

    Each lyric character is spoken once by the OS voice (cached, since songs
    reuse characters), then for every note it is pitch-shifted onto the melody
    note and time-stretched to the note's duration. Rests render as silence; a
    note whose character cannot be spoken keeps the melody via a synth tone.
    Returns ``(samples, sample_rate)``, or ``None`` when no syllable could be
    rendered (no OS voice) so the caller can fall back to the synthetic voice.
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
        spoken = _system_tts(char)
        if spoken is None:
            cache[char] = None
            continue
        samples, srate = spoken
        if rate == 0:
            rate = srate
        samples = _resample_rate(samples, srate, rate)
        cache[char] = (samples, _estimate_median_hz(samples, rate))

    if rate == 0:
        return None  # nothing rendered — let the caller use the synth voice

    gap = np.zeros(max(1, int(rate * 0.02)), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, note in enumerate(notes):
        duration = note.get("duration")
        seconds = float(duration) if duration else 0.45
        seconds = max(0.12, min(2.5, seconds))
        count = max(1, int(rate * seconds))

        freq = note.get("freq")
        char = str(note.get("char") or "").strip()
        entry = cache.get(char)
        if not freq or freq <= 0:
            pieces.append(np.zeros(count, dtype=np.float32))  # rest
        elif entry is None:
            # Character could not be spoken — keep the melody with a synth tone.
            pieces.append(_synthesize_song([note], rate))
        else:
            syllable, source_hz = entry
            target = float(min(_MAX_VOICE_HZ, max(_MIN_VOICE_HZ, freq)))
            semitones = 12.0 * math.log2(target / source_hz) if source_hz > 0 else 0.0
            semitones = max(-_MAX_SING_SHIFT_SEMITONES, min(_MAX_SING_SHIFT_SEMITONES, semitones))
            pitched = _pitch_shift(syllable, semitones)
            pieces.append(_apply_edge_fade(_fit_duration(pitched, count), rate))
        if index != len(notes) - 1:
            pieces.append(gap)

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


def _retune_to_target(samples: np.ndarray, sample_rate: int, target_hz: float) -> np.ndarray:
    """Shift real speech toward the target voice's pitch (kept modest/natural)."""
    if samples.size < _STFT_SIZE:
        return samples
    source_hz = _estimate_median_hz(samples, sample_rate)
    semitones = _shift_semitones(source_hz, target_hz)
    semitones = float(max(-7.0, min(7.0, semitones)))
    if abs(semitones) < 0.5:
        return samples
    return _pitch_shift(samples, semitones)


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
