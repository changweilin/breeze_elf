"""Anonymous, in-session speaker labelling (diarization) for VAD utterances.

Each recognised utterance gets a speaker tag (說話者 1／2…) by turning its audio
into a speaker embedding and assigning it to a running, online cluster of the
voices heard *so far in this connection*. Everything is **local** and **opt-in**
(``BREEZE_DIARIZE=on``); a missing model / runtime degrades to
:class:`NullDiarizer` (a no-op) so recognition is never affected.

Design, mirroring :mod:`breeze_elf.enhance`:

- :class:`OnnxSpeakerEmbedder` — a torch-free embedder: a pure-numpy log-mel
  front-end feeds an ONNX speaker-embedding model over ``onnxruntime`` (the same
  runtime faster-whisper already bundles for Silero VAD, so no new heavy
  dependency). The user supplies a permissively-licensed ONNX model; without it
  the factory returns :class:`NullDiarizer`.
- :class:`OnlineSpeakerClusterer` — pure numpy: L2-normalised running centroids,
  cosine assignment, a new speaker when the best match is below threshold (capped
  at ``BREEZE_DIARIZE_MAX_SPEAKERS``). One clusterer is built **per connection** so
  labels reset every session (no cross-session voiceprint linkage — a voiceprint
  is more sensitive than the transcript, so it never persists).

Privacy: embeddings are computed from the **raw** utterance audio and are never
stored; only the anonymous integer label reaches the transcript.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings, get_settings

LOGGER = logging.getLogger("breeze_elf.diarize")

_MODEL_SR = 16_000


class SpeakerEmbedder(Protocol):
    """Turns one utterance's audio into a fixed-length speaker embedding.

    ``embed`` returns a 1-D float vector, or ``None`` when no usable embedding can
    be produced (empty audio, model unavailable, or an inference failure) so the
    caller simply omits the speaker label rather than breaking."""

    name: str

    @property
    def available(self) -> bool:
        ...

    def load(self) -> None:
        ...

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        ...


class NullDiarizer:
    """No-op embedder — the default when diarization is off / the model is absent."""

    name = "off"

    @property
    def available(self) -> bool:
        return False

    @property
    def ready(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        return None


def _unit_normalize(vector: np.ndarray) -> np.ndarray | None:
    """L2-normalise to a finite unit vector, or ``None`` for a non-finite / zero
    vector so it can never poison a centroid (critique fix: reject non-finite)."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return vec / norm


class OnlineSpeakerClusterer:
    """Assign each embedding to a session speaker with online cosine clustering.

    Keeps one L2-normalised centroid per speaker. A new embedding joins its nearest
    centroid when their cosine similarity clears ``threshold``; otherwise it starts
    a new speaker, up to ``max_speakers`` (after which everything folds into the
    nearest existing speaker). Assignment updates the matched centroid as a running
    mean so a speaker's model tracks their voice across the session.

    Returns a **0-based** speaker index (the UI shows ``index + 1``), or ``None``
    for an unusable (non-finite) embedding.
    """

    def __init__(self, max_speakers: int = 6, threshold: float = 0.75) -> None:
        self.max_speakers = max(1, int(max_speakers))
        self.threshold = float(threshold)
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    def assign(self, embedding: np.ndarray) -> int | None:
        vec = _unit_normalize(embedding)
        if vec is None:
            return None

        if not self._centroids:
            return self._add_speaker(vec)

        sims = [float(np.dot(vec, centroid)) for centroid in self._centroids]
        best = int(np.argmax(sims))
        if sims[best] < self.threshold and len(self._centroids) < self.max_speakers:
            return self._add_speaker(vec)

        count = self._counts[best]
        merged = _unit_normalize((self._centroids[best] * count + vec) / (count + 1))
        if merged is not None:
            self._centroids[best] = merged
        self._counts[best] = count + 1
        return best

    def _add_speaker(self, unit_vec: np.ndarray) -> int:
        self._centroids.append(unit_vec)
        self._counts.append(1)
        return len(self._centroids) - 1


# ── numpy log-mel front-end (torch-free) ──────────────────────────────────────


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, fmin: float, fmax: float
) -> np.ndarray:
    """A ``(n_mels, n_fft//2 + 1)`` triangular HTK mel filterbank."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    bin_freqs = _mel_to_hz(mel_points)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        low, center, high = bin_freqs[m], bin_freqs[m + 1], bin_freqs[m + 2]
        if high <= low:
            continue
        rising = (fft_freqs - low) / max(center - low, 1e-9)
        falling = (high - fft_freqs) / max(high - center, 1e-9)
        filters[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return filters


def _resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == _MODEL_SR or sample_rate <= 0 or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / sample_rate
    target_len = max(1, int(round(duration * _MODEL_SR)))
    src_idx = np.linspace(0.0, samples.size - 1, num=samples.size)
    dst_idx = np.linspace(0.0, samples.size - 1, num=target_len)
    return np.interp(dst_idx, src_idx, samples).astype(np.float32)


def log_mel_fbank(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_mels: int = 80,
    n_fft: int = 400,
    hop: int = 160,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """A ``(frames, n_mels)`` log-mel spectrogram, mean-variance normalised per
    band. Torch-free; the audio is resampled to 16 kHz first (the rate speaker
    models expect). Returns an empty ``(0, n_mels)`` array for too-short input."""
    audio = _resample_to_16k(np.asarray(samples, dtype=np.float32).reshape(-1), sample_rate)
    if audio.size < n_fft:
        return np.empty((0, n_mels), dtype=np.float32)

    window = np.hanning(n_fft).astype(np.float64)
    starts = range(0, audio.size - n_fft + 1, hop)
    frames = np.stack([audio[start : start + n_fft].astype(np.float64) for start in starts])
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2  # (frames, n_freqs)

    fb = _mel_filterbank(n_mels, n_fft, _MODEL_SR, fmin, fmax or _MODEL_SR / 2.0)
    mel = spectrum @ fb.T  # (frames, n_mels)
    log_mel = np.log(np.maximum(mel, 1e-10))

    mean = log_mel.mean(axis=0, keepdims=True)
    std = log_mel.std(axis=0, keepdims=True)
    normalized = (log_mel - mean) / np.maximum(std, 1e-5)
    return normalized.astype(np.float32)


class OnnxSpeakerEmbedder:
    """Speaker embeddings from an ONNX model over onnxruntime, torch-free.

    The model's expected feature layout varies (``(batch, frames, mels)`` vs
    ``(batch, mels, frames)``); the input node name is read from the session and
    both layouts are tried, so a range of exported embedding models work without
    a per-model adapter. Any load / inference failure disables the embedder (it
    then behaves like :class:`NullDiarizer`) rather than breaking recognition.
    """

    name = "onnx"

    def __init__(self, model_path: Path, *, device: str = "cpu", n_mels: int = 80) -> None:
        self._model_path = Path(model_path)
        self._device = device
        self.n_mels = int(n_mels)
        self._session = None
        self._input_name: str | None = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        if importlib.util.find_spec("onnxruntime") is None:
            return False
        return self._model_path.is_file()

    @property
    def ready(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None or self._failed:
            return
        with self._lock:
            if self._session is not None or self._failed:
                return
            try:
                import onnxruntime

                opts = onnxruntime.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                opts.log_severity_level = 4
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self._device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self._session = onnxruntime.InferenceSession(
                    str(self._model_path), providers=providers, sess_options=opts
                )
                self._input_name = self._session.get_inputs()[0].name
            except Exception as exc:  # pragma: no cover - depends on model/runtime
                self._failed = True
                LOGGER.warning("speaker ONNX load failed; diarization disabled: %s", exc)

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        if samples is None or samples.size == 0:
            return None
        self.load()
        if self._session is None or self._input_name is None:
            return None
        try:
            with self._lock:
                return self._run(samples, sample_rate)
        except Exception as exc:  # pragma: no cover - inference guard
            LOGGER.warning("speaker embedding failed; skipping label: %s", exc)
            return None

    def _run(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        feats = log_mel_fbank(samples, sample_rate, n_mels=self.n_mels)
        if feats.shape[0] == 0:
            return None
        # Try (batch, frames, mels) then, on a shape error, (batch, mels, frames).
        for array in (feats[np.newaxis, :, :], feats.T[np.newaxis, :, :]):
            try:
                output = self._session.run(None, {self._input_name: array.astype(np.float32)})
            except Exception:
                continue
            return np.asarray(output[0], dtype=np.float32).reshape(-1)
        raise ValueError("speaker model rejected both feature layouts")


def build_diarizer(settings: Settings | None = None, *, base_dir: Path | None = None):
    """The shared, module-scope embedder. :class:`NullDiarizer` unless
    ``BREEZE_DIARIZE=on`` *and* onnxruntime + the model file are present, so the
    base install never needs a speaker model."""
    settings = settings or get_settings()
    if not settings.diarize_enabled:
        return NullDiarizer()
    model_path = Path(settings.diarize_model).expanduser()
    if not model_path.is_absolute() and base_dir is not None:
        model_path = Path(base_dir) / model_path
    embedder = OnnxSpeakerEmbedder(
        model_path, device=settings.diarize_device, n_mels=settings.diarize_n_mels
    )
    if not embedder.available:
        LOGGER.warning(
            "diarization enabled but model/onnxruntime unavailable (%s); disabled", model_path
        )
        return NullDiarizer()
    return embedder


def build_clusterer(settings: Settings | None = None) -> OnlineSpeakerClusterer:
    """A fresh per-connection clusterer (labels reset every session)."""
    settings = settings or get_settings()
    return OnlineSpeakerClusterer(
        max_speakers=settings.diarize_max_speakers, threshold=settings.diarize_threshold
    )
