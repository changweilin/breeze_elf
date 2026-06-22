from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 簡譜 (jianpu) maps each scale degree, measured in semitones above the tonic,
# to a number 1-7. Chromatic degrees borrow the lower number with a sharp.
_JIANPU_DEGREES = {
    0: "1",
    1: "#1",
    2: "2",
    3: "#2",
    4: "3",
    5: "4",
    6: "#4",
    7: "5",
    8: "#5",
    9: "6",
    10: "#6",
    11: "7",
}
_JIANPU_DOT_ABOVE = "̇"  # combining dot above (higher octave)
_JIANPU_DOT_BELOW = "̣"  # combining dot below (lower octave)
_JIANPU_GLIDE_UP = "↗"  # rising portamento between two degrees
_JIANPU_GLIDE_DOWN = "↘"  # falling portamento between two degrees


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start_seconds: float
    end_seconds: float
    samples: np.ndarray
    rms: float
    is_speech: bool
    kind: str = "window"


@dataclass(frozen=True)
class PitchPoint:
    offset_seconds: float
    hz: float
    confidence: float


@dataclass(frozen=True)
class PitchSummary:
    median_hz: float | None
    min_hz: float | None
    max_hz: float | None
    voiced_ratio: float
    points: tuple[PitchPoint, ...]


@dataclass(frozen=True)
class SegmentAnalysis:
    """Pitch and loudness behaviour measured over a single short segment.

    ``start_hz`` / ``end_hz`` are the median pitch of the leading and trailing
    edges of the segment so a portamento (slide) can be told apart from a
    steady note. ``intensity_start`` / ``intensity_end`` are the RMS of the two
    halves so a crescendo or decay is visible.
    """

    median_hz: float | None
    min_hz: float | None
    max_hz: float | None
    start_hz: float | None
    end_hz: float | None
    intensity: float
    intensity_start: float
    intensity_end: float


@dataclass(frozen=True)
class AudioPreprocessConfig:
    highpass_hz: float = 0.0
    noise_reduction_db: float = 0.0
    normalize_target_rms: float = 0.0
    normalize_max_gain_db: float = 0.0
    normalize_max_cut_db: float = 0.0
    peak_headroom: float = 0.98


_AUDIO_PREPROCESS_PROFILES = {
    "off": AudioPreprocessConfig(),
    "natural": AudioPreprocessConfig(
        highpass_hz=65.0,
        noise_reduction_db=4.5,
        normalize_target_rms=0.08,
        normalize_max_gain_db=6.0,
        normalize_max_cut_db=3.0,
    ),
    "speech": AudioPreprocessConfig(
        highpass_hz=80.0,
        noise_reduction_db=7.0,
        normalize_target_rms=0.1,
        normalize_max_gain_db=10.0,
        normalize_max_cut_db=6.0,
    ),
}


def calculate_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    values = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(values * values)))


def pcm16le_to_float32(payload: bytes) -> np.ndarray:
    if len(payload) < 2:
        return np.empty(0, dtype=np.float32)
    if len(payload) % 2:
        payload = payload[:-1]
    pcm = np.frombuffer(payload, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def prepare_asr_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    profile: str = "natural",
) -> np.ndarray:
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)

    profile_name = profile.strip().lower()
    config = _AUDIO_PREPROCESS_PROFILES.get(profile_name, _AUDIO_PREPROCESS_PROFILES["natural"])
    output = samples.astype(np.float32, copy=False)
    if config == _AUDIO_PREPROCESS_PROFILES["off"]:
        return output

    output = output.copy()
    if config.highpass_hz > 0:
        output = _highpass_filter(output, sample_rate, config.highpass_hz)
    if config.noise_reduction_db > 0:
        output = _soft_noise_floor_reduction(output, sample_rate, config.noise_reduction_db)
    if config.normalize_target_rms > 0:
        output = _normalize_rms(
            output,
            target_rms=config.normalize_target_rms,
            max_gain_db=config.normalize_max_gain_db,
            max_cut_db=config.normalize_max_cut_db,
            peak_headroom=config.peak_headroom,
        )
    return output.astype(np.float32, copy=False)


def _highpass_filter(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if sample_rate <= 0 or cutoff_hz <= 0 or samples.size < 2:
        return samples

    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    output = np.empty_like(samples)
    previous_input = float(samples[0])
    previous_output = 0.0
    output[0] = 0.0
    for index in range(1, samples.size):
        current_input = float(samples[index])
        current_output = alpha * (previous_output + current_input - previous_input)
        output[index] = current_output
        previous_input = current_input
        previous_output = current_output
    return output


def _soft_noise_floor_reduction(
    samples: np.ndarray,
    sample_rate: int,
    reduction_db: float,
) -> np.ndarray:
    if sample_rate <= 0 or reduction_db <= 0 or samples.size == 0:
        return samples

    frame_samples = max(1, round(sample_rate * 0.02))
    hop_samples = max(1, round(sample_rate * 0.01))
    if samples.size < frame_samples * 2:
        return samples

    starts = np.arange(0, samples.size - frame_samples + 1, hop_samples)
    if starts.size < 3:
        return samples

    frame_rms = np.array(
        [calculate_rms(samples[start : start + frame_samples]) for start in starts],
        dtype=np.float32,
    )
    active_rms = frame_rms[frame_rms > 1e-5]
    if active_rms.size < 3:
        return samples

    noise_floor = float(np.percentile(active_rms, 20))
    voice_floor = float(np.percentile(active_rms, 70))
    if noise_floor <= 1e-5 or voice_floor <= noise_floor * 1.4:
        return samples

    threshold = max(noise_floor * 2.6, noise_floor + 1e-5)
    floor_gain = 10 ** (-reduction_db / 20.0)
    ratios = np.clip((frame_rms - noise_floor) / (threshold - noise_floor), 0.0, 1.0)
    smooth = ratios * ratios * (3.0 - 2.0 * ratios)
    frame_gains = floor_gain + smooth * (1.0 - floor_gain)
    centers = starts + frame_samples / 2
    sample_indices = np.arange(samples.size, dtype=np.float32)
    gain_curve = np.interp(sample_indices, centers, frame_gains).astype(np.float32)
    return samples * gain_curve


def _normalize_rms(
    samples: np.ndarray,
    *,
    target_rms: float,
    max_gain_db: float,
    max_cut_db: float,
    peak_headroom: float,
) -> np.ndarray:
    rms = calculate_rms(samples)
    if rms <= 1e-6:
        return samples

    desired_gain = target_rms / rms
    max_gain = 10 ** (max_gain_db / 20.0)
    min_gain = 10 ** (-max_cut_db / 20.0)
    gain = float(np.clip(desired_gain, min_gain, max_gain))
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        gain = min(gain, peak_headroom / peak)

    if abs(gain - 1.0) < 0.01:
        return samples
    return np.clip(samples * gain, -peak_headroom, peak_headroom).astype(np.float32)


def summarize_pitch(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = 40,
    hop_ms: int = 20,
    min_hz: float = 70.0,
    max_hz: float = 500.0,
    rms_threshold: float = 0.01,
    confidence_threshold: float = 0.35,
    yin_threshold: float = 0.15,
    max_points: int = 80,
) -> PitchSummary:
    """Per-frame f0 track via the YIN algorithm (cumulative mean normalized
    difference).

    YIN picks the *first* period dip rather than the largest correlation peak,
    so it avoids the octave / sub-harmonic errors plain autocorrelation makes —
    the main cause of the 逐字稿 音準 drifting by an octave on some syllables.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if hop_ms <= 0:
        raise ValueError("hop_ms must be positive")
    if min_hz <= 0 or max_hz <= min_hz:
        raise ValueError("max_hz must be greater than min_hz")

    frame_samples = max(1, round(sample_rate * frame_ms / 1000))
    hop_samples = max(1, round(sample_rate * hop_ms / 1000))
    min_lag = max(1, round(sample_rate / max_hz))
    max_lag = max(min_lag + 1, round(sample_rate / min_hz))
    if samples.size < frame_samples or frame_samples <= max_lag:
        return PitchSummary(None, None, None, 0.0, ())

    points: list[PitchPoint] = []
    hz_values: list[float] = []
    frame_count = 0

    for start in range(0, samples.size - frame_samples + 1, hop_samples):
        frame_count += 1
        frame = samples[start : start + frame_samples].astype(np.float64, copy=True)
        frame -= np.mean(frame)
        rms = calculate_rms(frame)
        if rms < rms_threshold:
            continue

        hz, confidence = _yin_pitch(frame, sample_rate, min_lag, max_lag, yin_threshold)
        if hz <= 0 or confidence < confidence_threshold:
            continue
        if hz < min_hz or hz > max_hz:
            continue

        hz_values.append(hz)
        points.append(
            PitchPoint(
                offset_seconds=(start + frame_samples / 2) / sample_rate,
                hz=hz,
                confidence=confidence,
            )
        )

    if not hz_values:
        return PitchSummary(None, None, None, 0.0, ())

    values = np.array(hz_values, dtype=np.float64)
    voiced_ratio = len(hz_values) / frame_count if frame_count else 0.0
    return PitchSummary(
        median_hz=float(np.median(values)),
        min_hz=float(np.percentile(values, 10)),
        max_hz=float(np.percentile(values, 90)),
        voiced_ratio=float(voiced_ratio),
        points=tuple(_thin_pitch_points(points, max_points)),
    )


def hz_to_jianpu(hz: float | None, tonic_hz: float | None) -> str:
    """Convert a pitch in Hz to a 簡譜 (jianpu) number relative to a tonic.

    The tonic maps to ``1`` (do). Pitches above the tonic octave gain a
    combining dot above, pitches below gain a dot below. Returns an empty
    string when the pitch or tonic is missing or non-positive.
    """
    if not hz or not tonic_hz or hz <= 0 or tonic_hz <= 0:
        return ""

    semitones = round(12.0 * math.log2(hz / tonic_hz))
    octave, degree = divmod(semitones, 12)
    digit = _JIANPU_DEGREES[degree]
    if octave > 0:
        return digit + _JIANPU_DOT_ABOVE * octave
    if octave < 0:
        return digit + _JIANPU_DOT_BELOW * (-octave)
    return digit


def jianpu_glide(start_hz: float | None, end_hz: float | None, tonic_hz: float | None) -> str:
    """Render a 簡譜 glide between two pitches.

    When the leading and trailing pitch of a segment land on the same scale
    degree the note is steady and the single degree is returned. When they
    differ (a portamento / 滑音) the two degrees are joined with an arrow that
    shows the slide direction, e.g. ``3↗5`` or ``5↘1``.
    """
    start = hz_to_jianpu(start_hz, tonic_hz)
    end = hz_to_jianpu(end_hz, tonic_hz)
    if not start:
        return end
    if not end or start == end:
        return start
    arrow = _JIANPU_GLIDE_UP if (end_hz or 0.0) >= (start_hz or 0.0) else _JIANPU_GLIDE_DOWN
    return f"{start}{arrow}{end}"


# Base scale degrees 1-7 (no accidental) → semitones above the tonic.
_JIANPU_BASE_SEMITONES = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11}


def jianpu_to_semitones(jianpu: str | None) -> float | None:
    """Parse a 簡譜 token back into semitones above the tonic (inverse of
    :func:`hz_to_jianpu`).

    Understands the digits 1-7, a leading ``#``/``b`` accidental, octave marks
    (combining dot above/below as produced by :func:`hz_to_jianpu`, or the ASCII
    shortcuts ``'``/``^`` for up and ``,``/``_`` for down), and a glide such as
    ``3↗5`` (the leading degree is used). Returns ``None`` for rests/blanks or
    anything unparseable so the caller can treat it as silence.
    """
    if not jianpu:
        return None
    token = jianpu.strip()
    for arrow in (_JIANPU_GLIDE_UP, _JIANPU_GLIDE_DOWN):
        if arrow in token:
            token = token.split(arrow, 1)[0]
            break

    octave = 0
    body = ""
    for char in token:
        if char in (_JIANPU_DOT_ABOVE, "'", "^", "˙", "̇"):
            octave += 1
        elif char in (_JIANPU_DOT_BELOW, ",", "_", "̣"):
            octave -= 1
        elif not char.isspace():
            body += char

    accidental = 0
    while body[:1] in ("#", "♯", "b", "♭"):
        accidental += 1 if body[0] in ("#", "♯") else -1
        body = body[1:]
    if not body or body[0] not in _JIANPU_BASE_SEMITONES:
        return None
    base = _JIANPU_BASE_SEMITONES[body[0]]
    return float(base + accidental + 12 * octave)


def pitch_cents_off(hz: float | None, tonic_hz: float | None) -> float | None:
    """Signed distance, in cents, from the nearest scale degree of the tonic.

    ``0`` means perfectly in tune with the tonic's equal-tempered grid; the
    value approaches ±50 cents at the midpoint between two degrees. Returns
    ``None`` when the pitch or tonic is missing.
    """
    if not hz or not tonic_hz or hz <= 0 or tonic_hz <= 0:
        return None
    semitones = 12.0 * math.log2(hz / tonic_hz)
    return (semitones - round(semitones)) * 100.0


def analyze_segment(
    samples: np.ndarray,
    sample_rate: int,
    *,
    edge_fraction: float = 0.34,
) -> SegmentAnalysis:
    """Measure pitch range, slide edges, and loudness envelope of a segment."""
    if samples.size < 2:
        return SegmentAnalysis(None, None, None, None, None, 0.0, 0.0, 0.0)

    summary = summarize_pitch(samples, sample_rate)
    hz_points = [point.hz for point in summary.points if point.hz]
    start_hz: float | None = None
    end_hz: float | None = None
    if hz_points:
        edge = max(1, round(len(hz_points) * edge_fraction))
        start_hz = float(np.median(hz_points[:edge]))
        end_hz = float(np.median(hz_points[-edge:]))

    half = max(1, samples.size // 2)
    return SegmentAnalysis(
        median_hz=summary.median_hz,
        min_hz=summary.min_hz,
        max_hz=summary.max_hz,
        start_hz=start_hz,
        end_hz=end_hz,
        intensity=calculate_rms(samples),
        intensity_start=calculate_rms(samples[:half]),
        intensity_end=calculate_rms(samples[half:]),
    )


def _yin_pitch(
    frame: np.ndarray,
    sample_rate: int,
    min_lag: int,
    max_lag: int,
    threshold: float,
) -> tuple[float, float]:
    """Estimate one frame's f0 with YIN. Returns ``(hz, confidence)``.

    ``confidence`` is ``1 - d'(tau)`` at the chosen period (≈1 for a clean
    periodic frame, ≈0 for noise), so callers can gate on it like the old
    autocorrelation confidence.
    """
    n = frame.size
    max_lag = min(max_lag, n - 1)
    if max_lag <= min_lag:
        return 0.0, 0.0

    # Raw autocorrelation r[tau] for tau in [0, max_lag] via FFT (fast).
    size = 1
    while size < 2 * n:
        size <<= 1
    spectrum = np.fft.rfft(frame, size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), size)[: max_lag + 1]

    # Exact difference function d[tau] = A[tau] + B[tau] - 2 r[tau], where A/B are
    # the energies of the two (shrinking) windows compared at lag tau.
    squared = frame * frame
    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    total = prefix[n]
    taus = np.arange(max_lag + 1)
    energy_head = prefix[n - taus]
    energy_tail = total - prefix[taus]
    difference = np.maximum(energy_head + energy_tail - 2.0 * autocorr, 0.0)

    # Cumulative mean normalized difference: d'[tau] = d[tau]*tau / sum_{1..tau} d.
    cmnd = np.ones(max_lag + 1, dtype=np.float64)
    running = np.cumsum(difference[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        cmnd[1:] = difference[1:] * taus[1:] / running
    cmnd[~np.isfinite(cmnd)] = 1.0

    window = cmnd[min_lag : max_lag + 1]
    if window.size == 0:
        return 0.0, 0.0

    # Absolute threshold: the first local minimum that dips below ``threshold``;
    # falling back to the global minimum keeps a (lower-confidence) estimate.
    is_local_min = np.ones(window.size, dtype=bool)
    is_local_min[1:] &= window[1:] <= window[:-1]
    is_local_min[:-1] &= window[:-1] <= window[1:]
    candidates = np.where((window < threshold) & is_local_min)[0]
    best = int(candidates[0]) if candidates.size else int(np.argmin(window))
    best_lag = min_lag + best

    refined = _parabolic_lag(cmnd, best_lag)
    if refined <= 0:
        return 0.0, 0.0
    confidence = float(max(0.0, min(1.0, 1.0 - cmnd[best_lag])))
    return sample_rate / refined, confidence


def _parabolic_lag(autocorr: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= autocorr.size - 1:
        return float(lag)

    left = float(autocorr[lag - 1])
    center = float(autocorr[lag])
    right = float(autocorr[lag + 1])
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-12:
        return float(lag)
    return float(lag + 0.5 * (left - right) / denominator)


def _thin_pitch_points(points: list[PitchPoint], max_points: int) -> list[PitchPoint]:
    if max_points <= 0 or len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, max_points)
    return [points[round(float(index))] for index in indices]


class _SampleRingBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer = np.empty(max(1, capacity), dtype=np.float32)
        self._start = 0
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def append(self, samples: np.ndarray) -> None:
        if not samples.size:
            return
        self._ensure_capacity(self._length + samples.size)
        end = (self._start + self._length) % self._buffer.size
        first = min(samples.size, self._buffer.size - end)
        self._buffer[end : end + first] = samples[:first]
        remaining = samples.size - first
        if remaining:
            self._buffer[:remaining] = samples[first:]
        self._length += samples.size

    def copy(self, count: int) -> np.ndarray:
        count = min(count, self._length)
        output = np.empty(count, dtype=np.float32)
        first = min(count, self._buffer.size - self._start)
        output[:first] = self._buffer[self._start : self._start + first]
        remaining = count - first
        if remaining:
            output[first:] = self._buffer[:remaining]
        return output

    def drop(self, count: int) -> None:
        count = min(count, self._length)
        self._start = (self._start + count) % self._buffer.size
        self._length -= count
        if self._length == 0:
            self._start = 0

    def take(self, count: int) -> np.ndarray:
        samples = self.copy(count)
        self.drop(count)
        return samples

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._buffer.size:
            return

        new_capacity = max(required, self._buffer.size * 2)
        next_buffer = np.empty(new_capacity, dtype=np.float32)
        if self._length:
            next_buffer[: self._length] = self.copy(self._length)
        self._buffer = next_buffer
        self._start = 0


class AudioWindowBuffer:
    def __init__(
        self,
        sample_rate: int = 16_000,
        window_seconds: float = 2.0,
        overlap_seconds: float = 0.5,
        rms_threshold: float = 0.008,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if overlap_seconds < 0 or overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be less than window_seconds")

        self.sample_rate = sample_rate
        self.window_samples = round(sample_rate * window_seconds)
        self.overlap_samples = round(sample_rate * overlap_seconds)
        self.step_samples = self.window_samples - self.overlap_samples
        self.rms_threshold = rms_threshold

        initial_capacity = max(self.window_samples * 2, self.step_samples * 4, 1)
        self._samples = _SampleRingBuffer(initial_capacity)
        self._absolute_start = 0
        self._next_index = 0

    @property
    def buffered_seconds(self) -> float:
        return self._samples.length / self.sample_rate

    def append_pcm16(self, payload: bytes) -> list[AudioWindow]:
        chunk = pcm16le_to_float32(payload)
        if chunk.size:
            self._samples.append(chunk)
        return self.pop_ready()

    def pop_ready(self) -> list[AudioWindow]:
        windows: list[AudioWindow] = []
        while self._samples.length >= self.window_samples:
            samples = self._samples.copy(self.window_samples)
            rms = calculate_rms(samples)
            start = self._absolute_start / self.sample_rate
            end = (self._absolute_start + self.window_samples) / self.sample_rate
            windows.append(
                AudioWindow(
                    index=self._next_index,
                    start_seconds=start,
                    end_seconds=end,
                    samples=samples,
                    rms=rms,
                    is_speech=rms >= self.rms_threshold,
                )
            )
            self._samples.drop(self.step_samples)
            self._absolute_start += self.step_samples
            self._next_index += 1
        return windows


class AudioUtteranceBuffer:
    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 100,
        pre_roll_ms: int = 300,
        end_silence_ms: int = 700,
        max_segment_seconds: float = 12.0,
        rms_threshold: float = 0.008,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if pre_roll_ms < 0:
            raise ValueError("pre_roll_ms must not be negative")
        if end_silence_ms <= 0:
            raise ValueError("end_silence_ms must be positive")
        if max_segment_seconds <= 0:
            raise ValueError("max_segment_seconds must be positive")

        self.sample_rate = sample_rate
        self.frame_samples = max(1, round(sample_rate * frame_ms / 1000))
        self.pre_roll_frames = round(pre_roll_ms / frame_ms)
        self.end_silence_frames = max(1, round(end_silence_ms / frame_ms))
        self.max_segment_samples = max(self.frame_samples, round(sample_rate * max_segment_seconds))
        self.rms_threshold = rms_threshold

        self._pending = _SampleRingBuffer(max(self.frame_samples * 8, 1))
        self._pre_roll: list[np.ndarray] = []
        self._segment_chunks: list[np.ndarray] = []
        self._segment_start_sample: int | None = None
        self._segment_sample_count = 0
        self._silence_frames = 0
        self._absolute_position = 0
        self._next_index = 0

    @property
    def buffered_seconds(self) -> float:
        active = self._segment_sample_count + self._pending.length
        return active / self.sample_rate

    def append_pcm16(self, payload: bytes) -> list[AudioWindow]:
        chunk = pcm16le_to_float32(payload)
        if chunk.size:
            self._pending.append(chunk)

        windows: list[AudioWindow] = []
        while self._pending.length >= self.frame_samples:
            frame_start = self._absolute_position
            frame = self._pending.take(self.frame_samples)
            self._absolute_position += frame.size
            emitted = self._process_frame(frame, frame_start)
            if emitted is not None:
                windows.append(emitted)
        return windows

    def flush(self) -> list[AudioWindow]:
        if self._pending.length:
            frame_start = self._absolute_position
            frame = self._pending.take(self._pending.length)
            self._absolute_position += frame.size
            emitted = self._process_frame(frame, frame_start)
            if emitted is not None:
                return [emitted]

        emitted = self._flush_active(remember_tail=False)
        return [emitted] if emitted is not None else []

    def _process_frame(self, frame: np.ndarray, frame_start: int) -> AudioWindow | None:
        rms = calculate_rms(frame)
        is_speech = rms >= self.rms_threshold
        emitted: AudioWindow | None = None

        if is_speech:
            if not self._segment_chunks:
                pre_roll_samples = sum(chunk.size for chunk in self._pre_roll)
                self._segment_start_sample = max(0, frame_start - pre_roll_samples)
                self._segment_chunks = [chunk.copy() for chunk in self._pre_roll]
                self._segment_sample_count = pre_roll_samples
            self._append_segment_frame(frame)
            self._silence_frames = 0
        elif self._segment_chunks:
            self._append_segment_frame(frame)
            self._silence_frames += 1
            if self._silence_frames >= self.end_silence_frames:
                emitted = self._flush_active(remember_tail=True)
        else:
            self._remember_pre_roll(frame)

        if self._segment_sample_count >= self.max_segment_samples:
            emitted = self._flush_active(remember_tail=True)

        return emitted

    def _append_segment_frame(self, frame: np.ndarray) -> None:
        self._segment_chunks.append(frame)
        self._segment_sample_count += frame.size

    def _remember_pre_roll(self, frame: np.ndarray) -> None:
        if self.pre_roll_frames <= 0:
            return
        self._pre_roll.append(frame.copy())
        if len(self._pre_roll) > self.pre_roll_frames:
            del self._pre_roll[: len(self._pre_roll) - self.pre_roll_frames]

    def _flush_active(self, remember_tail: bool) -> AudioWindow | None:
        if not self._segment_chunks or self._segment_start_sample is None:
            return None

        tail = (
            self._segment_chunks[-self.pre_roll_frames :]
            if remember_tail and self.pre_roll_frames
            else []
        )
        samples = np.concatenate(self._segment_chunks).astype(np.float32, copy=False)
        rms = calculate_rms(samples)
        start_sample = self._segment_start_sample
        end_sample = start_sample + samples.size
        window = AudioWindow(
            index=self._next_index,
            start_seconds=start_sample / self.sample_rate,
            end_seconds=end_sample / self.sample_rate,
            samples=samples,
            rms=rms,
            is_speech=True,
            kind="utterance",
        )
        self._next_index += 1

        self._segment_chunks = []
        self._segment_start_sample = None
        self._segment_sample_count = 0
        self._silence_frames = 0
        self._pre_roll = []
        for chunk in tail:
            self._remember_pre_roll(chunk)
        return window
