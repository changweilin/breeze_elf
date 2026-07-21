from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

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
    # Where the slide happens, as a fraction (0–1) of the segment, found from the
    # pitch's rate of change; ``None`` for a steady (non-glide) note.
    glide_position: float | None = None


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


def compute_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    *,
    max_hz: float = 2000.0,
    time_bins: int = 160,
    freq_bins: int = 64,
    n_fft: int = 1024,
    min_hz: float = 70.0,
    f0_max_hz: float = 500.0,
    rms_threshold: float = 0.01,
    yin_threshold: float = 0.15,
    confidence_threshold: float = 0.4,
) -> dict | None:
    """STFT magnitude spectrogram + an aligned per-frame f0 track for 基頻分析.

    Returns a compact, JSON-friendly payload: the magnitudes are pooled to
    ``time_bins × freq_bins``, converted to dB, normalized to 0–1 and quantized
    to ``uint8`` (base64). ``f0`` holds one fundamental per time bin (``None``
    where the frame is non-voiced) sharing the spectrogram's time axis, so the
    frontend can overlay the continuous 基頻 curve directly on the image.
    """
    if sample_rate <= 0 or samples.size == 0 or n_fft <= 0:
        return None
    window = np.hanning(n_fft).astype(np.float64)
    if samples.size >= n_fft:
        usable = samples.size - n_fft
        hop = max(1, usable // max(1, time_bins - 1)) if time_bins > 1 else usable + 1
        starts = list(range(0, usable + 1, hop))[:time_bins]
    else:
        starts = [0]
    if not starts:
        starts = [0]

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    keep = freqs <= max_hz
    nkeep = int(np.count_nonzero(keep))
    if nkeep < 2:
        return None

    min_lag = max(1, round(sample_rate / f0_max_hz))
    max_lag = max(min_lag + 1, round(sample_rate / min_hz))
    can_track_f0 = n_fft > max_lag

    mags = np.empty((len(starts), nkeep), dtype=np.float64)
    f0: list[float | None] = []
    intensity: list[float] = []
    times: list[float] = []
    for index, start in enumerate(starts):
        raw = samples[start : start + n_fft]
        segment = raw
        if segment.size < n_fft:
            segment = np.pad(segment, (0, n_fft - segment.size))
        segment = segment.astype(np.float64)
        mags[index] = np.abs(np.fft.rfft(segment * window))[keep]
        intensity.append(round(calculate_rms(raw), 5))
        times.append(round((start + n_fft / 2) / sample_rate, 3))

        hz: float | None = None
        centered = segment - segment.mean()
        if can_track_f0 and calculate_rms(centered) >= rms_threshold:
            candidate, confidence = _yin_pitch(
                centered, sample_rate, min_lag, max_lag, yin_threshold
            )
            in_range = min_hz <= candidate <= f0_max_hz
            if candidate > 0 and confidence >= confidence_threshold and in_range:
                hz = round(float(candidate), 1)
        f0.append(hz)

    if nkeep > freq_bins:
        edges = np.linspace(0, nkeep, freq_bins + 1).astype(int)
        pooled = np.empty((mags.shape[0], freq_bins), dtype=np.float64)
        for band in range(freq_bins):
            low = edges[band]
            high = max(low + 1, edges[band + 1])
            pooled[:, band] = mags[:, low:high].mean(axis=1)
        mags = pooled
        out_freq_bins = freq_bins
    else:
        out_freq_bins = nkeep

    db = 20.0 * np.log10(np.maximum(mags, 1e-6))
    db -= float(db.max())
    db = np.clip(db, -80.0, 0.0)
    quant = np.clip((db + 80.0) / 80.0 * 255.0, 0.0, 255.0).astype(np.uint8)

    return {
        "timeBins": int(quant.shape[0]),
        "freqBins": int(out_freq_bins),
        "maxHz": round(float(freqs[keep][-1]), 1),
        "f0MaxHz": float(f0_max_hz),
        "magnitudes": base64.b64encode(np.ascontiguousarray(quant).tobytes()).decode("ascii"),
        "f0": f0,
        "intensity": intensity,
        "times": times,
    }


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


def _glide_edges(
    hz_points: list[float], edge_fraction: float
) -> tuple[float | None, float | None, float | None]:
    """Locate a slide from the pitch track's *rate of change*.

    Returns ``(start_hz, end_hz, position)``. The start/end pitches are the medians
    of the stable plateaus before and after the move (so they are not skewed by
    the slide itself, unlike a fixed front-/back-third), and ``position`` is the
    fraction of the segment where the pitch crosses halfway between them — the real
    place the glide happens. ``position`` is ``None`` for a steady note (the
    endpoints sit within a semitone, so it is not a glide)."""
    n = len(hz_points)
    if n == 0:
        return None, None, None
    if n < 4:
        edge = max(1, round(n * edge_fraction))
        return float(np.median(hz_points[:edge])), float(np.median(hz_points[-edge:])), None

    logs = np.log2(np.asarray(hz_points, dtype=np.float64))
    total = float(logs[-1] - logs[0])
    if abs(total) * 12.0 < 1.0:  # < 1 semitone end-to-end → steady, not a glide
        med = float(np.median(hz_points))
        return med, med, None

    tol = 1.0 / 12.0  # one semitone, in octaves
    s_end = 0
    while s_end + 1 < n and abs(logs[s_end + 1] - logs[0]) <= tol:
        s_end += 1
    e_start = n - 1
    while e_start - 1 >= 0 and abs(logs[e_start - 1] - logs[-1]) <= tol:
        e_start -= 1
    if e_start <= s_end:  # plateaus meet (a continuous slide) → split at the middle
        s_end = max(0, n // 2 - 1)
        e_start = min(n - 1, s_end + 1)

    start_hz = float(np.median(hz_points[: s_end + 1]))
    end_hz = float(np.median(hz_points[e_start:]))
    half = logs[0] + total / 2.0
    cross = n // 2
    for index in range(n):
        if (total > 0 and logs[index] >= half) or (total < 0 and logs[index] <= half):
            cross = index
            break
    position = min(0.9, max(0.1, cross / (n - 1)))
    return start_hz, end_hz, position


def analyze_segment(
    samples: np.ndarray,
    sample_rate: int,
    *,
    edge_fraction: float = 0.34,
) -> SegmentAnalysis:
    """Measure pitch range, slide edges/position, and loudness envelope."""
    if samples.size < 2:
        return SegmentAnalysis(None, None, None, None, None, 0.0, 0.0, 0.0)

    summary = summarize_pitch(samples, sample_rate)
    hz_points = [point.hz for point in summary.points if point.hz]
    start_hz, end_hz, glide_position = _glide_edges(hz_points, edge_fraction)

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
        glide_position=glide_position,
    )


def estimate_noise_floor(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    percentile: float = 10.0,
) -> float:
    """The room-tone RMS: the ``percentile``-th quietest short frame.

    Word edges (unvoiced consonants, aspiration) sit *below* the speech threshold
    but clearly *above* this floor, so :func:`extend_voiced_span` can tell them
    apart from true silence."""
    if sample_rate <= 0 or samples.size == 0:
        return 0.0
    frame = max(1, round(sample_rate * frame_ms / 1000.0))
    if samples.size < frame:
        return calculate_rms(samples)
    starts = range(0, samples.size - frame + 1, frame)
    rms = np.array([calculate_rms(samples[start : start + frame]) for start in starts])
    if rms.size == 0:
        return calculate_rms(samples)
    return float(np.percentile(rms, percentile))


def extend_voiced_span(
    samples: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    *,
    floor_sample: int,
    ceil_sample: int,
    noise_floor: float,
    frame_ms: float = 10.0,
    energy_margin: float = 1.6,
) -> tuple[int, int]:
    """Grow ``[start_sample, end_sample]`` outward through audio that still belongs
    to the 字.

    The live VAD onsets/offsets a segment on an RMS *speech* threshold, so a 字's
    leading unvoiced consonant (s/sh/f/h…) or trailing breath — loud enough to be
    word structure but quieter than the threshold — gets clipped. Here, during
    post-processing, each edge is walked one short frame at a time and kept while
    its RMS stays above ``noise_floor * energy_margin`` (above room tone), stopping
    at the first truly silent frame or at ``[floor_sample, ceil_sample]`` (the
    neighbour midpoints / block edges, so adjacent 字 never merge)."""
    n = samples.size
    floor_sample = max(0, floor_sample)
    ceil_sample = min(n, ceil_sample)
    start_sample = max(floor_sample, min(start_sample, n))
    end_sample = max(start_sample, min(end_sample, ceil_sample))
    if n == 0:
        return start_sample, end_sample

    frame = max(1, round(sample_rate * frame_ms / 1000.0))
    threshold = max(noise_floor * energy_margin, 1e-6)

    new_start = start_sample
    while new_start - frame >= floor_sample:
        if calculate_rms(samples[new_start - frame : new_start]) < threshold:
            break
        new_start -= frame

    new_end = end_sample
    while new_end + frame <= ceil_sample:
        if calculate_rms(samples[new_end : new_end + frame]) < threshold:
            break
        new_end += frame

    return new_start, new_end


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


class SpeechDetector(Protocol):
    """Per-frame speech/not-speech decision the utterance segmenter onsets on.

    ``is_speech`` is called once per fixed-size frame in stream order; a detector
    may keep state across calls (Silero's LSTM does), so a fresh detector is built
    per connection and never shared between streams."""

    name: str

    def is_speech(self, frame: np.ndarray) -> bool:
        ...


# A speech segment's *offset* is gated a factor below its *onset* by default, so a
# sentence-final syllable trailing off in volume (自然的句尾漸弱) is still held as
# speech down to this lower level instead of being cut the instant it dips under the
# onset gate. Onset stays at the full threshold, so quiet steady noise can't start a
# segment — only the tail of already-detected speech is extended.
_RMS_RELEASE_RATIO = 0.5


class RmsSpeechDetector:
    """The original energy gate, now with attack/release hysteresis (a Schmitt
    trigger). A frame *starts* speech when its RMS clears ``rms_threshold`` and
    *keeps* being speech until RMS falls below the lower ``release_threshold``.

    Cheap, and can't tell speech from equally-loud steady noise (冷氣/路噪) — the
    reason the Silero detector exists — but the asymmetric gate stops a naturally
    decaying 句尾 syllable (whose RMS slides under the onset gate while it is still
    being spoken) from being clipped, which the single-threshold version dropped.
    Onset is unchanged, so noise rejection at the start of a segment is unaffected."""

    name = "rms"

    def __init__(self, rms_threshold: float, *, release_threshold: float | None = None) -> None:
        self.rms_threshold = float(rms_threshold)
        release = (
            self.rms_threshold * _RMS_RELEASE_RATIO
            if release_threshold is None
            else float(release_threshold)
        )
        # Keep release in [0, onset]: above onset would invert the hysteresis (offset
        # harder than onset), below 0 is meaningless.
        self.release_threshold = max(0.0, min(release, self.rms_threshold))
        self._active = False

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = calculate_rms(frame)
        # Once speaking, hold speech down to the lower release gate; when silent,
        # require the full onset gate to (re)start — the asymmetry is the fix.
        self._active = rms >= (self.release_threshold if self._active else self.rms_threshold)
        return self._active


# Silero v6 (as bundled by faster-whisper) is a 16 kHz model that scores exactly
# 512-sample (32 ms) chunks, each prepended with the previous chunk's last 64
# samples as context, and carries an LSTM hidden/cell state (1, 1, 128) forward.
_SILERO_CHUNK_SAMPLES = 512
_SILERO_CONTEXT_SAMPLES = 64
_SILERO_STATE = (1, 1, 128)


def _default_silero_model_path() -> str | None:
    """Path to the silero VAD ONNX that ships inside faster-whisper's assets, or
    ``None`` when faster-whisper isn't importable. Globbed (not hard-coded) so a
    version bump renaming ``silero_vad_v6.onnx`` doesn't silently break loading."""
    try:
        from faster_whisper.utils import get_assets_path
    except Exception:  # pragma: no cover - faster-whisper optional in dev
        return None

    import glob
    import os

    assets = get_assets_path()
    for pattern in ("silero*vad*.onnx", "*silero*.onnx", "*vad*.onnx"):
        matches = sorted(glob.glob(os.path.join(assets, pattern)))
        if matches:
            return matches[0]
    return None


class SileroSpeechDetector:
    """Streaming Silero VAD over onnxruntime. Judges *voice*, not loudness, so the
    far-field phone false-triggers that energy VAD produces on steady noise go away.

    faster-whisper bundles both the ONNX model and onnxruntime, so this adds no new
    dependency. Its own ``SileroVADModel`` resets state every call (batch use); here
    the LSTM state and the 64-sample context are carried across frames for true
    streaming. Construction never raises — an unavailable model/runtime is reported
    via :attr:`available` so the factory can fall back to RMS, and any *runtime*
    inference failure degrades this detector to an internal RMS gate (logged once).
    """

    name = "silero"

    def __init__(
        self,
        *,
        rms_threshold: float,
        rms_release_threshold: float | None = None,
        speech_threshold: float = 0.5,
        neg_threshold: float | None = None,
        model_path: str | None = None,
    ) -> None:
        # The RMS fallback (below) shares the same attack/release hysteresis as the
        # standalone RmsSpeechDetector so a mid-stream degrade doesn't regress the
        # 句尾漸弱 handling back to a bare threshold.
        self._rms_gate = RmsSpeechDetector(rms_threshold, release_threshold=rms_release_threshold)
        self.rms_threshold = rms_threshold
        self.speech_threshold = float(speech_threshold)
        # Hysteresis low edge: between neg and speech thresholds the previous
        # decision is held, so a wavering probability doesn't chatter the segmenter.
        self.neg_threshold = (
            float(neg_threshold)
            if neg_threshold is not None
            else max(0.0, self.speech_threshold - 0.15)
        )
        self.available = False
        self.unavailable_reason: str | None = None
        self._degraded = False
        self._session = None
        self._pending = np.empty(0, dtype=np.float32)
        self._context = np.zeros(_SILERO_CONTEXT_SAMPLES, dtype=np.float32)
        self._h = np.zeros(_SILERO_STATE, dtype=np.float32)
        self._c = np.zeros(_SILERO_STATE, dtype=np.float32)
        self._last_speech = False

        path = model_path or _default_silero_model_path()
        if not path:
            self.unavailable_reason = "silero VAD model not found in faster-whisper assets"
            return
        try:
            import onnxruntime
        except Exception as exc:  # pragma: no cover - onnxruntime optional in dev
            self.unavailable_reason = f"onnxruntime import failed: {exc}"
            return
        try:
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.enable_cpu_mem_arena = False
            opts.log_severity_level = 4
            self._session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            self.unavailable_reason = f"onnxruntime session failed: {exc}"

    def is_speech(self, frame: np.ndarray) -> bool:
        if self._degraded or self._session is None:
            return self._rms_gate.is_speech(frame)

        if frame.dtype != np.float32:
            frame = frame.astype(np.float32, copy=False)
        self._pending = (
            frame if self._pending.size == 0 else np.concatenate([self._pending, frame])
        )

        max_prob: float | None = None
        while self._pending.size >= _SILERO_CHUNK_SAMPLES:
            chunk = self._pending[:_SILERO_CHUNK_SAMPLES]
            self._pending = self._pending[_SILERO_CHUNK_SAMPLES:]
            try:
                prob = self._infer_chunk(chunk)
            except Exception as exc:  # pragma: no cover - depends on runtime
                logger.warning("silero VAD inference failed; falling back to RMS: %s", exc)
                self._degraded = True
                return self._rms_gate.is_speech(frame)
            max_prob = prob if max_prob is None else max(max_prob, prob)

        if max_prob is None:
            # Fewer than one chunk buffered so far — hold the previous decision.
            return self._last_speech
        if max_prob >= self.speech_threshold:
            self._last_speech = True
        elif max_prob < self.neg_threshold:
            self._last_speech = False
        # else: within the hysteresis band — keep _last_speech unchanged.
        return self._last_speech

    def _infer_chunk(self, chunk: np.ndarray) -> float:
        model_input = np.concatenate([self._context, chunk]).reshape(1, -1).astype(np.float32)
        output, self._h, self._c = self._session.run(
            None, {"input": model_input, "h": self._h, "c": self._c}
        )
        self._context = chunk[-_SILERO_CONTEXT_SAMPLES:].copy()
        return float(np.asarray(output).reshape(-1)[-1])


def make_speech_detector(
    kind: str,
    *,
    rms_threshold: float,
    rms_release_threshold: float | None = None,
    speech_threshold: float = 0.5,
    neg_threshold: float | None = None,
    model_path: str | None = None,
) -> SpeechDetector:
    """Build the requested detector, degrading to RMS when Silero can't load so a
    missing model/runtime never breaks recognition — only reverts to energy VAD."""
    if kind == "silero":
        detector = SileroSpeechDetector(
            rms_threshold=rms_threshold,
            rms_release_threshold=rms_release_threshold,
            speech_threshold=speech_threshold,
            neg_threshold=neg_threshold,
            model_path=model_path,
        )
        if detector.available:
            return detector
        logger.warning(
            "silero VAD unavailable (%s); using RMS energy VAD instead",
            detector.unavailable_reason,
        )
    return RmsSpeechDetector(rms_threshold, release_threshold=rms_release_threshold)


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


# When a held phrase forces a max-segment split, the cut is searched within this
# fraction of the segment (its tail region) so the head stays close to the cap yet
# still ends at a quiet point rather than through a sung note.
_SPLIT_SEARCH_LO = 0.75
_SPLIT_SEARCH_HI = 0.98


class AudioUtteranceBuffer:
    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 100,
        pre_roll_ms: int = 300,
        end_silence_ms: int = 700,
        max_segment_seconds: float = 12.0,
        rms_threshold: float = 0.008,
        speech_detector: SpeechDetector | None = None,
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
        self._detector = speech_detector or RmsSpeechDetector(rms_threshold)

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

    @property
    def detector_name(self) -> str:
        return getattr(self._detector, "name", "rms")

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
        # Voice-vs-not (Silero) or energy-vs-threshold (RMS) — the detector decides;
        # the segment/pre-roll/silence machinery below is identical either way.
        is_speech = self._detector.is_speech(frame)
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
            emitted = self._force_split()

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

    def _build_window(self, chunks: list[np.ndarray], start_sample: int) -> AudioWindow:
        samples = np.concatenate(chunks).astype(np.float32, copy=False)
        end_sample = start_sample + samples.size
        window = AudioWindow(
            index=self._next_index,
            start_seconds=start_sample / self.sample_rate,
            end_seconds=end_sample / self.sample_rate,
            samples=samples,
            rms=calculate_rms(samples),
            is_speech=True,
            kind="utterance",
        )
        self._next_index += 1
        return window

    def _reset_segment(self) -> None:
        self._segment_chunks = []
        self._segment_start_sample = None
        self._segment_sample_count = 0
        self._silence_frames = 0
        self._pre_roll = []

    def _flush_active(self, remember_tail: bool) -> AudioWindow | None:
        if not self._segment_chunks or self._segment_start_sample is None:
            return None

        tail = (
            self._segment_chunks[-self.pre_roll_frames :]
            if remember_tail and self.pre_roll_frames
            else []
        )
        window = self._build_window(self._segment_chunks, self._segment_start_sample)
        self._reset_segment()
        for chunk in tail:
            self._remember_pre_roll(chunk)
        return window

    def _force_split(self) -> AudioWindow | None:
        """Flush at the max-segment cap, but cut at the quietest recent frame (a
        syllable / word gap) instead of through whatever sound is playing, and carry
        the remainder over as the start of the next utterance.

        The live VAD never ends a continuously sung phrase (no 700 ms silence), so it
        previously hard-cut at exactly the cap — clipping a held note mid-sound and
        leaving the 歌詞 unfinished. Splitting at a quiet point keeps each piece a
        whole phrase and lets the carried-over tail finish the line in the next block.
        """
        if not self._segment_chunks or self._segment_start_sample is None:
            return None

        samples = np.concatenate(self._segment_chunks).astype(np.float32, copy=False)
        cut = self._best_split_sample(samples)
        if cut <= 0 or cut >= samples.size:
            return self._flush_active(remember_tail=True)

        start_sample = self._segment_start_sample
        head = samples[:cut]
        tail = samples[cut:]
        window = self._build_window([head], start_sample)

        # The tail continues contiguously (no overlap), so it is a clean new segment.
        # Re-chunk it into frame-sized pieces so the silence / pre-roll bookkeeping
        # (which counts in frames) keeps working.
        self._segment_chunks = [
            tail[index : index + self.frame_samples].copy()
            for index in range(0, tail.size, self.frame_samples)
        ] or [tail.copy()]
        self._segment_start_sample = start_sample + int(cut)
        self._segment_sample_count = int(tail.size)
        self._silence_frames = 0
        return window

    def _best_split_sample(self, samples: np.ndarray) -> int:
        """Sample to cut at when force-splitting: the *latest* point in the segment's
        tail region quiet enough to be a syllable / word gap (so the head stays close
        to the cap), or the deepest dip there when a held phrase never falls that
        quiet. Probed on a fine (~25 ms) window so short, legato dips that straddle
        frame boundaries are still found."""
        n = samples.size
        if n < self.frame_samples * 4:
            return n
        lo = int(n * _SPLIT_SEARCH_LO)
        hi = int(n * _SPLIT_SEARCH_HI)
        probe = max(1, self.frame_samples // 4)
        hop = max(1, probe // 2)
        quiet_threshold = max(self.rms_threshold * 1.5, calculate_rms(samples) * 0.5)

        latest_gap: int | None = None
        min_center = lo
        min_rms: float | None = None
        position = lo
        while position <= hi:
            value = calculate_rms(samples[position : position + probe])
            center = position + probe // 2
            if value <= quiet_threshold:
                latest_gap = center
            if min_rms is None or value < min_rms:
                min_rms = value
                min_center = center
            position += hop
        return latest_gap if latest_gap is not None else min_center
