from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    max_points: int = 80,
) -> PitchSummary:
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

    analysis_window = np.hanning(frame_samples)
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

        autocorr = np.correlate(frame * analysis_window, frame * analysis_window, mode="full")
        autocorr = autocorr[frame_samples - 1 :]
        energy = float(autocorr[0])
        search_end = min(max_lag, autocorr.size - 1)
        if energy <= 0 or search_end <= min_lag:
            continue

        search = autocorr[min_lag : search_end + 1]
        lag = min_lag + int(np.argmax(search))
        confidence = float(autocorr[lag] / energy)
        if confidence < confidence_threshold:
            continue

        refined_lag = _parabolic_lag(autocorr, lag)
        if refined_lag <= 0:
            continue

        hz = sample_rate / refined_lag
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
