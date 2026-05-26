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
        self._buffer = np.empty(initial_capacity, dtype=np.float32)
        self._start = 0
        self._length = 0
        self._absolute_start = 0
        self._next_index = 0

    @property
    def buffered_seconds(self) -> float:
        return self._length / self.sample_rate

    def append_pcm16(self, payload: bytes) -> list[AudioWindow]:
        chunk = pcm16le_to_float32(payload)
        if chunk.size:
            self._append_samples(chunk)
        return self.pop_ready()

    def pop_ready(self) -> list[AudioWindow]:
        windows: list[AudioWindow] = []
        while self._length >= self.window_samples:
            samples = self._copy_samples(self.window_samples)
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
            self._drop_samples(self.step_samples)
            self._absolute_start += self.step_samples
            self._next_index += 1
        return windows

    def _append_samples(self, samples: np.ndarray) -> None:
        self._ensure_capacity(self._length + samples.size)
        end = (self._start + self._length) % self._buffer.size
        first = min(samples.size, self._buffer.size - end)
        self._buffer[end : end + first] = samples[:first]
        remaining = samples.size - first
        if remaining:
            self._buffer[:remaining] = samples[first:]
        self._length += samples.size

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._buffer.size:
            return

        new_capacity = max(required, self._buffer.size * 2)
        next_buffer = np.empty(new_capacity, dtype=np.float32)
        if self._length:
            next_buffer[: self._length] = self._copy_samples(self._length)
        self._buffer = next_buffer
        self._start = 0

    def _copy_samples(self, count: int) -> np.ndarray:
        count = min(count, self._length)
        output = np.empty(count, dtype=np.float32)
        first = min(count, self._buffer.size - self._start)
        output[:first] = self._buffer[self._start : self._start + first]
        remaining = count - first
        if remaining:
            output[first:] = self._buffer[:remaining]
        return output

    def _drop_samples(self, count: int) -> None:
        count = min(count, self._length)
        self._start = (self._start + count) % self._buffer.size
        self._length -= count
        if self._length == 0:
            self._start = 0
