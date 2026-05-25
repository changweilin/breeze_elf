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

        self._samples = np.empty(0, dtype=np.float32)
        self._absolute_start = 0
        self._next_index = 0

    @property
    def buffered_seconds(self) -> float:
        return self._samples.size / self.sample_rate

    def append_pcm16(self, payload: bytes) -> list[AudioWindow]:
        chunk = pcm16le_to_float32(payload)
        if chunk.size:
            self._samples = np.concatenate((self._samples, chunk))
        return self.pop_ready()

    def pop_ready(self) -> list[AudioWindow]:
        windows: list[AudioWindow] = []
        while self._samples.size >= self.window_samples:
            samples = self._samples[: self.window_samples].copy()
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
            self._samples = self._samples[self.step_samples :]
            self._absolute_start += self.step_samples
            self._next_index += 1
        return windows

