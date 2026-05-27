import unittest

import numpy as np

from breeze_elf.audio import (
    AudioUtteranceBuffer,
    AudioWindowBuffer,
    calculate_rms,
    pcm16le_to_float32,
    summarize_pitch,
)


class AudioTests(unittest.TestCase):
    def test_pcm16le_to_float32(self):
        payload = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        samples = pcm16le_to_float32(payload)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(float(samples[0]), -1.0, places=5)
        self.assertAlmostEqual(float(samples[1]), 0.0, places=5)
        self.assertAlmostEqual(float(samples[2]), 32767 / 32768, places=5)

    def test_summarize_pitch_detects_sine_frequency(self):
        sample_rate = 16_000
        seconds = 0.6
        frequency = 220.0
        time_axis = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
        samples = (np.sin(2 * np.pi * frequency * time_axis) * 0.4).astype(np.float32)

        summary = summarize_pitch(samples, sample_rate)

        self.assertIsNotNone(summary.median_hz)
        self.assertAlmostEqual(summary.median_hz, frequency, delta=4.0)
        self.assertGreater(summary.voiced_ratio, 0.9)
        self.assertGreater(len(summary.points), 0)

    def test_summarize_pitch_returns_empty_for_silence(self):
        summary = summarize_pitch(np.zeros(16_000, dtype=np.float32), 16_000)

        self.assertIsNone(summary.median_hz)
        self.assertEqual(summary.voiced_ratio, 0.0)
        self.assertEqual(summary.points, ())

    def test_silence_gate(self):
        buffer = AudioWindowBuffer(
            sample_rate=4,
            window_seconds=1.0,
            overlap_seconds=0.25,
            rms_threshold=0.1,
        )
        payload = np.zeros(4, dtype="<i2").tobytes()
        windows = buffer.append_pcm16(payload)
        self.assertEqual(len(windows), 1)
        self.assertFalse(windows[0].is_speech)
        self.assertEqual(calculate_rms(windows[0].samples), 0.0)

    def test_emits_overlapping_windows(self):
        buffer = AudioWindowBuffer(
            sample_rate=4,
            window_seconds=1.0,
            overlap_seconds=0.25,
            rms_threshold=0.0,
        )
        payload = np.full(7, 1000, dtype="<i2").tobytes()
        windows = buffer.append_pcm16(payload)
        self.assertEqual([window.index for window in windows], [0, 1])
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[1].start_seconds, 0.75)
        self.assertAlmostEqual(buffer.buffered_seconds, 0.25)

    def test_ring_buffer_preserves_overlapping_samples_after_wrap(self):
        buffer = AudioWindowBuffer(
            sample_rate=4,
            window_seconds=1.0,
            overlap_seconds=0.5,
            rms_threshold=0.0,
        )

        first = buffer.append_pcm16(np.array([1, 2, 3, 4], dtype="<i2").tobytes())[0]
        second = buffer.append_pcm16(np.array([5, 6], dtype="<i2").tobytes())[0]
        third = buffer.append_pcm16(np.array([7, 8], dtype="<i2").tobytes())[0]
        fourth = buffer.append_pcm16(np.array([9, 10], dtype="<i2").tobytes())[0]

        scale = 1 / 32768
        np.testing.assert_allclose(first.samples, np.array([1, 2, 3, 4], dtype=np.float32) * scale)
        np.testing.assert_allclose(second.samples, np.array([3, 4, 5, 6], dtype=np.float32) * scale)
        np.testing.assert_allclose(third.samples, np.array([5, 6, 7, 8], dtype=np.float32) * scale)
        np.testing.assert_allclose(
            fourth.samples,
            np.array([7, 8, 9, 10], dtype=np.float32) * scale,
        )
        self.assertAlmostEqual(buffer.buffered_seconds, 0.5)

    def test_utterance_buffer_emits_speech_after_trailing_silence(self):
        buffer = AudioUtteranceBuffer(
            sample_rate=10,
            frame_ms=100,
            pre_roll_ms=200,
            end_silence_ms=200,
            rms_threshold=0.01,
        )
        payload = np.array([0, 0, 1000, 1000, 1000, 0, 0, 0], dtype="<i2").tobytes()

        windows = buffer.append_pcm16(payload)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].kind, "utterance")
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].end_seconds, 0.7)
        self.assertEqual(windows[0].samples.size, 7)
        self.assertTrue(windows[0].is_speech)

    def test_utterance_buffer_flushes_active_speech(self):
        buffer = AudioUtteranceBuffer(
            sample_rate=10,
            frame_ms=100,
            pre_roll_ms=100,
            end_silence_ms=300,
            rms_threshold=0.01,
        )

        self.assertEqual(buffer.append_pcm16(np.array([0, 1000, 1000], dtype="<i2").tobytes()), [])
        windows = buffer.flush()

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].samples.size, 3)


if __name__ == "__main__":
    unittest.main()
