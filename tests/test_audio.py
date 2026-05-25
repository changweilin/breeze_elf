import unittest

import numpy as np

from breeze_elf.audio import AudioWindowBuffer, calculate_rms, pcm16le_to_float32


class AudioTests(unittest.TestCase):
    def test_pcm16le_to_float32(self):
        payload = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        samples = pcm16le_to_float32(payload)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(float(samples[0]), -1.0, places=5)
        self.assertAlmostEqual(float(samples[1]), 0.0, places=5)
        self.assertAlmostEqual(float(samples[2]), 32767 / 32768, places=5)

    def test_silence_gate(self):
        buffer = AudioWindowBuffer(sample_rate=4, window_seconds=1.0, overlap_seconds=0.25, rms_threshold=0.1)
        payload = np.zeros(4, dtype="<i2").tobytes()
        windows = buffer.append_pcm16(payload)
        self.assertEqual(len(windows), 1)
        self.assertFalse(windows[0].is_speech)
        self.assertEqual(calculate_rms(windows[0].samples), 0.0)

    def test_emits_overlapping_windows(self):
        buffer = AudioWindowBuffer(sample_rate=4, window_seconds=1.0, overlap_seconds=0.25, rms_threshold=0.0)
        payload = np.full(7, 1000, dtype="<i2").tobytes()
        windows = buffer.append_pcm16(payload)
        self.assertEqual([window.index for window in windows], [0, 1])
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[1].start_seconds, 0.75)
        self.assertAlmostEqual(buffer.buffered_seconds, 0.25)


if __name__ == "__main__":
    unittest.main()

