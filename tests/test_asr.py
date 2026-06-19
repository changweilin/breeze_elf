import unittest

import numpy as np

from breeze_elf.asr import MockASR


class ASRTests(unittest.TestCase):
    def test_mock_asr(self):
        engine = MockASR()
        result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000, "zh")
        self.assertEqual(result.backend, "mock")
        self.assertIn("測試字幕", result.text)
        self.assertEqual(result.language, "zh")

    def test_mock_asr_emits_monotonic_word_timings(self):
        engine = MockASR()
        result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000, "zh")

        self.assertGreater(len(result.words), 0)
        self.assertTrue(all(" " != word.word for word in result.words))
        self.assertAlmostEqual(result.words[0].start, 0.0, places=3)
        for previous, nxt in zip(result.words, result.words[1:]):
            self.assertLessEqual(previous.start, nxt.start)
            self.assertLessEqual(previous.end, nxt.end + 1e-6)


if __name__ == "__main__":
    unittest.main()

