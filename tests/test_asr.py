import unittest
from types import SimpleNamespace

import numpy as np

from breeze_elf.asr import TRADITIONAL_CHINESE_PROMPT, FasterWhisperASR, MockASR


class _StubWhisperModel:
    """Records the kwargs faster-whisper would receive, no real model loaded."""

    def __init__(self, detected_language="en"):
        self.detected_language = detected_language
        self.calls = []

    def transcribe(self, samples, **kwargs):
        del samples
        self.calls.append(kwargs)
        info = SimpleNamespace(language=self.detected_language)
        return [], info


def _fake_engine(model):
    engine = FasterWhisperASR()
    engine._model = model
    engine._converter = None
    engine.device = "cpu"
    engine.compute_type = "int8"
    return engine


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

    def test_auto_language_detects_without_chinese_prompt(self):
        model = _StubWhisperModel(detected_language="en")
        engine = _fake_engine(model)

        result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000, "auto")

        call = model.calls[-1]
        self.assertIsNone(call["language"])
        self.assertIsNone(call["initial_prompt"])
        self.assertEqual(result.language, "en")

    def test_explicit_language_forces_chinese_prompt(self):
        model = _StubWhisperModel(detected_language="zh")
        engine = _fake_engine(model)

        engine.transcribe(np.zeros(16000, dtype=np.float32), 16000, "zh")

        call = model.calls[-1]
        self.assertEqual(call["language"], "zh")
        self.assertEqual(call["initial_prompt"], TRADITIONAL_CHINESE_PROMPT)


if __name__ == "__main__":
    unittest.main()

