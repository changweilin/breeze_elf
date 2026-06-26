import unittest
from types import SimpleNamespace

import numpy as np

from breeze_elf.asr import TRADITIONAL_CHINESE_PROMPT, FasterWhisperASR, MockASR


class _StubWhisperModel:
    """Records the kwargs faster-whisper would receive, no real model loaded."""

    def __init__(self, detected_language="en", language_probs=None):
        self.detected_language = detected_language
        # Ranked (code, prob) pairs detect_language() returns; None → the model
        # has no detect_language (older faster-whisper) so the stub omits it.
        self.language_probs = language_probs
        self.calls = []
        if language_probs is not None:
            self.detect_language = self._detect_language

    def _detect_language(self, samples):
        del samples
        top = self.language_probs[0]
        return top[0], top[1], list(self.language_probs)

    def transcribe(self, samples, **kwargs):
        del samples
        self.calls.append(kwargs)
        info = SimpleNamespace(language=kwargs.get("language") or self.detected_language)
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

    def test_multi_language_restricts_detection_to_allowed_set(self):
        # Detection ranks 日文 highest, but it is not in the allowed 繁中/英文 set,
        # so recognition is forced to the best *allowed* language (英文).
        model = _StubWhisperModel(
            language_probs=[("ja", 0.6), ("en", 0.3), ("zh", 0.1)],
        )
        engine = _fake_engine(model)

        engine.transcribe(
            np.zeros(16000, dtype=np.float32), 16000, "zh", languages=("zh", "en")
        )

        self.assertEqual(model.calls[-1]["language"], "en")

    def test_multi_language_falls_back_to_primary_without_detection(self):
        # No detect_language available → never escapes the user's languages:
        # fall back to the primary (first) one.
        model = _StubWhisperModel(detected_language="zh")
        engine = _fake_engine(model)

        engine.transcribe(
            np.zeros(16000, dtype=np.float32), 16000, "zh", languages=("zh", "en")
        )

        self.assertEqual(model.calls[-1]["language"], "zh")

    def test_glossary_terms_extend_initial_prompt(self):
        model = _StubWhisperModel(detected_language="zh")
        engine = _fake_engine(model)

        engine.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            "zh",
            languages=("zh",),
            prompt_terms=("Mike 學習", "布雷茲"),
        )

        prompt = model.calls[-1]["initial_prompt"]
        self.assertTrue(prompt.startswith(TRADITIONAL_CHINESE_PROMPT))
        self.assertIn("Mike 學習", prompt)
        self.assertIn("布雷茲", prompt)


if __name__ == "__main__":
    unittest.main()

