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

    def test_context_precedes_base_prompt_and_glossary(self):
        model = _StubWhisperModel(detected_language="zh")
        engine = _fake_engine(model)

        engine.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            "zh",
            languages=("zh",),
            prompt_terms=("布雷茲",),
            context="精靈 是主角",
        )

        prompt = model.calls[-1]["initial_prompt"]
        # Context comes FIRST (whitespace-collapsed); the base prompt + glossary
        # sit at the tail so they survive Whisper's initial_prompt truncation.
        self.assertTrue(prompt.startswith("精靈 是主角"))
        self.assertIn(TRADITIONAL_CHINESE_PROMPT, prompt)
        self.assertTrue(prompt.endswith("布雷茲"))

    def test_context_dropped_in_free_detect_mode(self):
        # Free detection drops the whole prompt, context included, so it cannot
        # bias the language guess.
        model = _StubWhisperModel(detected_language="en")
        engine = _fake_engine(model)

        engine.transcribe(
            np.zeros(16000, dtype=np.float32), 16000, "auto", context="先前的中文脈絡"
        )

        self.assertIsNone(model.calls[-1]["initial_prompt"])


class _StubWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _StubSegment:
    def __init__(self, text, start, end, words=()):
        self.text = text
        self.start = start
        self.end = end
        self.words = words


class _StubBatched:
    """Stand-in for faster-whisper's BatchedInferencePipeline."""

    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        del audio
        self.calls.append(kwargs)
        info = SimpleNamespace(language=kwargs.get("language") or "zh")
        return list(self.segments), info


class BatchedFileASRTests(unittest.TestCase):
    def test_mock_transcribe_file_returns_one_segment(self):
        result = MockASR().transcribe_file(np.zeros(16000, dtype=np.float32), 16000, "zh")
        self.assertEqual(len(result.segments), 1)
        self.assertIn("批次測試", result.segments[0].text)
        self.assertEqual(result.language, "zh")

    def test_transcribe_file_maps_batched_segments_and_passes_batch_size(self):
        engine = _fake_engine(_StubWhisperModel(detected_language="zh"))
        words = [_StubWord("你好", 0.0, 0.5), _StubWord("世界", 0.5, 1.0)]
        engine._batched = _StubBatched([_StubSegment("你好世界", 0.0, 1.0, words)])

        result = engine.transcribe_file(
            np.zeros(16000, dtype=np.float32), 16000, "auto", batch_size=8
        )

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "你好世界")
        self.assertEqual(result.segments[0].start, 0.0)
        self.assertEqual(result.segments[0].end, 1.0)
        self.assertEqual(len(result.segments[0].words), 2)
        # Auto (free-detect) drops the prompt; the batch size is threaded through.
        self.assertIsNone(engine._batched.calls[-1]["initial_prompt"])
        self.assertEqual(engine._batched.calls[-1]["batch_size"], 8)

    def test_transcribe_file_drops_empty_segments(self):
        engine = _fake_engine(_StubWhisperModel(detected_language="zh"))
        engine._batched = _StubBatched(
            [_StubSegment("  ", 0.0, 0.5), _StubSegment("有字", 0.5, 1.0)]
        )

        result = engine.transcribe_file(np.zeros(16000, dtype=np.float32), 16000, "auto")

        self.assertEqual([seg.text for seg in result.segments], ["有字"])


if __name__ == "__main__":
    unittest.main()

