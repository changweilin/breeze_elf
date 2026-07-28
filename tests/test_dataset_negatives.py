"""C 層負樣本: the instrumental gaps the dataset builder exports with empty text.

Written as ``unittest.TestCase`` so both the CI runner (``unittest discover``) and
pytest collect them.
"""

import unittest

import numpy as np

from breeze_elf.dataset_builder import (
    Chunk,
    PipelineConfig,
    _write_negatives,
    instrumental_chunks,
    passes_quality,
)


class RecordingWriter:
    """The one method ``_write_negatives`` needs, without touching soundfile."""

    def __init__(self, accept=True):
        self.calls = []
        self.accept = accept

    def add_chunk(self, samples, sr, text, lang, source_id, index):
        self.calls.append(
            {"seconds": len(samples) / sr, "text": text, "source": source_id, "index": index}
        )
        return self.accept


def config(**overrides) -> PipelineConfig:
    return PipelineConfig(output_dir=None, **overrides)


class InstrumentalChunkTests(unittest.TestCase):
    def test_finds_intro_interlude_and_outro(self):
        chunks = [Chunk(10.0, 20.0, "第一段"), Chunk(35.0, 45.0, "第二段")]
        gaps = instrumental_chunks(chunks, 60.0, min_dur=3.0, max_dur=20.0, pad=0.1)
        spans = [(round(gap.start, 2), round(gap.end, 2)) for gap in gaps]
        self.assertEqual(spans, [(0.1, 9.9), (20.1, 34.9), (45.1, 59.9)])
        self.assertTrue(all(gap.text == "" for gap in gaps))

    def test_padding_keeps_the_gap_off_the_vocal_tail(self):
        # Same padding the sung chunks grow by, so the two never overlap.
        chunks = [Chunk(0.0, 10.0, "詞")]
        gaps = instrumental_chunks(chunks, 20.0, min_dur=3.0, max_dur=20.0, pad=0.5)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start, 10.5)
        self.assertAlmostEqual(gaps[0].end, 19.5)

    def test_breath_between_lines_is_not_an_interlude(self):
        chunks = [Chunk(0.0, 5.0, "一"), Chunk(6.0, 11.0, "二")]
        self.assertEqual(instrumental_chunks(chunks, 11.0, min_dur=3.0, max_dur=20.0, pad=0.1), [])

    def test_long_outro_is_split_into_equal_usable_pieces(self):
        chunks = [Chunk(0.0, 5.0, "詞")]
        gaps = instrumental_chunks(chunks, 65.0, min_dur=3.0, max_dur=20.0, pad=0.0)
        self.assertEqual(len(gaps), 3)
        lengths = {round(gap.end - gap.start, 6) for gap in gaps}
        self.assertEqual(len(lengths), 1)
        self.assertLessEqual(max(lengths), 20.0)
        # Contiguous, and covering the whole outro rather than truncating it.
        self.assertAlmostEqual(gaps[0].start, 5.0)
        self.assertAlmostEqual(gaps[-1].end, 65.0)

    def test_split_that_would_produce_unusable_pieces_is_dropped(self):
        # 21 s against max 20 would split into two 10.5 s pieces; with a 12 s
        # floor neither survives, so the gap yields nothing rather than a stub.
        chunks = [Chunk(0.0, 1.0, "詞")]
        gaps = instrumental_chunks(chunks, 22.0, min_dur=12.0, max_dur=20.0, pad=0.0)
        self.assertEqual(gaps, [])


class WriteNegativeTests(unittest.TestCase):
    def audio(self, seconds, sr=16000, level=0.1):
        return np.full(int(seconds * sr), level, dtype=np.float32)

    def test_ratio_caps_how_many_are_written(self):
        writer = RecordingWriter()
        chunks = [Chunk(float(i), float(i) + 1.0, "詞") for i in range(10)]
        written = _write_negatives(
            writer,
            self.audio(200.0),
            16000,
            chunks,
            200.0,
            lang="zh-TW",
            source_id="song",
            cfg=config(negative_ratio=0.08),
        )
        # 1 negative beside 10 sung chunks is 9.1% of the export — inside §3.3's 5–10%.
        self.assertEqual(written, 1)
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(writer.calls[0]["text"], "")

    def test_indices_never_collide_with_the_sung_chunks(self):
        writer = RecordingWriter()
        chunks = [Chunk(float(i), float(i) + 1.0, "詞") for i in range(40)]
        _write_negatives(
            writer,
            self.audio(400.0),
            16000,
            chunks,
            400.0,
            lang="zh-TW",
            source_id="song",
            cfg=config(negative_ratio=0.08),
        )
        self.assertTrue(all(call["index"] >= len(chunks) for call in writer.calls))

    def test_silent_gap_is_rejected(self):
        writer = RecordingWriter()
        chunks = [Chunk(float(i), float(i) + 1.0, "詞") for i in range(10)]
        written = _write_negatives(
            writer,
            self.audio(200.0, level=0.0),
            16000,
            chunks,
            200.0,
            lang="zh-TW",
            source_id="song",
            cfg=config(negative_ratio=0.08),
        )
        # Silence is what the RMS gate already handles; the training signal that
        # is missing is *loud* non-speech.
        self.assertEqual(written, 0)

    def test_disabled_by_zero_ratio(self):
        writer = RecordingWriter()
        chunks = [Chunk(float(i), float(i) + 1.0, "詞") for i in range(10)]
        self.assertEqual(
            _write_negatives(
                writer,
                self.audio(200.0),
                16000,
                chunks,
                200.0,
                lang="zh-TW",
                source_id="song",
                cfg=config(negative_ratio=0.0),
            ),
            0,
        )
        self.assertEqual(writer.calls, [])

    def test_song_without_lyric_chunks_writes_nothing(self):
        writer = RecordingWriter()
        self.assertEqual(
            _write_negatives(
                writer,
                self.audio(200.0),
                16000,
                [],
                200.0,
                lang="zh-TW",
                source_id="song",
                cfg=config(negative_ratio=0.08),
            ),
            0,
        )


class EmptyTargetQualityTests(unittest.TestCase):
    def test_empty_text_needs_allow_empty(self):
        samples = np.full(16000 * 4, 0.1, dtype=np.float32)
        cfg = config()
        self.assertFalse(passes_quality(samples, 16000, "", cfg))
        self.assertTrue(passes_quality(samples, 16000, "", cfg, allow_empty=True))

    def test_allow_empty_still_enforces_duration_and_rms(self):
        cfg = config()
        short = np.full(int(16000 * 0.2), 0.1, dtype=np.float32)
        self.assertFalse(passes_quality(short, 16000, "", cfg, allow_empty=True))
        quiet = np.full(16000 * 4, 0.0, dtype=np.float32)
        self.assertFalse(passes_quality(quiet, 16000, "", cfg, allow_empty=True))


if __name__ == "__main__":
    unittest.main()
