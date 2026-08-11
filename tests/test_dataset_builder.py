"""Offline lyric-dataset builder: parsing, normalisation, chunking, export, negatives.

All ``unittest.TestCase`` so ``unittest discover`` -- the CI runner -- actually collects
them. (This file was previously pytest-style module-level functions, which meant CI
reported it green while running none of it.)
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from breeze_elf.dataset_builder import (
    Chunk,
    DatasetWriter,
    LyricLine,
    PipelineConfig,
    _decode_text,
    _write_negatives,
    instrumental_chunks,
    is_section_tag_line,
    lang_joiner,
    lrc_to_lines,
    merge_lines_to_chunks,
    normalize_text,
    parse_lrc,
    passes_quality,
    sanitize_id,
    strip_furigana,
)

LRC = """[ti:測試]
[ar:someone]
[00:01.50]第一句歌詞
[00:04.00]第二句歌詞
[00:12.34][01:02.00]重複的副歌
[00:20.00]
"""

HAS_SOUNDFILE = importlib.util.find_spec("soundfile") is not None


def config(**overrides) -> PipelineConfig:
    return PipelineConfig(output_dir=None, **overrides)


def tone(seconds: float, sr: int = 16000, hz: float = 220.0, level: float = 0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class LrcParseTests(unittest.TestCase):
    def test_skips_meta_and_expands_repeats(self):
        entries = parse_lrc(LRC)
        self.assertEqual([t for t, _ in entries], [1.5, 4.0, 12.34, 20.0, 62.0])
        self.assertEqual(entries[2][1], "重複的副歌")

    def test_lines_get_ends_and_are_capped_by_audio_duration(self):
        lines = lrc_to_lines(parse_lrc(LRC), audio_duration=70.0)
        # The empty 20.0 s line is dropped; each end = the next start, or the cap.
        self.assertEqual(lines[0], LyricLine(1.5, 4.0, "第一句歌詞"))
        self.assertEqual(lines[1].end, 12.34)
        self.assertEqual(lines[-1].start, 62.0)
        self.assertEqual(lines[-1].end, 70.0)

    def test_section_tag_lines(self):
        self.assertTrue(is_section_tag_line("[Verse 1]"))
        self.assertTrue(is_section_tag_line("【副歌】"))
        self.assertFalse(is_section_tag_line("not a [tag] only line"))

    def test_decode_text_falls_back_to_big5(self):
        # MIR-1K lyrics ship as cp950.
        self.assertEqual(_decode_text("歌詞".encode("cp950")), "歌詞")
        self.assertEqual(_decode_text("歌詞".encode()), "歌詞")


class NormalizeTextTests(unittest.TestCase):
    def test_zh_tw_converts_vocabulary_and_strips_tags(self):
        out = normalize_text("[Chorus] 软件里的鼠标", "zh-TW")
        self.assertIn("軟體", out)
        self.assertIn("滑鼠", out)
        self.assertNotIn("[", out)

    def test_ja_strips_furigana_only(self):
        self.assertEqual(strip_furigana("漢字(かんじ)を読む"), "漢字を読む")
        self.assertEqual(normalize_text("漢字(かんじ)を読む", "ja"), "漢字を読む")
        self.assertEqual(strip_furigana("Tokyo(TYO)"), "Tokyo(TYO)")

    def test_nan_mapping_applies_longest_source_first(self):
        mapping = [("袂記", "袂記"), ("未记", "袂記"), ("未", "袂")]
        mapping.sort(key=lambda pair: len(pair[0]), reverse=True)
        self.assertEqual(normalize_text("我未记你", "nan", nan_mapping=mapping), "我袂記你")

    def test_nan_strips_tailo_reading_annotations(self):
        # Common Voice nan-tw style: Han text carrying a Tai-lo reading gloss. These
        # are ~48% of the label characters and wreck both CER and the training target.
        self.assertEqual(normalize_text("蘋果派(phōng-kó-phài)真好食", "nan"), "蘋果派真好食")
        self.assertEqual(normalize_text("大武(Tāi-bú)", "nan"), "大武")
        # A pure Tai-lo sentence is content, not a gloss, and survives untouched.
        self.assertEqual(normalize_text("Thinn-tíng ê gue̍h-niû", "nan"), "Thinn-tíng ê gue̍h-niû")


class ChunkingTests(unittest.TestCase):
    def test_merges_short_lines_until_the_gap_is_too_large(self):
        lines = [
            LyricLine(0.0, 0.5, "a"),
            LyricLine(0.6, 1.4, "b"),
            LyricLine(5.0, 7.0, "c"),  # gap > 2 s starts a new chunk
        ]
        chunks = merge_lines_to_chunks(lines, min_dur=1.0, max_dur=20.0, joiner=" ")
        self.assertEqual(chunks, [Chunk(0.0, 1.4, "a b"), Chunk(5.0, 7.0, "c")])

    def test_merge_respects_max_duration(self):
        lines = [LyricLine(0.0, 0.4, "a"), LyricLine(0.5, 25.0, "b")]
        self.assertEqual(len(merge_lines_to_chunks(lines, min_dur=1.0, max_dur=20.0)), 2)

    def test_lang_joiner(self):
        self.assertEqual(lang_joiner("zh-TW"), "")
        self.assertEqual(lang_joiner("ja"), "")
        self.assertEqual(lang_joiner("en"), " ")
        self.assertEqual(lang_joiner("nan"), " ")


class QualityGateTests(unittest.TestCase):
    def test_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(output_dir=Path(tmp))
            sr = 16000
            loud = tone(2.0, sr)
            self.assertTrue(passes_quality(loud, sr, "text", cfg))
            self.assertFalse(passes_quality(loud[: sr // 2], sr, "text", cfg))  # too short
            self.assertFalse(passes_quality(np.zeros(sr * 2, np.float32), sr, "text", cfg))
            self.assertFalse(passes_quality(loud, sr, "   ", cfg))  # empty text

    def test_empty_text_needs_allow_empty(self):
        samples = np.full(16000 * 4, 0.1, dtype=np.float32)
        cfg = config()
        self.assertFalse(passes_quality(samples, 16000, "", cfg))
        self.assertTrue(passes_quality(samples, 16000, "", cfg, allow_empty=True))

    def test_allow_empty_still_enforces_duration_and_rms(self):
        cfg = config()
        short = np.full(int(16000 * 0.2), 0.1, dtype=np.float32)
        self.assertFalse(passes_quality(short, 16000, "", cfg, allow_empty=True))
        quiet = np.zeros(16000 * 4, dtype=np.float32)
        self.assertFalse(passes_quality(quiet, 16000, "", cfg, allow_empty=True))

    def test_sanitize_id(self):
        self.assertEqual(sanitize_id("阿妹/聽海 (live)"), "live")  # non-ascii collapses
        self.assertEqual(sanitize_id("abc DEF-1.mp3"), "abc_DEF-1.mp3")
        self.assertEqual(sanitize_id("///"), "song")


@unittest.skipUnless(HAS_SOUNDFILE, "soundfile is only in the optional `dataset` extra")
class DatasetWriterTests(unittest.TestCase):
    def test_appends_and_dedupes_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = tone(1.0, hz=440.0, level=0.2)
            writer = DatasetWriter(root)
            self.assertTrue(writer.add_chunk(clip, 16000, "你好", "zh-TW", "song1", 0))
            self.assertFalse(writer.add_chunk(clip, 16000, "你好", "zh-TW", "song1", 0))

            # A fresh instance re-reads metadata.csv and still dedupes.
            reopened = DatasetWriter(root)
            self.assertFalse(reopened.add_chunk(clip, 16000, "你好", "zh-TW", "song1", 0))
            self.assertTrue(reopened.add_chunk(clip, 16000, "hello", "en", "song1", 1))

            content = (root / "metadata.csv").read_text(encoding="utf-8")
            self.assertEqual(content.count("chunks/song1_0000.wav"), 1)
            self.assertIn("hello", content)
            self.assertIn("你好", content)
            self.assertTrue((root / "chunks" / "song1_0001.wav").is_file())

    def test_has_source_matches_whole_ids_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = DatasetWriter(root)
            writer.add_chunk(tone(1.0), 16000, "你好", "zh-TW", "song1", 0)
            self.assertTrue(writer.has_source("song1"))
            self.assertFalse(writer.has_source("song"))  # prefix must not match
            self.assertFalse(writer.has_source("other"))


# --- C-layer negatives (TRAINING_PLAN.md §3.3) --------------------------------------


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
        # 21 s against max 20 would split into two 10.5 s pieces; with a 12 s floor
        # neither survives, so the gap yields nothing rather than a stub.
        chunks = [Chunk(0.0, 1.0, "詞")]
        self.assertEqual(instrumental_chunks(chunks, 22.0, min_dur=12.0, max_dur=20.0, pad=0.0), [])


class WriteNegativeTests(unittest.TestCase):
    def audio(self, seconds, sr=16000, level=0.1):
        return np.full(int(seconds * sr), level, dtype=np.float32)

    def write(self, writer, *, seconds=200.0, sung=10, level=0.1, ratio=0.08):
        chunks = [Chunk(float(i), float(i) + 1.0, "詞") for i in range(sung)]
        return chunks, _write_negatives(
            writer,
            self.audio(seconds, level=level),
            16000,
            chunks,
            seconds,
            lang="zh-TW",
            source_id="song",
            cfg=config(negative_ratio=ratio),
        )

    def test_ratio_caps_how_many_are_written(self):
        writer = RecordingWriter()
        _, written = self.write(writer)
        # 1 negative beside 10 sung chunks is 9.1% of the export -- inside §3.3's 5-10%.
        self.assertEqual(written, 1)
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(writer.calls[0]["text"], "")

    def test_indices_never_collide_with_the_sung_chunks(self):
        writer = RecordingWriter()
        chunks, _ = self.write(writer, seconds=400.0, sung=40)
        self.assertTrue(all(call["index"] >= len(chunks) for call in writer.calls))

    def test_silent_gap_is_rejected(self):
        # Silence is what the RMS gate already handles; the training signal that is
        # missing is *loud* non-speech.
        writer = RecordingWriter()
        _, written = self.write(writer, level=0.0)
        self.assertEqual(written, 0)

    def test_disabled_by_zero_ratio(self):
        writer = RecordingWriter()
        _, written = self.write(writer, ratio=0.0)
        self.assertEqual(written, 0)
        self.assertEqual(writer.calls, [])

    def test_song_without_lyric_chunks_writes_nothing(self):
        writer = RecordingWriter()
        _, written = self.write(writer, sung=0)
        self.assertEqual(written, 0)


if __name__ == "__main__":
    unittest.main()
