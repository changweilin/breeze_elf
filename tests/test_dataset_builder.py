import numpy as np
import pytest

from breeze_elf.dataset_builder import (
    Chunk,
    DatasetWriter,
    LyricLine,
    PipelineConfig,
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


def test_parse_lrc_skips_meta_and_expands_repeats():
    entries = parse_lrc(LRC)
    times = [t for t, _ in entries]
    assert times == [1.5, 4.0, 12.34, 20.0, 62.0]
    assert entries[2][1] == "重複的副歌"


def test_lrc_to_lines_ends_and_caps():
    lines = lrc_to_lines(parse_lrc(LRC), audio_duration=70.0)
    # The 20.0 s empty line is dropped; each end = next start (or capped).
    assert lines[0] == LyricLine(1.5, 4.0, "第一句歌詞")
    assert lines[1].end == 12.34
    assert lines[-1].start == 62.0 and lines[-1].end == 70.0


def test_normalize_zh_tw_converts_and_strips_tags():
    out = normalize_text("[Chorus] 软件里的鼠标", "zh-TW")
    assert "軟體" in out and "滑鼠" in out
    assert "[" not in out


def test_normalize_ja_strips_furigana():
    assert strip_furigana("漢字(かんじ)を読む") == "漢字を読む"
    assert normalize_text("漢字(かんじ)を読む", "ja") == "漢字を読む"
    # Non-kana parentheses survive.
    assert strip_furigana("Tokyo(TYO)") == "Tokyo(TYO)"


def test_normalize_nan_applies_mapping_longest_first():
    mapping = [("袂記", "袂記"), ("未记", "袂記"), ("未", "袂")]
    mapping.sort(key=lambda p: len(p[0]), reverse=True)
    assert normalize_text("我未记你", "nan", nan_mapping=mapping) == "我袂記你"


def test_section_tag_lines():
    assert is_section_tag_line("[Verse 1]")
    assert is_section_tag_line("【副歌】")
    assert not is_section_tag_line("not a [tag] only line")


def test_merge_lines_to_chunks_merges_short_lines():
    lines = [
        LyricLine(0.0, 0.5, "a"),
        LyricLine(0.6, 1.4, "b"),
        LyricLine(5.0, 7.0, "c"),  # gap > 2 s starts a new chunk
    ]
    chunks = merge_lines_to_chunks(lines, min_dur=1.0, max_dur=20.0, joiner=" ")
    assert chunks == [Chunk(0.0, 1.4, "a b"), Chunk(5.0, 7.0, "c")]


def test_merge_respects_max_duration():
    lines = [LyricLine(0.0, 0.4, "a"), LyricLine(0.5, 25.0, "b")]
    chunks = merge_lines_to_chunks(lines, min_dur=1.0, max_dur=20.0)
    assert len(chunks) == 2  # merging would exceed max_dur


def test_lang_joiner():
    assert lang_joiner("zh-TW") == "" and lang_joiner("ja") == ""
    assert lang_joiner("en") == " " and lang_joiner("nan") == " "


def test_passes_quality_bounds(tmp_path):
    cfg = PipelineConfig(output_dir=tmp_path)
    sr = 16000
    loud = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    assert passes_quality(loud, sr, "text", cfg)
    assert not passes_quality(loud[: int(0.5 * sr)], sr, "text", cfg)  # too short
    assert not passes_quality(np.zeros(sr * 2, np.float32), sr, "text", cfg)  # silence
    assert not passes_quality(loud, sr, "   ", cfg)  # empty text


def test_sanitize_id():
    assert sanitize_id("阿妹/聽海 (live)") == "live"  # non-ascii collapses, still non-empty
    assert sanitize_id("abc DEF-1.mp3") == "abc_DEF-1.mp3"
    assert sanitize_id("///") == "song"


def test_dataset_writer_appends_and_dedupes(tmp_path):
    pytest.importorskip("soundfile")
    sr = 16000
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    writer = DatasetWriter(tmp_path)
    assert writer.add_chunk(tone, sr, "你好", "zh-TW", "song1", 0)
    assert not writer.add_chunk(tone, sr, "你好", "zh-TW", "song1", 0)  # dedupe
    # A fresh writer instance re-reads the csv and still dedupes.
    writer2 = DatasetWriter(tmp_path)
    assert not writer2.add_chunk(tone, sr, "你好", "zh-TW", "song1", 0)
    assert writer2.add_chunk(tone, sr, "hello", "en", "song1", 1)
    content = (tmp_path / "metadata.csv").read_text(encoding="utf-8")
    assert content.count("chunks/song1_0000.wav") == 1
    assert "hello" in content and "你好" in content
    assert (tmp_path / "chunks" / "song1_0001.wav").exists()
    assert writer2.has_source("song1")
    assert not writer2.has_source("song")  # prefix must not match song1
    assert not writer2.has_source("other")
