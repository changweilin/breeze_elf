import unittest

from breeze_elf.lyrics import (
    TimedChar,
    align_lyrics,
    parse_lyric_lines,
    timed_chars_from_blocks,
)


def _recognised(text, *, start=0.0, step=0.5):
    """One recognised character per 0.5 s, in order."""
    return [
        TimedChar(char, start + index * step, start + (index + 1) * step)
        for index, char in enumerate(text)
    ]


class ParseLyricLinesTests(unittest.TestCase):
    def test_drops_blanks_and_section_markers(self):
        lyrics = "[Verse 1]\n一閃一閃亮晶晶\n\n  \n【副歌】\n滿天都是小星星\n(Outro)"
        self.assertEqual(parse_lyric_lines(lyrics), ["一閃一閃亮晶晶", "滿天都是小星星"])

    def test_keeps_a_line_that_merely_contains_brackets(self):
        self.assertEqual(parse_lyric_lines("你說 (輕輕地) 再見了"), ["你說 (輕輕地) 再見了"])


class AlignLyricsTests(unittest.TestCase):
    def test_perfect_recognition_keeps_every_measured_timing(self):
        recognised = _recognised("一閃一閃亮晶晶")
        result = align_lyrics("一閃一閃亮晶晶", recognised)
        self.assertEqual(result.anchored_ratio, 1.0)
        self.assertEqual(result.dropped, 0)
        line = result.lines[0]
        self.assertEqual("".join(char.char for char in line.characters), "一閃一閃亮晶晶")
        for aligned, source in zip(line.characters, recognised):
            self.assertTrue(aligned.anchored)
            self.assertAlmostEqual(aligned.start, source.start)
            self.assertAlmostEqual(aligned.end, source.end)

    def test_mishearings_keep_their_timing_and_gain_the_real_words(self):
        # Same syllable count, wrong characters — the singing case. The words
        # come from the lyrics, the timings stay measured.
        result = align_lyrics("滿天都是小星星", _recognised("滿天都四小心星"))
        line = result.lines[0]
        self.assertEqual("".join(char.char for char in line.characters), "滿天都是小星星")
        self.assertTrue(all(char.anchored for char in line.characters))
        # 是 was misheard as 四 but keeps the time that syllable was sung; the
        # repeated 星星 stays in order rather than sliding a note early.
        self.assertAlmostEqual(line.characters[3].start, 1.5)
        self.assertAlmostEqual(line.characters[3].end, 2.0)
        self.assertAlmostEqual(line.characters[5].start, 2.5)
        self.assertAlmostEqual(line.characters[6].start, 3.0)
        self.assertEqual([char.matched for char in line.characters],
                         [True, True, True, False, True, False, True])

    def test_hallucinated_credits_are_dropped(self):
        recognised = _recognised("一閃一閃") + _recognised("字幕由社群提供", start=10.0)
        result = align_lyrics("一閃一閃", recognised)
        self.assertEqual(result.dropped, 7)
        self.assertEqual(len(result.lines[0].characters), 4)
        self.assertAlmostEqual(result.lines[0].end, 2.0)

    def test_a_missed_line_is_interpolated_across_the_gap(self):
        # The recogniser lost the middle line entirely (a loud chorus).
        recognised = _recognised("一閃一閃") + _recognised("滿天都是", start=6.0)
        result = align_lyrics("一閃一閃\n亮晶晶\n滿天都是", recognised)
        missed = result.lines[1]
        self.assertEqual(missed.text, "亮晶晶")
        self.assertEqual(missed.anchored_ratio, 0.0)
        self.assertFalse(any(char.anchored for char in missed.characters))
        # Spread over the silence between the two anchored lines, in order.
        self.assertAlmostEqual(missed.start, 2.0)
        self.assertAlmostEqual(missed.end, 6.0)
        starts = [char.start for char in missed.characters]
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(result.matched_ratio, 0.7)

    def test_lyrics_that_do_not_match_the_audio_report_no_agreement(self):
        # Substituting every character is cheaper than dropping and re-inserting
        # them, so a mismatched song still anchors timings — only the agreement
        # rate exposes it.
        result = align_lyrics("完全不同的一首歌詞", _recognised("一閃一閃亮晶晶"))
        self.assertLess(result.matched_ratio, 0.3)
        self.assertGreater(result.anchored_ratio, 0.5)

    def test_punctuation_is_kept_in_the_text_but_takes_no_timing(self):
        result = align_lyrics("一閃，一閃！", _recognised("一閃一閃"))
        line = result.lines[0]
        self.assertEqual(line.text, "一閃，一閃！")
        self.assertEqual("".join(char.char for char in line.characters), "一閃一閃")
        self.assertTrue(all(char.anchored for char in line.characters))

    def test_recognised_punctuation_does_not_drag_the_match(self):
        result = align_lyrics("一閃一閃亮晶晶", _recognised("一閃，一閃、亮晶晶"))
        self.assertEqual(result.matched_ratio, 1.0)

    def test_simplified_lyrics_match_traditional_recognition(self):
        result = align_lyrics("满天都是小星星", _recognised("滿天都是小星星"))
        # Without 繁簡 folding every character would read as a mismatch.
        self.assertGreater(result.matched_ratio, 0.7)

    def test_english_case_and_width_fold_together(self):
        result = align_lyrics("Twinkle", _recognised("ＴＷＩＮＫＬＥ"))
        self.assertEqual(result.matched_ratio, 1.0)

    def test_timeline_never_goes_backwards(self):
        recognised = [
            TimedChar("一", 0.0, 1.0),
            TimedChar("閃", 0.5, 1.2),  # overlapping windows can produce this
            TimedChar("亮", 1.1, 1.4),
        ]
        result = align_lyrics("一閃亮", recognised)
        chars = result.lines[0].characters
        for previous, following in zip(chars, chars[1:]):
            self.assertGreaterEqual(following.start, previous.end)
            self.assertGreaterEqual(following.end, following.start)

    def test_no_recognised_characters_returns_words_without_timings(self):
        result = align_lyrics("一閃一閃", [])
        self.assertEqual(result.anchored_ratio, 0.0)
        self.assertEqual(result.matched_ratio, 0.0)
        self.assertEqual(result.lines[0].text, "一閃一閃")
        self.assertEqual(result.lines[0].characters, ())
        self.assertIsNone(result.lines[0].start)

    def test_empty_lyrics(self):
        result = align_lyrics("   \n[Verse]\n", _recognised("一閃"))
        self.assertEqual(result.lines, ())
        self.assertEqual(result.dropped, 2)


class TimedCharsFromBlocksTests(unittest.TestCase):
    def test_flattens_and_sorts_skipping_untimed_characters(self):
        blocks = [
            {
                "characters": [
                    {"char": "閃", "startSeconds": 1.0, "endSeconds": 1.5},
                    {"char": "一", "startSeconds": 0.0, "endSeconds": 0.5},
                    {"char": "無", "startSeconds": None, "endSeconds": 2.0},
                    {"char": " ", "startSeconds": 3.0, "endSeconds": 3.5},
                ]
            },
            {"characters": [{"char": "亮", "startSeconds": 2.0, "endSeconds": 2.5}]},
            {"text": "no characters key"},
        ]
        chars = timed_chars_from_blocks(blocks)
        self.assertEqual([char.char for char in chars], ["一", "閃", "亮"])
        self.assertEqual(chars[0].start, 0.0)

    def test_end_is_never_before_start(self):
        chars = timed_chars_from_blocks(
            [{"characters": [{"char": "一", "startSeconds": 2.0, "endSeconds": 1.0}]}]
        )
        self.assertEqual(chars[0].end, 2.0)


if __name__ == "__main__":
    unittest.main()
