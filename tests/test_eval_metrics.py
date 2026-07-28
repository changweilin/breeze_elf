"""The four metrics `tools/eval_asr.py` reports beside CER/WER.

`tools/` is a script directory, not a package, so it is put on the path the same
way `tools/rescore.py` does it.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from eval_metrics import (  # noqa: E402
    AlignmentCounter,
    HallucinationCounter,
    RtfCounter,
    reference_words,
)

GATE = {"no_speech_threshold": 0.6, "rms_threshold": 0.02}


class HallucinationTests(unittest.TestCase):
    def test_silent_model_scores_zero(self):
        counter = HallucinationCounter()
        counter.add_clip([("", 0.9)], rms=0.3, **GATE)
        counter.add_clip([("  ", None)], rms=0.3, **GATE)
        summary = counter.summary()
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["hallucination_rate"], 0.0)
        self.assertEqual(summary["hallucination_rate_gated"], 0.0)

    def test_credit_line_over_loud_music_is_caught_by_the_gate(self):
        counter = HallucinationCounter()
        counter.add_clip([("字幕由 Amara.org 社群提供", 0.05)], rms=0.4, **GATE)
        summary = counter.summary()
        # Raw counts it — the model did invent text over an interlude — but the
        # production gate removes it, which is the number users actually see.
        self.assertEqual(summary["hallucination_rate"], 1.0)
        self.assertEqual(summary["hallucination_rate_gated"], 0.0)
        self.assertEqual(summary["credit_rate"], 1.0)

    def test_invented_lyrics_survive_the_gate_and_are_reported(self):
        counter = HallucinationCounter()
        counter.add_clip([("月光灑在你身上", 0.05)], rms=0.4, **GATE)
        summary = counter.summary()
        self.assertEqual(summary["hallucination_rate"], 1.0)
        self.assertEqual(summary["hallucination_rate_gated"], 1.0)
        self.assertEqual(summary["credit_rate"], 0.0)

    def test_a_clip_leaks_once_however_many_segments_do(self):
        counter = HallucinationCounter()
        counter.add_clip([("", 0.9), ("憑空一句", 0.05), ("又一句", 0.05)], rms=0.4, **GATE)
        summary = counter.summary()
        self.assertEqual(summary["hallucination_rate_gated"], 1.0)
        # Segment rates keep the severity visible next to the clip rate.
        self.assertEqual(summary["segment_rate"], round(2 / 3, 5))


class RtfTests(unittest.TestCase):
    def test_totals_not_mean_of_ratios(self):
        counter = RtfCounter()
        counter.add(decode_seconds=0.4, audio_seconds=0.4)  # RTF 1.0 on a tiny clip
        counter.add(decode_seconds=2.0, audio_seconds=20.0)  # RTF 0.1 on a long one
        # A mean of per-clip ratios would say 0.55; what matters is 2.4 s of
        # decoding for 20.4 s of audio.
        self.assertAlmostEqual(counter.rtf, 2.4 / 20.4)

    def test_no_audio_is_none_not_zero(self):
        self.assertIsNone(RtfCounter().rtf)


class AlignmentTests(unittest.TestCase):
    def test_perfect_timing_scores_zero(self):
        counter = AlignmentCounter()
        reference = [("一", 1.0, 1.5), ("閃", 1.5, 2.0)]
        counter.add(reference, list(reference))
        summary = counter.summary()
        self.assertEqual(summary["start_median_abs_ms"], 0.0)
        self.assertEqual(summary["end_median_abs_ms"], 0.0)
        self.assertEqual(summary["matched_ratio"], 1.0)

    def test_constant_drift_is_reported_in_ms(self):
        counter = AlignmentCounter()
        counter.add(
            [("一", 1.0, 1.5), ("閃", 1.5, 2.0), ("一", 2.0, 2.5)],
            [("一", 1.3, 1.8), ("閃", 1.8, 2.3), ("一", 2.3, 2.8)],
        )
        self.assertEqual(counter.summary()["start_median_abs_ms"], 300.0)

    def test_repeated_tokens_align_optimally(self):
        # 星星 twice over: a longest-common-subsequence matcher pairs the wrong
        # 星 and invents a whole beat of drift.
        counter = AlignmentCounter()
        counter.add(
            [("星", 0.0, 0.5), ("星", 0.5, 1.0), ("亮", 1.0, 1.5), ("星", 1.5, 2.0)],
            [("星", 0.0, 0.5), ("星", 0.5, 1.0), ("量", 1.0, 1.5), ("星", 1.5, 2.0)],
        )
        summary = counter.summary()
        self.assertEqual(summary["start_median_abs_ms"], 0.0)
        # 亮 was misrecognised, so its timing is not scored — but it still counts
        # against coverage.
        self.assertEqual(summary["n_matched_tokens"], 3)
        self.assertEqual(summary["n_reference_tokens"], 4)
        self.assertEqual(summary["matched_ratio"], 0.75)

    def test_a_model_that_emits_nothing_scores_no_error_but_no_coverage(self):
        counter = AlignmentCounter()
        counter.add([("一", 1.0, 1.5), ("閃", 1.5, 2.0)], [])
        summary = counter.summary()
        self.assertIsNone(summary["start_median_abs_ms"])
        self.assertEqual(summary["matched_ratio"], 0.0)
        self.assertEqual(summary["n_reference_tokens"], 2)

    def test_case_and_spacing_do_not_count_as_mismatches(self):
        counter = AlignmentCounter()
        counter.add([("Moon", 1.0, 1.5)], [(" moon", 1.0, 1.5)])
        self.assertEqual(counter.summary()["n_matched_tokens"], 1)


class ReferenceWordTests(unittest.TestCase):
    def test_reads_annotated_rows(self):
        row = {"words": [{"word": "月", "start": 1.0, "end": 1.4}]}
        self.assertEqual(reference_words(row), [("月", 1.0, 1.4)])

    def test_accepts_char_key(self):
        row = {"words": [{"char": "月", "start": 1.0, "end": 1.4}]}
        self.assertEqual(reference_words(row), [("月", 1.0, 1.4)])

    def test_unannotated_or_malformed_rows_contribute_nothing(self):
        self.assertEqual(reference_words({}), [])
        self.assertEqual(reference_words({"words": "月"}), [])
        self.assertEqual(reference_words({"words": [{"word": "月"}]}), [])
        self.assertEqual(reference_words({"words": [{"start": 1.0, "end": 1.4}]}), [])


if __name__ == "__main__":
    unittest.main()
