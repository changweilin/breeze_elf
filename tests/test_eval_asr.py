import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import eval_asr  # noqa: E402

try:
    import jiwer  # noqa: F401

    _HAS_JIWER = True
except Exception:
    _HAS_JIWER = False


class SourceConfigTests(unittest.TestCase):
    def test_source_cfg_is_four_tuple_and_has_prefix(self):
        # rescore.py unpacks (lang, metric, use_cc, kind); every source needs a prefix.
        for source, cfg in eval_asr.SOURCE_CFG.items():
            self.assertEqual(len(cfg), 4, source)
            _lang, metric, _use_cc, kind = cfg
            self.assertIn(kind, {"asr", "negative"})
            self.assertIn(source, eval_asr.PREFIX)
            if kind == "negative":
                self.assertEqual(metric, "hallucination")

    def test_negative_prefix_matches_manifest_convention(self):
        self.assertEqual(eval_asr.PREFIX["negatives"], "neg_")


class DecodeKnobsTests(unittest.TestCase):
    def test_defaults_mirror_production(self):
        knobs = eval_asr.DecodeKnobs()
        self.assertEqual(knobs.beam, 1)
        self.assertFalse(knobs.vad_filter)
        self.assertFalse(knobs.condition_on_previous_text)
        # None -> arg omitted -> faster-whisper default (what production does).
        self.assertIsNone(knobs.temperature)
        self.assertIsNone(knobs.compression_ratio_threshold)
        self.assertIsNone(knobs.initial_prompt)
        self.assertFalse(knobs.word_timestamps)

    def test_parse_temperature(self):
        self.assertIsNone(eval_asr._parse_temperature(""))
        self.assertIsNone(eval_asr._parse_temperature("   "))
        self.assertEqual(eval_asr._parse_temperature("0.0"), (0.0,))
        self.assertEqual(eval_asr._parse_temperature("0.0,0.2,0.4"), (0.0, 0.2, 0.4))


class NormalizeTests(unittest.TestCase):
    def test_cjk_drops_whitespace_and_punct(self):
        self.assertEqual(eval_asr.normalize("你好，世界！", cjk=True), "你好世界")

    def test_latin_lowercases(self):
        # Spaces are in _PUNCT, so WER normalisation is space-free (pre-existing rule).
        self.assertEqual(eval_asr.normalize("Hello, World!", cjk=False), "helloworld")


class MetricTests(unittest.TestCase):
    def test_rtf(self):
        self.assertEqual(eval_asr._rtf(1.0, 4.0), 0.25)
        self.assertIsNone(eval_asr._rtf(1.0, 0.0))

    def test_median(self):
        self.assertEqual(eval_asr._median([3, 1, 2]), 2)
        self.assertEqual(eval_asr._median([1, 2, 3, 4]), 2.5)

    def test_alignment_error_matches_and_skips_insertions(self):
        pred = [
            {"word": "一", "start": 0.00, "end": 0.10},
            {"word": "X", "start": 0.10, "end": 0.12},  # pred insertion -> skipped
            {"word": "二", "start": 0.20, "end": 0.30},
        ]
        gold = [
            {"word": "一", "start": 0.05, "end": 0.15},
            {"word": "二", "start": 0.25, "end": 0.35},
        ]
        # every |Δ| is 50 ms -> median 50.0
        self.assertEqual(eval_asr.alignment_error_ms(pred, gold), 50.0)

    def test_alignment_error_none_when_nothing_matches(self):
        gold = [{"word": "一", "start": 0.0, "end": 1.0}]
        self.assertIsNone(eval_asr.alignment_error_ms([], gold))
        self.assertIsNone(
            eval_asr.alignment_error_ms([{"word": "甲", "start": 0.0, "end": 1.0}], gold)
        )

    def test_alignment_matches_case_insensitively(self):
        # jamendo (en) tokens must match regardless of case, like their WER scoring.
        pred = [{"word": "love", "start": 1.05, "end": 2.05}]
        gold = [{"word": "Love", "start": 1.00, "end": 2.00}]
        self.assertEqual(eval_asr.alignment_error_ms(pred, gold), 50.0)

    def test_hallucination_stats(self):
        recs = [
            {"hyp": "", "no_speech_prob": 0.9, "rms": 0.001},  # silent -> empty (good)
            {"hyp": "字幕由社群提供", "no_speech_prob": 0.1, "rms": 0.5},  # credit -> dropped
            {"hyp": "憑空一句歌詞", "no_speech_prob": 0.1, "rms": 0.5},  # leaks the gate
            {"hyp": "嗯", "no_speech_prob": 0.95, "rms": 0.001},  # quiet -> dropped
        ]
        stats = eval_asr.hallucination_stats(recs, drop_no_speech=0.6, drop_rms=0.02)
        self.assertEqual(
            stats,
            {
                "n": 4,
                "nonempty_rate": 0.75,
                "credit_rate": 0.25,
                "post_gate_leak_rate": 0.25,
            },
        )

    def test_hallucination_stats_empty(self):
        self.assertEqual(
            eval_asr.hallucination_stats([], drop_no_speech=0.6, drop_rms=0.02), {"n": 0}
        )


class CharCerMerTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_JIWER, "jiwer not installed (dataset/eval extra only)")
    def test_mer_is_character_level_not_degenerate(self):
        # One wrong char per line. Word-level jiwer.mer on whitespace-stripped CJK
        # returns the degenerate 1.0; char-level MER must be small and match CER here.
        import jiwer

        refs = ["今天天氣很好", "我喜歡聽音樂"]
        hyps = ["今天天氣真好", "我喜歡看音樂"]
        cer, mer = eval_asr.char_cer_mer(refs, hyps)
        self.assertAlmostEqual(cer, round(jiwer.cer(refs, hyps), 5), places=5)  # CER preserved
        self.assertLess(mer, 0.5)  # not the degenerate ~1.0
        self.assertAlmostEqual(mer, cer, places=5)  # no insertions -> mer == cer

    @unittest.skipUnless(_HAS_JIWER, "jiwer not installed (dataset/eval extra only)")
    def test_mer_counts_insertions_below_cer(self):
        # Pure insertion: CER = I/N, char-MER = I/(N+I) -> strictly less than CER.
        cer, mer = eval_asr.char_cer_mer(["abc"], ["abcd"])
        self.assertGreater(cer, 0)
        self.assertLess(mer, cer)


class AlignmentGoldTests(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        self.assertIsInstance(eval_asr.load_alignment_gold(), dict)

    def test_parses_and_skips_blank_and_empty_words(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "align_gold.jsonl"
            path.write_text(
                json.dumps({"id": "x", "words": [{"word": "一", "start": 0, "end": 1}]})
                + "\n\n"  # blank line ignored
                + json.dumps({"id": "y", "words": []})  # no words -> skipped
                + "\n",
                encoding="utf-8",
            )
            original = eval_asr.ALIGN_GOLD
            eval_asr.ALIGN_GOLD = path
            try:
                gold = eval_asr.load_alignment_gold()
            finally:
                eval_asr.ALIGN_GOLD = original
            self.assertEqual(list(gold), ["x"])
            self.assertEqual(gold["x"][0]["word"], "一")


if __name__ == "__main__":
    unittest.main()
