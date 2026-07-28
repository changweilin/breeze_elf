import unittest

from breeze_elf.hallucination import (
    is_common_silence_hallucination,
    is_credit_hallucination,
    normalize_hallucination_text,
    should_drop_text,
)


class NormalizeTests(unittest.TestCase):
    def test_strips_punctuation_and_unifies_zan(self):
        self.assertEqual(normalize_hallucination_text("按讚，分享!"), "按贊分享")

    def test_empty_when_only_punctuation(self):
        self.assertEqual(normalize_hallucination_text(" 。，!? "), "")


class CreditTests(unittest.TestCase):
    def test_matches_wording_variants(self):
        for text in (
            "字幕由 Amara.org 社群提供",
            "字幕提供由 XXX 社群提供的字",
            "請不吝點贊 訂閱 轉發 打賞支持明鏡與點點欄目",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_credit_hallucination(text))

    def test_real_speech_survives(self):
        self.assertFalse(is_credit_hallucination("今天的字幕是我自己打的"))

    def test_empty_text_is_not_a_hallucination(self):
        self.assertFalse(is_credit_hallucination("   "))


class SilenceFragmentTests(unittest.TestCase):
    def test_outro_fragment_matched_through_punctuation(self):
        self.assertTrue(is_common_silence_hallucination("歡迎訂閱、按讚、分享!"))

    def test_unrelated_text_not_matched(self):
        self.assertFalse(is_common_silence_hallucination("我們下週再分享進度"))


class GateTests(unittest.TestCase):
    def gate(self, text, *, rms, no_speech_prob):
        return should_drop_text(
            text,
            rms=rms,
            no_speech_prob=no_speech_prob,
            no_speech_threshold=0.6,
            rms_threshold=0.02,
        )

    def test_credit_dropped_even_when_loud_and_confident(self):
        # The 唱歌 case: loud, and the model is sure it is speech.
        self.assertTrue(self.gate("字幕由 Amara.org 社群提供", rms=0.4, no_speech_prob=0.01))

    def test_outro_needs_quiet_or_doubt(self):
        self.assertFalse(self.gate("歡迎訂閱按讚分享", rms=0.4, no_speech_prob=0.01))
        self.assertTrue(self.gate("歡迎訂閱按讚分享", rms=0.001, no_speech_prob=0.01))
        self.assertTrue(self.gate("歡迎訂閱按讚分享", rms=0.4, no_speech_prob=0.9))

    def test_quiet_and_doubted_text_dropped(self):
        self.assertTrue(self.gate("嗯", rms=0.001, no_speech_prob=0.9))

    def test_quiet_but_confident_speech_kept(self):
        # A genuinely-spoken far-field utterance: quiet, but Whisper is sure.
        self.assertFalse(self.gate("會議紀錄請寄給我", rms=0.001, no_speech_prob=0.02))

    def test_missing_no_speech_prob_never_counts_as_doubt(self):
        self.assertFalse(self.gate("會議紀錄請寄給我", rms=0.0, no_speech_prob=None))


if __name__ == "__main__":
    unittest.main()
