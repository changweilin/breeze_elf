import unittest

from breeze_elf.hallucination import (
    is_common_silence_hallucination,
    is_credit_hallucination,
    normalize_hallucination_text,
    should_drop_text,
)

# Thresholds mirror the production defaults (config.py asr_no_speech_prob_threshold /
# asr_hallucination_rms_threshold) so these read like the real gate.
NO_SPEECH = 0.6
RMS = 0.02


class HallucinationTextTests(unittest.TestCase):
    def test_normalize_strips_punct_and_unifies_zan(self):
        # 讚/赞 both fold to 贊; punctuation and spaces drop out.
        self.assertEqual(normalize_hallucination_text("點, 讚！"), "點贊")
        self.assertEqual(normalize_hallucination_text("點赞"), "點贊")
        self.assertEqual(normalize_hallucination_text("   "), "")

    def test_credit_matches_wording_variants(self):
        self.assertTrue(is_credit_hallucination("字幕由社群提供"))
        self.assertTrue(is_credit_hallucination("字幕提供由某某社群提供的字"))
        self.assertTrue(is_credit_hallucination("請不吝點贊訂閱轉發打賞"))
        self.assertTrue(is_credit_hallucination("明鏡與點點欄目"))
        self.assertFalse(is_credit_hallucination("今天天氣很好"))
        self.assertFalse(is_credit_hallucination(""))

    def test_common_silence_fragment(self):
        self.assertTrue(is_common_silence_hallucination("歡迎訂閱按讚分享"))
        self.assertFalse(is_common_silence_hallucination("歡迎大家來聽"))


class ShouldDropTests(unittest.TestCase):
    def test_credit_dropped_even_loud_and_confident(self):
        # The loud-歌詞 case: high volume, low no_speech_prob, but still boilerplate.
        self.assertTrue(
            should_drop_text(
                "字幕由社群提供", 0.0, 0.9, no_speech_threshold=NO_SPEECH, rms_threshold=RMS
            )
        )

    def test_quiet_no_speech_dropped(self):
        self.assertTrue(
            should_drop_text("嗯", 0.9, 0.001, no_speech_threshold=NO_SPEECH, rms_threshold=RMS)
        )

    def test_loud_invented_lyric_leaks(self):
        # No credit match, loud, confident -> the gate cannot catch it (this is the
        # hallucination the C-layer negative metric exists to surface).
        self.assertFalse(
            should_drop_text(
                "憑空一句歌詞", 0.1, 0.5, no_speech_threshold=NO_SPEECH, rms_threshold=RMS
            )
        )

    def test_common_fragment_needs_only_one_of_energy_or_no_speech(self):
        # no_speech only
        self.assertTrue(
            should_drop_text(
                "歡迎訂閱按讚分享", 0.9, 0.5, no_speech_threshold=NO_SPEECH, rms_threshold=RMS
            )
        )
        # low energy only
        self.assertTrue(
            should_drop_text(
                "歡迎訂閱按讚分享", 0.1, 0.001, no_speech_threshold=NO_SPEECH, rms_threshold=RMS
            )
        )
        # neither -> kept
        self.assertFalse(
            should_drop_text(
                "歡迎訂閱按讚分享", 0.1, 0.5, no_speech_threshold=NO_SPEECH, rms_threshold=RMS
            )
        )

    def test_missing_no_speech_prob_is_not_no_speech(self):
        self.assertFalse(
            should_drop_text("嗯", None, 0.001, no_speech_threshold=NO_SPEECH, rms_threshold=RMS)
        )


if __name__ == "__main__":
    unittest.main()
