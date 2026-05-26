import os
import unittest
from unittest.mock import patch

from breeze_elf.config import get_settings


class ConfigTests(unittest.TestCase):
    def test_get_settings_reads_environment_at_call_time(self):
        with patch.dict(
            os.environ,
            {
                "BREEZE_PORT": "9999",
                "BREEZE_ASR_PROVIDER": "mock",
                "BREEZE_ASR_CONCURRENCY": "3",
                "BREEZE_SEGMENTER": "window",
                "BREEZE_VAD_END_SILENCE_MS": "900",
                "BREEZE_STOP_DRAIN_TIMEOUT_SECONDS": "12.5",
                "BREEZE_ASR_NO_SPEECH_PROB_THRESHOLD": "0.7",
                "BREEZE_ASR_HALLUCINATION_RMS_THRESHOLD": "0.03",
                "BREEZE_REMOTE_STORAGE_DIR": "saved_transcripts",
            },
        ):
            settings = get_settings()

        self.assertEqual(settings.port, 9999)
        self.assertEqual(settings.asr_provider, "mock")
        self.assertEqual(settings.asr_concurrency, 3)
        self.assertEqual(settings.segmenter, "window")
        self.assertEqual(settings.vad_end_silence_ms, 900)
        self.assertEqual(settings.stop_drain_timeout_seconds, 12.5)
        self.assertEqual(settings.asr_no_speech_prob_threshold, 0.7)
        self.assertEqual(settings.asr_hallucination_rms_threshold, 0.03)
        self.assertEqual(settings.remote_storage_dir, "saved_transcripts")

    def test_queue_and_asr_concurrency_are_at_least_one(self):
        with patch.dict(
            os.environ,
            {
                "BREEZE_MAX_QUEUE_WINDOWS": "0",
                "BREEZE_ASR_CONCURRENCY": "0",
            },
        ):
            settings = get_settings()

        self.assertEqual(settings.max_queue_windows, 1)
        self.assertEqual(settings.asr_concurrency, 1)


if __name__ == "__main__":
    unittest.main()
