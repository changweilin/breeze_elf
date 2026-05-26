import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from breeze_elf import main
from breeze_elf.asr import ASRResult
from breeze_elf.asr_queue import QueuedASRResult
from breeze_elf.audio import AudioUtteranceBuffer, AudioWindow, AudioWindowBuffer
from breeze_elf.main import (
    StreamState,
    _handle_audio_payload,
    _handle_text_message,
    _novel_text,
    _should_drop_asr_result,
)


class ImmediateASRQueue:
    backend = "test"
    device = "cpu"
    queue_depth = 0

    async def transcribe(self, samples, sample_rate, language):
        del samples, sample_rate
        return QueuedASRResult(
            result=ASRResult(
                text="最後一句",
                language=language,
                duration_ms=0,
                backend=self.backend,
                device=self.device,
            ),
            queue_wait_ms=0,
            queue_depth=0,
        )


class MainTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_transcript_endpoint_saves_to_configured_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(main, "settings", replace(main.settings, remote_storage_dir=tmp)):
                response = await main.create_remote_transcript(
                    main.TranscriptSaveRequest(text="遠端儲存測試")
                )

            data = json.loads(response.body)

            self.assertTrue(data["ok"])
            self.assertTrue(data["filename"].endswith(".txt"))
            self.assertEqual(
                (Path(tmp) / data["filename"]).read_text(encoding="utf-8"),
                "遠端儲存測試\n",
            )

    async def test_audio_payload_reports_backpressure_when_queue_drops_window(self):
        state = StreamState(
            started=True,
            segmenter=AudioWindowBuffer(
                sample_rate=4,
                window_seconds=1.0,
                overlap_seconds=0.0,
                rms_threshold=0.0,
            ),
            queue=asyncio.Queue(maxsize=1),
        )
        events = []

        async def send_json(payload):
            events.append(payload)

        payload = np.full(8, 1000, dtype="<i2").tobytes()
        await _handle_audio_payload(payload, state, send_json)

        self.assertEqual(state.dropped_windows, 1)
        self.assertEqual(state.queue.qsize(), 1)
        self.assertTrue(any(event.get("backpressure") for event in events))

    async def test_audio_payload_rejects_audio_before_start(self):
        events = []

        async def send_json(payload):
            events.append(payload)

        await _handle_audio_payload(b"1234", StreamState(), send_json)

        self.assertEqual(events[0]["type"], "error")
        self.assertIn("before start", events[0]["message"])

    async def test_stop_flushes_active_utterance_before_reporting_stopped(self):
        events = []

        async def send_json(payload):
            events.append(payload)

        state = StreamState(
            started=True,
            segmenter=AudioUtteranceBuffer(
                sample_rate=10,
                frame_ms=100,
                pre_roll_ms=0,
                end_silence_ms=500,
                rms_threshold=0.01,
            ),
            queue=asyncio.Queue(maxsize=4),
        )
        state.processor_task = asyncio.create_task(
            main._process_windows(state, send_json, ImmediateASRQueue(), "zh")
        )
        state.segmenter.append_pcm16(np.array([1000, 1000, 1000], dtype="<i2").tobytes())

        should_stop = await _handle_text_message(
            '{"type":"stop","reason":"test"}',
            state,
            send_json,
            ImmediateASRQueue(),
        )

        self.assertTrue(should_stop)
        final_index = next(index for index, event in enumerate(events) if event["type"] == "final")
        stopped_index = next(
            index for index, event in enumerate(events) if event.get("stopped")
        )
        self.assertLess(final_index, stopped_index)
        self.assertEqual(events[final_index]["text"], "最後一句")
        self.assertEqual(events[stopped_index]["reason"], "test")


class NovelTextTests(unittest.TestCase):
    def test_novel_text_removes_overlap(self):
        self.assertEqual(_novel_text("今天 天氣很好", "天氣很好 我們出門"), "我們出門")

    def test_novel_text_ignores_duplicate_tail(self):
        self.assertEqual(_novel_text("今天 天氣很好", "天氣很好"), "")

    def test_novel_text_ignores_punctuation_and_spacing_when_deduping(self):
        self.assertEqual(_novel_text("今天，天氣很好。", "今天 天氣很好"), "")

    def test_novel_text_removes_overlap_with_spacing_difference(self):
        self.assertEqual(_novel_text("今天，天氣很好", "天氣 很好 我們出門"), "我們出門")


class SilenceHallucinationTests(unittest.TestCase):
    def test_drops_low_energy_high_no_speech_result(self):
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=np.zeros(16000, dtype=np.float32),
            rms=0.01,
            is_speech=True,
        )
        result = ASRResult(
            text="這是一段幻覺文字",
            language="zh",
            duration_ms=1,
            backend="test",
            device="cpu",
            no_speech_prob=0.8,
        )

        self.assertTrue(_should_drop_asr_result(window, result))

    def test_drops_common_sponsor_hallucination_when_low_energy(self):
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=np.zeros(16000, dtype=np.float32),
            rms=0.01,
            is_speech=True,
        )
        result = ASRResult(
            text="請不吝點讚訂閱轉發打賞支持明鏡與點點欄目",
            language="zh",
            duration_ms=1,
            backend="test",
            device="cpu",
            no_speech_prob=0.1,
        )

        self.assertTrue(_should_drop_asr_result(window, result))

    def test_keeps_normal_speech(self):
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=np.ones(16000, dtype=np.float32) * 0.05,
            rms=0.05,
            is_speech=True,
        )
        result = ASRResult(
            text="今天下午三點開會",
            language="zh",
            duration_ms=1,
            backend="test",
            device="cpu",
            no_speech_prob=0.1,
        )

        self.assertFalse(_should_drop_asr_result(window, result))


class StaticAssetsTests(unittest.TestCase):
    def test_web_dir_points_to_existing_static_assets(self):
        self.assertTrue((Path(main.WEB_DIR) / "index.html").is_file())

    def test_root_static_assets_are_whitelisted_and_present(self):
        for asset_name in main.ROOT_STATIC_MEDIA_TYPES:
            self.assertTrue((Path(main.WEB_DIR) / asset_name).is_file(), asset_name)


if __name__ == "__main__":
    unittest.main()
