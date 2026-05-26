import asyncio
import unittest

import numpy as np

from breeze_elf.audio import AudioWindowBuffer
from breeze_elf.main import StreamState, _handle_audio_payload, _novel_text


class MainTests(unittest.IsolatedAsyncioTestCase):
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


class NovelTextTests(unittest.TestCase):
    def test_novel_text_removes_overlap(self):
        self.assertEqual(_novel_text("今天 天氣很好", "天氣很好 我們出門"), "我們出門")

    def test_novel_text_ignores_duplicate_tail(self):
        self.assertEqual(_novel_text("今天 天氣很好", "天氣很好"), "")


if __name__ == "__main__":
    unittest.main()
