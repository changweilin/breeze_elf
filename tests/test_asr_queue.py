import asyncio
import threading
import time
import unittest

import numpy as np

from breeze_elf.asr import ASRResult
from breeze_elf.asr_queue import ASRQueue


class RecordingASR:
    backend = "recording"
    device = "cpu"

    def __init__(self, delay_seconds=0.01):
        self.delay_seconds = delay_seconds
        self.active = 0
        self.calls = 0
        self.max_active = 0
        self.last_context = None
        self.lock = threading.Lock()

    def load(self):
        return None

    def transcribe(
        self, samples, sample_rate, language, *, languages=(), prompt_terms=(), context=""
    ):
        del samples, sample_rate, languages, prompt_terms
        with self.lock:
            self.active += 1
            self.calls += 1
            call = self.calls
            self.max_active = max(self.max_active, self.active)
            self.last_context = context
        try:
            time.sleep(self.delay_seconds)
            return ASRResult(
                text=f"job {call}",
                language=language,
                duration_ms=round(self.delay_seconds * 1000),
                backend=self.backend,
                device=self.device,
            )
        finally:
            with self.lock:
                self.active -= 1


class BlockingASR:
    backend = "blocking"
    device = "cpu"

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def load(self):
        return None

    def transcribe(
        self, samples, sample_rate, language, *, languages=(), prompt_terms=(), context=""
    ):
        del samples, sample_rate, languages, prompt_terms, context
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=1)
        return ASRResult(
            text="done",
            language=language,
            duration_ms=0,
            backend=self.backend,
            device=self.device,
        )


class ASRQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_serializes_transcription_when_concurrency_is_one(self):
        asr = RecordingASR()
        queue = ASRQueue(asr, concurrency=1)
        try:
            results = await asyncio.gather(
                queue.transcribe(np.zeros(4, dtype=np.float32), 4, "zh"),
                queue.transcribe(np.zeros(4, dtype=np.float32), 4, "zh"),
                queue.transcribe(np.zeros(4, dtype=np.float32), 4, "zh"),
            )
        finally:
            await queue.stop()

        self.assertEqual(asr.max_active, 1)
        self.assertEqual([item.result.text for item in results], ["job 1", "job 2", "job 3"])
        self.assertEqual(results[-1].queue_depth, 0)

    async def test_threads_context_to_engine(self):
        asr = RecordingASR()
        queue = ASRQueue(asr, concurrency=1)
        try:
            await queue.transcribe(np.zeros(4, dtype=np.float32), 4, "zh", context="先前脈絡")
        finally:
            await queue.stop()

        self.assertEqual(asr.last_context, "先前脈絡")

    async def test_cancelled_queued_job_is_not_transcribed(self):
        asr = BlockingASR()
        queue = ASRQueue(asr, concurrency=1)
        first = asyncio.create_task(queue.transcribe(np.zeros(4, dtype=np.float32), 4, "zh"))
        try:
            self.assertTrue(await asyncio.to_thread(asr.started.wait, 1))
            second = asyncio.create_task(queue.transcribe(np.zeros(4, dtype=np.float32), 4, "zh"))
            for _ in range(50):
                if queue.queue_depth >= 1:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(queue.queue_depth, 1)

            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second

            asr.release.set()
            await first
            await asyncio.sleep(0)
            self.assertEqual(asr.calls, 1)
        finally:
            asr.release.set()
            await queue.stop()


if __name__ == "__main__":
    unittest.main()
