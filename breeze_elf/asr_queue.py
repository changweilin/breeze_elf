from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

import numpy as np

from .asr import ASREngine, ASRResult


@dataclass(frozen=True)
class QueuedASRResult:
    result: ASRResult
    queue_wait_ms: int
    queue_depth: int


@dataclass
class _ASRJob:
    samples: np.ndarray
    sample_rate: int
    language: str
    submitted_at: float
    future: asyncio.Future[QueuedASRResult]
    languages: tuple[str, ...] = ()
    prompt_terms: tuple[str, ...] = ()


class ASRQueue:
    def __init__(self, asr: ASREngine, concurrency: int = 1) -> None:
        self.asr = asr
        self.concurrency = max(1, concurrency)
        self._queue: asyncio.Queue[_ASRJob] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    @property
    def backend(self) -> str:
        return self.asr.backend

    @property
    def device(self) -> str:
        return self.asr.device

    @property
    def model(self) -> str:
        return getattr(self.asr, "model_name", "unknown")

    @property
    def compute_type(self) -> str:
        return getattr(self.asr, "compute_type", "unknown")

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._workers:
            return
        for index in range(self.concurrency):
            self._workers.append(asyncio.create_task(self._worker(index)))

    async def stop(self) -> None:
        while True:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.cancel()
            self._queue.task_done()

        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str,
        *,
        languages: tuple[str, ...] = (),
        prompt_terms: tuple[str, ...] = (),
    ) -> QueuedASRResult:
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[QueuedASRResult] = loop.create_future()
        await self._queue.put(
            _ASRJob(
                samples=samples,
                sample_rate=sample_rate,
                language=language,
                submitted_at=time.perf_counter(),
                future=future,
                languages=languages,
                prompt_terms=prompt_terms,
            )
        )
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _worker(self, index: int) -> None:
        del index
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            try:
                if job.future.cancelled():
                    continue

                queue_wait_ms = round((time.perf_counter() - job.submitted_at) * 1000)
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda job=job: self.asr.transcribe(
                            job.samples,
                            job.sample_rate,
                            job.language,
                            languages=job.languages,
                            prompt_terms=job.prompt_terms,
                        ),
                    )
                except asyncio.CancelledError:
                    if not job.future.done():
                        job.future.cancel()
                    raise
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                    continue

                if not job.future.done():
                    job.future.set_result(
                        QueuedASRResult(
                            result=result,
                            queue_wait_ms=queue_wait_ms,
                            queue_depth=self.queue_depth,
                        )
                    )
            finally:
                self._queue.task_done()
