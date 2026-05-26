from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from .asr import build_asr_from_env
from .asr_queue import ASRQueue
from .audio import AudioUtteranceBuffer, AudioWindow, AudioWindowBuffer
from .config import get_settings
from .protocol import (
    PingMessage,
    ProtocolError,
    StartMessage,
    StopMessage,
    parse_client_text,
    server_event,
)

LOGGER = logging.getLogger("breeze_elf")
ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_WEB_DIR = Path(__file__).resolve().parent / "web"
PROJECT_WEB_DIR = ROOT_DIR / "web"
WEB_DIR = PACKAGE_WEB_DIR if PACKAGE_WEB_DIR.exists() else PROJECT_WEB_DIR


settings = get_settings()
asr_engine = build_asr_from_env(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.asr = asr_engine
    app.state.asr_error = None
    app.state.asr_queue = ASRQueue(asr_engine, settings.asr_concurrency)
    await app.state.asr_queue.start()

    if settings.asr_load_on_startup:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, app.state.asr.load)
        except Exception as exc:  # pragma: no cover - depends on local ASR setup
            app.state.asr_error = str(exc)
            LOGGER.exception("ASR startup load failed")

    try:
        yield
    finally:
        await app.state.asr_queue.stop()


app = FastAPI(title="Breeze Elf", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/manifest.webmanifest")
async def manifest() -> FileResponse:
    return FileResponse(WEB_DIR / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/service-worker.js")
async def service_worker() -> FileResponse:
    return FileResponse(WEB_DIR / "service-worker.js", media_type="application/javascript")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "sampleRate": settings.sample_rate,
            "segmenter": settings.segmenter,
            "vadFrameMs": settings.vad_frame_ms,
            "vadEndSilenceMs": settings.vad_end_silence_ms,
            "asrBackend": asr_engine.backend,
            "asrDevice": asr_engine.device,
            "asrConcurrency": settings.asr_concurrency,
            "asrQueueDepth": app.state.asr_queue.queue_depth,
            "asrError": app.state.asr_error,
        }
    )


@dataclass
class StreamState:
    started: bool = False
    segmenter: AudioWindowBuffer | AudioUtteranceBuffer | None = None
    queue: asyncio.Queue[AudioWindow] | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    processor_task: asyncio.Task[None] | None = None
    transcript: str = ""
    dropped_windows: int = 0
    started_at: float = field(default_factory=time.monotonic)


@app.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    state = StreamState()
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        if websocket.client_state != WebSocketState.CONNECTED:
            return
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except RuntimeError:
                return

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            payload = message.get("bytes")

            if text is not None:
                should_stop = await _handle_text_message(
                    text,
                    state,
                    send_json,
                    websocket.app.state.asr_queue,
                )
                if should_stop:
                    break
                continue

            if payload is not None:
                await _handle_audio_payload(payload, state, send_json)

    except WebSocketDisconnect:
        pass
    finally:
        await _stop_state(state)


async def _handle_text_message(
    raw: str,
    state: StreamState,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
    asr_queue: ASRQueue,
) -> bool:
    try:
        message = parse_client_text(raw)
    except ProtocolError as exc:
        await send_json(server_event("error", message=str(exc)))
        return False

    if isinstance(message, PingMessage):
        uptime_ms = round((time.monotonic() - state.started_at) * 1000)
        await send_json(server_event("stats", uptimeMs=uptime_ms))
        return False

    if isinstance(message, StopMessage):
        await send_json(server_event("stats", stopped=True, reason=message.reason))
        return True

    if isinstance(message, StartMessage):
        if state.started:
            await send_json(server_event("error", message="stream already started"))
            return False

        state.segmenter = _build_segmenter(message.sample_rate)
        state.queue = asyncio.Queue(maxsize=settings.max_queue_windows)
        state.stop_event.clear()
        state.processor_task = asyncio.create_task(
            _process_windows(state, send_json, asr_queue, message.language)
        )
        state.started = True
        await send_json(
            server_event(
                "ready",
                sampleRate=message.sample_rate,
                language=message.language,
                windowSeconds=settings.window_seconds,
                overlapSeconds=settings.overlap_seconds,
                segmenter=settings.segmenter,
                backend=asr_queue.backend,
                device=asr_queue.device,
            )
        )
        return False

    return False


async def _handle_audio_payload(
    payload: bytes,
    state: StreamState,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    if not state.started or state.segmenter is None or state.queue is None:
        await send_json(server_event("error", message="audio received before start"))
        return

    windows = state.segmenter.append_pcm16(payload)
    await _enqueue_windows(windows, state, send_json)


async def _enqueue_windows(
    windows: list[AudioWindow],
    state: StreamState,
    send_json: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> None:
    if state.queue is None:
        return

    dropped = 0
    for window in windows:
        if state.queue.full():
            try:
                state.queue.get_nowait()
                state.queue.task_done()
                state.dropped_windows += 1
                dropped += 1
            except asyncio.QueueEmpty:
                pass
        await state.queue.put(window)

    if dropped and send_json is not None:
        await send_json(
            server_event(
                "stats",
                backpressure=True,
                droppedWindows=state.dropped_windows,
                queueDepth=state.queue.qsize(),
            )
        )


async def _process_windows(
    state: StreamState,
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
    asr_queue: ASRQueue,
    language: str,
) -> None:
    assert state.queue is not None

    while True:
        try:
            window = await asyncio.wait_for(state.queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            if state.stop_event.is_set() and state.queue.empty():
                return
            continue

        try:
            if not window.is_speech:
                await send_json(
                    server_event(
                        "stats",
                        windowIndex=window.index,
                        speech=False,
                        rms=round(window.rms, 5),
                        droppedWindows=state.dropped_windows,
                        queueDepth=state.queue.qsize(),
                        asrQueueDepth=asr_queue.queue_depth,
                    )
                )
                continue

            asr_queue_wait_ms = 0
            try:
                queued_result = await asr_queue.transcribe(
                    window.samples,
                    settings.sample_rate,
                    language,
                )
                result = queued_result.result
                asr_queue_wait_ms = queued_result.queue_wait_ms
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await send_json(
                    server_event(
                        "error",
                        message=str(exc),
                        windowIndex=window.index,
                        asrQueueWaitMs=asr_queue_wait_ms,
                        asrQueueDepth=asr_queue.queue_depth,
                    )
                )
                continue

            if result.text:
                await send_json(
                    server_event(
                        "partial",
                        text=result.text,
                        language=result.language,
                        windowIndex=window.index,
                        segmentKind=window.kind,
                        startSeconds=round(window.start_seconds, 2),
                        endSeconds=round(window.end_seconds, 2),
                    )
                )
                novel_text = _novel_text(state.transcript, result.text)
                if novel_text:
                    state.transcript = f"{state.transcript}{novel_text}".strip()
                    await send_json(
                        server_event(
                            "final",
                            text=novel_text,
                            transcript=state.transcript,
                            language=result.language,
                            windowIndex=window.index,
                            segmentKind=window.kind,
                        )
                    )

            await send_json(
                server_event(
                    "stats",
                    windowIndex=window.index,
                    segmentKind=window.kind,
                    speech=True,
                    rms=round(window.rms, 5),
                    asrMs=result.duration_ms,
                    asrQueueWaitMs=asr_queue_wait_ms,
                    backend=result.backend,
                    device=result.device,
                    droppedWindows=state.dropped_windows,
                    queueDepth=state.queue.qsize(),
                    asrQueueDepth=asr_queue.queue_depth,
                )
            )
        finally:
            state.queue.task_done()


async def _stop_state(state: StreamState) -> None:
    if state.segmenter is not None and hasattr(state.segmenter, "flush"):
        flushed = state.segmenter.flush()
        await _enqueue_windows(flushed, state)

    state.stop_event.set()
    if state.processor_task is None:
        return
    try:
        await asyncio.wait_for(state.processor_task, timeout=2.0)
    except asyncio.TimeoutError:
        state.processor_task.cancel()
    except asyncio.CancelledError:
        pass


def _build_segmenter(sample_rate: int) -> AudioWindowBuffer | AudioUtteranceBuffer:
    if settings.segmenter == "window":
        return AudioWindowBuffer(
            sample_rate=sample_rate,
            window_seconds=settings.window_seconds,
            overlap_seconds=settings.overlap_seconds,
            rms_threshold=settings.rms_threshold,
        )

    return AudioUtteranceBuffer(
        sample_rate=sample_rate,
        frame_ms=settings.vad_frame_ms,
        pre_roll_ms=settings.vad_pre_roll_ms,
        end_silence_ms=settings.vad_end_silence_ms,
        max_segment_seconds=settings.vad_max_segment_seconds,
        rms_threshold=settings.rms_threshold,
    )


def _novel_text(transcript: str, current: str) -> str:
    current = " ".join(current.split())
    if not current:
        return ""
    if not transcript:
        return current

    tail = transcript[-160:]
    normalized_tail = _normalize_for_dedupe(tail)
    normalized_current = _normalize_for_dedupe(current)
    if normalized_current and normalized_current in normalized_tail:
        return ""

    overlap_end = _overlap_end_index(tail, current)
    if overlap_end is not None:
        return current[overlap_end:].lstrip(" ，,。.!?！？")
    return f" {current}"


def _overlap_end_index(tail: str, current: str) -> int | None:
    tail_chars = _dedupe_chars(tail)
    current_chars = _dedupe_chars(current)
    if not tail_chars or not current_chars:
        return None

    min_overlap = _min_overlap_size(tail_chars, current_chars)
    max_overlap = min(len(tail_chars), len(current_chars))
    normalized_tail = "".join(char for char, _ in tail_chars)
    normalized_current = "".join(char for char, _ in current_chars)
    for size in range(max_overlap, min_overlap - 1, -1):
        if normalized_tail.endswith(normalized_current[:size]):
            return current_chars[size - 1][1]
    return None


def _min_overlap_size(
    tail_chars: list[tuple[str, int]],
    current_chars: list[tuple[str, int]],
) -> int:
    probe = "".join(char for char, _ in tail_chars[-24:] + current_chars[:24])
    if any("\u4e00" <= char <= "\u9fff" for char in probe):
        return 2
    return 6


def _normalize_for_dedupe(text: str) -> str:
    return "".join(char for char, _ in _dedupe_chars(text))


def _dedupe_chars(text: str) -> list[tuple[str, int]]:
    ignored = set(" \t\r\n，,。.!?！？、；;：:\"'“”‘’（）()[]【】")
    chars: list[tuple[str, int]] = []
    for index, char in enumerate(text):
        if char in ignored:
            continue
        chars.append((char.casefold(), index + 1))
    return chars


def run() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("breeze_elf.main:app", host=settings.host, port=settings.port, reload=False)
