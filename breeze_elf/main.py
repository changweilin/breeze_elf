from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from .asr import ASREngine, build_asr_from_env
from .audio import AudioWindow, AudioWindowBuffer
from .config import Settings, get_settings
from .protocol import PingMessage, ProtocolError, StartMessage, StopMessage, parse_client_text, server_event


LOGGER = logging.getLogger("breeze_elf")
ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"


settings = get_settings()
asr_engine = build_asr_from_env(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.asr = asr_engine
    app.state.asr_error = None
    app.state.asr_semaphore = asyncio.Semaphore(max(1, settings.asr_concurrency))

    if settings.asr_load_on_startup:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, app.state.asr.load)
        except Exception as exc:  # pragma: no cover - depends on local ASR setup
            app.state.asr_error = str(exc)
            LOGGER.exception("ASR startup load failed")

    yield


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
            "asrBackend": asr_engine.backend,
            "asrDevice": asr_engine.device,
            "asrConcurrency": settings.asr_concurrency,
            "asrError": app.state.asr_error,
        }
    )


@dataclass
class StreamState:
    started: bool = False
    segmenter: AudioWindowBuffer | None = None
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
                    websocket.app.state.asr,
                    websocket.app.state.asr_semaphore,
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
    asr: ASREngine,
    asr_semaphore: asyncio.Semaphore,
) -> bool:
    try:
        message = parse_client_text(raw)
    except ProtocolError as exc:
        await send_json(server_event("error", message=str(exc)))
        return False

    if isinstance(message, PingMessage):
        await send_json(server_event("stats", uptimeMs=round((time.monotonic() - state.started_at) * 1000)))
        return False

    if isinstance(message, StopMessage):
        await send_json(server_event("stats", stopped=True, reason=message.reason))
        return True

    if isinstance(message, StartMessage):
        if state.started:
            await send_json(server_event("error", message="stream already started"))
            return False

        state.segmenter = AudioWindowBuffer(
            sample_rate=message.sample_rate,
            window_seconds=settings.window_seconds,
            overlap_seconds=settings.overlap_seconds,
            rms_threshold=settings.rms_threshold,
        )
        state.queue = asyncio.Queue(maxsize=settings.max_queue_windows)
        state.stop_event.clear()
        state.processor_task = asyncio.create_task(
            _process_windows(state, send_json, asr, asr_semaphore, message.language)
        )
        state.started = True
        await send_json(
            server_event(
                "ready",
                sampleRate=message.sample_rate,
                language=message.language,
                windowSeconds=settings.window_seconds,
                overlapSeconds=settings.overlap_seconds,
                backend=asr.backend,
                device=asr.device,
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

    if dropped:
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
    asr: ASREngine,
    asr_semaphore: asyncio.Semaphore,
    language: str,
) -> None:
    assert state.queue is not None
    loop = asyncio.get_running_loop()

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
                    )
                )
                continue

            asr_queue_wait_ms = 0
            try:
                wait_started = time.perf_counter()
                async with asr_semaphore:
                    asr_queue_wait_ms = round((time.perf_counter() - wait_started) * 1000)
                    result = await loop.run_in_executor(
                        None,
                        asr.transcribe,
                        window.samples,
                        settings.sample_rate,
                        language,
                    )
            except Exception as exc:
                await send_json(
                    server_event(
                        "error",
                        message=str(exc),
                        windowIndex=window.index,
                        asrQueueWaitMs=asr_queue_wait_ms,
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
                        )
                    )

            await send_json(
                server_event(
                    "stats",
                    windowIndex=window.index,
                    speech=True,
                    rms=round(window.rms, 5),
                    asrMs=result.duration_ms,
                    asrQueueWaitMs=asr_queue_wait_ms,
                    backend=result.backend,
                    device=result.device,
                    droppedWindows=state.dropped_windows,
                    queueDepth=state.queue.qsize(),
                )
            )
        finally:
            state.queue.task_done()


async def _stop_state(state: StreamState) -> None:
    state.stop_event.set()
    if state.processor_task is None:
        return
    try:
        await asyncio.wait_for(state.processor_task, timeout=2.0)
    except asyncio.TimeoutError:
        state.processor_task.cancel()
    except asyncio.CancelledError:
        pass


def _novel_text(transcript: str, current: str) -> str:
    current = " ".join(current.split())
    if not current:
        return ""
    if not transcript:
        return current

    tail = transcript[-120:]
    if current in tail:
        return ""

    min_overlap = _min_overlap_size(tail, current)
    max_overlap = min(len(tail), len(current))
    for size in range(max_overlap, min_overlap - 1, -1):
        if tail.endswith(current[:size]):
            return current[size:].lstrip(" ，,。.!?！？")
    return f" {current}"


def _min_overlap_size(tail: str, current: str) -> int:
    probe = f"{tail[-24:]}{current[:24]}"
    if any("\u4e00" <= char <= "\u9fff" for char in probe):
        return 2
    return 6


def run() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("breeze_elf.main:app", host=settings.host, port=settings.port, reload=False)
