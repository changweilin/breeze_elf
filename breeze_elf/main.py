from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from .asr import ASRResult, WordTiming, build_asr_from_env
from .asr_queue import ASRQueue
from .audio import (
    AudioUtteranceBuffer,
    AudioWindow,
    AudioWindowBuffer,
    PitchSummary,
    SegmentAnalysis,
    analyze_segment,
    hz_to_jianpu,
    jianpu_glide,
    jianpu_to_semitones,
    pitch_cents_off,
    prepare_asr_audio,
    summarize_pitch,
)
from .config import get_settings
from .protocol import (
    PingMessage,
    ProtocolError,
    StartMessage,
    StopMessage,
    parse_client_text,
    server_event,
)
from .storage import save_transcript
from .voice import build_voice_from_env
from .voice_storage import (
    decode_wav,
    delete_voice,
    encode_wav,
    list_voices,
    load_embedding,
    save_voice,
    save_voice_output,
    update_voice,
    voice_sample_path,
)

LOGGER = logging.getLogger("breeze_elf")
ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_WEB_DIR = Path(__file__).resolve().parent / "web"
PROJECT_WEB_DIR = ROOT_DIR / "web"
WEB_DIR = PACKAGE_WEB_DIR if PACKAGE_WEB_DIR.exists() else PROJECT_WEB_DIR
ROOT_STATIC_MEDIA_TYPES = {
    "app.js": "application/javascript",
    "voice.js": "application/javascript",
    "audio-utils.js": "application/javascript",
    "audio-worklet.js": "application/javascript",
    "favicon.svg": "image/svg+xml",
    "logo.svg": "image/svg+xml",
    "apple-touch-icon.png": "image/png",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
}

COMMON_SILENCE_HALLUCINATION_FRAGMENTS = (
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "請不吝點讚訂閱轉發打賞支持明鏡與點點欄目",
    "字幕由 Amara.org 社群提供",
    "由 Amara.org 社群提供的字幕",
    "歡迎訂閱按讚分享",
)
HALLUCINATION_TEXT_TRANSLATION = str.maketrans({"讚": "贊", "赞": "贊"})
_JIANPU_GLIDE_UP = "↗"
_JIANPU_GLIDE_DOWN = "↘"


class TranscriptCharacter(BaseModel):
    char: str
    startSeconds: float | None = None
    endSeconds: float | None = None
    durationSeconds: float | None = None
    hz: float | None = None
    jianpu: str | None = None
    jianpuStart: str | None = None
    jianpuEnd: str | None = None
    minHz: float | None = None
    maxHz: float | None = None
    startHz: float | None = None
    endHz: float | None = None
    isGlide: bool | None = None
    centsOff: float | None = None
    intensity: float | None = None
    intensityStart: float | None = None
    intensityEnd: float | None = None


class TranscriptBlock(BaseModel):
    text: str
    startSeconds: float | None = None
    endSeconds: float | None = None
    segmentKind: str | None = None
    pitch: dict[str, Any] | None = None
    characters: list[TranscriptCharacter] = Field(default_factory=list)


class TranscriptSaveRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str | None = None
    blocks: list[TranscriptBlock] | None = None
    sampleRate: int | None = None
    audioBase64: str | None = None


class VoiceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    audioBase64: str = Field(min_length=1)
    favorite: bool = False


class VoiceUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    favorite: bool | None = None


class VoiceConvertRequest(BaseModel):
    voiceId: str = Field(min_length=1)
    audioBase64: str = Field(min_length=1)


class VoiceTtsRequest(BaseModel):
    voiceId: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)
    language: str | None = None


class VoiceOutputSaveRequest(BaseModel):
    audioBase64: str = Field(min_length=1)
    kind: str = Field(default="convert", max_length=24)
    voiceId: str | None = None
    text: str | None = Field(default=None, max_length=2000)


class VoiceSingNote(BaseModel):
    char: str = Field(default="", max_length=8)
    jianpu: str | None = Field(default=None, max_length=16)
    hz: float | None = None
    durationSeconds: float | None = None


class VoiceSingRequest(BaseModel):
    voiceId: str = Field(min_length=1)
    notes: list[VoiceSingNote] = Field(min_length=1, max_length=1000)
    tonicHz: float | None = None
    useMeasuredHz: bool = False


@dataclass
class VoiceLoadState:
    """Thread-safe snapshot of voice-model loading for the progress bar.

    The engine loads in an executor thread while the status is polled from the
    async request handlers, so every field is guarded by a lock.
    """

    status: str = "idle"  # idle | loading | ready | error
    progress: float = 0.0
    stage: str = ""
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self) -> bool:
        with self.lock:
            if self.status in {"loading", "ready"}:
                return False
            self.status = "loading"
            self.progress = 0.0
            self.stage = "準備中"
            self.error = None
            return True

    def report(self, fraction: float, stage: str) -> None:
        with self.lock:
            self.progress = max(0.0, min(1.0, float(fraction)))
            self.stage = stage

    def finish(self) -> None:
        with self.lock:
            self.status = "ready"
            self.progress = 1.0
            self.stage = "完成"
            self.error = None

    def fail(self, message: str) -> None:
        with self.lock:
            self.status = "error"
            self.error = message

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "progress": round(self.progress, 3),
                "stage": self.stage,
                "error": self.error,
            }


settings = get_settings()
asr_engine = build_asr_from_env(settings)
voice_engine = build_voice_from_env(settings)
voice_load_state = VoiceLoadState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.asr = asr_engine
    app.state.asr_error = None
    app.state.voice = voice_engine
    app.state.voice_load = voice_load_state
    app.state.voice_load_task = None
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


# The PWA shell + service worker must always revalidate, otherwise a browser
# (or an installed home-screen app) keeps serving a stale page after a deploy —
# which masks frontend fixes until the user manually clears the cache.
_NO_CACHE = {"Cache-Control": "no-cache, max-age=0"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers=_NO_CACHE)


@app.get("/manifest.webmanifest")
async def manifest() -> FileResponse:
    return FileResponse(
        WEB_DIR / "manifest.webmanifest",
        media_type="application/manifest+json",
        headers=_NO_CACHE,
    )


@app.get("/service-worker.js")
async def service_worker() -> FileResponse:
    return FileResponse(
        WEB_DIR / "service-worker.js",
        media_type="application/javascript",
        headers=_NO_CACHE,
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "sampleRate": settings.sample_rate,
            "segmenter": settings.segmenter,
            "audioPreprocess": settings.audio_preprocess,
            "vadFrameMs": settings.vad_frame_ms,
            "vadEndSilenceMs": settings.vad_end_silence_ms,
            "asrBackend": asr_engine.backend,
            "asrDevice": asr_engine.device,
            "asrModel": getattr(asr_engine, "model_name", "unknown"),
            "asrComputeType": getattr(asr_engine, "compute_type", "unknown"),
            "asrConcurrency": settings.asr_concurrency,
            "asrQueueDepth": app.state.asr_queue.queue_depth,
            "asrError": app.state.asr_error,
            "voiceProvider": settings.voice_provider,
            "voiceBackend": voice_engine.backend,
            "voiceModel": getattr(voice_engine, "model_name", "unknown"),
            "voiceLoadStatus": voice_load_state.snapshot()["status"],
        }
    )


@app.get("/{asset_name}", include_in_schema=False)
async def root_static_asset(asset_name: str) -> FileResponse:
    media_type = ROOT_STATIC_MEDIA_TYPES.get(asset_name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="static asset not found")
    return FileResponse(WEB_DIR / asset_name, media_type=media_type)


@app.post("/api/transcripts")
async def create_remote_transcript(payload: TranscriptSaveRequest) -> JSONResponse:
    structured = _structured_payload(payload)
    audio = _decode_audio(payload.audioBase64)

    try:
        stored = save_transcript(
            payload.text,
            _remote_storage_dir(),
            title=payload.title,
            structured=structured,
            audio=audio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        LOGGER.exception("Remote transcript storage failed")
        raise HTTPException(status_code=500, detail="remote transcript storage failed") from exc

    return JSONResponse(
        {
            "ok": True,
            "id": stored.id,
            "filename": stored.filename,
            "jsonFilename": stored.json_filename,
            "audioFilename": stored.audio_filename,
            "createdAt": stored.created_at,
            "sizeBytes": stored.size_bytes,
        }
    )


def _structured_payload(payload: TranscriptSaveRequest) -> dict[str, Any] | None:
    if payload.blocks is None and not payload.audioBase64:
        return None
    return {
        "text": payload.text,
        "title": payload.title,
        "sampleRate": payload.sampleRate,
        "blocks": [block.model_dump() for block in (payload.blocks or [])],
    }


def _decode_audio(audio_base64: str | None) -> bytes | None:
    if not audio_base64:
        return None
    try:
        return base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid audio encoding") from exc


def _remote_storage_dir() -> Path:
    configured = Path(settings.remote_storage_dir).expanduser()
    if configured.is_absolute():
        return configured
    return ROOT_DIR / configured


def _voice_storage_dir() -> Path:
    configured = Path(settings.voice_storage_dir).expanduser()
    if configured.is_absolute():
        return configured
    return ROOT_DIR / configured


def _voice_output_dir() -> Path:
    configured = Path(settings.voice_output_dir).expanduser()
    if configured.is_absolute():
        return configured
    return ROOT_DIR / configured


def _voice_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "provider": settings.voice_provider,
        "backend": voice_engine.backend,
        "device": getattr(voice_engine, "device", "unknown"),
        "model": getattr(voice_engine, "model_name", "unknown"),
        "sampleRate": settings.voice_sample_rate,
        "language": settings.voice_language,
    }
    if extra:
        meta.update(extra)
    return meta


def _run_voice_load() -> None:
    try:
        voice_engine.load(progress=voice_load_state.report)
    except Exception as exc:  # pragma: no cover - depends on optional engine
        LOGGER.exception("Voice model load failed")
        voice_load_state.fail(str(exc))
    else:
        voice_load_state.finish()


@app.post("/api/voice/load")
async def load_voice_model() -> JSONResponse:
    if voice_load_state.begin():
        loop = asyncio.get_running_loop()
        app.state.voice_load_task = loop.run_in_executor(None, _run_voice_load)
    snapshot = voice_load_state.snapshot()
    return JSONResponse({"ok": True, **snapshot, **_voice_meta()})


@app.get("/api/voice/status")
async def voice_status() -> JSONResponse:
    snapshot = voice_load_state.snapshot()
    meta = _voice_meta({"voiceCount": len(list_voices(_voice_storage_dir()))})
    return JSONResponse({"ok": True, **snapshot, **meta})


@app.get("/api/voices")
async def list_saved_voices() -> JSONResponse:
    voices = [voice.to_public() for voice in list_voices(_voice_storage_dir())]
    return JSONResponse({"ok": True, "voices": voices})


@app.post("/api/voices")
async def create_saved_voice(payload: VoiceCreateRequest) -> JSONResponse:
    samples, sample_rate = _decode_wav_payload(payload.audioBase64)
    audio_bytes = _decode_audio(payload.audioBase64)
    loop = asyncio.get_running_loop()
    try:
        embedding = await loop.run_in_executor(
            None, voice_engine.extract_embedding, samples, sample_rate
        )
    except Exception as exc:
        LOGGER.exception("Voice embedding extraction failed")
        raise HTTPException(status_code=500, detail=f"voice extraction failed: {exc}") from exc

    duration = samples.size / sample_rate if sample_rate else 0.0
    try:
        stored = save_voice(
            payload.name,
            embedding,
            _voice_storage_dir(),
            sample_audio=audio_bytes,
            sample_rate=sample_rate,
            duration_seconds=duration,
            favorite=payload.favorite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        LOGGER.exception("Voice storage failed")
        raise HTTPException(status_code=500, detail="voice storage failed") from exc

    return JSONResponse({"ok": True, "voice": stored.to_public()})


@app.patch("/api/voices/{voice_id}")
async def patch_saved_voice(voice_id: str, payload: VoiceUpdateRequest) -> JSONResponse:
    try:
        stored = update_voice(
            voice_id,
            _voice_storage_dir(),
            name=payload.name,
            favorite=payload.favorite,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="voice not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "voice": stored.to_public()})


@app.delete("/api/voices/{voice_id}")
async def delete_saved_voice(voice_id: str) -> JSONResponse:
    try:
        removed = delete_voice(voice_id, _voice_storage_dir())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="voice not found")
    return JSONResponse({"ok": True, "id": voice_id})


@app.get("/api/voices/{voice_id}/sample")
async def get_voice_sample(voice_id: str) -> FileResponse:
    try:
        path = voice_sample_path(voice_id, _voice_storage_dir())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="voice sample not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/voice/convert")
async def convert_voice(payload: VoiceConvertRequest) -> JSONResponse:
    samples, sample_rate = _decode_wav_payload(payload.audioBase64)
    embedding = _require_embedding(payload.voiceId)
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, voice_engine.convert, samples, sample_rate, embedding
        )
    except Exception as exc:
        LOGGER.exception("Voice conversion failed")
        raise HTTPException(status_code=500, detail=f"voice conversion failed: {exc}") from exc
    return JSONResponse({"ok": True, **_audio_response(result)})


@app.post("/api/voice/outputs")
async def save_voice_output_endpoint(payload: VoiceOutputSaveRequest) -> JSONResponse:
    audio = _decode_audio(payload.audioBase64)
    if not audio:
        raise HTTPException(status_code=400, detail="audio payload is empty")
    try:
        stored = save_voice_output(
            audio,
            _voice_output_dir(),
            kind=payload.kind,
            voice_id=payload.voiceId,
            text=payload.text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        LOGGER.exception("Voice output storage failed")
        raise HTTPException(status_code=500, detail="voice output storage failed") from exc
    return JSONResponse(
        {
            "ok": True,
            "id": stored.id,
            "filename": stored.filename,
            "kind": stored.kind,
            "createdAt": stored.created_at,
            "sizeBytes": stored.size_bytes,
        }
    )


@app.post("/api/voice/tts")
async def voice_tts(payload: VoiceTtsRequest) -> JSONResponse:
    embedding = _require_embedding(payload.voiceId)
    language = (payload.language or settings.voice_language).strip() or settings.voice_language
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, voice_engine.synthesize, payload.text, language, embedding
        )
    except Exception as exc:
        LOGGER.exception("Voice synthesis failed")
        raise HTTPException(status_code=500, detail=f"voice synthesis failed: {exc}") from exc
    return JSONResponse({"ok": True, **_audio_response(result)})


@app.post("/api/voice/sing")
async def voice_sing(payload: VoiceSingRequest) -> JSONResponse:
    embedding = _require_embedding(payload.voiceId)
    if not hasattr(voice_engine, "synthesize_song"):
        raise HTTPException(status_code=400, detail="singing not supported by this engine")
    notes = [note.model_dump() for note in payload.notes]
    singable = any(
        (note.get("hz") or 0) > 0 or jianpu_to_semitones(note.get("jianpu")) is not None
        for note in notes
    )
    if not singable:
        raise HTTPException(status_code=400, detail="no singable notes (need 簡譜 or hz)")
    tonic = payload.tonicHz or 0.0
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: voice_engine.synthesize_song(notes, tonic, embedding, payload.useMeasuredHz),
        )
    except Exception as exc:
        LOGGER.exception("Voice singing failed")
        raise HTTPException(status_code=500, detail=f"voice singing failed: {exc}") from exc
    return JSONResponse({"ok": True, **_audio_response(result)})


def _decode_wav_payload(audio_base64: str):
    audio = _decode_audio(audio_base64)
    if not audio:
        raise HTTPException(status_code=400, detail="audio payload is empty")
    try:
        return decode_wav(audio)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure maps to a 400
        raise HTTPException(status_code=400, detail=f"invalid WAV audio: {exc}") from exc


def _require_embedding(voice_id: str) -> bytes:
    try:
        return load_embedding(voice_id, _voice_storage_dir())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="voice not found") from exc


def _audio_response(result) -> dict[str, Any]:
    wav_bytes = encode_wav(result.samples, result.sample_rate)
    duration = result.samples.size / result.sample_rate if result.sample_rate else 0.0
    return {
        "audioBase64": base64.b64encode(wav_bytes).decode("ascii"),
        "sampleRate": result.sample_rate,
        "durationSeconds": round(duration, 3),
    }


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
        await _stop_state(state, drain_timeout=settings.stop_drain_timeout_seconds)
        await send_json(server_event("stats", stopped=True, reason=message.reason))
        return True

    if isinstance(message, StartMessage):
        if state.started:
            await send_json(server_event("error", message="stream already started"))
            return False

        state.segmenter = _build_segmenter(message.sample_rate)
        # File analysis streams a whole recording at once; an unbounded queue
        # keeps every window so nothing is dropped to backpressure. Live mic
        # streaming stays bounded so latency cannot snowball.
        queue_size = 0 if message.mode == "file" else settings.max_queue_windows
        state.queue = asyncio.Queue(maxsize=queue_size)
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
                model=asr_queue.model,
                computeType=asr_queue.compute_type,
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
                asr_samples = prepare_asr_audio(
                    window.samples,
                    settings.sample_rate,
                    profile=settings.audio_preprocess,
                )
                queued_result = await asr_queue.transcribe(
                    asr_samples,
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

            filtered_as_silence = bool(result.text and _should_drop_asr_result(window, result))
            if result.text and not filtered_as_silence:
                sample_rate = _window_sample_rate(window)
                summary = summarize_pitch(window.samples, sample_rate)
                pitch = _pitch_summary_payload(summary)
                characters = _character_payloads(window, result.words, sample_rate, summary)
                await send_json(
                    server_event(
                        "partial",
                        text=result.text,
                        language=result.language,
                        windowIndex=window.index,
                        segmentKind=window.kind,
                        startSeconds=round(window.start_seconds, 2),
                        endSeconds=round(window.end_seconds, 2),
                        pitch=pitch,
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
                            startSeconds=round(window.start_seconds, 2),
                            endSeconds=round(window.end_seconds, 2),
                            pitch=pitch,
                            characters=characters,
                        )
                    )

            await send_json(
                server_event(
                    "stats",
                    windowIndex=window.index,
                    segmentKind=window.kind,
                    speech=not filtered_as_silence,
                    filtered=filtered_as_silence,
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


async def _stop_state(state: StreamState, drain_timeout: float = 2.0) -> None:
    if not state.stop_event.is_set() and state.segmenter is not None and hasattr(
        state.segmenter,
        "flush",
    ):
        flushed = state.segmenter.flush()
        await _enqueue_windows(flushed, state)

    state.stop_event.set()
    if state.processor_task is None:
        return
    try:
        await asyncio.wait_for(state.processor_task, timeout=drain_timeout)
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


def _should_drop_asr_result(window: AudioWindow, result: ASRResult) -> bool:
    likely_no_speech = (
        result.no_speech_prob is not None
        and result.no_speech_prob >= settings.asr_no_speech_prob_threshold
    )
    low_energy = window.rms <= settings.asr_hallucination_rms_threshold
    common_hallucination = _is_common_silence_hallucination(result.text)

    return (likely_no_speech and low_energy) or (
        common_hallucination and (likely_no_speech or low_energy)
    )


def _character_payloads(
    window: AudioWindow,
    words: tuple[WordTiming, ...],
    sample_rate: int,
    summary: PitchSummary,
) -> list[dict[str, Any]]:
    """Per-character timing, pitch, slide, tuning, and intensity analysis."""
    segments = _split_words_to_chars(words)
    if not segments:
        return []

    analyzed = [
        (char, start, end, _segment_pitch(window.samples, sample_rate, start, end))
        for char, start, end in segments
    ]
    voiced = [seg.median_hz for *_, seg in analyzed if seg.median_hz]
    tonic = float(np.median(voiced)) if voiced else (summary.median_hz or 0.0)

    return [
        _character_payload(window, char, start, end, seg, tonic)
        for char, start, end, seg in analyzed
    ]


def _character_payload(
    window: AudioWindow,
    char: str,
    start: float,
    end: float,
    seg: SegmentAnalysis,
    tonic: float,
) -> dict[str, Any]:
    median = seg.median_hz
    start_hz = seg.start_hz or median
    end_hz = seg.end_hz or median
    single = hz_to_jianpu(median, tonic)
    glide = jianpu_glide(start_hz, end_hz, tonic) if (start_hz and end_hz) else single
    is_glide = glide != single and (_JIANPU_GLIDE_UP in glide or _JIANPU_GLIDE_DOWN in glide)
    return {
        "char": char,
        "startSeconds": round(window.start_seconds + start, 3),
        "endSeconds": round(window.start_seconds + end, 3),
        "durationSeconds": round(max(0.0, end - start), 3),
        "hz": _round_pitch(median),
        "minHz": _round_pitch(seg.min_hz),
        "maxHz": _round_pitch(seg.max_hz),
        "startHz": _round_pitch(start_hz),
        "endHz": _round_pitch(end_hz),
        "jianpu": glide if is_glide else single,
        "jianpuStart": hz_to_jianpu(start_hz, tonic),
        "jianpuEnd": hz_to_jianpu(end_hz, tonic),
        "isGlide": is_glide,
        "centsOff": _round_cents(pitch_cents_off(median, tonic)),
        "intensity": _round_intensity(seg.intensity),
        "intensityStart": _round_intensity(seg.intensity_start),
        "intensityEnd": _round_intensity(seg.intensity_end),
    }


def _split_words_to_chars(
    words: tuple[WordTiming, ...],
) -> list[tuple[str, float, float]]:
    segments: list[tuple[str, float, float]] = []
    for word in words:
        chars = [char for char in word.word if not char.isspace()]
        if not chars:
            continue
        start = max(0.0, word.start)
        end = max(start, word.end)
        step = (end - start) / len(chars)
        for index, char in enumerate(chars):
            char_start = start + index * step
            char_end = end if index == len(chars) - 1 else start + (index + 1) * step
            segments.append((char, char_start, char_end))
    return segments


def _segment_pitch(
    samples: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
) -> SegmentAnalysis:
    begin = max(0, int(start_seconds * sample_rate))
    finish = min(samples.size, int(math.ceil(end_seconds * sample_rate)))
    return analyze_segment(samples[begin:finish], sample_rate)


def _window_sample_rate(window: AudioWindow) -> int:
    duration = window.end_seconds - window.start_seconds
    if duration <= 0:
        return settings.sample_rate
    return max(1, round(window.samples.size / duration))


def _pitch_summary_payload(summary: PitchSummary) -> dict[str, Any]:
    return {
        "medianHz": _round_pitch(summary.median_hz),
        "minHz": _round_pitch(summary.min_hz),
        "maxHz": _round_pitch(summary.max_hz),
        "voicedRatio": round(summary.voiced_ratio, 3),
        "points": [
            {
                "offsetSeconds": round(point.offset_seconds, 3),
                "hz": _round_pitch(point.hz),
                "confidence": round(point.confidence, 3),
            }
            for point in summary.points
        ],
    }


def _round_pitch(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 1)


def _round_cents(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value)


def _round_intensity(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(float(value), 4)


def _is_common_silence_hallucination(text: str) -> bool:
    normalized = _normalize_hallucination_text(text)
    if not normalized:
        return False
    return any(
        _normalize_hallucination_text(fragment) in normalized
        for fragment in COMMON_SILENCE_HALLUCINATION_FRAGMENTS
    )


def _normalize_hallucination_text(text: str) -> str:
    ignored = set(" \t\r\n，,。.!?！？、；;：:\"'“”‘’（）()[]【】<>《》·-_/")
    return "".join(
        char.casefold()
        for char in text.translate(HALLUCINATION_TEXT_TRANSLATION)
        if char not in ignored
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
