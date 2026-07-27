from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import re
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

from .asr import (
    ASRResult,
    FasterWhisperASR,
    WordTiming,
    build_asr_from_env,
    resolve_breeze_model_dir,
)
from .asr_models import active_model_id, find_asr_model, list_asr_models
from .asr_queue import ASRQueue
from .audio import (
    AudioUtteranceBuffer,
    AudioWindow,
    AudioWindowBuffer,
    PitchSummary,
    SegmentAnalysis,
    analyze_segment,
    compute_spectrogram,
    estimate_noise_floor,
    estimate_tempo,
    estimate_tonic,
    extend_voiced_span,
    hz_to_jianpu,
    jianpu_glide,
    jianpu_to_semitones,
    make_speech_detector,
    pcm16le_to_float32,
    pitch_cents_off,
    prepare_asr_audio,
    quantize_beats,
    summarize_pitch,
)
from .config import get_settings
from .diarize import OnlineSpeakerClusterer, build_clusterer, build_diarizer
from .enhance import build_denoiser, build_enhancer, build_separator, preload_torch_cudnn
from .lyrics import align_lyrics, timed_chars_from_blocks
from .protocol import (
    GlossaryEntry,
    PingMessage,
    ProtocolError,
    StartMessage,
    StopMessage,
    _parse_glossary,
    _parse_languages,
    parse_client_text,
    server_event,
)
from .search_index import build_search_index
from .storage import load_transcript, safe_transcript_id, save_transcript
from .summarize import build_summarizer
from .translate import build_translator
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

# Pure subtitle-credit / sponsor boilerplate Whisper emits on *non-speech* (music,
# singing, silence). No real speaker utters these, so they are dropped regardless of
# loudness — this is what catches loud 歌詞, which the energy gate (built for quiet
# silence) structurally cannot: singing is loud and the model is confident, so
# ``low_energy``/``likely_no_speech`` never fire. Matched as regex against the
# punctuation-stripped, casefolded form (see ``_normalize_hallucination_text``) so
# org-name / wording variants ("字幕提供由 XXX 社群提供的字") still hit.
HALLUCINATION_CREDIT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"字幕.{0,20}社群提供",
        r"社群提供.{0,8}字幕",
        r"請不吝.{0,24}打賞",
        r"明鏡與點點",
    )
)
# Softer channel-outro phrases a real speaker *might* actually say; only dropped when
# the audio is also quiet / no-speech so genuine speech is never lost.
COMMON_SILENCE_HALLUCINATION_FRAGMENTS = ("歡迎訂閱按讚分享",)
HALLUCINATION_TEXT_TRANSLATION = str.maketrans({"讚": "贊", "赞": "贊"})
_JIANPU_GLIDE_UP = "↗"
_JIANPU_GLIDE_DOWN = "↘"
# How many leading characters of a *disjoint* VAD utterance may be trimmed as a
# boundary duplicate. Disjoint utterances never share speech, so the only real
# overlap is the pre-roll re-included when a >max_segment 字句 is force-split; that
# is at most ``vad_pre_roll_ms`` of audio (a couple of 字). Capping the trim here
# keeps a repeated 歌詞 (a chorus, or the same short line sung twice) from being
# mistaken for an overlap and dropped.
_VAD_BOUNDARY_OVERLAP_CHARS = 6


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
    glideMid: float | None = None
    centsOff: float | None = None
    intensity: float | None = None
    intensityStart: float | None = None
    intensityEnd: float | None = None
    # Rhythm, from the offline pass only (the beat is measured over the whole
    # recording). Kept on the model so they survive a save/restore round trip.
    beats: float | None = None
    noteValue: str | None = None


class TranscriptBlock(BaseModel):
    text: str
    startSeconds: float | None = None
    endSeconds: float | None = None
    segmentKind: str | None = None
    pitch: dict[str, Any] | None = None
    characters: list[TranscriptCharacter] = Field(default_factory=list)
    # Optional post-recognition annotations: the translated line (#4) and the
    # anonymous per-session speaker index (#3, 0-based; UI shows index + 1).
    translation: str | None = None
    speaker: int | None = None


class TranscriptSaveRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str | None = None
    blocks: list[TranscriptBlock] | None = None
    sampleRate: int | None = None
    audioBase64: str | None = None
    # Optional 基頻分析 export (time,hz,intensity,text rows) saved as a sibling .csv.
    pitchCsv: str | None = Field(default=None, max_length=8_000_000)


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


class SeparateRequest(BaseModel):
    # Raw 16-bit little-endian mono PCM (base64) at ``sampleRate`` — the same
    # format the audio WebSocket streams, so the client needs no WAV encoder.
    pcmBase64: str = Field(min_length=1)
    sampleRate: int = Field(default=16_000, ge=8_000, le=48_000)
    scenario: str = Field(default="music", max_length=16)


class DenoiseRequest(BaseModel):
    # Same raw 16-bit LE mono PCM upload as SeparateRequest; no scenario needed.
    pcmBase64: str = Field(min_length=1)
    sampleRate: int = Field(default=16_000, ge=8_000, le=48_000)


class VoiceTtsRequest(BaseModel):
    voiceId: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)
    language: str | None = None
    baseHz: float | None = Field(default=None, ge=50.0, le=600.0)
    speed: float | None = Field(default=None, ge=0.25, le=4.0)


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
    # 滑音 (glide) endpoints + measured pitch contour / 氣音 support.
    jianpuStart: str | None = Field(default=None, max_length=16)
    jianpuEnd: str | None = Field(default=None, max_length=16)
    startHz: float | None = None
    endHz: float | None = None
    glideMid: float | None = None
    contour: list[float] | None = Field(default=None, max_length=64)
    kind: str | None = Field(default=None, max_length=12)
    intensity: float | None = None


class VoiceSingRequest(BaseModel):
    voiceId: str = Field(min_length=1)
    notes: list[VoiceSingNote] = Field(min_length=1, max_length=1000)
    tonicHz: float | None = None
    useMeasuredHz: bool = False
    speed: float | None = Field(default=None, ge=0.25, le=4.0)


class TranscriptAnalyzeRequest(BaseModel):
    """Re-analyze a recorded transcript's per-character 音準 from its audio.

    The blocks carry each character's *absolute* start/end seconds (as produced
    live and persisted in the 逐字稿), so the offline pass can re-measure pitch
    straight from the saved WAV with the more accurate YIN tracker and a single
    global 主音, fixing the octave / tonic drift the real-time pass can have.
    """

    audioBase64: str = Field(min_length=1)
    sampleRate: int | None = None
    blocks: list[TranscriptBlock] = Field(default_factory=list, max_length=4000)


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


@dataclass
class AnalyzeState:
    """Thread-safe progress for the offline 逐字稿 音準 re-analysis pass.

    Like :class:`VoiceLoadState` the work runs in an executor thread while the
    frontend polls a status endpoint to drive a real progress bar; unlike it,
    the job can be re-run (after ``done``/``error``) and carries its result.
    """

    status: str = "idle"  # idle | running | done | error
    progress: float = 0.0
    stage: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self) -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.progress = 0.0
            self.stage = "準備中"
            self.error = None
            self.result = None
            return True

    def report(self, fraction: float, stage: str) -> None:
        with self.lock:
            self.progress = max(0.0, min(1.0, float(fraction)))
            self.stage = stage

    def finish(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.status = "done"
            self.progress = 1.0
            self.stage = "完成"
            self.error = None
            self.result = result

    def fail(self, message: str) -> None:
        with self.lock:
            self.status = "error"
            self.error = message
            self.result = None

    def snapshot(self, *, include_result: bool = False) -> dict[str, Any]:
        with self.lock:
            data = {
                "status": self.status,
                "progress": round(self.progress, 3),
                "stage": self.stage,
                "error": self.error,
            }
            if include_result and self.status == "done":
                data["result"] = self.result
            return data


@dataclass
class ASRSwitchState:
    """Thread-safe progress for a live ASR model hot-swap (模型與演算法 switcher).

    Re-runnable like :class:`AnalyzeState`: the new model loads in an executor
    thread while the frontend polls a status endpoint to drive a progress bar.
    """

    status: str = "idle"  # idle | loading | ready | error
    progress: float = 0.0
    stage: str = ""
    error: str | None = None
    target_id: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self, target_id: str) -> bool:
        with self.lock:
            if self.status == "loading":
                return False
            self.status = "loading"
            self.progress = 0.0
            self.stage = "準備中"
            self.error = None
            self.target_id = target_id
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

    @property
    def is_loading(self) -> bool:
        with self.lock:
            return self.status == "loading"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "progress": round(self.progress, 3),
                "stage": self.stage,
                "error": self.error,
                "targetId": self.target_id,
            }


settings = get_settings()
asr_engine = build_asr_from_env(settings)
# Resolve the effective VAD detector once at startup so /health reports what will
# actually run — silero silently degrades to rms when the model/onnxruntime is
# missing — and so that fallback warning is logged at boot, not per connection.
effective_vad_detector = make_speech_detector(
    settings.vad_detector,
    rms_threshold=settings.rms_threshold,
    rms_release_threshold=settings.rms_threshold * settings.vad_rms_release_ratio,
    speech_threshold=settings.vad_speech_threshold,
    neg_threshold=settings.vad_neg_threshold,
    model_path=settings.vad_silero_model_path or None,
).name
# Neural enhancement stage in front of Whisper (NullEnhancer when off / no extra).
# ``live``/``file`` can differ so the latency-sensitive mic path and the offline
# file path are tuned independently; the separator is the whole-file music pass.
live_enhancer = build_enhancer("live", settings)
file_enhancer = build_enhancer("file", settings)
separator = build_separator(settings)
# Whole-recording denoise for the 原始/降噪 A/B compare — always DeepFilterNet,
# regardless of the ASR enhance settings.
denoiser = build_denoiser(settings)
summarizer = build_summarizer(
    settings.summary_provider,
    model=settings.summary_model,
    ollama_url=settings.summary_ollama_url,
    timeout=settings.summary_timeout_seconds,
)
# Post-recognition NLLB translator (NullTranslator when off / no model). Shared
# module-scope; its own lock serialises inference across streams.
translator = build_translator(settings, base_dir=ROOT_DIR)
translate_target = settings.translate_target
# Shared speaker-embedding model (NullDiarizer when off / no model). The embedder
# is shared (one model, own lock); each connection gets its own clusterer so
# speaker labels are anonymous and reset per session.
diarizer = build_diarizer(settings, base_dir=ROOT_DIR)
voice_engine = build_voice_from_env(settings)
voice_load_state = VoiceLoadState()
analyze_state = AnalyzeState()
asr_switch_state = ASRSwitchState()
# Live streams currently attached to the ASR queue. A model hot-swap is rejected
# while > 0 so we never stop a queue out from under an in-flight recording. Mutated
# only from event-loop coroutines (StartMessage / _stop_state), so a plain int is safe.
_active_stream_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.asr = asr_engine
    app.state.asr_error = None
    app.state.voice = voice_engine
    app.state.voice_load = voice_load_state
    app.state.voice_load_task = None
    app.state.analyze = analyze_state
    app.state.analyze_task = None
    app.state.asr_switch_task = None
    app.state.asr_queue = ASRQueue(asr_engine, settings.asr_concurrency)
    await app.state.asr_queue.start()
    loop = asyncio.get_running_loop()

    # When GPU enhancement/separation is in play, load torch's cuDNN before
    # ctranslate2 (Whisper) so the two do not fight over cudnn64_9.dll on Windows.
    # Must run before any Whisper load (startup or lazy). See preload_torch_cudnn.
    torch_gpu_in_use = settings.enhance_device != "cpu" and (
        live_enhancer.name != "off"
        or file_enhancer.name != "off"
        or separator.available
    )
    if torch_gpu_in_use:
        await loop.run_in_executor(None, preload_torch_cudnn, settings.enhance_device)

    if settings.asr_load_on_startup:
        try:
            await loop.run_in_executor(None, app.state.asr.load)
        except Exception as exc:  # pragma: no cover - depends on local ASR setup
            app.state.asr_error = str(exc)
            LOGGER.exception("ASR startup load failed")
        # Warm the live enhancer too so the first utterance is not slow; failure
        # is non-fatal (the enhancer self-disables and falls back to raw audio).
        if live_enhancer.name != "off":
            try:
                await loop.run_in_executor(None, live_enhancer.load)
            except Exception:  # pragma: no cover - depends on optional extra
                LOGGER.exception("Live enhancer startup load failed")

    # Reconcile the search index once at startup so transcripts added or deleted
    # out-of-band are picked up / purged without waiting for the user to search.
    if search_index.available:
        try:
            await loop.run_in_executor(None, search_index.sync, _remote_storage_dir())
        except Exception:  # pragma: no cover - sync already self-guards
            LOGGER.exception("Search index startup sync failed")

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
    # Offload the SQLite count so a poll never blocks the event loop on disk I/O.
    search_indexed = await asyncio.get_event_loop().run_in_executor(
        None, search_index.indexed_count
    )
    return JSONResponse(
        {
            "ok": True,
            "sampleRate": settings.sample_rate,
            "segmenter": settings.segmenter,
            "vadDetector": settings.vad_detector,
            "vadDetectorEffective": effective_vad_detector,
            "audioPreprocess": settings.audio_preprocess,
            "vadFrameMs": settings.vad_frame_ms,
            "vadEndSilenceMs": settings.vad_end_silence_ms,
            "asrBackend": asr_engine.backend,
            "asrDevice": asr_engine.device,
            "asrModel": getattr(asr_engine, "model_name", "unknown"),
            "asrModelId": active_model_id(settings, getattr(asr_engine, "model_name", "")),
            "asrComputeType": getattr(asr_engine, "compute_type", "unknown"),
            "asrConcurrency": settings.asr_concurrency,
            "asrQueueDepth": app.state.asr_queue.queue_depth,
            "asrError": app.state.asr_error,
            "asrFileBatch": settings.asr_file_batch_size,
            "fileBatchAvailable": settings.asr_file_batch_size > 0
            and hasattr(asr_engine, "transcribe_file"),
            "enhanceLive": live_enhancer.name,
            "enhanceFile": file_enhancer.name,
            "enhanceDevice": settings.enhance_device,
            "separatorAvailable": separator.available,
            "denoiseAvailable": denoiser.available,
            "searchEnabled": search_index.available,
            "searchIndexed": search_indexed,
            "summaryProvider": summarizer.name,
            "translateProvider": settings.translate_provider,
            "translateAvailable": translator.available,
            "translateTarget": settings.translate_target,
            "diarizeEnabled": settings.diarize_enabled,
            "diarizeAvailable": diarizer.available,
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

    pitch_csv = payload.pitchCsv.strip() if payload.pitchCsv else None
    try:
        stored = save_transcript(
            payload.text,
            _remote_storage_dir(),
            title=payload.title,
            structured=structured,
            audio=audio,
            pitch_csv=pitch_csv or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        LOGGER.exception("Remote transcript storage failed")
        raise HTTPException(status_code=500, detail="remote transcript storage failed") from exc

    # Best-effort incremental index — an indexing failure must never fail the save
    # (a later search sync() self-heals from the files on disk).
    await asyncio.get_event_loop().run_in_executor(
        None, _index_saved_transcript, stored.id
    )

    return JSONResponse(
        {
            "ok": True,
            "id": stored.id,
            "filename": stored.filename,
            "jsonFilename": stored.json_filename,
            "audioFilename": stored.audio_filename,
            "csvFilename": stored.csv_filename,
            "createdAt": stored.created_at,
            "sizeBytes": stored.size_bytes,
        }
    )


def _index_saved_transcript(doc_id: str) -> None:
    try:
        record = load_transcript(_remote_storage_dir(), doc_id)
        if record is not None:
            search_index.upsert(record)
    except Exception:  # pragma: no cover - indexing must never break a save
        LOGGER.exception("Transcript indexing failed for %s", doc_id)


def _search_transcripts(query: str, limit: int) -> list[dict]:
    # Stat-based sync self-heals for out-of-band adds/deletes and backfills old
    # transcripts on first use; cheap for the few hundred files this app produces.
    storage_dir = _remote_storage_dir()
    search_index.sync(storage_dir)
    return search_index.search(query, limit)


# NOTE: /search MUST be declared before /{doc_id} or FastAPI routes "search" as an id.
@app.get("/api/transcripts/search")
async def search_remote_transcripts(q: str = "", limit: int | None = None) -> JSONResponse:
    if not search_index.available:
        raise HTTPException(status_code=503, detail="transcript search is unavailable")
    capped = min(limit or settings.search_max_results, settings.search_max_results)
    results = await asyncio.get_event_loop().run_in_executor(
        None, _search_transcripts, q, capped
    )
    return JSONResponse({"ok": True, "query": q, "count": len(results), "results": results})


@app.get("/api/transcripts")
async def list_remote_transcripts(limit: int | None = None) -> JSONResponse:
    if not search_index.available:
        raise HTTPException(status_code=503, detail="transcript search is unavailable")
    capped = min(limit or settings.search_max_results, settings.search_max_results)
    results = await asyncio.get_event_loop().run_in_executor(
        None, _search_transcripts, "", capped
    )
    return JSONResponse({"ok": True, "count": len(results), "results": results})


@app.get("/api/transcripts/{doc_id}")
async def read_remote_transcript(doc_id: str) -> JSONResponse:
    try:
        safe = safe_transcript_id(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = await asyncio.get_event_loop().run_in_executor(
        None, load_transcript, _remote_storage_dir(), safe
    )
    if record is None:
        raise HTTPException(status_code=404, detail="transcript not found")

    return JSONResponse(
        {
            "ok": True,
            "id": record.id,
            "filename": record.filename,
            "title": record.title,
            "createdAt": record.created_at,
            "text": record.text,
            "hasAudio": record.has_audio,
            "hasJianpu": record.has_jianpu,
            "audioFilename": record.audio_filename,
            "blocks": record.blocks,
        }
    )


class SummaryRequest(BaseModel):
    text: str
    maxSentences: int | None = None


@app.post("/api/summary")
async def summarize_transcript(payload: SummaryRequest) -> JSONResponse:
    if not summarizer.available:
        raise HTTPException(status_code=503, detail="summary is disabled")
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    # Cap input so a very long transcript can't stall the LLM / blow the context.
    text = text[: settings.summary_max_chars]
    max_sentences = payload.maxSentences or settings.summary_max_sentences
    max_sentences = max(1, min(int(max_sentences), 20))

    summary = await asyncio.get_event_loop().run_in_executor(
        None, lambda: summarizer.summarize(text, max_sentences=max_sentences)
    )
    return JSONResponse({"ok": True, "provider": summarizer.name, "summary": summary})


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


def _decode_pcm_payload(pcm_base64: str) -> np.ndarray:
    """Decode a base64 16-bit PCM upload to float32, bounded so a huge body can't
    OOM the host. The size is checked *before* decoding (413), then the bytes are
    decoded (400 on garbage) and rejected if empty (400). Shared by the whole-file
    endpoints (/api/transcribe/file, /api/enhance/separate)."""
    max_bytes = settings.max_audio_upload_bytes
    if max_bytes and (len(pcm_base64) * 3) // 4 > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"audio payload exceeds {max_bytes} bytes"
        )
    try:
        raw = base64.b64decode(pcm_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid PCM payload") from exc
    samples = pcm16le_to_float32(raw)
    if samples.size == 0:
        raise HTTPException(status_code=400, detail="audio payload is empty")
    return samples


def _remote_storage_dir() -> Path:
    configured = Path(settings.remote_storage_dir).expanduser()
    if configured.is_absolute():
        return configured
    return ROOT_DIR / configured


# Cross-transcript search index over the saved bundles. Degrades to a disabled
# no-op when the runtime SQLite lacks FTS5/trigram (see SearchIndex).
search_index = build_search_index(
    _remote_storage_dir() / "_index.sqlite3", enabled=settings.search_enabled
)


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


class ASRModelSwitchRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64)


def _asr_models_payload() -> dict[str, Any]:
    """The switcher's view of the world: the presets, which one is loaded, and
    enough backend detail to rebuild the footer label after a swap."""
    current_model = getattr(asr_engine, "model_name", "")
    return {
        "models": [opt.to_public() for opt in list_asr_models(settings, current_model)],
        "activeId": active_model_id(settings, current_model),
        "activeModel": current_model,
        "backend": asr_engine.backend,
        "device": asr_engine.device,
        "computeType": getattr(asr_engine, "compute_type", ""),
        "segmenter": settings.segmenter,
        "provider": settings.asr_provider,
        "streaming": _active_stream_count > 0,
        "switch": asr_switch_state.snapshot(),
    }


@app.get("/api/asr/models")
async def list_asr_model_options() -> JSONResponse:
    return JSONResponse({"ok": True, **_asr_models_payload()})


@app.get("/api/asr/model/status")
async def asr_model_switch_status() -> JSONResponse:
    return JSONResponse({"ok": True, **_asr_models_payload()})


async def _run_asr_switch(option) -> None:
    """Load ``option`` into a fresh engine, then atomically swap it (and a new
    queue) in. The new model is loaded *before* the old queue is stopped, so a
    load failure leaves the running model untouched — at the cost of both models
    briefly co-residing in VRAM (fine for medium/large-v3 on a 12 GB card)."""
    global asr_engine
    loop = asyncio.get_running_loop()
    try:
        load_ref = option.model
        if option.kind == "breeze":
            # Offline-first: breeze is a LOCAL CTranslate2 directory. Resolve it
            # (CWD-independent) and fail with a clear message instead of letting
            # faster-whisper fall through to a confusing HuggingFace 404 — and an
            # unwanted network call — when the directory is absent.
            breeze_dir = resolve_breeze_model_dir(settings)
            if not breeze_dir.exists():
                raise RuntimeError(
                    f"找不到 Breeze 模型目錄:{breeze_dir}(請放入 CTranslate2 模型,"
                    "或用 BREEZE_ASR_BREEZE_MODEL 指定路徑)"
                )
            load_ref = str(breeze_dir)
        asr_switch_state.report(0.1, "建立引擎")
        # Identity/display uses the preset value (e.g. the relative breeze dir) so the
        # switcher matches it back to its option; ``load_ref`` is the absolute path the
        # local model is actually loaded from (CWD-independent).
        new_engine = FasterWhisperASR(option.model, settings.asr_device, load_ref=load_ref)
        asr_switch_state.report(0.25, f"載入 {option.label}")
        await loop.run_in_executor(None, new_engine.load)

        asr_switch_state.report(0.85, "切換佇列")
        new_queue = ASRQueue(new_engine, settings.asr_concurrency)
        await new_queue.start()
        old_queue = app.state.asr_queue
        app.state.asr_queue = new_queue
        app.state.asr = new_engine
        app.state.asr_error = None
        asr_engine = new_engine
        # Drain and release the previous engine's queue; dropping the last reference
        # lets CTranslate2 free the old model's VRAM on garbage collection.
        await old_queue.stop()
    except Exception as exc:  # noqa: BLE001 - surface any load/swap failure to the poller
        LOGGER.exception("ASR model switch failed")
        asr_switch_state.fail(str(exc))
    else:
        asr_switch_state.finish()


@app.post("/api/asr/model")
async def switch_asr_model(payload: ASRModelSwitchRequest) -> JSONResponse:
    current_model = getattr(asr_engine, "model_name", "")
    option = find_asr_model(settings, current_model, payload.id)
    if option is None:
        raise HTTPException(status_code=404, detail=f"未知的模型:{payload.id}")
    # Already the running model (and nothing loading) → nothing to do.
    if option.model == current_model and not asr_switch_state.is_loading:
        return JSONResponse({"ok": True, "noop": True, **_asr_models_payload()})
    if _active_stream_count > 0:
        raise HTTPException(status_code=409, detail="辨識進行中,請先停止再切換模型")
    if not asr_switch_state.begin(option.id):
        # A switch is already in flight; report its progress instead of racing.
        return JSONResponse({"ok": True, **_asr_models_payload()})

    app.state.asr_switch_task = asyncio.create_task(_run_asr_switch(option))
    return JSONResponse({"ok": True, **_asr_models_payload()})


def _run_transcript_analysis(
    samples: np.ndarray, sample_rate: int, blocks: list[dict[str, Any]]
) -> None:
    try:
        result = _analyze_blocks_pitch(samples, sample_rate, blocks, analyze_state.report)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the poller
        LOGGER.exception("Transcript pitch analysis failed")
        analyze_state.fail(str(exc))
    else:
        analyze_state.finish(result)


@app.post("/api/transcript/analyze")
async def analyze_transcript(payload: TranscriptAnalyzeRequest) -> JSONResponse:
    samples, sample_rate = _decode_wav_payload(payload.audioBase64)
    if samples.size == 0:
        raise HTTPException(status_code=400, detail="audio payload is empty")
    blocks = [block.model_dump() for block in payload.blocks]
    if not blocks:
        raise HTTPException(status_code=400, detail="no transcript blocks to analyze")

    # One pass at a time; a poll on the status endpoint drives the progress bar.
    if analyze_state.begin():
        loop = asyncio.get_running_loop()
        app.state.analyze_task = loop.run_in_executor(
            None, _run_transcript_analysis, samples, sample_rate, blocks
        )
    return JSONResponse({"ok": True, **analyze_state.snapshot()})


@app.get("/api/transcript/analyze/status")
async def analyze_transcript_status() -> JSONResponse:
    return JSONResponse({"ok": True, **analyze_state.snapshot(include_result=True)})


class LyricsAlignRequest(BaseModel):
    """Replace a recognised transcript with the song's real lyrics, timed.

    The blocks are the recognised transcript (with the per-character timings the
    live pass measured); ``lyrics`` is what the user pasted, one sung line per
    text line. No model runs — see :mod:`breeze_elf.lyrics`.
    """

    lyrics: str = Field(min_length=1, max_length=20_000)
    blocks: list[TranscriptBlock] = Field(min_length=1, max_length=4000)


@app.post("/api/transcript/lyrics")
async def align_transcript_lyrics(payload: LyricsAlignRequest) -> JSONResponse:
    recognised = timed_chars_from_blocks([block.model_dump() for block in payload.blocks])
    try:
        alignment = align_lyrics(payload.lyrics, recognised)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not alignment.lines:
        raise HTTPException(status_code=400, detail="歌詞是空的")

    blocks = [
        {
            "text": line.text,
            "startSeconds": round(line.start, 2) if line.start is not None else None,
            "endSeconds": round(line.end, 2) if line.end is not None else None,
            "segmentKind": "lyrics",
            "characters": [
                {
                    "char": char.char,
                    "startSeconds": round(char.start, 3),
                    "endSeconds": round(char.end, 3),
                    "durationSeconds": round(max(0.0, char.end - char.start), 3),
                }
                for char in line.characters
            ],
            "matchedRatio": round(line.matched_ratio, 3),
            "anchoredRatio": round(line.anchored_ratio, 3),
        }
        for line in alignment.lines
    ]
    return JSONResponse(
        {
            "ok": True,
            "blocks": blocks,
            "transcript": "\n".join(line.text for line in alignment.lines),
            "matchedRatio": round(alignment.matched_ratio, 3),
            "anchoredRatio": round(alignment.anchored_ratio, 3),
            "dropped": alignment.dropped,
        }
    )


class FileTranscribeRequest(BaseModel):
    # Raw 16-bit little-endian mono PCM (base64) — the same format the audio
    # WebSocket streams, so the client reuses its decoder; the whole recording is
    # sent once for a single batched pass. 16 kHz only (like the streaming path).
    pcmBase64: str = Field(min_length=1)
    sampleRate: int = 16_000
    languages: list[str] | None = None
    glossary: list[dict[str, str]] | None = None
    translate: bool = False
    translateTarget: str | None = None


# The batched file endpoint runs OUTSIDE the live ASRQueue on purpose: a long file
# must not head-of-line-block live-mic utterances stuck behind it in a single-worker
# queue. This lock instead serialises concurrent *file* requests so two batched
# pipelines can't run at once and double the shared model's VRAM.
_file_batch_lock = asyncio.Lock()


@app.post("/api/transcribe/file")
async def transcribe_file_endpoint(payload: FileTranscribeRequest) -> JSONResponse:
    """Whole-file batched transcription (opt-in ``BREEZE_ASR_FILE_BATCH_SIZE`` > 0).

    Unlike the streaming file path, this returns every block at once from a single
    ``BatchedInferencePipeline`` pass (3-4x on long files). Per-character 基頻/簡譜 is
    left to the offline ``/api/transcript/analyze`` pass, which re-measures pitch
    against a single global 主音 and is more accurate than the live per-window pass.
    """
    batch_size = settings.asr_file_batch_size
    if batch_size <= 0 or not hasattr(asr_engine, "transcribe_file"):
        raise HTTPException(
            status_code=503,
            detail="batched file transcription is disabled (set BREEZE_ASR_FILE_BATCH_SIZE>0)",
        )
    # 16 kHz only: the model is fixed-rate and the samples are fed in verbatim, so a
    # different rate would silently produce a garbled (wrong-speed) transcript.
    if payload.sampleRate != settings.sample_rate:
        raise HTTPException(
            status_code=400, detail=f"sampleRate must be {settings.sample_rate}"
        )
    samples = _decode_pcm_payload(payload.pcmBase64)

    languages = _parse_languages(payload.languages, "auto")
    glossary = _parse_glossary(payload.glossary)
    replacements = sorted(
        ((entry.source, entry.target) for entry in glossary),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    prompt_terms = tuple(dict.fromkeys(entry.target for entry in glossary))
    want_translate = payload.translate and translator.available
    target = (payload.translateTarget or "").strip().lower() or translate_target

    loop = asyncio.get_running_loop()
    try:
        async with _file_batch_lock:
            transcription = await loop.run_in_executor(
                None,
                lambda: asr_engine.transcribe_file(
                    samples,
                    payload.sampleRate,
                    languages[0],
                    languages=languages,
                    prompt_terms=prompt_terms,
                    batch_size=batch_size,
                ),
            )
    except Exception as exc:
        LOGGER.exception("Batched file transcription failed")
        raise HTTPException(status_code=500, detail=f"file transcription failed: {exc}") from exc

    blocks: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    for segment in transcription.segments:
        display_text = _apply_glossary(segment.text, replacements)
        if not display_text.strip():
            continue
        # A loaded song file hallucinates the same subtitle-credit lines per segment;
        # the streaming gate never runs here, so filter them out by text alone.
        if _is_credit_hallucination(display_text):
            continue
        translation = ""
        if want_translate:
            translation = await loop.run_in_executor(
                None, translator.translate, display_text, transcription.language, target
            )
        characters = [
            {"char": char, "startSeconds": round(start, 3), "endSeconds": round(end, 3)}
            for char, start, end in _split_words_to_chars(segment.words)
        ]
        blocks.append(
            {
                "text": display_text,
                "startSeconds": round(segment.start, 2),
                "endSeconds": round(segment.end, 2),
                "segmentKind": "utterance",
                "characters": characters,
                "translation": translation or None,
            }
        )
        transcript_parts.append(display_text)

    return JSONResponse(
        {
            "ok": True,
            "language": transcription.language,
            "blocks": blocks,
            "transcript": " ".join(transcript_parts).strip(),
            "backend": transcription.backend,
            "device": transcription.device,
            "batchSize": batch_size,
            "translate": want_translate,
            "translateTarget": target if want_translate else "",
        }
    )


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


@app.post("/api/enhance/separate")
async def separate_audio(payload: SeparateRequest) -> JSONResponse:
    """Isolate the vocals stem from a whole music recording (Demucs htdemucs).

    The 音樂 post-processing pass: the client sends the full decoded PCM once,
    before streaming, so recognition runs on isolated 歌詞 instead of a full mix.
    """
    if not separator.available:
        raise HTTPException(
            status_code=503,
            detail="source separation not installed; install torch (CUDA) + demucs "
            "(see README 'Speech Enhancement & Source Separation')",
        )
    samples = _decode_pcm_payload(payload.pcmBase64)

    loop = asyncio.get_running_loop()
    try:
        vocals = await loop.run_in_executor(
            None, separator.separate_vocals, samples, payload.sampleRate
        )
    except Exception as exc:
        LOGGER.exception("Source separation failed")
        raise HTTPException(status_code=500, detail=f"source separation failed: {exc}") from exc

    pcm = (np.clip(vocals, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return JSONResponse(
        {
            "ok": True,
            "pcmBase64": base64.b64encode(pcm).decode("ascii"),
            "sampleRate": payload.sampleRate,
            "durationSeconds": round(vocals.size / payload.sampleRate, 3)
            if payload.sampleRate
            else 0.0,
            "model": separator.name,
            "device": separator.device,
        }
    )


@app.post("/api/enhance/denoise")
async def denoise_audio(payload: DenoiseRequest) -> JSONResponse:
    """DeepFilterNet3 denoise+dereverb over a whole recording, for A/B compare.

    Runs the full clip through the same model the ASR pipeline uses per utterance,
    but in one pass (no window-edge artifacts) — so the returned clip is a fair,
    slightly cleaner stand-in for what recognition hears. Independent of the ASR
    enhance settings so the compare works even when live/file enhancement is off.
    """
    if not denoiser.available:
        raise HTTPException(
            status_code=503,
            detail="speech denoise not installed; install torch + deepfilternet "
            "(see README 'Speech Enhancement & Source Separation')",
        )
    samples = _decode_pcm_payload(payload.pcmBase64)

    loop = asyncio.get_running_loop()
    try:
        enhanced = await loop.run_in_executor(
            None, denoiser.enhance, samples, payload.sampleRate
        )
    except Exception as exc:
        LOGGER.exception("Speech denoise failed")
        raise HTTPException(status_code=500, detail=f"speech denoise failed: {exc}") from exc

    pcm = (np.clip(enhanced, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return JSONResponse(
        {
            "ok": True,
            "pcmBase64": base64.b64encode(pcm).decode("ascii"),
            "sampleRate": payload.sampleRate,
            "durationSeconds": round(enhanced.size / payload.sampleRate, 3)
            if payload.sampleRate
            else 0.0,
            "model": denoiser.name,
            "device": denoiser.device,
        }
    )


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
    speed = payload.speed or 1.0
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: voice_engine.synthesize(
                payload.text, language, embedding, base_hz=payload.baseHz, speed=speed
            ),
        )
    except Exception as exc:
        LOGGER.exception("Voice synthesis failed")
        raise HTTPException(status_code=500, detail=f"voice synthesis failed: {exc}") from exc
    return JSONResponse({"ok": True, **_audio_response(result)})


def _note_has_pitch(note: dict[str, Any]) -> bool:
    """A note can be sung when it carries a pitch in any form (single Hz, a
    contour, glide endpoints, or a parseable 簡譜). Breath/rest notes do not."""
    if (note.get("hz") or 0) > 0:
        return True
    contour = note.get("contour")
    if isinstance(contour, list) and any((hz or 0) > 0 for hz in contour):
        return True
    if (note.get("startHz") or 0) > 0 and (note.get("endHz") or 0) > 0:
        return True
    for token in (note.get("jianpu"), note.get("jianpuStart"), note.get("jianpuEnd")):
        if token and jianpu_to_semitones(token) is not None:
            return True
    return False


@app.post("/api/voice/sing")
async def voice_sing(payload: VoiceSingRequest) -> JSONResponse:
    embedding = _require_embedding(payload.voiceId)
    if not hasattr(voice_engine, "synthesize_song"):
        raise HTTPException(status_code=400, detail="singing not supported by this engine")
    notes = [note.model_dump() for note in payload.notes]
    if not any(_note_has_pitch(note) for note in notes):
        raise HTTPException(status_code=400, detail="no singable notes (need 簡譜 or hz)")
    tonic = payload.tonicHz or 0.0
    speed = payload.speed or 1.0
    target_median = _voice_median_hz(payload.voiceId)
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: voice_engine.synthesize_song(
                notes,
                tonic,
                embedding,
                payload.useMeasuredHz,
                speed=speed,
                target_median_hz=target_median,
            ),
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


_VOICE_MEDIAN_CACHE: dict[str, float] = {}


def _voice_median_hz(voice_id: str) -> float:
    """The target voice's median pitch, measured from its reference sample and
    cached. Used to calibrate sung output into the target's register; returns 0
    when no sample is available (then the register correction is a no-op)."""
    if voice_id in _VOICE_MEDIAN_CACHE:
        return _VOICE_MEDIAN_CACHE[voice_id]
    median = 0.0
    try:
        path = voice_sample_path(voice_id, _voice_storage_dir())
        if path is not None:
            samples, rate = decode_wav(path.read_bytes())
            summary = summarize_pitch(samples, rate)
            if summary.median_hz and math.isfinite(summary.median_hz):
                median = float(summary.median_hz)
    except (OSError, ValueError):
        median = 0.0
    _VOICE_MEDIAN_CACHE[voice_id] = median
    return median


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
    mode: str = "live"
    segmenter: AudioWindowBuffer | AudioUtteranceBuffer | None = None
    queue: asyncio.Queue[AudioWindow] | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    processor_task: asyncio.Task[None] | None = None
    transcript: str = ""
    dropped_windows: int = 0
    # Whether this stream has been tallied into ``_active_stream_count`` (so the
    # decrement in ``_stop_state`` fires exactly once, even when called twice).
    counted_active: bool = False
    started_at: float = field(default_factory=time.monotonic)
    # Per-connection speaker clusterer (diarization). Built at start only when the
    # embedder is available, so labels are anonymous and reset every session.
    clusterer: OnlineSpeakerClusterer | None = None
    # End (absolute seconds) of the last speech segment that reached the dedupe,
    # so the next utterance can tell a disjoint segment (real repeat → keep) from
    # a force-split continuation that re-includes the pre-roll (overlap → trim).
    last_segment_end: float | None = None


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
        # Closes the race where a stream starts mid-swap and captures a queue that
        # ``_run_asr_switch`` is about to stop out from under it.
        if asr_switch_state.is_loading:
            await send_json(server_event("error", message="模型切換中,請稍後再試"))
            return False

        state.mode = message.mode
        state.segmenter = _build_segmenter(message.sample_rate)
        # A fresh clusterer per connection → anonymous, session-local speaker labels.
        state.clusterer = build_clusterer(settings) if diarizer.available else None
        # Translate only when both the client asked and the server has a model.
        want_translate = message.translate and translator.available
        target = message.translate_target or translate_target
        # File analysis streams a whole recording at once; an unbounded queue
        # keeps every window so nothing is dropped to backpressure. Live mic
        # streaming stays bounded so latency cannot snowball.
        queue_size = 0 if message.mode == "file" else settings.max_queue_windows
        state.queue = asyncio.Queue(maxsize=queue_size)
        state.stop_event.clear()
        state.processor_task = asyncio.create_task(
            _process_windows(
                state,
                send_json,
                asr_queue,
                message.languages,
                message.glossary,
                translate=want_translate,
                translate_target=target,
            )
        )
        state.started = True
        global _active_stream_count
        _active_stream_count += 1
        state.counted_active = True
        await send_json(
            server_event(
                "ready",
                sampleRate=message.sample_rate,
                language=message.language,
                languages=list(message.languages),
                windowSeconds=settings.window_seconds,
                overlapSeconds=settings.overlap_seconds,
                segmenter=settings.segmenter,
                # AudioWindowBuffer has no detector (always energy) → report "rms".
                vadDetector=getattr(state.segmenter, "detector_name", "rms"),
                backend=asr_queue.backend,
                device=asr_queue.device,
                model=asr_queue.model,
                computeType=asr_queue.compute_type,
                enhance=(file_enhancer if message.mode == "file" else live_enhancer).name,
                translate=want_translate,
                translateTarget=target if want_translate else "",
                diarize=state.clusterer is not None,
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
    languages: tuple[str, ...],
    glossary: tuple[GlossaryEntry, ...],
    *,
    translate: bool = False,
    translate_target: str = "",
) -> None:
    assert state.queue is not None

    loop = asyncio.get_running_loop()
    # Fixed per stream: the file path may use a different (or no) enhancer than
    # the latency-sensitive live path.
    enhancer = file_enhancer if state.mode == "file" else live_enhancer
    primary_language = languages[0] if languages else "zh"
    # ``source→target`` replacements (longest source first so a longer 詞 wins
    # over a substring), and the preferred spellings fed to the Whisper prompt.
    replacements = sorted(
        ((entry.source, entry.target) for entry in glossary),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    prompt_terms = tuple(dict.fromkeys(entry.target for entry in glossary))

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

            # Remember where the previous speech segment ended before advancing,
            # so the dedupe can tell a disjoint utterance from a force-split one.
            prev_segment_end = state.last_segment_end
            state.last_segment_end = window.end_seconds

            asr_queue_wait_ms = 0
            try:
                asr_samples = prepare_asr_audio(
                    window.samples,
                    settings.sample_rate,
                    profile=settings.audio_preprocess,
                )
                # Neural denoise+dereverb runs in a worker thread so the model's
                # GPU/CPU work never blocks the event loop; it self-disables to a
                # passthrough on failure, so this never breaks the ASR feed.
                if enhancer.name != "off":
                    asr_samples = await loop.run_in_executor(
                        None, enhancer.enhance, asr_samples, settings.sample_rate
                    )
                # Cross-segment 脈絡: seed the prompt with the tail of what has
                # already been committed so proper nouns stay consistent. Off by
                # default (chars == 0); the ASR engine drops it in free-detect mode.
                context = (
                    state.transcript[-settings.asr_context_chars :]
                    if settings.asr_context_chars > 0
                    else ""
                )
                queued_result = await asr_queue.transcribe(
                    asr_samples,
                    settings.sample_rate,
                    primary_language,
                    languages=languages,
                    prompt_terms=prompt_terms,
                    context=context,
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
                # Apply the learned 慣用詞庫 corrections to the recognised text so
                # both the live partial and the accumulated transcript carry the
                # user's preferred spellings (the prompt biasing is best-effort).
                display_text = _apply_glossary(result.text, replacements)
                sample_rate = _window_sample_rate(window)
                summary = summarize_pitch(window.samples, sample_rate)
                pitch = _pitch_summary_payload(summary)
                characters = _character_payloads(window, result.words, sample_rate, summary)
                await send_json(
                    server_event(
                        "partial",
                        text=display_text,
                        language=result.language,
                        windowIndex=window.index,
                        segmentKind=window.kind,
                        startSeconds=round(window.start_seconds, 2),
                        endSeconds=round(window.end_seconds, 2),
                        pitch=pitch,
                    )
                )
                novel_text = _novel_text(
                    state.transcript,
                    display_text,
                    max_overlap_chars=_dedupe_overlap_cap(window, prev_segment_end),
                )
                if novel_text:
                    state.transcript = f"{state.transcript}{novel_text}".strip()
                    # Both run only on the committed (final) segment, never the
                    # low-latency partial. Translation is of the block's own text so
                    # the bilingual view stays 1:1 with blocks (no overlap-window
                    # duplication). Speaker uses the RAW utterance audio.
                    translation = ""
                    if translate:
                        translation = await loop.run_in_executor(
                            None,
                            translator.translate,
                            novel_text,
                            result.language,
                            translate_target,
                        )
                    speaker = await _assign_speaker(loop, state, window, sample_rate)
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
                            translation=translation or None,
                            speaker=speaker,
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


async def _assign_speaker(
    loop: asyncio.AbstractEventLoop,
    state: StreamState,
    window: AudioWindow,
    sample_rate: int,
) -> int | None:
    """The anonymous session speaker index for this utterance, or ``None``.

    Very short or low-energy utterances are gated out (their embedding is noisy
    and would pollute a centroid). The embedding is taken from the **raw**
    ``window.samples`` — never the denoised ``asr_samples``, whose dereverb/denoise
    strips the very voiceprint cues the speaker model relies on.
    """
    clusterer = state.clusterer
    if clusterer is None:
        return None
    duration = window.end_seconds - window.start_seconds
    if duration < settings.diarize_min_duration or window.rms < settings.rms_threshold:
        return None
    embedding = await loop.run_in_executor(None, diarizer.embed, window.samples, sample_rate)
    if embedding is None:
        return None
    return clusterer.assign(embedding)


async def _stop_state(state: StreamState, drain_timeout: float = 2.0) -> None:
    if state.counted_active:
        global _active_stream_count
        _active_stream_count = max(0, _active_stream_count - 1)
        state.counted_active = False
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

    detector = make_speech_detector(
        settings.vad_detector,
        rms_threshold=settings.rms_threshold,
        rms_release_threshold=settings.rms_threshold * settings.vad_rms_release_ratio,
        speech_threshold=settings.vad_speech_threshold,
        neg_threshold=settings.vad_neg_threshold,
        model_path=settings.vad_silero_model_path or None,
    )
    return AudioUtteranceBuffer(
        sample_rate=sample_rate,
        frame_ms=settings.vad_frame_ms,
        pre_roll_ms=settings.vad_pre_roll_ms,
        end_silence_ms=settings.vad_end_silence_ms,
        max_segment_seconds=settings.vad_max_segment_seconds,
        rms_threshold=settings.rms_threshold,
        speech_detector=detector,
    )


def _should_drop_asr_result(window: AudioWindow, result: ASRResult) -> bool:
    # Subtitle-credit / sponsor boilerplate is never real speech, so drop it whatever
    # the energy. This is the loud-歌詞 case: singing produces these at high volume and
    # high confidence, where the quiet-silence gate below can never fire.
    if _is_credit_hallucination(result.text):
        return True
    # Note (silero VAD): the silero detector can onset a segment on quiet far-field
    # voice whose whole-window RMS is below asr_hallucination_rms_threshold, so
    # ``low_energy`` is True for windows the RMS segmenter would never have emitted.
    # This stays gated behind ``likely_no_speech``/``common_hallucination`` on
    # purpose: a genuinely-spoken quiet utterance carries a *low* no_speech_prob and
    # non-filler text, so it is kept; only when Whisper itself doubts it (or emits a
    # known silence phrase) is it dropped — loosening this would re-admit exactly the
    # silence hallucinations this backstop exists to remove.
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
    # 主音 from key detection over this utterance's notes, weighted by how long
    # each is held; falls back to the median pitch when one utterance is too
    # short to show a key. The offline 校準音準 pass re-estimates it once over
    # the whole piece, which is where the estimate is actually reliable.
    notes = [(seg.median_hz, end - start) for _, start, end, seg in analyzed]
    tonic = estimate_tonic(notes).hz or (summary.median_hz or 0.0)

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
    return _analyzed_character_payload(
        char, window.start_seconds + start, window.start_seconds + end, seg, tonic
    )


def _analyzed_character_payload(
    char: str,
    start: float,
    end: float,
    seg: SegmentAnalysis,
    tonic: float,
    beat_seconds: float | None = None,
) -> dict[str, Any]:
    """Per-character timing, pitch, slide, tuning, and intensity, against a
    given 主音 (``tonic``). Shared by the live window pass and the offline pass.

    ``beat_seconds`` adds the note value (八分音符 / 四分音符 …). Only the offline
    pass has it: the beat is measured over the whole recording, which one live
    utterance cannot see."""
    median = seg.median_hz
    start_hz = seg.start_hz or median
    end_hz = seg.end_hz or median
    single = hz_to_jianpu(median, tonic)
    glide = jianpu_glide(start_hz, end_hz, tonic) if (start_hz and end_hz) else single
    is_glide = glide != single and (_JIANPU_GLIDE_UP in glide or _JIANPU_GLIDE_DOWN in glide)
    beats, note_value = quantize_beats(max(0.0, end - start), beat_seconds)
    return {
        "char": char,
        "startSeconds": round(start, 3),
        "endSeconds": round(end, 3),
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
        "glideMid": round(seg.glide_position, 3) if (is_glide and seg.glide_position) else None,
        "centsOff": _round_cents(pitch_cents_off(median, tonic)),
        "intensity": _round_intensity(seg.intensity),
        "intensityStart": _round_intensity(seg.intensity_start),
        "intensityEnd": _round_intensity(seg.intensity_end),
        "beats": beats,
        "noteValue": note_value or None,
    }


def _padded_char_spans(
    block: dict[str, Any],
    samples: np.ndarray | None = None,
    sample_rate: int = 0,
    noise_floor: float = 0.0,
) -> list[tuple[str, float, float]]:
    """Each 字's analysis window grown outward so 基頻/簡譜 cover the whole 字.

    The base growth is a fixed attack (before its onset) and release (after its
    tail). When the block audio is supplied, the window is grown *further* through
    any adjacent sub-threshold-but-structured audio — an unvoiced consonant or a
    breath that the live RMS VAD clipped — using :func:`extend_voiced_span`. Both
    are capped at half the gap to each neighbour (and to the block edges), so
    adjacent 字 never overlap / merge."""
    attack = settings.char_attack_ms / 1000.0
    release = settings.char_release_ms / 1000.0
    chars = [
        (str(c.get("char") or "").strip(), float(c["startSeconds"]), float(c["endSeconds"]))
        for c in (block.get("characters") or [])
        if c.get("char") and c.get("startSeconds") is not None and c.get("endSeconds") is not None
    ]
    block_start = block.get("startSeconds")
    block_end = block.get("endSeconds")
    have_audio = samples is not None and sample_rate > 0 and samples.size > 0
    spans: list[tuple[str, float, float]] = []
    for index, (char, start, end) in enumerate(chars):
        prev_end = chars[index - 1][2] if index > 0 else None
        next_start = chars[index + 1][1] if index < len(chars) - 1 else None
        # Leftmost / rightmost second this 字 may claim: the neighbour midpoints,
        # then the block edges. Both the fixed pad and the content walk obey it.
        floor_seconds = (start + prev_end) / 2.0 if prev_end is not None else 0.0
        ceil_seconds = (end + next_start) / 2.0 if next_start is not None else end + release
        if block_start is not None:
            floor_seconds = max(floor_seconds, float(block_start))
        if block_end is not None:
            ceil_seconds = min(ceil_seconds, float(block_end))
        floor_seconds = max(0.0, floor_seconds)

        new_start = max(floor_seconds, start - attack)
        new_end = min(ceil_seconds, end + release)
        if have_audio:
            grown_start, grown_end = extend_voiced_span(
                samples,
                sample_rate,
                round(start * sample_rate),
                round(end * sample_rate),
                floor_sample=round(floor_seconds * sample_rate),
                ceil_sample=round(ceil_seconds * sample_rate),
                noise_floor=noise_floor,
                energy_margin=settings.char_voiceless_margin,
            )
            new_start = min(new_start, grown_start / sample_rate)
            new_end = max(new_end, grown_end / sample_rate)
        spans.append((char, new_start, max(new_start, new_end)))
    return spans


def _analyze_blocks_pitch(
    samples: np.ndarray,
    sample_rate: int,
    blocks: list[dict[str, Any]],
    progress: Callable[[float, str], None],
) -> dict[str, Any]:
    """Run every post-processable step in one pass: re-measure each character's
    pitch from the audio, rebuild the 簡譜 against a single global 主音 (so it
    stays consistent across the whole piece), and compute each block's STFT
    spectrogram + 基頻 curve for the 基頻分析 view."""
    total = sum(len(block.get("characters") or []) for block in blocks) or 1
    done = 0
    measured: list[list[tuple[str, float, float, SegmentAnalysis]]] = []
    all_medians: list[float] = []
    notes: list[tuple[float, float]] = []
    # Room-tone floor for the content-aware boundary growth (task: cover the
    # sub-threshold-but-字詞 audio the live RMS VAD clipped).
    noise_floor = estimate_noise_floor(samples, sample_rate)

    # Pass 1 (0.02–0.5): measure every character's pitch from its (attack/release +
    # content-grown) audio slice so the 基頻 covers the 字's onset and tail.
    progress(0.02, "分析音高")
    for block in blocks:
        block_measured: list[tuple[str, float, float, SegmentAnalysis]] = []
        for char, start, end in _padded_char_spans(block, samples, sample_rate, noise_floor):
            done += 1
            seg = _segment_pitch(samples, sample_rate, start, end)
            block_measured.append((char, start, end, seg))
            if seg.median_hz:
                all_medians.append(seg.median_hz)
                notes.append((seg.median_hz, end - start))
            if done % 8 == 0 or done == total:
                progress(0.02 + 0.48 * min(1.0, done / total), f"分析音高 {done}/{total}")
        measured.append(block_measured)

    # One global 主音 for the whole piece, from key detection over every note in
    # it — the median pitch is not the tonic, so writing 簡譜 against it shifts
    # every degree. Thin / tuneless material falls back to the median.
    tonic_estimate = estimate_tonic(notes)
    tonic = tonic_estimate.hz or 0.0
    # One pulse for the whole piece, from the recording's onset pattern. Free
    # tempo / speech reports no beat at all, and the note values stay off.
    tempo = estimate_tempo(samples, sample_rate)

    # Pass 2 (0.5–1.0): rebuild 簡譜 vs the global 主音 + per-block 基頻分析.
    block_total = len(blocks) or 1
    out_blocks: list[dict[str, Any]] = []
    for block_index, (block, block_measured) in enumerate(zip(blocks, measured)):
        characters = [
            _analyzed_character_payload(char, start, end, seg, tonic, tempo.beat_seconds)
            for char, start, end, seg in block_measured
        ]
        out_blocks.append(
            {
                "text": block.get("text") or "",
                "startSeconds": block.get("startSeconds"),
                "endSeconds": block.get("endSeconds"),
                "segmentKind": block.get("segmentKind") or "",
                "pitch": _block_pitch_from_audio(samples, sample_rate, block),
                "characters": characters,
                "spectrogram": _block_spectrogram(samples, sample_rate, block, noise_floor),
            }
        )
        spectro_done = (block_index + 1) / block_total
        progress(0.5 + 0.5 * spectro_done, f"基頻分析 {block_index + 1}/{block_total}")

    progress(1.0, "完成")
    return {
        "blocks": out_blocks,
        "tonicHz": round(tonic, 1) if tonic else None,
        "tonicConfidence": (
            round(tonic_estimate.confidence, 2) if tonic_estimate.source == "key" else None
        ),
        "tempoBpm": round(tempo.bpm, 1) if tempo.bpm else None,
        "tempoConfidence": round(tempo.confidence, 2) if tempo.bpm else None,
        "characterCount": len(all_medians),
    }


def _block_spectrogram(
    samples: np.ndarray, sample_rate: int, block: dict[str, Any], noise_floor: float = 0.0
) -> dict[str, Any] | None:
    start = block.get("startSeconds")
    end = block.get("endSeconds")
    if start is None or end is None:
        return None
    begin = max(0, int(float(start) * sample_rate))
    finish = min(samples.size, int(math.ceil(float(end) * sample_rate)))
    if finish - begin < sample_rate // 50:
        return None
    payload = compute_spectrogram(samples[begin:finish], sample_rate, clean_f0=settings.f0_clean)
    if payload is None:
        return None
    payload["durationSeconds"] = round((finish - begin) / sample_rate, 3)
    # Tag each time bin with the character sounding then (per-point 文字 for the
    # 基頻分析 / CSV), using the attack/release-padded character windows (absolute
    # seconds) so brief 字 still claim a bin and the lyric reads completely.
    slice_start = begin / sample_rate
    spans = [
        (start, end, char)
        for char, start, end in _padded_char_spans(block, samples, sample_rate, noise_floor)
    ]
    payload["text"] = [
        _char_at_time(spans, slice_start + relative) for relative in payload.get("times", [])
    ]
    return payload


def _char_at_time(spans: list[tuple[float, float, str]], moment: float) -> str:
    for start, end, char in spans:
        if start <= moment < end:
            return char
    return ""


def _block_pitch_from_audio(
    samples: np.ndarray, sample_rate: int, block: dict[str, Any]
) -> dict[str, Any] | None:
    start = block.get("startSeconds")
    end = block.get("endSeconds")
    if start is None or end is None:
        return None
    begin = max(0, int(float(start) * sample_rate))
    finish = min(samples.size, int(math.ceil(float(end) * sample_rate)))
    if finish - begin < sample_rate // 20:
        return None
    return _pitch_summary_payload(summarize_pitch(samples[begin:finish], sample_rate))


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


def _is_credit_hallucination(text: str) -> bool:
    """True when the text is subtitle-credit / sponsor boilerplate. Dropped regardless
    of energy (unlike the silence fragments) because loud singing emits these at high
    volume + high confidence, where the energy / no_speech gate can never fire."""
    normalized = _normalize_hallucination_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in HALLUCINATION_CREDIT_PATTERNS)


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


def _dedupe_overlap_cap(window: AudioWindow, prev_segment_end: float | None) -> int | None:
    """Decide how aggressively this segment may be deduped against the transcript.

    ``None`` (the overlapping ``window`` segmenter) keeps the full dedupe: those
    windows literally re-cover the previous window's audio, so a verbatim repeat
    is an artefact to drop. Disjoint VAD utterances never share speech, so a
    repeat there is a real 歌詞 — they get ``0`` (no dedupe). The one exception is
    a >max_segment 字句 force-split, where the continuation overlaps the previous
    segment by the re-included pre-roll; that small boundary duplicate is trimmed.
    """
    if window.kind != "utterance":
        return None
    overlaps_previous = (
        prev_segment_end is not None and window.start_seconds < prev_segment_end - 1e-3
    )
    return _VAD_BOUNDARY_OVERLAP_CHARS if overlaps_previous else 0


def _apply_glossary(text: str, replacements: list[tuple[str, str]]) -> str:
    """Substitute each learned ``source→target`` 慣用詞 in the recognised text.

    ``replacements`` is pre-sorted longest-source-first so a longer 詞 is taken
    before any substring of it; empty sources are already filtered upstream.
    """
    if not text or not replacements:
        return text
    for source, target in replacements:
        if source:
            text = text.replace(source, target)
    return text


def _novel_text(transcript: str, current: str, *, max_overlap_chars: int | None = None) -> str:
    current = " ".join(current.split())
    if not current:
        return ""
    if not transcript:
        return current

    tail = transcript[-160:]
    normalized_tail = _normalize_for_dedupe(tail)
    normalized_current = _normalize_for_dedupe(current)
    # Only the overlapping ``window`` segmenter (no cap) can emit a window whose new
    # audio is silence and so repeats the previous text verbatim — drop it. A capped
    # (disjoint VAD) segment that happens to match recent text is a real 歌詞 repeat.
    if max_overlap_chars is None and normalized_current and normalized_current in normalized_tail:
        return ""

    overlap_end = _overlap_end_index(tail, current, max_overlap_chars=max_overlap_chars)
    if overlap_end is not None:
        return current[overlap_end:].lstrip(" ，,。.!?！？")
    return f" {current}"


def _overlap_end_index(
    tail: str, current: str, *, max_overlap_chars: int | None = None
) -> int | None:
    tail_chars = _dedupe_chars(tail)
    current_chars = _dedupe_chars(current)
    if not tail_chars or not current_chars:
        return None

    min_overlap = _min_overlap_size(tail_chars, current_chars)
    max_overlap = min(len(tail_chars), len(current_chars))
    if max_overlap_chars is not None:
        max_overlap = min(max_overlap, max_overlap_chars)
    if max_overlap < min_overlap:
        return None
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


def run(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="breeze-elf", description="Breeze Elf local ASR server.")
    sub = parser.add_subparsers(dest="command")
    doctor_parser = sub.add_parser(
        "doctor", help="Check GPU/torch/cuDNN/models and print an actionable report."
    )
    doctor_parser.add_argument(
        "--load",
        action="store_true",
        help="Also load the Whisper model to confirm the resolved device/compute type.",
    )
    args = parser.parse_args(argv)

    if args.command == "doctor":
        from .doctor import run_doctor

        raise SystemExit(run_doctor(load_model=args.load))

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("breeze_elf.main:app", host=settings.host, port=settings.port, reload=False)
