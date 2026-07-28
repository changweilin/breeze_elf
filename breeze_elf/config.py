from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _choice_env(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in choices else default


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8788
    sample_rate: int = 16_000
    window_seconds: float = 2.0
    overlap_seconds: float = 0.5
    rms_threshold: float = 0.008
    audio_preprocess: str = "natural"
    # Neural speech enhancement in front of Whisper (needs the ``[enhance]``
    # extra). ``off`` keeps the torch-free DSP-only path. ``deepfilter`` runs
    # DeepFilterNet3 denoise+dereverb per utterance on the live / file path.
    enhance_live: str = "off"
    enhance_file: str = "off"
    enhance_device: str = "auto"
    max_queue_windows: int = 4
    segmenter: str = "vad"
    # VAD onset gate for the "vad" segmenter. ``rms`` is the energy threshold;
    # ``silero`` runs the neural voice detector bundled with faster-whisper
    # (onnxruntime, no new dependency) and silently falls back to ``rms`` when the
    # model/runtime is missing. ``BREEZE_SEGMENTER=silero`` is accepted as an alias.
    vad_detector: str = "rms"
    vad_speech_threshold: float = 0.5
    vad_neg_threshold: float = 0.35
    # RMS VAD attack/release hysteresis: a segment ends only once RMS falls below
    # ``rms_threshold * vad_rms_release_ratio`` (onset still uses the full threshold).
    # < 1.0 keeps a decaying 句尾 syllable attached to its utterance; 1.0 restores the
    # old single-threshold gate. Clamped to [0, 1].
    vad_rms_release_ratio: float = 0.5
    vad_silero_model_path: str = ""
    vad_frame_ms: int = 100
    vad_pre_roll_ms: int = 300
    vad_end_silence_ms: int = 700
    vad_max_segment_seconds: float = 18.0
    # Per-character VAD-style attack/release: each 字's analysis window is grown a
    # little before its onset and after its tail (so 基頻 covers the whole 字),
    # bounded by half the gap to its neighbours so adjacent 字 never merge.
    char_attack_ms: int = 40
    char_release_ms: int = 90
    # Post-processing only: each 字's analysis window is also grown outward through
    # adjacent audio that is below the speech threshold but above the room noise
    # floor (an unvoiced consonant / breath) — kept while RMS stays above
    # ``noise_floor * char_voiceless_margin`` — so 基頻/簡譜 cover the whole 字.
    char_voiceless_margin: float = 1.6
    # 基頻分析 post-processing: drop erratic f0 blobs that never settle and have no
    # 平穩音高 (stable pitch) on either side — pitch-detection artifacts, not singing.
    # Conservative by design (滑音/抖音 and other techniques are preserved); ``0``
    # keeps the raw per-bin YIN track for the 基頻 curve.
    f0_clean: bool = True
    language: str = "zh"
    # Default recognition model. ``breeze`` is a preset sentinel resolved to the local
    # CTranslate2 dir in ``asr_breeze_model`` by ``build_asr_from_env`` (falls back to
    # Whisper ``medium`` when that dir is absent, so a fresh clone still boots). Any
    # other value is passed straight to faster-whisper (a size, HF id, or CT2 path).
    asr_model: str = "breeze"
    asr_device: str = "auto"
    asr_provider: str = "faster-whisper"
    # Local CTranslate2 directory the "breeze" model preset resolves to (the model
    # switcher in 模型與演算法). Offline-first — faster-whisper can only load a CT2
    # model, so this points at a converted Breeze ASR dir on disk, not a HF id.
    asr_breeze_model: str = "models/breeze-asr-25-ct2"
    # LoRA post-trained variant (nan/台語 + zh-TW 歌唱域適應) offered as an A/B preset
    # in the model switcher — surfaced only when its CT2 dir exists (see asr_models).
    asr_breeze_nan_model: str = "models/breeze-asr-25-nan-ct2"
    # JSON registry of models deployed by tools/deploy_model.py, appended to the
    # switcher's builtin presets. Read per request, so a deploy shows up without a
    # restart; lives under models/ (git-ignored) because it is machine-local state.
    asr_presets_file: str = "models/presets.json"
    asr_load_on_startup: bool = True
    asr_concurrency: int = 1
    asr_no_speech_prob_threshold: float = 0.6
    asr_hallucination_rms_threshold: float = 0.02
    # Cross-segment context: how many trailing characters of the committed
    # transcript to feed the next utterance's Whisper ``initial_prompt`` (alongside
    # the 慣用詞庫) so proper nouns stay consistent across segments. ``0`` (default)
    # keeps the current stateless behaviour — opt-in because seeding recent text can
    # coax Whisper into echoing it on a short/quiet utterance. Bounded to cap prompt
    # growth; only applies when a language is fixed (free-detect drops the prompt).
    asr_context_chars: int = 0
    # Loaded-file transcription via faster-whisper's BatchedInferencePipeline
    # (3-4x throughput on long files by batching the whole recording's VAD chunks).
    # ``0`` (default) keeps the per-utterance streaming file path; > 0 enables the
    # POST /api/transcribe/file batched endpoint and is the batch_size. Live mic
    # streaming is never affected. Bounded to keep GPU memory sane.
    asr_file_batch_size: int = 0
    # Beam width for the batched whole-file endpoint. 5 (default) — the §2 sweep showed
    # beam 5 beats greedy on sung audio (mir1k CER -16%, jamendo WER lower) at ~2-3x the
    # decode cost, which file mode can spend since it is offline. Live streaming stays at
    # beam 1 (hardcoded in ASREngine.transcribe) so latency is untouched.
    asr_file_beam: int = 5
    # Upper bound (decoded bytes) on a base64 PCM upload to the whole-file endpoints
    # (/api/transcribe/file, /api/enhance/separate), checked before decoding so an
    # oversized body can't OOM the host. Default ~256 MB ≈ 2.2 h of 16 kHz mono;
    # ``0`` disables the cap. This is a local single-user app, so the bound is loose.
    max_audio_upload_bytes: int = 256_000_000
    stop_drain_timeout_seconds: float = 60.0
    remote_storage_dir: str = "remote_transcripts"
    search_enabled: bool = True
    search_max_results: int = 50
    # Post-meeting summary. ``extractive`` is stdlib-only (no model/VRAM/network);
    # ``ollama`` calls a LOCAL Ollama daemon (transcript stays on the box) and
    # degrades to extractive on failure; ``off`` disables it. No cloud path exists.
    summary_provider: str = "extractive"
    summary_model: str = "qwen3:4b-instruct"
    summary_ollama_url: str = "http://127.0.0.1:11434"
    summary_timeout_seconds: float = 60.0
    summary_max_chars: int = 8000
    summary_max_sentences: int = 5
    # Post-recognition translation. ``off`` (default) keeps the base install free of
    # sentencepiece; ``nllb`` runs NLLB-200 on the ctranslate2 runtime faster-whisper
    # already ships (no torch). NLLB-200 is CC-BY-NC-4.0 (non-commercial) → opt-in.
    # The model loads from a LOCAL dir only (offline-first — never downloaded on demand).
    translate_provider: str = "off"
    translate_target: str = "zh"
    translate_model: str = "models/nllb-200-distilled-600M-ct2"
    translate_spm: str = ""
    translate_device: str = "auto"
    translate_compute_type: str = "auto"
    translate_beam: int = 1
    # Anonymous in-session speaker diarization. ``off`` (default) means no speaker
    # labels; ``on`` needs a local ONNX speaker-embedding model + onnxruntime (the
    # runtime faster-whisper already bundles), otherwise it degrades to no-op.
    diarize_enabled: bool = False
    diarize_model: str = "models/speaker_embedding.onnx"
    diarize_max_speakers: int = 6
    diarize_threshold: float = 0.75
    diarize_min_duration: float = 0.4
    diarize_device: str = "cpu"
    diarize_n_mels: int = 80
    voice_provider: str = "mock"
    voice_storage_dir: str = "voices"
    voice_output_dir: str = "voice_outputs"
    voice_os_tts: bool = True
    voice_sample_rate: int = 16_000
    voice_language: str = "zh"
    voice_checkpoints_dir: str = "checkpoints_v2"
    voice_mock_warmup_seconds: float = 0.9


def _resolve_vad() -> tuple[str, str]:
    """(segmenter, vad_detector). ``BREEZE_SEGMENTER=silero`` is sugar for the vad
    segmenter with the silero detector, so users can flip one env var. It takes
    precedence: ``BREEZE_SEGMENTER=silero`` forces the silero detector even if
    ``BREEZE_VAD_DETECTOR`` is set to something else."""
    raw_segmenter = os.getenv("BREEZE_SEGMENTER", "vad").strip().lower()
    detector = _choice_env("BREEZE_VAD_DETECTOR", "rms", {"rms", "silero"})
    if raw_segmenter == "silero":
        return "vad", "silero"
    return raw_segmenter, detector


def get_settings() -> Settings:
    segmenter, vad_detector = _resolve_vad()
    # Silero probabilities live in [0, 1]; clamp so a fat-fingered threshold can't
    # silently disable detection (speech>1 -> nothing is ever speech), and keep
    # neg <= speech or the hysteresis band inverts and stops holding decisions.
    vad_speech_threshold = min(1.0, max(0.0, _float_env("BREEZE_VAD_SPEECH_THRESHOLD", 0.5)))
    vad_neg_threshold = min(
        vad_speech_threshold, max(0.0, _float_env("BREEZE_VAD_NEG_THRESHOLD", 0.35))
    )
    return Settings(
        host=os.getenv("BREEZE_HOST", "127.0.0.1"),
        port=_int_env("BREEZE_PORT", 8788),
        sample_rate=_int_env("BREEZE_SAMPLE_RATE", 16_000),
        window_seconds=_float_env("BREEZE_WINDOW_SECONDS", 2.0),
        overlap_seconds=_float_env("BREEZE_OVERLAP_SECONDS", 0.5),
        rms_threshold=_float_env("BREEZE_RMS_THRESHOLD", 0.008),
        audio_preprocess=_choice_env(
            "BREEZE_AUDIO_PREPROCESS",
            "natural",
            {"off", "natural", "speech"},
        ),
        enhance_live=_choice_env("BREEZE_ENHANCE_LIVE", "off", {"off", "deepfilter"}),
        enhance_file=_choice_env("BREEZE_ENHANCE_FILE", "off", {"off", "deepfilter"}),
        enhance_device=_choice_env("BREEZE_ENHANCE_DEVICE", "auto", {"auto", "cuda", "cpu"}),
        max_queue_windows=max(1, _int_env("BREEZE_MAX_QUEUE_WINDOWS", 4)),
        segmenter=segmenter,
        vad_detector=vad_detector,
        vad_speech_threshold=vad_speech_threshold,
        vad_neg_threshold=vad_neg_threshold,
        vad_rms_release_ratio=min(1.0, max(0.0, _float_env("BREEZE_VAD_RMS_RELEASE_RATIO", 0.5))),
        vad_silero_model_path=os.getenv("BREEZE_VAD_SILERO_MODEL", ""),
        vad_frame_ms=max(1, _int_env("BREEZE_VAD_FRAME_MS", 100)),
        vad_pre_roll_ms=max(0, _int_env("BREEZE_VAD_PRE_ROLL_MS", 300)),
        vad_end_silence_ms=max(1, _int_env("BREEZE_VAD_END_SILENCE_MS", 700)),
        vad_max_segment_seconds=max(0.1, _float_env("BREEZE_VAD_MAX_SEGMENT_SECONDS", 18.0)),
        char_attack_ms=max(0, _int_env("BREEZE_CHAR_ATTACK_MS", 40)),
        char_release_ms=max(0, _int_env("BREEZE_CHAR_RELEASE_MS", 90)),
        char_voiceless_margin=max(1.0, _float_env("BREEZE_CHAR_VOICELESS_MARGIN", 1.6)),
        f0_clean=_bool_env("BREEZE_F0_CLEAN", True),
        language=os.getenv("BREEZE_LANGUAGE", "zh"),
        asr_model=os.getenv("BREEZE_ASR_MODEL", "breeze"),
        asr_device=os.getenv("BREEZE_ASR_DEVICE", "auto"),
        asr_provider=os.getenv("BREEZE_ASR_PROVIDER", "faster-whisper"),
        asr_breeze_model=os.getenv("BREEZE_ASR_BREEZE_MODEL", "models/breeze-asr-25-ct2"),
        asr_breeze_nan_model=os.getenv(
            "BREEZE_ASR_BREEZE_NAN_MODEL", "models/breeze-asr-25-nan-ct2"
        ),
        asr_presets_file=os.getenv("BREEZE_ASR_PRESETS_FILE", "models/presets.json"),
        asr_load_on_startup=_bool_env("BREEZE_ASR_LOAD_ON_STARTUP", True),
        asr_concurrency=max(1, _int_env("BREEZE_ASR_CONCURRENCY", 1)),
        asr_no_speech_prob_threshold=_float_env("BREEZE_ASR_NO_SPEECH_PROB_THRESHOLD", 0.6),
        asr_hallucination_rms_threshold=_float_env("BREEZE_ASR_HALLUCINATION_RMS_THRESHOLD", 0.02),
        asr_context_chars=min(2000, max(0, _int_env("BREEZE_ASR_CONTEXT_CHARS", 0))),
        asr_file_batch_size=min(32, max(0, _int_env("BREEZE_ASR_FILE_BATCH_SIZE", 0))),
        asr_file_beam=max(1, _int_env("BREEZE_ASR_FILE_BEAM", 5)),
        max_audio_upload_bytes=max(0, _int_env("BREEZE_MAX_AUDIO_UPLOAD_BYTES", 256_000_000)),
        stop_drain_timeout_seconds=max(0.1, _float_env("BREEZE_STOP_DRAIN_TIMEOUT_SECONDS", 60.0)),
        remote_storage_dir=os.getenv("BREEZE_REMOTE_STORAGE_DIR", "remote_transcripts"),
        search_enabled=_bool_env("BREEZE_SEARCH_ENABLED", True),
        search_max_results=max(1, _int_env("BREEZE_SEARCH_MAX_RESULTS", 50)),
        summary_provider=_choice_env(
            "BREEZE_SUMMARY_PROVIDER", "extractive", {"off", "extractive", "ollama"}
        ),
        summary_model=os.getenv("BREEZE_SUMMARY_MODEL", "qwen3:4b-instruct"),
        summary_ollama_url=os.getenv("BREEZE_SUMMARY_OLLAMA_URL", "http://127.0.0.1:11434"),
        summary_timeout_seconds=max(1.0, _float_env("BREEZE_SUMMARY_TIMEOUT_SECONDS", 60.0)),
        summary_max_chars=max(200, _int_env("BREEZE_SUMMARY_MAX_CHARS", 8000)),
        summary_max_sentences=max(1, _int_env("BREEZE_SUMMARY_MAX_SENTENCES", 5)),
        translate_provider=_choice_env("BREEZE_TRANSLATE", "off", {"off", "nllb"}),
        translate_target=os.getenv("BREEZE_TRANSLATE_TARGET", "zh"),
        translate_model=os.getenv(
            "BREEZE_TRANSLATE_MODEL", "models/nllb-200-distilled-600M-ct2"
        ),
        translate_spm=os.getenv("BREEZE_TRANSLATE_SPM", ""),
        translate_device=_choice_env(
            "BREEZE_TRANSLATE_DEVICE", "auto", {"auto", "cuda", "cpu"}
        ),
        translate_compute_type=os.getenv("BREEZE_TRANSLATE_COMPUTE_TYPE", "auto"),
        translate_beam=max(1, _int_env("BREEZE_TRANSLATE_BEAM", 1)),
        diarize_enabled=_bool_env("BREEZE_DIARIZE", False),
        diarize_model=os.getenv("BREEZE_DIARIZE_MODEL", "models/speaker_embedding.onnx"),
        diarize_max_speakers=max(1, _int_env("BREEZE_DIARIZE_MAX_SPEAKERS", 6)),
        diarize_threshold=min(1.0, max(0.0, _float_env("BREEZE_DIARIZE_THRESHOLD", 0.75))),
        diarize_min_duration=max(0.0, _float_env("BREEZE_DIARIZE_MIN_DURATION", 0.4)),
        diarize_device=_choice_env("BREEZE_DIARIZE_DEVICE", "cpu", {"cpu", "cuda"}),
        diarize_n_mels=max(8, _int_env("BREEZE_DIARIZE_N_MELS", 80)),
        voice_provider=_choice_env(
            "BREEZE_VOICE_PROVIDER",
            "mock",
            {"mock", "openvoice"},
        ),
        voice_storage_dir=os.getenv("BREEZE_VOICE_STORAGE_DIR", "voices"),
        voice_output_dir=os.getenv("BREEZE_VOICE_OUTPUT_DIR", "voice_outputs"),
        voice_os_tts=_bool_env("BREEZE_VOICE_OS_TTS", True),
        voice_sample_rate=_int_env("BREEZE_VOICE_SAMPLE_RATE", 16_000),
        voice_language=os.getenv("BREEZE_VOICE_LANGUAGE", "zh"),
        voice_checkpoints_dir=os.getenv("BREEZE_VOICE_CHECKPOINTS_DIR", "checkpoints_v2"),
        voice_mock_warmup_seconds=max(0.0, _float_env("BREEZE_VOICE_MOCK_WARMUP_SECONDS", 0.9)),
    )
