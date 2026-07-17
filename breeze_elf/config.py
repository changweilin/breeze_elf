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
    language: str = "zh"
    asr_model: str = "medium"
    asr_device: str = "auto"
    asr_provider: str = "faster-whisper"
    asr_load_on_startup: bool = True
    asr_concurrency: int = 1
    asr_no_speech_prob_threshold: float = 0.6
    asr_hallucination_rms_threshold: float = 0.02
    stop_drain_timeout_seconds: float = 60.0
    remote_storage_dir: str = "remote_transcripts"
    search_enabled: bool = True
    search_max_results: int = 50
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
        vad_silero_model_path=os.getenv("BREEZE_VAD_SILERO_MODEL", ""),
        vad_frame_ms=max(1, _int_env("BREEZE_VAD_FRAME_MS", 100)),
        vad_pre_roll_ms=max(0, _int_env("BREEZE_VAD_PRE_ROLL_MS", 300)),
        vad_end_silence_ms=max(1, _int_env("BREEZE_VAD_END_SILENCE_MS", 700)),
        vad_max_segment_seconds=max(0.1, _float_env("BREEZE_VAD_MAX_SEGMENT_SECONDS", 18.0)),
        char_attack_ms=max(0, _int_env("BREEZE_CHAR_ATTACK_MS", 40)),
        char_release_ms=max(0, _int_env("BREEZE_CHAR_RELEASE_MS", 90)),
        char_voiceless_margin=max(1.0, _float_env("BREEZE_CHAR_VOICELESS_MARGIN", 1.6)),
        language=os.getenv("BREEZE_LANGUAGE", "zh"),
        asr_model=os.getenv("BREEZE_ASR_MODEL", "medium"),
        asr_device=os.getenv("BREEZE_ASR_DEVICE", "auto"),
        asr_provider=os.getenv("BREEZE_ASR_PROVIDER", "faster-whisper"),
        asr_load_on_startup=_bool_env("BREEZE_ASR_LOAD_ON_STARTUP", True),
        asr_concurrency=max(1, _int_env("BREEZE_ASR_CONCURRENCY", 1)),
        asr_no_speech_prob_threshold=_float_env("BREEZE_ASR_NO_SPEECH_PROB_THRESHOLD", 0.6),
        asr_hallucination_rms_threshold=_float_env("BREEZE_ASR_HALLUCINATION_RMS_THRESHOLD", 0.02),
        stop_drain_timeout_seconds=max(0.1, _float_env("BREEZE_STOP_DRAIN_TIMEOUT_SECONDS", 60.0)),
        remote_storage_dir=os.getenv("BREEZE_REMOTE_STORAGE_DIR", "remote_transcripts"),
        search_enabled=_bool_env("BREEZE_SEARCH_ENABLED", True),
        search_max_results=max(1, _int_env("BREEZE_SEARCH_MAX_RESULTS", 50)),
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
