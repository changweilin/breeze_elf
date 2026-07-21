"""Neural speech enhancement + source separation for the ASR pipeline.

Two optional models sit in front of Whisper, both lazy-loaded so the base
install stays torch-free and both fault-tolerant so a load/inference failure
falls back to the unmodified audio instead of breaking recognition:

- :class:`DeepFilterEnhancer` — DeepFilterNet3 full-band denoise + dereverb.
  Runs faster than real time, so it is applied per VAD utterance on both the
  live-mic and file paths. Targets the far-mic / reverb / low-level phone
  pickup that degrades recognition.
- :class:`DemucsSeparator` — htdemucs music source separation. A whole-file
  post-processing pass (not per window) that isolates the ``vocals`` stem so
  Whisper transcribes the 歌詞 instead of hallucinating over a full music mix.

Everything is gated behind the ``[enhance]`` extra (torch/torchaudio/
deepfilternet/demucs). Without it the factories return :class:`NullEnhancer`
and :attr:`DemucsSeparator.available` is ``False``.
"""

from __future__ import annotations

import importlib.util
import logging
import threading

import numpy as np

from .config import Settings, get_settings

LOGGER = logging.getLogger("breeze_elf.enhance")

# Explicit scenario tags for the post-processing path (the UI 情境 toggle).
SCENARIO_GENERAL = "general"
SCENARIO_MUSIC = "music"


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _resolve_device(preference: str) -> str:
    pref = (preference or "auto").strip().lower()
    if pref in {"cuda", "cpu"}:
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch optional
        return "cpu"


def preload_torch_cudnn(device_preference: str = "auto") -> bool:
    """Force torch's cuDNN to load before ctranslate2's (Windows DLL conflict).

    torch and ctranslate2 both ship ``cudnn64_9.dll``, but ctranslate2's monolithic
    build lacks symbols torch needs (e.g. ``cudnnGetLibConfig`` → "Could not load
    symbol … Error code 127"). Only one ``cudnn64_9.dll`` loads per process —
    whichever is requested first wins. Whisper (ctranslate2) normally loads at
    startup before any enhancer, so its older cuDNN wins and torch then fails.

    Running one tiny cuDNN conv on CUDA here, *before* Whisper loads, makes torch's
    newer cuDNN win; ctranslate2 then shares it (a superset) and both work. A no-op
    off CUDA or without the ``[enhance]`` extra, and non-fatal on any failure.
    """
    if _resolve_device(device_preference) != "cuda":
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        with torch.no_grad():
            x = torch.zeros(1, 1, 8, 8, device="cuda")
            weight = torch.zeros(1, 1, 3, 3, device="cuda")
            torch.nn.functional.conv2d(x, weight)  # a conv forces the cuDNN load
        torch.cuda.synchronize()
        LOGGER.info("Loaded torch cuDNN ahead of ctranslate2")
        return True
    except Exception as exc:  # pragma: no cover - depends on optional extra
        LOGGER.warning("torch cuDNN preload failed: %s", exc)
        return False


def _resample(audio, orig_sr: int, target_sr: int):
    """Resample a ``(channels, time)`` torch tensor; a no-op at equal rates."""
    if orig_sr == target_sr:
        return audio
    import torchaudio.functional as AF

    return AF.resample(audio, orig_sr, target_sr)


class NullEnhancer:
    """No-op enhancer — the default when ``[enhance]`` is not installed / off."""

    name = "off"

    def load(self) -> None:
        return None

    @property
    def ready(self) -> bool:
        return True

    def enhance(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return samples


class DeepFilterEnhancer:
    """DeepFilterNet3 denoise + dereverb.

    The model runs at 48 kHz internally; 16 kHz ASR audio is resampled up,
    enhanced, and resampled back. A single ``_lock`` serializes both load and
    inference so one CUDA model is safe to share across concurrent streams.
    """

    name = "deepfilter"

    def __init__(self, device: str = "auto") -> None:
        self._device_preference = device
        self.device = "unloaded"
        self._model = None
        self._state = None
        self._sr = 48_000
        self._lock = threading.Lock()
        self._failed = False

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        """True when torch + DeepFilterNet are importable, without loading weights.

        Lets the whole-file denoise endpoint answer 503 up front instead of
        silently returning the input unchanged (the passthrough fallback).
        """
        return _torch_available() and importlib.util.find_spec("df") is not None

    def load(self) -> None:
        if self._model is not None or self._failed:
            return
        with self._lock:
            if self._model is not None or self._failed:
                return
            try:
                from df.enhance import init_df

                model, state, _ = init_df()
                self.device = _resolve_device(self._device_preference)
                try:
                    model = model.to(self.device)
                except Exception:  # pragma: no cover - device move best effort
                    self.device = "cpu"
                self._model = model
                self._state = state
                self._sr = int(state.sr())
            except Exception as exc:  # pragma: no cover - depends on optional extra
                self._failed = True
                LOGGER.warning("DeepFilterNet load failed; enhancement disabled: %s", exc)

    def enhance(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if samples.size == 0:
            return samples
        self.load()
        if self._model is None:
            return samples
        try:
            with self._lock:
                return self._run(samples, sample_rate)
        except Exception as exc:  # pragma: no cover - inference guard
            LOGGER.warning("DeepFilterNet enhance failed; using raw audio: %s", exc)
            return samples

    def _run(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        from df.enhance import enhance as df_enhance

        with torch.no_grad():
            audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
            audio = audio.unsqueeze(0)  # (1, time)
            audio = _resample(audio, sample_rate, self._sr)
            enhanced = df_enhance(self._model, self._state, audio)
            enhanced = _resample(enhanced, self._sr, sample_rate)
        out = enhanced.squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        # Length can drift by a few samples through the resample round-trip; the
        # ASR feed does not require an exact match, so return as-is.
        return out


class DemucsSeparator:
    """htdemucs source separation, returning the isolated ``vocals`` stem.

    A whole-file operation: it must be given the complete recording (not per
    window), so it is driven from the ``/api/enhance/separate`` endpoint before
    the file is streamed through the segmenter.
    """

    name = "demucs"

    def __init__(self, device: str = "auto", model_name: str = "htdemucs") -> None:
        self._device_preference = device
        self.device = "unloaded"
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True when the optional deps are importable, without loading weights."""
        return _torch_available() and importlib.util.find_spec("demucs") is not None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from demucs.pretrained import get_model

            self.device = _resolve_device(self._device_preference)
            model = get_model(self._model_name)
            model.to(self.device)
            model.eval()
            self._model = model

    def separate_vocals(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Isolated vocals as mono ``float32`` at the input ``sample_rate``."""
        if samples.size == 0:
            return samples
        self.load()
        with self._lock:
            return self._run(samples, sample_rate)

    def _run(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        from demucs.apply import apply_model

        model = self._model
        model_sr = int(model.samplerate)
        channels = int(model.audio_channels)

        with torch.no_grad():
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
            wav = wav.unsqueeze(0)  # (1, time)
            wav = _resample(wav, sample_rate, model_sr)
            if channels == 2:
                wav = wav.repeat(2, 1)  # mono → duplicated stereo for the model

            ref = wav.mean(0)
            mean = ref.mean()
            std = ref.std() + 1e-8
            wav = (wav - mean) / std

            sources = apply_model(
                model,
                wav[None],
                device=self.device,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
            sources = sources * std + mean

            vocals = sources[model.sources.index("vocals")]  # (channels, time)
            mono = vocals.mean(0, keepdim=True)
            mono = _resample(mono, model_sr, sample_rate)
        return mono.squeeze(0).contiguous().cpu().numpy().astype(np.float32)


def build_enhancer(mode: str, settings: Settings | None = None):
    """The per-utterance enhancer for ``"live"`` or ``"file"`` mode.

    Returns :class:`NullEnhancer` when the mode's env choice is ``off`` (the
    default) so the base install never needs torch.
    """
    settings = settings or get_settings()
    choice = settings.enhance_live if mode == "live" else settings.enhance_file
    if choice == "deepfilter":
        return DeepFilterEnhancer(settings.enhance_device)
    return NullEnhancer()


def build_separator(settings: Settings | None = None) -> DemucsSeparator:
    settings = settings or get_settings()
    return DemucsSeparator(settings.enhance_device)


def build_denoiser(settings: Settings | None = None) -> DeepFilterEnhancer:
    """A DeepFilterNet enhancer for the whole-recording A/B compare endpoint.

    Independent of ``enhance_live``/``enhance_file`` so the compare still produces
    a denoised clip even when the ASR pipeline runs enhancement ``off``.
    """
    settings = settings or get_settings()
    return DeepFilterEnhancer(settings.enhance_device)
