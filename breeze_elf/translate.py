"""Any-language → Traditional-Chinese translation of the transcript.

An optional NLLB-200 translator sits *after* Whisper: each recognised segment's
text is translated once and shown as a bilingual line. It runs on the
``ctranslate2`` runtime faster-whisper already pulls in — so **no torch**, no new
heavy CUDA/DLL to fight with — lazy-loads from a *local* model directory, and is
fault-tolerant: a missing model / dependency / OOM degrades to
:class:`NullTranslator` (a no-op) so recognition is never affected.

Gated behind the ``[translate]`` extra (``sentencepiece``) plus a locally-present
CT2-converted NLLB model. NLLB-200 is **CC-BY-NC-4.0 (non-commercial)**, so it is
strictly opt-in (``BREEZE_TRANSLATE=nllb``) and never enabled by default.

Tokenisation note (the sentencepiece-only recipe's one landmine): the flores
language codes (``zho_Hant``, ``eng_Latn`` …) are **not** pieces in the
``sentencepiece`` model — they are added special tokens in the CT2 shared
vocabulary. So the source is fed as *token strings* ``[src_lang] + pieces +
["</s>"]`` and the target is primed with ``target_prefix=[tgt_lang]``; the leading
target-language token is stripped back off the hypothesis before decoding. See
:func:`NllbTranslator._run` and the round-trip test that locks this exact shape.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path

from .config import Settings, get_settings

LOGGER = logging.getLogger("breeze_elf.translate")

# The NLLB end-of-sentence piece and the sentencepiece file names shipped with the
# public CT2 conversions (either the flores tokenizer or the raw BPE model).
_EOS_TOKEN = "</s>"
_SPM_GLOBS = ("*sentencepiece*.model", "*flores*.model", "*.model", "*.spm")

# Map the ISO-639-1 codes Whisper reports (and the UI's language picker uses) to
# the flores-200 codes NLLB expects. ``zh`` → Traditional because this app's
# transcript output is already OpenCC-converted to Traditional Chinese; the target
# default is likewise Traditional. Codes already in flores form pass through.
_FLORES = {
    "zh": "zho_Hant",
    "zh-tw": "zho_Hant",
    "zh-hant": "zho_Hant",
    "zh-cn": "zho_Hans",
    "zh-hans": "zho_Hans",
    "yue": "yue_Hant",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "tl": "tgl_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "fa": "pes_Arab",
}


def to_flores(code: str | None) -> str | None:
    """A flores-200 code for ``code`` (an ISO-639-1 / picker code, or an already
    flores code), or ``None`` when it is empty / ``auto`` / unrecognised."""
    if not code:
        return None
    key = code.strip().lower().replace("_", "-")
    if not key or key == "auto":
        return None
    if key in _FLORES:
        return _FLORES[key]
    # Accept an already-flores code passed straight through (e.g. "zho_Hant").
    if "_" in code and len(code) >= 7:
        return code.strip()
    return None


class NullTranslator:
    """No-op translator — the default when ``BREEZE_TRANSLATE=off`` or the extra /
    model is absent. ``translate`` always returns ``""`` so callers add no line."""

    name = "off"
    device = "none"
    compute_type = "none"

    @property
    def available(self) -> bool:
        return False

    @property
    def ready(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def translate(self, text: str, source_lang: str | None, target_lang: str | None) -> str:
        return ""


class NllbTranslator:
    """NLLB-200 translation over ``ctranslate2`` + ``sentencepiece``.

    Lazy-loaded and thread-safe: a single ``_lock`` serialises both load and
    inference so one CT2 model is safe to share across concurrent streams, exactly
    like :class:`~breeze_elf.enhance.DeepFilterEnhancer`. Only translates *between
    different* languages — a same-language pair short-circuits to ``""``.
    """

    name = "nllb"

    def __init__(
        self,
        model_dir: Path,
        *,
        spm_path: Path | None = None,
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 1,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._spm_path = Path(spm_path) if spm_path else None
        self._device_preference = device
        self._compute_preference = compute_type
        self.beam_size = max(1, int(beam_size))
        self.device = "unloaded"
        self.compute_type = "unloaded"
        self._translator = None
        self._sp = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        """True when the deps import *and* a local model + sentencepiece file are
        present — decided without loading any weights, so ``/health`` is cheap."""
        if importlib.util.find_spec("ctranslate2") is None:
            return False
        if importlib.util.find_spec("sentencepiece") is None:
            return False
        return self._model_dir.is_dir() and self._resolve_spm() is not None

    @property
    def ready(self) -> bool:
        return self._translator is not None

    def _resolve_spm(self) -> Path | None:
        if self._spm_path is not None:
            return self._spm_path if self._spm_path.is_file() else None
        if not self._model_dir.is_dir():
            return None
        for pattern in _SPM_GLOBS:
            matches = sorted(self._model_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _resolve_ct2_device(self) -> tuple[str, str]:
        pref = (self._device_preference or "auto").strip().lower()
        if pref == "cpu":
            device = "cpu"
        elif pref == "cuda":
            device = "cuda"
        else:
            device = "cuda" if _cuda_available() else "cpu"
        compute = (self._compute_preference or "auto").strip().lower()
        if compute in {"", "auto"}:
            compute = "default"
        return device, compute

    def load(self) -> None:
        if self._translator is not None or self._failed:
            return
        with self._lock:
            if self._translator is not None or self._failed:
                return
            spm_path = self._resolve_spm()
            if spm_path is None:
                self._failed = True
                LOGGER.warning("NLLB sentencepiece model not found under %s", self._model_dir)
                return
            try:
                import ctranslate2
                import sentencepiece as spm

                device, compute = self._resolve_ct2_device()
                self._translator = ctranslate2.Translator(
                    str(self._model_dir), device=device, compute_type=compute
                )
                processor = spm.SentencePieceProcessor()
                processor.Load(str(spm_path))
                self._sp = processor
                self.device = device
                self.compute_type = compute
            except Exception as exc:  # pragma: no cover - depends on optional extra
                self._failed = True
                LOGGER.warning("NLLB load failed; translation disabled: %s", exc)

    def translate(self, text: str, source_lang: str | None, target_lang: str | None) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        src = to_flores(source_lang)
        tgt = to_flores(target_lang)
        # No usable language pair, or already the target language → nothing to do.
        if not src or not tgt or src == tgt:
            return ""
        self.load()
        if self._translator is None or self._sp is None:
            return ""
        try:
            with self._lock:
                return self._run(text, src, tgt)
        except Exception as exc:  # pragma: no cover - inference guard
            LOGGER.warning("NLLB translate failed; skipping line: %s", exc)
            return ""

    def _run(self, text: str, src_lang: str, tgt_lang: str) -> str:
        pieces = list(self._sp.encode(text, out_type=str))
        source = [src_lang, *pieces, _EOS_TOKEN]
        results = self._translator.translate_batch(
            [source],
            target_prefix=[[tgt_lang]],
            beam_size=self.beam_size,
            max_decoding_length=max(64, len(source) * 3),
        )
        hypotheses = results[0].hypotheses[0] if results and results[0].hypotheses else []
        tokens = [tok for tok in hypotheses if tok not in (tgt_lang, _EOS_TOKEN)]
        return self._sp.decode(tokens).strip()


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # pragma: no cover - ctranslate2 optional in dev
        return False


def build_translator(settings: Settings | None = None, *, base_dir: Path | None = None):
    """The post-recognition translator. :class:`NullTranslator` unless
    ``BREEZE_TRANSLATE=nllb`` (so the base install never touches sentencepiece)."""
    settings = settings or get_settings()
    if settings.translate_provider != "nllb":
        return NullTranslator()
    model_dir = Path(settings.translate_model).expanduser()
    if not model_dir.is_absolute() and base_dir is not None:
        model_dir = Path(base_dir) / model_dir
    spm_path = Path(settings.translate_spm).expanduser() if settings.translate_spm else None
    if spm_path is not None and not spm_path.is_absolute() and base_dir is not None:
        spm_path = Path(base_dir) / spm_path
    return NllbTranslator(
        model_dir,
        spm_path=spm_path,
        device=settings.translate_device,
        compute_type=settings.translate_compute_type,
        beam_size=settings.translate_beam,
    )
