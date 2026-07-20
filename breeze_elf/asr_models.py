"""Switchable ASR model presets for the 模型與演算法 model switcher.

``model`` is what faster-whisper's ``WhisperModel`` receives: a size string
(``medium``), a HF repo id, or a local CTranslate2 directory (Breeze). The active
model is always derived from the running engine's ``model_name`` — there is no
separate source of truth to drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class ASRModelOption:
    id: str
    label: str
    model: str
    kind: str  # "breeze" | "whisper" | "custom"

    def to_public(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "model": self.model, "kind": self.kind}


def builtin_asr_models(settings: Settings) -> list[ASRModelOption]:
    """The curated switchable presets, in display order. Breeze is a local CT2
    directory (offline-first); the Whisper sizes are fetched/cached by
    faster-whisper on first use."""
    return [
        ASRModelOption("breeze", "Breeze ASR", settings.asr_breeze_model, "breeze"),
        ASRModelOption("medium", "Whisper medium", "medium", "whisper"),
        ASRModelOption("large-v3", "Whisper large-v3", "large-v3", "whisper"),
    ]


def list_asr_models(settings: Settings, active_model: str) -> list[ASRModelOption]:
    """Builtin presets plus, when the running model matches none of them, a
    trailing ``current`` entry so the UI always shows what is actually loaded
    (e.g. a ``BREEZE_ASR_MODEL=small`` override)."""
    options = builtin_asr_models(settings)
    active = (active_model or "").strip()
    if active and not any(opt.model == active for opt in options):
        options.append(ASRModelOption("current", f"目前:{active}", active, "custom"))
    return options


def active_model_id(settings: Settings, active_model: str) -> str:
    for opt in list_asr_models(settings, active_model):
        if opt.model == active_model:
            return opt.id
    return "current"


def find_asr_model(
    settings: Settings, active_model: str, option_id: str
) -> ASRModelOption | None:
    for opt in list_asr_models(settings, active_model):
        if opt.id == option_id:
            return opt
    return None
