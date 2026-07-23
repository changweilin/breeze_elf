"""Machine-local registry of *deployed* ASR model presets.

The presets in :mod:`asr_models` are code — the models you train and deploy are
data, so ``tools/deploy_model.py`` writes them to a JSON file that the switcher
reads on every request. A new post-trained model therefore needs no code change
and no server restart to appear in 模型與演算法 on the phone.

The file is read defensively: it lives outside git (``models/``) and is edited by
a CLI, so a malformed or half-written file must degrade to "no extra presets"
rather than take the switcher (and with it every recording session) down.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = 1
# Preset ids travel through URLs, JSON and DOM dataset attributes — keep them boring.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
VALID_KINDS = ("breeze", "whisper")
# What faster-whisper needs next to model.bin to load a CTranslate2 Whisper dir.
CT2_REQUIRED_FILES = ("model.bin", "tokenizer.json", "preprocessor_config.json")


@dataclass(frozen=True)
class RegisteredModel:
    id: str
    label: str
    model: str
    kind: str = "breeze"
    note: str = ""

    def to_json(self) -> dict[str, str]:
        entry = {"id": self.id, "label": self.label, "model": self.model, "kind": self.kind}
        if self.note:
            entry["note"] = self.note
        return entry


class RegistryError(ValueError):
    """A caller-supplied entry is invalid — raised by the deploy CLI, never by
    the server's read path (which drops bad entries instead)."""


def resolve_path(value: str, base_dir: Path | None = None) -> Path:
    """A configured path, resolved CWD-independently against the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir or ROOT_DIR) / path
    return path


def registry_path(settings: Settings, base_dir: Path | None = None) -> Path:
    return resolve_path(settings.asr_presets_file, base_dir)


def validate_entry(entry: RegisteredModel) -> RegisteredModel:
    if not ID_PATTERN.match(entry.id):
        raise RegistryError(
            f"invalid preset id {entry.id!r}: use lowercase letters, digits, '.', '-', '_'"
        )
    if not entry.label.strip():
        raise RegistryError(f"preset {entry.id!r} needs a non-empty label")
    if not entry.model.strip():
        raise RegistryError(f"preset {entry.id!r} needs a model path or name")
    if entry.kind not in VALID_KINDS:
        raise RegistryError(
            f"preset {entry.id!r} has unknown kind {entry.kind!r} (expected one of {VALID_KINDS})"
        )
    return entry


def verify_ct2_dir(path: Path) -> list[str]:
    """The CT2 files missing from ``path`` — empty means faster-whisper can load it.

    Registering a dir that is missing e.g. ``tokenizer.json`` would fail only later,
    at switch time, as a confusing HuggingFace lookup; the deploy CLI checks here."""
    if not path.is_dir():
        return list(CT2_REQUIRED_FILES)
    return [name for name in CT2_REQUIRED_FILES if not (path / name).is_file()]


def _parse_entry(raw: object) -> RegisteredModel | None:
    if not isinstance(raw, dict):
        return None
    try:
        entry = RegisteredModel(
            id=str(raw["id"]),
            label=str(raw["label"]),
            model=str(raw["model"]),
            kind=str(raw.get("kind", "breeze")),
            note=str(raw.get("note", "")),
        )
        return validate_entry(entry)
    except (KeyError, TypeError, RegistryError) as exc:
        LOGGER.warning("Ignoring invalid deployed-model entry %r: %s", raw, exc)
        return None


def load_registry(path: Path) -> list[RegisteredModel]:
    """Deployed presets from ``path``; ``[]`` when the file is absent or unreadable."""
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Deployed-model registry %s is unreadable: %s", path, exc)
        return []
    raw_entries = data.get("presets") if isinstance(data, dict) else data
    if not isinstance(raw_entries, list):
        LOGGER.warning("Deployed-model registry %s has no 'presets' list", path)
        return []
    entries: list[RegisteredModel] = []
    seen: set[str] = set()
    for raw in raw_entries:
        entry = _parse_entry(raw)
        if entry is None or entry.id in seen:
            continue
        seen.add(entry.id)
        entries.append(entry)
    return entries


def save_registry(path: Path, entries: list[RegisteredModel]) -> None:
    """Write the registry atomically — the server reads this file on every
    ``GET /api/asr/models``, so it must never observe a half-written file."""
    payload = {
        "version": SCHEMA_VERSION,
        "presets": [entry.to_json() for entry in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def upsert_entry(
    entries: list[RegisteredModel], entry: RegisteredModel
) -> list[RegisteredModel]:
    """Replace the entry with the same id (keeping its position), else append.
    Re-deploying the same id is the normal case — a retrained model reuses it."""
    validate_entry(entry)
    updated = [entry if existing.id == entry.id else existing for existing in entries]
    if all(existing.id != entry.id for existing in entries):
        updated.append(entry)
    return updated


def remove_entry(entries: list[RegisteredModel], preset_id: str) -> list[RegisteredModel]:
    return [entry for entry in entries if entry.id != preset_id]
