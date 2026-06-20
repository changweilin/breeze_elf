"""Server-side storage for saved voice profiles (A's 聲音特徵 / 我的最愛).

Each saved voice is a trio of sibling files under the storage directory sharing
one stem:

* ``{id}.json``      — metadata (name, favorite flag, timestamps, audio info)
* ``{id}.embedding`` — the opaque embedding blob produced by the voice engine
* ``{id}.wav``       — the reference recording, kept so the UI can play A back

The embedding is engine-defined bytes (JSON for the mock, a torch tensor for
OpenVoice), so this layer never interprets it.
"""

from __future__ import annotations

import json
import secrets
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np

# Persisted on disk in snake_case; surfaced to the API in camelCase to match the
# rest of Breeze Elf's JSON contract.
_PUBLIC_KEYS = {
    "id": "id",
    "name": "name",
    "favorite": "favorite",
    "created_at": "createdAt",
    "updated_at": "updatedAt",
    "sample_rate": "sampleRate",
    "duration_seconds": "durationSeconds",
    "has_sample": "hasSample",
}

_EMBEDDING_SUFFIX = ".embedding"
_SAMPLE_SUFFIX = ".wav"
_META_SUFFIX = ".json"


@dataclass(frozen=True)
class StoredVoice:
    id: str
    name: str
    favorite: bool
    created_at: str
    updated_at: str
    sample_rate: int
    duration_seconds: float
    has_sample: bool

    def to_public(self) -> dict:
        """API-facing dict with camelCase keys."""
        storage = asdict(self)
        return {public: storage[snake] for snake, public in _PUBLIC_KEYS.items()}


def save_voice(
    name: str,
    embedding: bytes,
    storage_dir: str | Path,
    *,
    sample_audio: bytes | None = None,
    sample_rate: int = 16_000,
    duration_seconds: float = 0.0,
    favorite: bool = False,
    now: datetime | None = None,
) -> StoredVoice:
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("voice name must not be empty")
    if not embedding:
        raise ValueError("voice embedding must not be empty")

    directory = _resolve_dir(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    created_at = _now_iso(now)
    voice_id = _unique_id(directory, created_at)

    (directory / f"{voice_id}{_EMBEDDING_SUFFIX}").write_bytes(embedding)

    has_sample = False
    if sample_audio:
        (directory / f"{voice_id}{_SAMPLE_SUFFIX}").write_bytes(sample_audio)
        has_sample = True

    stored = StoredVoice(
        id=voice_id,
        name=clean_name,
        favorite=bool(favorite),
        created_at=created_at,
        updated_at=created_at,
        sample_rate=int(sample_rate),
        duration_seconds=round(float(duration_seconds), 3),
        has_sample=has_sample,
    )
    _write_meta(directory, stored)
    return stored


def list_voices(storage_dir: str | Path) -> list[StoredVoice]:
    directory = _resolve_dir(storage_dir)
    if not directory.exists():
        return []
    voices: list[StoredVoice] = []
    for meta_path in directory.glob(f"*{_META_SUFFIX}"):
        stored = _read_meta(meta_path)
        if stored is not None:
            voices.append(stored)
    # Favorites first, then newest. Both keys descend so favorites bubble up.
    voices.sort(key=lambda voice: (voice.favorite, voice.created_at), reverse=True)
    return voices


def get_voice(voice_id: str, storage_dir: str | Path) -> StoredVoice | None:
    directory = _resolve_dir(storage_dir)
    return _read_meta(directory / f"{_safe_id(voice_id)}{_META_SUFFIX}")


def load_embedding(voice_id: str, storage_dir: str | Path) -> bytes:
    directory = _resolve_dir(storage_dir)
    path = directory / f"{_safe_id(voice_id)}{_EMBEDDING_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(f"voice embedding not found: {voice_id}")
    return path.read_bytes()


def voice_sample_path(voice_id: str, storage_dir: str | Path) -> Path | None:
    directory = _resolve_dir(storage_dir)
    path = directory / f"{_safe_id(voice_id)}{_SAMPLE_SUFFIX}"
    return path if path.exists() else None


def update_voice(
    voice_id: str,
    storage_dir: str | Path,
    *,
    name: str | None = None,
    favorite: bool | None = None,
    now: datetime | None = None,
) -> StoredVoice:
    directory = _resolve_dir(storage_dir)
    stored = _read_meta(directory / f"{_safe_id(voice_id)}{_META_SUFFIX}")
    if stored is None:
        raise FileNotFoundError(f"voice not found: {voice_id}")

    new_name = stored.name
    if name is not None:
        new_name = name.strip()
        if not new_name:
            raise ValueError("voice name must not be empty")

    updated = StoredVoice(
        id=stored.id,
        name=new_name,
        favorite=stored.favorite if favorite is None else bool(favorite),
        created_at=stored.created_at,
        updated_at=_now_iso(now),
        sample_rate=stored.sample_rate,
        duration_seconds=stored.duration_seconds,
        has_sample=stored.has_sample,
    )
    _write_meta(directory, updated)
    return updated


def delete_voice(voice_id: str, storage_dir: str | Path) -> bool:
    directory = _resolve_dir(storage_dir)
    safe = _safe_id(voice_id)
    removed = False
    for suffix in (_META_SUFFIX, _EMBEDDING_SUFFIX, _SAMPLE_SUFFIX):
        path = directory / f"{safe}{suffix}"
        if path.exists():
            path.unlink()
            removed = True
    return removed


# --------------------------------------------------------------------------- #
# WAV helpers (stdlib only, mono 16-bit PCM)
# --------------------------------------------------------------------------- #


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes into mono ``float32`` samples in [-1, 1]."""
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except wave.Error as exc:
        raise ValueError(f"malformed WAV data: {exc}") from exc

    if width != 2:
        raise ValueError("only 16-bit PCM WAV is supported")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sample_rate)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono ``float32`` samples in [-1, 1] into 16-bit PCM WAV bytes."""
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=tuple(range(1, mono.ndim)))
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _resolve_dir(storage_dir: str | Path) -> Path:
    return Path(storage_dir).expanduser()


def _now_iso(now: datetime | None) -> str:
    moment = now or datetime.now(timezone.utc).astimezone()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat(timespec="seconds")


def _unique_id(directory: Path, created_at: str) -> str:
    stamp = created_at.replace(":", "").replace("-", "")[:15].replace("T", "-")
    for _ in range(10_000):
        candidate = f"voice-{stamp}-{secrets.token_hex(3)}"
        if not (directory / f"{candidate}{_META_SUFFIX}").exists():
            return candidate
    raise OSError("could not allocate a unique voice id")


def _safe_id(voice_id: str) -> str:
    # Voice ids are server-generated slugs; refuse anything that could escape
    # the storage directory if a client sends a crafted id.
    cleaned = (voice_id or "").strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("invalid voice id")
    return cleaned


def _write_meta(directory: Path, voice: StoredVoice) -> None:
    path = directory / f"{voice.id}{_META_SUFFIX}"
    path.write_text(json.dumps(asdict(voice), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_meta(path: Path) -> StoredVoice | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "id" not in data:
        return None
    return StoredVoice(
        id=str(data["id"]),
        name=str(data.get("name", "")),
        favorite=bool(data.get("favorite", False)),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", data.get("created_at", ""))),
        sample_rate=int(data.get("sample_rate", 16_000)),
        duration_seconds=float(data.get("duration_seconds", 0.0)),
        has_sample=bool(data.get("has_sample", False)),
    )
