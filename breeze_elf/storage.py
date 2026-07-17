from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StoredTranscript:
    id: str
    filename: str
    created_at: str
    size_bytes: int
    json_filename: str | None = None
    audio_filename: str | None = None
    csv_filename: str | None = None


# Sibling audio extensions save_transcript may have written (audio_ext defaults to
# "wav"; kept broad so an id is flagged has_audio regardless of the chosen format).
_AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac")


@dataclass(frozen=True)
class TranscriptRecord:
    id: str
    filename: str
    title: str
    created_at: str
    text: str
    has_audio: bool
    has_jianpu: bool
    mtime: float
    blocks: list[dict[str, Any]]
    audio_filename: str | None = None


def safe_transcript_id(doc_id: str) -> str:
    """Validate a client-supplied transcript id before it is joined to a path.

    Transcript ids are server-generated stems; refuse anything that could escape
    the storage directory (path separators / ``..``) so ``GET /{id}`` can't be
    turned into an arbitrary-file read."""
    cleaned = (doc_id or "").strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("invalid transcript id")
    return cleaned


def iter_transcript_ids(storage_dir: str | Path) -> list[str]:
    """The stems of every saved ``.txt`` transcript, excluding sidecar files."""
    directory = Path(storage_dir).expanduser()
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.txt") if path.is_file())


def load_transcript(storage_dir: str | Path, doc_id: str) -> TranscriptRecord | None:
    """Read a saved transcript back by id: the ``.txt`` body plus, when present,
    the sibling ``.json`` (title, per-character blocks) and audio. Returns ``None``
    when the id has no ``.txt`` on disk. ``doc_id`` is caller-validated — this does
    not escape ``storage_dir`` because it only ever appends ``id + suffix``."""
    directory = Path(storage_dir).expanduser()
    txt_path = directory / f"{doc_id}.txt"
    if not txt_path.is_file():
        return None

    text = txt_path.read_text(encoding="utf-8").strip()
    mtime = txt_path.stat().st_mtime

    title = ""
    created_at = ""
    blocks: list[dict[str, Any]] = []
    json_path = directory / f"{doc_id}.json"
    if json_path.is_file():
        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            document = {}
        if isinstance(document, dict):
            title = (document.get("title") or "").strip()
            created_at = (document.get("createdAt") or "").strip()
            raw_blocks = document.get("blocks")
            if isinstance(raw_blocks, list):
                blocks = [block for block in raw_blocks if isinstance(block, dict)]

    if not title:
        title = _first_line(text) or doc_id
    if not created_at:
        created_at = _created_at(datetime.fromtimestamp(mtime, timezone.utc)).isoformat(
            timespec="seconds"
        )

    audio_filename = next(
        (f"{doc_id}{ext}" for ext in _AUDIO_EXTS if (directory / f"{doc_id}{ext}").is_file()),
        None,
    )
    has_jianpu = any(
        isinstance(char, dict) and str(char.get("jianpu") or "").strip()
        for block in blocks
        for char in (block.get("characters") or [])
    )

    return TranscriptRecord(
        id=doc_id,
        filename=txt_path.name,
        title=title,
        created_at=created_at,
        text=text,
        has_audio=audio_filename is not None,
        has_jianpu=has_jianpu,
        mtime=mtime,
        blocks=blocks,
        audio_filename=audio_filename,
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return ""


def save_transcript(
    text: str,
    storage_dir: str | Path,
    title: str | None = None,
    now: datetime | None = None,
    *,
    structured: dict[str, Any] | None = None,
    audio: bytes | None = None,
    audio_ext: str = "wav",
    pitch_csv: str | None = None,
) -> StoredTranscript:
    """Persist a transcript and, when provided, its 簡譜/timing metadata and audio.

    The plain ``.txt`` transcript is always written. When ``structured`` is
    given a sibling ``.json`` holds the per-character timing, duration, and
    jianpu; when ``audio`` is given a sibling ``.<audio_ext>`` holds the
    recording. All files share one stem so they stay grouped together.
    """
    transcript = text.strip()
    if not transcript:
        raise ValueError("transcript text must not be empty")

    created_at = _created_at(now)
    directory = Path(storage_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    stamp = created_at.strftime("%Y%m%d-%H%M%S")
    slug = _slugify(title or transcript[:80])
    target = _unique_path(directory, f"breeze-elf-{stamp}-{slug}")
    payload = f"{transcript}\n".encode()
    target.write_bytes(payload)
    stem = target.stem

    json_filename: str | None = None
    if structured is not None:
        document = {**structured, "createdAt": created_at.isoformat(timespec="seconds")}
        json_path = directory / f"{stem}.json"
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        json_filename = json_path.name

    audio_filename: str | None = None
    if audio:
        audio_path = directory / f"{stem}.{audio_ext.lstrip('.')}"
        audio_path.write_bytes(audio)
        audio_filename = audio_path.name

    csv_filename: str | None = None
    if pitch_csv:
        csv_path = directory / f"{stem}.csv"
        csv_path.write_text(pitch_csv, encoding="utf-8")
        csv_filename = csv_path.name

    return StoredTranscript(
        id=stem,
        filename=target.name,
        created_at=created_at.isoformat(timespec="seconds"),
        size_bytes=len(payload),
        json_filename=json_filename,
        audio_filename=audio_filename,
        csv_filename=csv_filename,
    )


def _created_at(now: datetime | None) -> datetime:
    created_at = now or datetime.now(timezone.utc).astimezone()
    if created_at.tzinfo is None:
        return created_at.replace(tzinfo=timezone.utc)
    return created_at


def _unique_path(directory: Path, stem: str) -> Path:
    candidate = directory / f"{stem}.txt"
    if not candidate.exists():
        return candidate

    for index in range(2, 10_000):
        candidate = directory / f"{stem}-{index}.txt"
        if not candidate.exists():
            return candidate

    raise OSError("could not allocate a unique transcript filename")


def _slugify(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug[:48].strip("-") or "transcript"
