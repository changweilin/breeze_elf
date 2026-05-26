from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class StoredTranscript:
    id: str
    filename: str
    created_at: str
    size_bytes: int


def save_transcript(
    text: str,
    storage_dir: str | Path,
    title: str | None = None,
    now: datetime | None = None,
) -> StoredTranscript:
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

    return StoredTranscript(
        id=target.stem,
        filename=target.name,
        created_at=created_at.isoformat(timespec="seconds"),
        size_bytes=len(payload),
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
