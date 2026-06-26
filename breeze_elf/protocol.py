from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    pass


# How many languages the client may restrict recognition to (mirrors the UI's
# 最多 4 國 picker) and the longest 慣用詞庫 term / number of entries we accept,
# so a malformed or oversized payload can't bloat the per-window Whisper prompt.
MAX_LANGUAGES = 4
MAX_GLOSSARY_ENTRIES = 200
MAX_GLOSSARY_TERM_CHARS = 80


@dataclass(frozen=True)
class GlossaryEntry:
    """One 慣用詞 correction learned from a user edit: ``source`` is what the
    model tends to produce, ``target`` is what the user wants instead."""

    source: str
    target: str


@dataclass(frozen=True)
class StartMessage:
    sample_rate: int
    language: str
    chunk_ms: int
    mode: str = "live"
    # The languages recognition is restricted to (``("auto",)`` means free
    # detection). ``language`` stays as the primary/back-compat single code.
    languages: tuple[str, ...] = ()
    glossary: tuple[GlossaryEntry, ...] = ()


@dataclass(frozen=True)
class StopMessage:
    reason: str = "client"


@dataclass(frozen=True)
class PingMessage:
    pass


ClientMessage = StartMessage | StopMessage | PingMessage


def parse_client_text(raw: str) -> ClientMessage:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid JSON message") from exc

    if not isinstance(payload, dict):
        raise ProtocolError("message must be a JSON object")

    message_type = payload.get("type")
    if message_type == "start":
        sample_rate = _coerce_int(payload.get("sampleRate", 16_000), "sampleRate")
        chunk_ms = _coerce_int(payload.get("chunkMs", 1_000), "chunkMs")
        language = str(payload.get("language", "zh"))
        if sample_rate != 16_000:
            raise ProtocolError("sampleRate must be 16000")
        if chunk_ms <= 0:
            raise ProtocolError("chunkMs must be positive")
        if not language:
            raise ProtocolError("language must not be empty")
        mode = str(payload.get("mode", "live")).strip().lower()
        if mode not in {"live", "file"}:
            mode = "live"
        languages = _parse_languages(payload.get("languages"), language)
        return StartMessage(
            sample_rate=sample_rate,
            language=languages[0],
            chunk_ms=chunk_ms,
            mode=mode,
            languages=languages,
            glossary=_parse_glossary(payload.get("glossary")),
        )

    if message_type == "stop":
        return StopMessage(reason=str(payload.get("reason", "client")))

    if message_type == "ping":
        return PingMessage()

    raise ProtocolError(f"unsupported message type: {message_type!r}")


def server_event(event_type: str, **fields: Any) -> dict[str, Any]:
    return {"type": event_type, **fields}


def _parse_languages(raw: Any, fallback: str) -> tuple[str, ...]:
    """The de-duplicated, lower-cased language codes recognition is limited to.

    Falls back to the single ``language`` field (back-compat) and finally to
    ``"zh"`` so the tuple is never empty; capped at :data:`MAX_LANGUAGES`.
    """
    if isinstance(raw, list):
        items: list[str] = []
        for value in raw:
            code = str(value).strip().lower()
            if code and code not in items:
                items.append(code)
            if len(items) >= MAX_LANGUAGES:
                break
        if items:
            return tuple(items)
    code = (fallback or "").strip().lower()
    return (code,) if code else ("zh",)


def _parse_glossary(raw: Any) -> tuple[GlossaryEntry, ...]:
    """Parse the 慣用詞庫 the client learned from edits into ``source→target``
    pairs, dropping malformed/oversized/no-op entries so an untrusted payload
    cannot bloat the Whisper prompt."""
    if not isinstance(raw, list):
        return ()
    entries: list[GlossaryEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = str(item.get("from", "")).strip()
        target = str(item.get("to", "")).strip()
        if not source or not target or source == target:
            continue
        if len(source) > MAX_GLOSSARY_TERM_CHARS or len(target) > MAX_GLOSSARY_TERM_CHARS:
            continue
        entries.append(GlossaryEntry(source=source, target=target))
        if len(entries) >= MAX_GLOSSARY_ENTRIES:
            break
    return tuple(entries)


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field_name} must be an integer") from exc

