from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class StartMessage:
    sample_rate: int
    language: str
    chunk_ms: int
    mode: str = "live"


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
        return StartMessage(
            sample_rate=sample_rate,
            language=language,
            chunk_ms=chunk_ms,
            mode=mode,
        )

    if message_type == "stop":
        return StopMessage(reason=str(payload.get("reason", "client")))

    if message_type == "ping":
        return PingMessage()

    raise ProtocolError(f"unsupported message type: {message_type!r}")


def server_event(event_type: str, **fields: Any) -> dict[str, Any]:
    return {"type": event_type, **fields}


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field_name} must be an integer") from exc

