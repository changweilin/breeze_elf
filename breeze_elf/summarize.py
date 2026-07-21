"""Post-meeting summary of a transcript.

Two fully-local tiers, chosen by ``BREEZE_SUMMARY_PROVIDER``:

- ``extractive`` (default): pick the most informative sentences with a pure-stdlib
  character-frequency score. No model, no VRAM, no network — ships everywhere.
- ``ollama``: abstractive summary via a **local** Ollama daemon (127.0.0.1) over
  stdlib ``urllib``; transcript text never leaves the machine. Any failure
  (daemon down, model missing, timeout) degrades to the extractive summary.

``off`` disables the feature. No cloud path exists here: sending transcripts to a
third-party API would break the app's privacy-first premise, so it is intentionally
absent rather than a hidden default.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Protocol

logger = logging.getLogger(__name__)

# Sentence boundaries: CJK 。！？…, ASCII .!?, and hard newlines. Terminators are
# kept with the sentence so the extract reads naturally.
_SENTENCE_RE = re.compile(r"[^。！？!?…\n]+[。！？!?…]*")
# Characters that carry no topical signal, excluded from the frequency score.
_SKIP_CHARS = set(" \t\r\n、，,。.！!？?…；;：:「」『』（）()【】《》\"'`~-—…")


def _split_sentences(text: str) -> list[str]:
    return [m.group().strip() for m in _SENTENCE_RE.finditer(text or "") if m.group().strip()]


class Summarizer(Protocol):
    name: str
    available: bool

    def summarize(self, text: str, *, max_sentences: int = 5) -> str:
        ...


class NullSummarizer:
    name = "off"
    available = False

    def summarize(self, text: str, *, max_sentences: int = 5) -> str:
        return ""


class ExtractiveSummarizer:
    """Frequency-based extractive summary: score each sentence by the mean topical
    frequency of its characters (so it favours sentences dense with the transcript's
    recurring content, not merely long ones), keep the top ``max_sentences``, and
    re-order them by original position so the summary still reads in sequence."""

    name = "extractive"
    available = True

    def summarize(self, text: str, *, max_sentences: int = 5) -> str:
        sentences = _split_sentences(text)
        if len(sentences) <= max_sentences:
            return "".join(sentences)

        freq: Counter[str] = Counter()
        for sentence in sentences:
            for char in sentence:
                if char not in _SKIP_CHARS:
                    freq[char] += 1

        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            informative = [char for char in sentence if char not in _SKIP_CHARS]
            if not informative:
                continue
            score = sum(freq[char] for char in informative) / len(informative)
            scored.append((score, index, sentence))

        if not scored:
            return "".join(sentences[:max_sentences])

        top = sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]
        top.sort(key=lambda item: item[1])  # restore reading order
        return "".join(sentence for _, _, sentence in top)


class OllamaSummarizer:
    """Abstractive summary via a local Ollama daemon. Never raises: on any transport
    or protocol failure it falls back to the extractive summary so the endpoint
    always returns something useful and the data stays on the box."""

    name = "ollama"
    available = True

    def __init__(
        self,
        url: str,
        model: str,
        *,
        timeout: float = 60.0,
        fallback: Summarizer | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._fallback = fallback or ExtractiveSummarizer()

    def summarize(self, text: str, *, max_sentences: int = 5) -> str:
        prompt = (
            "請用繁體中文,以條列式摘要以下逐字稿的重點,"
            f"最多 {max_sentences} 點,只輸出摘要本身:\n\n{text}"
        )
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            # A well-behaved daemon returns a JSON object; guard against a non-object
            # body (proxy error page, wrong service on the port) so ``body.get`` can't
            # raise and break the "never raises → extractive" contract.
            response_text = body.get("response") if isinstance(body, dict) else None
            summary = (response_text or "").strip()
            if summary:
                return summary
            logger.warning("ollama returned an empty summary; using extractive")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            logger.warning("ollama summary failed (%s); using extractive", exc)
        return self._fallback.summarize(text, max_sentences=max_sentences)


def build_summarizer(
    provider: str,
    *,
    model: str = "qwen3:4b-instruct",
    ollama_url: str = "http://127.0.0.1:11434",
    timeout: float = 60.0,
) -> Summarizer:
    kind = (provider or "").strip().lower()
    if kind == "off":
        return NullSummarizer()
    if kind == "ollama":
        return OllamaSummarizer(ollama_url, model, timeout=timeout)
    return ExtractiveSummarizer()
