"""The four metrics `eval_asr.py` reports beside CER/WER — as pure functions.

`TRAINING_PLAN.md §1.2` asks for five numbers per run and the tool only ever
produced one of them. The other four are here, separated from the decoding loop
so they are unit-testable without a GPU, a model, or a single wav file:

* **幻覺率** — of the C 層 negatives (empty target over instrumental audio), what
  share came back with text, before and after the production gate. CER cannot see
  this at all: a negative clip has no reference characters to be wrong about.
* **對齊誤差** — the median absolute error, in ms, between measured and
  hand-annotated word/character boundaries. 簡譜's whole output rides on these, so
  a fine-tune that quietly shifts them is a regression CER reports as an
  improvement.
* **RTF** — decode seconds per audio second.
* **說話回歸** — no maths of its own: a plain CER over a speech-only source, which
  is what catches catastrophic forgetting after training on singing.

Imports are numpy plus `breeze_elf`; no jiwer, no soundfile, no faster_whisper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from breeze_elf.hallucination import is_credit_hallucination, should_drop_text
from breeze_elf.lyrics import align_indices


@dataclass
class HallucinationCounter:
    """Non-empty output over audio whose reference text is empty.

    Segments are counted, not clips, and each one is passed through the same
    :func:`breeze_elf.hallucination.should_drop_text` production uses — the point
    of the metric is the leak rate *users* see, which is what survives the gate.
    A clip counts as leaking when any of its segments survives.
    """

    clips: int = 0
    #: Clips where the model emitted any text at all.
    raw_clips: int = 0
    #: Clips where text survived the hallucination gate.
    gated_clips: int = 0
    segments: int = 0
    raw_segments: int = 0
    gated_segments: int = 0
    #: Segments matching the subtitle-credit patterns — the signature failure on
    #: music, tracked separately because it is the one the gate is built for.
    credit_segments: int = 0

    def add_clip(
        self,
        segments: Sequence[tuple[str, float | None]],
        *,
        rms: float,
        no_speech_threshold: float,
        rms_threshold: float,
    ) -> None:
        """Record one negative clip's ``(text, no_speech_prob)`` segments."""
        self.clips += 1
        raw = gated = False
        for text, no_speech_prob in segments:
            self.segments += 1
            if not text.strip():
                continue
            raw = True
            self.raw_segments += 1
            if is_credit_hallucination(text):
                self.credit_segments += 1
            if not should_drop_text(
                text,
                rms=rms,
                no_speech_prob=no_speech_prob,
                no_speech_threshold=no_speech_threshold,
                rms_threshold=rms_threshold,
            ):
                gated = True
                self.gated_segments += 1
        self.raw_clips += int(raw)
        self.gated_clips += int(gated)

    def summary(self) -> dict[str, float | int]:
        clips = self.clips or 1
        segments = self.segments or 1
        return {
            "n": self.clips,
            "hallucination_rate": round(self.raw_clips / clips, 5),
            "hallucination_rate_gated": round(self.gated_clips / clips, 5),
            "segment_rate": round(self.raw_segments / segments, 5),
            "segment_rate_gated": round(self.gated_segments / segments, 5),
            "credit_rate": round(self.credit_segments / segments, 5),
        }


@dataclass
class RtfCounter:
    """Decode seconds per audio second — the ≤ baseline × 1.2 上線門檻.

    Totals are summed rather than per-clip RTFs averaged: a mean of ratios lets a
    0.4 s clip weigh as much as a 20 s one, which is not what "can this keep up
    with the audio" means.
    """

    decode_seconds: float = 0.0
    audio_seconds: float = 0.0

    def add(self, decode_seconds: float, audio_seconds: float) -> None:
        self.decode_seconds += max(0.0, decode_seconds)
        self.audio_seconds += max(0.0, audio_seconds)

    @property
    def rtf(self) -> float | None:
        if self.audio_seconds <= 0:
            return None
        return self.decode_seconds / self.audio_seconds


@dataclass
class AlignmentCounter:
    """Absolute error between measured and annotated token boundaries, in ms."""

    start_errors: list[float] = field(default_factory=list)
    end_errors: list[float] = field(default_factory=list)
    #: Reference tokens the hypothesis had no counterpart for. A model can look
    #: perfectly aligned by emitting two words and getting both right, so the
    #: coverage has to be reported next to the error.
    reference_tokens: int = 0
    matched_tokens: int = 0

    def add(
        self,
        reference: Sequence[tuple[str, float, float]],
        hypothesis: Sequence[tuple[str, float, float]],
    ) -> None:
        """Match one clip's ``(token, start, end)`` sequences and record the errors."""
        self.reference_tokens += len(reference)
        if not reference or not hypothesis:
            return
        # Same optimal alignment 歌詞對齊 uses. difflib would misplace repeated
        # tokens (歌詞 are full of them) and quietly inflate the error.
        mapping, _dropped = align_indices(
            [_token_key(token) for token, _s, _e in reference],
            [_token_key(token) for token, _s, _e in hypothesis],
        )
        for hyp_index, ref_index in enumerate(mapping):
            if ref_index is None:
                continue
            if _token_key(reference[ref_index][0]) != _token_key(hypothesis[hyp_index][0]):
                # Aligned but misrecognised: the timing belongs to a different
                # token, so scoring it would measure the substitution, not drift.
                continue
            self.matched_tokens += 1
            self.start_errors.append(
                abs(hypothesis[hyp_index][1] - reference[ref_index][1]) * 1000.0
            )
            self.end_errors.append(abs(hypothesis[hyp_index][2] - reference[ref_index][2]) * 1000.0)

    def summary(self) -> dict[str, float | int | None]:
        return {
            "n_reference_tokens": self.reference_tokens,
            "n_matched_tokens": self.matched_tokens,
            "matched_ratio": round(
                self.matched_tokens / self.reference_tokens if self.reference_tokens else 0.0, 5
            ),
            "start_median_abs_ms": _median(self.start_errors),
            "end_median_abs_ms": _median(self.end_errors),
            "start_p90_abs_ms": _percentile(self.start_errors, 0.9),
            "end_p90_abs_ms": _percentile(self.end_errors, 0.9),
        }


def reference_words(record: dict) -> list[tuple[str, float, float]]:
    """Hand-annotated timings off a manifest row, if it carries any.

    Expected shape, added by hand to `dataset/manifests/*.jsonl` for the clips
    that have been annotated::

        {"id": …, "text": …, "words": [{"word": "月", "start": 1.02, "end": 1.35}]}

    Rows without it simply do not contribute to 對齊誤差; the metric reports how
    many tokens it actually saw so a near-empty annotation set is visible rather
    than reading as a suspiciously good number.
    """
    words = record.get("words")
    if not isinstance(words, list):
        return []
    out: list[tuple[str, float, float]] = []
    for item in words:
        if not isinstance(item, dict):
            continue
        token = str(item.get("word") or item.get("char") or "").strip()
        start, end = item.get("start"), item.get("end")
        if not token or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        out.append((token, float(start), float(end)))
    return out


def _token_key(token: str) -> str:
    return token.strip().casefold()


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 1)
    return round((ordered[middle - 1] + ordered[middle]) / 2.0, 1)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 1)
