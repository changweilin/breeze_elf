"""The 幻覺閘門: which ASR output is an invention rather than something spoken.

Whisper-family models answer *something* for every window they are handed, so
non-speech audio — silence, a room tone, an instrumental interlude — comes back
as text. This module is the single place that decides which of it to throw away,
and it is deliberately free of FastAPI, settings and audio buffers so the
評測工具 can measure the very gate production runs (``TRAINING_PLAN.md §1.2``
指標 2) instead of a re-implementation that quietly drifts from it.

Two families, dropped on different terms:

* **Subtitle-credit / sponsor boilerplate** — dropped whatever the energy. This
  is the loud-歌詞 case, which the energy gate structurally cannot catch:
  singing is loud and the model is confident, so neither ``low_energy`` nor
  ``likely_no_speech`` ever fires.
* **Channel-outro phrases** a real speaker might genuinely say — dropped only
  when the audio is also quiet or the model itself doubts there is speech.
"""

from __future__ import annotations

import re

#: Matched as regex against the punctuation-stripped, casefolded form (see
#: :func:`normalize_hallucination_text`) so org-name / wording variants
#: ("字幕提供由 XXX 社群提供的字") still hit.
HALLUCINATION_CREDIT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"字幕.{0,20}社群提供",
        r"社群提供.{0,8}字幕",
        r"請不吝.{0,24}打賞",
        r"明鏡與點點",
    )
)
#: Softer channel-outro phrases a real speaker *might* actually say; only dropped
#: when the audio is also quiet / no-speech so genuine speech is never lost.
COMMON_SILENCE_HALLUCINATION_FRAGMENTS = ("歡迎訂閱按讚分享",)
HALLUCINATION_TEXT_TRANSLATION = str.maketrans({"讚": "贊", "赞": "贊"})

_IGNORED_CHARS = frozenset(" \t\r\n，,。.!?！？、；;：:\"'“”‘’（）()[]【】<>《》·-_/")


def normalize_hallucination_text(text: str) -> str:
    """Casefold, drop punctuation/whitespace, unify 讚/赞 — the form matched against."""
    return "".join(
        char.casefold()
        for char in text.translate(HALLUCINATION_TEXT_TRANSLATION)
        if char not in _IGNORED_CHARS
    )


def is_credit_hallucination(text: str) -> bool:
    """True when the text is subtitle-credit / sponsor boilerplate. Dropped regardless
    of energy (unlike the silence fragments) because loud singing emits these at high
    volume + high confidence, where the energy / no_speech gate can never fire."""
    normalized = normalize_hallucination_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in HALLUCINATION_CREDIT_PATTERNS)


def is_common_silence_hallucination(text: str) -> bool:
    """True for the softer outro phrases — only actionable together with quiet /
    no-speech audio, which :func:`should_drop_text` enforces."""
    normalized = normalize_hallucination_text(text)
    if not normalized:
        return False
    return any(
        normalize_hallucination_text(fragment) in normalized
        for fragment in COMMON_SILENCE_HALLUCINATION_FRAGMENTS
    )


def should_drop_text(
    text: str,
    *,
    rms: float,
    no_speech_prob: float | None,
    no_speech_threshold: float,
    rms_threshold: float,
) -> bool:
    """Whether this recognised text should be treated as silence and dropped.

    Note (silero VAD): the silero detector can onset a segment on quiet far-field
    voice whose whole-window RMS is below ``rms_threshold``, so ``low_energy`` is
    True for windows the RMS segmenter would never have emitted. That stays gated
    behind ``likely_no_speech``/``common_hallucination`` on purpose: a genuinely
    spoken quiet utterance carries a *low* no_speech_prob and non-filler text, so
    it is kept; only when Whisper itself doubts it (or emits a known silence
    phrase) is it dropped — loosening this would re-admit exactly the silence
    hallucinations this backstop exists to remove.
    """
    if is_credit_hallucination(text):
        return True
    likely_no_speech = no_speech_prob is not None and no_speech_prob >= no_speech_threshold
    low_energy = rms <= rms_threshold
    common_hallucination = is_common_silence_hallucination(text)
    return (likely_no_speech and low_energy) or (
        common_hallucination and (likely_no_speech or low_energy)
    )
