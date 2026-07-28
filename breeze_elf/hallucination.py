"""Whisper hallucination detection — the subtitle-credit / silence gate.

Shared by two callers so the offline hallucination metric measures exactly what
the runtime drops:

  * ``breeze_elf.main`` — the live drop gate (:func:`should_drop_text`).
  * ``tools/eval_asr.py`` — the C-layer negative-sample metric (non-empty rate,
    credit-hit rate, post-gate leak rate) per TRAINING_PLAN.md §1.2.

Pure ``re``/stdlib on purpose: the eval tool imports it without dragging in
FastAPI / faster-whisper, and it never grows an app dependency.
"""

from __future__ import annotations

import re

# Pure subtitle-credit / sponsor boilerplate Whisper emits on *non-speech* (music,
# singing, silence). No real speaker utters these, so they are dropped regardless of
# loudness — this is what catches loud 歌詞, which the energy gate (built for quiet
# silence) structurally cannot: singing is loud and the model is confident, so
# ``low_energy``/``likely_no_speech`` never fire. Matched as regex against the
# punctuation-stripped, casefolded form (see :func:`normalize_hallucination_text`) so
# org-name / wording variants ("字幕提供由 XXX 社群提供的字") still hit.
HALLUCINATION_CREDIT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"字幕.{0,20}社群提供",
        r"社群提供.{0,8}字幕",
        r"請不吝.{0,24}打賞",
        r"明鏡與點點",
    )
)
# Softer channel-outro phrases a real speaker *might* actually say; only dropped when
# the audio is also quiet / no-speech so genuine speech is never lost.
COMMON_SILENCE_HALLUCINATION_FRAGMENTS = ("歡迎訂閱按讚分享",)
HALLUCINATION_TEXT_TRANSLATION = str.maketrans({"讚": "贊", "赞": "贊"})


def normalize_hallucination_text(text: str) -> str:
    ignored = set(" \t\r\n，,。.!?！？、；;：:\"'“”‘’（）()[]【】<>《》·-_/")
    return "".join(
        char.casefold()
        for char in text.translate(HALLUCINATION_TEXT_TRANSLATION)
        if char not in ignored
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
    normalized = normalize_hallucination_text(text)
    if not normalized:
        return False
    return any(
        normalize_hallucination_text(fragment) in normalized
        for fragment in COMMON_SILENCE_HALLUCINATION_FRAGMENTS
    )


def should_drop_text(
    text: str,
    no_speech_prob: float | None,
    rms: float,
    *,
    no_speech_threshold: float,
    rms_threshold: float,
) -> bool:
    """Whether an ASR result should be dropped as a hallucination.

    Extracted from the runtime gate so the offline eval measures the identical
    decision. Inputs are the decoded ``text``, the segment ``no_speech_prob``,
    the window ``rms``, and the two production thresholds.
    """
    # Subtitle-credit / sponsor boilerplate is never real speech, so drop it whatever
    # the energy. This is the loud-歌詞 case: singing produces these at high volume and
    # high confidence, where the quiet-silence gate below can never fire.
    if is_credit_hallucination(text):
        return True
    # Note (silero VAD): the silero detector can onset a segment on quiet far-field
    # voice whose whole-window RMS is below rms_threshold, so ``low_energy`` is True
    # for windows the RMS segmenter would never have emitted. This stays gated behind
    # ``likely_no_speech``/``common_hallucination`` on purpose: a genuinely-spoken quiet
    # utterance carries a *low* no_speech_prob and non-filler text, so it is kept; only
    # when Whisper itself doubts it (or emits a known silence phrase) is it dropped —
    # loosening this would re-admit exactly the silence hallucinations this backstop
    # exists to remove.
    likely_no_speech = no_speech_prob is not None and no_speech_prob >= no_speech_threshold
    low_energy = rms <= rms_threshold
    common_hallucination = is_common_silence_hallucination(text)

    return (likely_no_speech and low_energy) or (
        common_hallucination and (likely_no_speech or low_energy)
    )
