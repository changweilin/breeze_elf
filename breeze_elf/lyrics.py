"""Align known lyrics onto a recognised transcript.

For a song the words are already known — what is missing is *when* each one is
sung. Free-running recognition instead solves the much harder problem of
guessing the words, and on singing it guesses badly: it mishears held vowels,
and it invents subtitle credits over instrumental stretches. Given the lyrics,
the recognised characters stop being the answer and become an *anchor track* —
the two character streams are matched against each other, the recognised
timings are transferred onto the real words, and anything the recogniser
invented is dropped.

Nothing here loads a model: it is text/timing surgery over the transcript
blocks the app already produces, so it costs one HTTP round trip and works
offline. The output blocks carry the same per-character ``startSeconds`` /
``endSeconds`` the live pass emits, which is exactly what ``/api/transcript/
analyze`` consumes — so 校準音準 then measures 基頻/簡譜 against the *correct*
words instead of the misheard ones.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Section markers on their own line — ``[Verse]``, ``【副歌】``, ``(Chorus)`` — are
# structure, not sung text, so they never become a transcript line.
_MARKER_EDGES = (("[", "]"), ("【", "】"), ("(", ")"), ("（", "）"))


@dataclass(frozen=True)
class TimedChar:
    """One recognised character with the timing measured from the audio."""

    char: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignedChar:
    char: str
    start: float
    end: float
    #: A recognised character stood at this position, so the timing is measured
    #: — whether or not it was the *same* character. False means the timing was
    #: interpolated across a stretch the recogniser missed: fine for display,
    #: but not ground truth, and not training data.
    anchored: bool
    #: The recogniser produced this very character here. This is the evidence
    #: that the lyrics belong to this audio at all; ``anchored`` is not, since a
    #: mishearing anchors a timing while agreeing with nothing.
    matched: bool


@dataclass(frozen=True)
class AlignedLine:
    text: str
    start: float | None
    end: float | None
    characters: tuple[AlignedChar, ...]
    anchored_ratio: float
    matched_ratio: float


@dataclass(frozen=True)
class LyricsAlignment:
    lines: tuple[AlignedLine, ...]
    #: Share of lyric characters whose timing is measured rather than
    #: interpolated.
    anchored_ratio: float
    #: Share of lyric characters the recogniser independently agreed with. This
    #: is the "are these the right lyrics for this audio" signal — a mismatched
    #: song still aligns (substituting every character is cheaper than dropping
    #: and re-inserting them), so it scores a high ``anchored_ratio`` and a
    #: near-zero ``matched_ratio``.
    matched_ratio: float
    #: Recognised characters with no lyric counterpart: mishearings over the
    #: accompaniment, and the hallucinated credits that singing provokes.
    dropped: int


def parse_lyric_lines(lyrics: str) -> list[str]:
    """Split pasted lyrics into sung lines, dropping blanks and section markers."""
    lines: list[str] = []
    for raw in (lyrics or "").splitlines():
        line = raw.strip()
        if not line or _is_section_marker(line):
            continue
        lines.append(line)
    return lines


def align_lyrics(lyrics: str, recognised: Sequence[TimedChar]) -> LyricsAlignment:
    """Transfer ``recognised`` timings onto the characters of ``lyrics``.

    The two character streams are aligned optimally over normalised keys (case,
    width, and 繁簡 differences do not break a match), and each lyric character
    then takes the timing of whatever was recognised in its place — including
    when that was a *different* character, since a mishearing still marks when
    the syllable was sung. Lyric characters with nothing aligned to them (a line
    the recogniser lost) are spread evenly across the surrounding gap and marked
    unanchored; recognised characters with nothing to align *to* (mishearings
    over the accompaniment, hallucinated credits) are dropped, and their time is
    absorbed by the neighbouring gap.

    Raises ``ValueError`` when the two streams are too long to align.
    """
    lines = parse_lyric_lines(lyrics)
    # (line index, character) for every *sounding* lyric character. Punctuation
    # is carried in the line text but is never sung, so it takes no timing and
    # must not take part in the match either.
    lyric_chars = [
        (index, char) for index, line in enumerate(lines) for char in line if _is_sounding(char)
    ]
    anchors = [char for char in recognised if _is_sounding(char.char)]
    if not lyric_chars:
        return LyricsAlignment((), 0.0, 0.0, len(anchors))
    if not anchors:
        # Nothing measured to align against; return the words with no timings
        # rather than inventing a timeline out of nothing.
        return LyricsAlignment(
            tuple(AlignedLine(line, None, None, (), 0.0, 0.0) for line in lines), 0.0, 0.0, 0
        )

    anchor_keys = [_match_key(char.char) for char in anchors]
    lyric_keys = [_match_key(char) for _, char in lyric_chars]
    mapping, dropped = _align_indices(anchor_keys, lyric_keys)
    spans: list[tuple[float, float, bool] | None] = [
        (anchors[index].start, anchors[index].end, True) if index is not None else None
        for index in mapping
    ]

    _interpolate_gaps(spans, anchors[0].start, anchors[-1].end)
    _enforce_monotonic(spans)

    aligned = tuple(
        AlignedChar(
            char,
            *(spans[index] or (0.0, 0.0, False)),
            matched=(
                mapping[index] is not None and anchor_keys[mapping[index]] == lyric_keys[index]
            ),
        )
        for index, (_, char) in enumerate(lyric_chars)
    )
    return LyricsAlignment(
        _group_by_line(lines, [line for line, _ in lyric_chars], aligned),
        sum(1 for char in aligned if char.anchored) / len(aligned),
        sum(1 for char in aligned if char.matched) / len(aligned),
        dropped,
    )


#: Cells of the alignment matrix to allow. 4M covers a ~2000-character recording
#: against ~2000 characters of lyrics — far past a song — at 16 MB.
_MAX_ALIGNMENT_CELLS = 4_000_000


def _align_indices(source: Sequence[str], target: Sequence[str]) -> tuple[list[int | None], int]:
    """Optimal character alignment: per target position, the source index at it.

    Needleman-Wunsch with unit costs, **not** ``difflib``. difflib maximises the
    longest common subsequence, which on repeated characters lines the wrong one
    up: mishearing 星 as 心 in ``小星星`` makes it match the *second* 星 against
    the first, so every later syllable is anchored one note early and the last
    one falls past the end of the audio. Repeated characters are everywhere in
    lyrics (一閃一閃, 慢慢, 星星), so the alignment has to be the optimal one.

    Ties are broken towards substitution (the diagonal), which keeps a misheard
    syllable anchored to the time it was actually sung instead of dropping it
    and interpolating a neighbour into its place.

    Returns ``(mapping, dropped)`` where ``dropped`` counts source characters
    with no target counterpart.
    """
    rows, columns = len(source), len(target)
    if rows * columns > _MAX_ALIGNMENT_CELLS:
        raise ValueError("歌詞或逐字稿過長,無法對齊")

    symbols: dict[str, int] = {}
    source_ids = np.fromiter(
        (symbols.setdefault(key, len(symbols)) for key in source), dtype=np.int32, count=rows
    )
    target_ids = np.fromiter(
        (symbols.setdefault(key, len(symbols)) for key in target), dtype=np.int32, count=columns
    )

    distance = np.empty((rows + 1, columns + 1), dtype=np.int32)
    indices = np.arange(columns + 1, dtype=np.int32)
    distance[0] = indices
    for row in range(1, rows + 1):
        previous = distance[row - 1]
        best = np.empty(columns + 1, dtype=np.int32)
        best[0] = row
        # Diagonal (match / substitute) against up (drop this source character).
        best[1:] = np.minimum(previous[:-1] + (source_ids[row - 1] != target_ids), previous[1:] + 1)
        # Then the left moves, which cost one each: the cheapest way to reach a
        # column is the best cell at or before it plus the steps to walk over.
        # Folding them in with a running minimum keeps the row vectorised.
        distance[row] = indices + np.minimum.accumulate(best - indices)

    mapping: list[int | None] = [None] * columns
    dropped = 0
    row, column = rows, columns
    while row > 0 and column > 0:
        cost = int(source_ids[row - 1] != target_ids[column - 1])
        if distance[row][column] == distance[row - 1][column - 1] + cost:
            mapping[column - 1] = row - 1
            row -= 1
            column -= 1
        elif distance[row][column] == distance[row - 1][column] + 1:
            dropped += 1
            row -= 1
        else:
            column -= 1
    return mapping, dropped + row


#: Public name for the optimal alignment above — the 簡譜 lyric timeline uses this
#: DP matcher. The 對齊誤差 metric in `tools/eval_asr.py` deliberately uses a lighter
#: greedy matcher over gold word timings; it is a diagnostic, not this product path.
align_indices = _align_indices


def _interpolate_gaps(
    spans: list[tuple[float, float, bool] | None], first_start: float, last_end: float
) -> None:
    """Fill every run of unplaced characters from the anchors around it."""
    index = 0
    while index < len(spans):
        if spans[index] is not None:
            index += 1
            continue
        end_index = index
        while end_index < len(spans) and spans[end_index] is None:
            end_index += 1
        previous = spans[index - 1] if index > 0 else None
        following = spans[end_index] if end_index < len(spans) else None
        start = previous[1] if previous else first_start
        finish = following[0] if following else last_end
        _spread(spans, index, end_index, start, max(start, finish))
        index = end_index


def _spread(
    spans: list[tuple[float, float, bool] | None],
    j1: int,
    j2: int,
    start: float,
    end: float,
) -> None:
    """Lay ``j2 - j1`` characters evenly across ``[start, end]``, unanchored."""
    count = j2 - j1
    if count <= 0:
        return
    step = max(0.0, end - start) / count
    for offset in range(count):
        char_start = start + offset * step
        spans[j1 + offset] = (char_start, char_start + step, False)


def _enforce_monotonic(spans: list[tuple[float, float, bool] | None]) -> None:
    """Keep the timeline non-decreasing.

    Overlapping windows and a mid-word diff block can hand back a character that
    starts before its predecessor ended; the 簡譜/基頻 passes slice the audio by
    these bounds, so a backwards span would read the wrong samples.
    """
    previous_end = 0.0
    for index, span in enumerate(spans):
        if span is None:
            continue
        start, end, anchored = span
        start = max(start, previous_end)
        end = max(end, start)
        spans[index] = (start, end, anchored)
        previous_end = end


def _group_by_line(
    lines: Sequence[str], line_indices: Sequence[int], aligned: Sequence[AlignedChar]
) -> tuple[AlignedLine, ...]:
    grouped: list[list[AlignedChar]] = [[] for _ in lines]
    for line_index, char in zip(line_indices, aligned):
        grouped[line_index].append(char)
    out: list[AlignedLine] = []
    for line, characters in zip(lines, grouped):
        count = len(characters) or 1
        out.append(
            AlignedLine(
                text=line,
                start=characters[0].start if characters else None,
                end=characters[-1].end if characters else None,
                characters=tuple(characters),
                anchored_ratio=sum(1 for char in characters if char.anchored) / count,
                matched_ratio=sum(1 for char in characters if char.matched) / count,
            )
        )
    return tuple(out)


def _is_section_marker(line: str) -> bool:
    return any(line.startswith(open_) and line.endswith(close) for open_, close in _MARKER_EDGES)


def _is_sounding(char: str) -> bool:
    """True for characters that are actually sung.

    Whitespace and punctuation carry no audio: they must not consume a timing
    slot, and — more importantly — must not take part in the match, where a
    stray recognised comma would otherwise pair with a lyric comma several
    syllables away and drag the alignment with it.
    """
    if not char or char.isspace():
        return False
    return not unicodedata.category(char).startswith("P")


@lru_cache(maxsize=4096)
def _match_key(char: str) -> str:
    """The form two characters are compared in.

    Width and case differences (``Ａ``/``a``) and 簡繁 differences are spelling,
    not different sung syllables: the recogniser emits Traditional while pasted
    lyrics are often Simplified, and without folding them together every such
    character would read as a mismatch and lose its anchor.
    """
    folded = unicodedata.normalize("NFKC", char).casefold()
    converter = _traditional_converter()
    if converter is None:
        return folded
    try:
        return converter.convert(folded)
    except Exception:  # noqa: BLE001 - a converter failure must not fail alignment
        return folded


@lru_cache(maxsize=1)
def _traditional_converter():
    """OpenCC simplified→traditional, or ``None`` when it is unavailable.

    Mirrors :mod:`breeze_elf.asr`, which converts recognised text the same way;
    the two must agree or the keys would not meet in the middle.
    """
    try:
        from opencc import OpenCC
    except Exception:  # noqa: BLE001 - optional at runtime, like every extra
        return None
    for config_name in ("s2twp", "s2tw", "s2t"):
        try:
            return OpenCC(config_name)
        except Exception:  # noqa: BLE001
            continue
    return None


def timed_chars_from_blocks(blocks: Iterable[dict]) -> list[TimedChar]:
    """Flatten transcript blocks into the recognised character track.

    Characters without both bounds carry no timing to transfer, so they are
    skipped rather than defaulted to zero.
    """
    out: list[TimedChar] = []
    for block in blocks:
        for char in block.get("characters") or ():
            text = (char.get("char") or "").strip()
            start = char.get("startSeconds")
            end = char.get("endSeconds")
            if not text or start is None or end is None:
                continue
            out.append(TimedChar(text, float(start), max(float(start), float(end))))
    out.sort(key=lambda char: (char.start, char.end))
    return out
