"""Text rules shared by manifest building and evaluation.

Common Voice nan-tw sentences carry a 台羅 pronunciation gloss in parentheses,
frequently with two variants:

    公館(Kong-kuán | Kong-koán)
    濁水溪公社(Lô-tsuí-khe Kong-siā | Lô-tsuí-khue Kong-siā)

It is a dictionary annotation, not speech — the speaker only utters the 漢字.
Leaving it in place corrupts both sides of the pipeline:

  * training: 48% of all nan label characters are gloss, so nearly half the
    decoder's target signal teaches it to emit parenthetical romanisation.
  * evaluation: a perfect transcript scores ~20 deletions. Measured on the nan
    test set this inflates CER from 0.481 to 0.661.

Stripping is safe: over the 16,135 nan train clips it empties 0 labels and
leaves only 9 pure-台羅 sentences (which are genuine romanised speech and are
kept as-is).
"""

from __future__ import annotations

import re

# A parenthesised run (half- or full-width) containing at least one ASCII letter.
_GLOSS = re.compile(r"[（(][^（()）]*[a-zA-Z][^（()）]*[)）]")


def strip_reference_gloss(text: str) -> str:
    """Remove 台羅 pronunciation glosses. Returns the original text if stripping
    would leave nothing (pure-romanisation sentences keep their content)."""
    stripped = _GLOSS.sub("", text).strip()
    return stripped if stripped else text.strip()
