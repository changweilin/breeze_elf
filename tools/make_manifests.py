"""Build leakage-safe train/dev/test manifests from dataset/metadata.csv.

Split rules (TRAINING_PLAN.md §切分規則):
  nan (cv_)      : reuse Common Voice official split by clip stem; leftover
                   validated clips join train ONLY if their speaker (client_id)
                   is absent from dev/test (else dropped — speaker-leak guard).
  MIR-1K (mir1k_): split by singer, 17 train / 1 dev / 1 test.
  Jamendo (jam_) : split by song, 16 train / 2 dev / 2 test.

Text target is metadata's `transcription` column verbatim (教育部漢字 for nan,
zh-TW lyrics for MIR-1K, lowercased en for Jamendo). Output: dataset/manifests/
{train,dev,test}.jsonl with {id, audio, text, lang, source, split, layer}.

`layer` is `lyric` or `negative`. Negatives are the empty-transcription chunks the
dataset builder cuts from instrumental gaps (C 層, TRAINING_PLAN.md §3.3) — they
are the training signal for "say nothing over an interlude" and the evaluation set
for 幻覺率, so they are kept, not dropped. An empty `cv_` row is a different thing
entirely (a hole in Common Voice's metadata, over read speech) and is still dropped.

Run: .venv/Scripts/python.exe tools/make_manifests.py   (NOT `uv run` — see
memory training-toolchain; irrelevant here but keep the habit consistent.)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import strip_reference_gloss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
NAN = DATASET / "nan-tw"
OUT = DATASET / "manifests"

# MIR-1K held-out singers (alphabetically last two of the 19 — deterministic &
# reproducible; both have ~5 song prefixes so eval size is adequate).
MIR_DEV = "titon"
MIR_TEST = "yifen"

# Non-test nan speakers sorted by clip count (desc); these ranks become the
# early-stopping dev set. Mid-sized so the big voices stay in training.
NAN_DEV_SPEAKER_RANKS = slice(9, 12)


def read_tsv_stems(path: Path) -> set[str]:
    stems: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stems.add(Path(row["path"]).stem)
    return stems


def read_stem_to_client(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m[Path(row["path"]).stem] = row["client_id"]
    return m


def cv_stem(chunk_stem: str) -> str:
    # cv_common_voice_nan-tw_30885968_0000 -> common_voice_nan-tw_30885968
    return re.sub(r"_\d+$", "", chunk_stem[len("cv_"):])


def main() -> int:
    # Common Voice's official split is pathological for our purpose: of the 274
    # validated speakers it puts 256 in test and 13 in dev, leaving TRAIN with
    # only 5 voices. Training Taiwanese on 5 speakers and testing on 256 unseen
    # ones is the dominant generalisation gap. We keep test EXACTLY as the
    # official test.tsv (so every previously measured number stays comparable)
    # and pool every non-test speaker into training, holding out 3 mid-sized
    # speakers for early stopping. Still strictly speaker-disjoint.
    test_stems = read_tsv_stems(NAN / "test.tsv")
    client = read_stem_to_client(NAN / "validated.tsv")
    test_spk = {client[s] for s in test_stems if s in client}
    clips_per_spk = Counter(client.values())
    nontest_spk = sorted(
        (s for s in clips_per_spk if s not in test_spk),
        key=lambda s: (-clips_per_spk[s], s),
    )
    dev_spk = set(nontest_spk[NAN_DEV_SPEAKER_RANKS])

    # Jamendo songs -> deterministic split by sorted song id.
    jam_songs = sorted(
        {
            row["source_dataset_or_song_id"]
            for row in _rows()
            if row["file_name"].split("/")[-1].startswith("jamendo_")
        }
    )
    jam_test = set(jam_songs[-2:])
    jam_dev = set(jam_songs[-4:-2])

    splits: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    dropped: Counter = Counter()
    negatives: Counter = Counter()
    mir_singer_split: dict[str, str] = {}

    for row in _rows():
        fn = row["file_name"]  # chunks/<id>.wav
        stem = Path(fn).stem
        text = (row["transcription"] or "").strip()
        negative = not text
        rec = {
            "id": stem,
            "audio": f"dataset/{fn}",
            "text": text,
            "lang": row["language"],
            "source": row["source_dataset_or_song_id"],
            "layer": "negative" if negative else "lyric",
        }
        # An empty nan label is missing metadata over read speech, not an
        # instrumental gap — there is nothing musical for the model to learn to
        # stay silent through, so it stays dropped.
        if negative and stem.startswith("cv_"):
            dropped["empty_text"] += 1
            continue

        if stem.startswith("cv_"):
            cs = cv_stem(stem)
            # Drop the 台羅 pronunciation gloss — it is annotation, not speech.
            cleaned = strip_reference_gloss(text)
            if cleaned != text:
                counts[("nan", "gloss_stripped")] += 1
            rec["text"] = cleaned
            rec["speaker"] = client.get(cs, "")
            spk = client.get(cs, "")
            if cs in test_stems:
                split = "test"
            elif spk in test_spk:
                # a test speaker's other clips would leak the test voices
                dropped["nan_speaker_leak"] += 1
                continue
            elif spk in dev_spk:
                split = "dev"
            else:
                split = "train"
            counts[("nan", split)] += 1
        elif stem.startswith("mir1k_"):
            m = re.match(r"mir1k_([A-Za-z]+)_\d", stem)
            if not m:
                dropped["mir_no_singer"] += 1
                continue
            singer = m.group(1)
            split = "test" if singer == MIR_TEST else "dev" if singer == MIR_DEV else "train"
            mir_singer_split[singer] = split
            counts[("mir1k", split)] += 1
        elif stem.startswith("jamendo_"):
            song = row["source_dataset_or_song_id"]
            split = "test" if song in jam_test else "dev" if song in jam_dev else "train"
            counts[("jamendo", split)] += 1
        else:
            dropped["unknown_prefix"] += 1
            continue

        rec["split"] = split
        if negative:
            negatives[split] += 1
        splits[split].append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        path = OUT / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in splits[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- report ---
    print("=== manifest counts (source x split) ===")
    for src in ("nan", "mir1k", "jamendo"):
        line = "  ".join(f"{sp}={counts[(src, sp)]:>6}" for sp in ("train", "dev", "test"))
        print(f"  {src:8} {line}")
    print("=== split totals ===")
    for split in ("train", "dev", "test"):
        print(f"  {split:6} {len(splits[split]):>6}")
    print("=== C 層負樣本 (empty target; TRAINING_PLAN §3.3 wants 5–10% of train) ===")
    for split in ("train", "dev", "test"):
        total = len(splits[split])
        share = negatives[split] / total if total else 0.0
        flag = ""
        if split == "train":
            flag = "  <-- 目標 5–10%" if not 0.05 <= share <= 0.10 else ""
        print(f"  {split:6} {negatives[split]:>6}  ({share:.1%}){flag}")
    if not sum(negatives.values()):
        print("  none — rebuild mixed songs with --negative_ratio to create them")
    spk_in = lambda sp: len({r["speaker"] for r in splits[sp] if r.get("speaker")})  # noqa: E731
    print("=== nan speakers (disjoint) ===")
    print(f"  train={spk_in('train')}  dev={spk_in('dev')}  test={spk_in('test')}")
    print(f"=== MIR-1K singer split: dev={MIR_DEV} test={MIR_TEST} (rest train) ===")
    print(f"=== Jamendo dev songs: {sorted(jam_dev)}")
    print(f"=== Jamendo test songs: {sorted(jam_test)}")
    if dropped:
        print("=== dropped ===")
        for k, v in dropped.items():
            print(f"  {k}: {v}")
    print(f"\nwrote -> {OUT}")
    return 0


_ROWS_CACHE: list[dict] | None = None


def _rows() -> list[dict]:
    global _ROWS_CACHE
    if _ROWS_CACHE is None:
        with open(DATASET / "metadata.csv", encoding="utf-8") as f:
            _ROWS_CACHE = list(csv.DictReader(f))
    return _ROWS_CACHE


if __name__ == "__main__":
    sys.exit(main())
