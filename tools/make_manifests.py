"""Build leakage-safe train/dev/test manifests from dataset/metadata.csv.

Split rules (TRAINING_PLAN.md §切分規則):
  nan (cv_)      : reuse Common Voice official split by clip stem; leftover
                   validated clips join train ONLY if their speaker (client_id)
                   is absent from dev/test (else dropped — speaker-leak guard).
  MIR-1K (mir1k_): split by singer, 17 train / 1 dev / 1 test.
  Jamendo (jam_) : split by song, 16 train / 2 dev / 2 test.

Text target is metadata's `transcription` column verbatim (教育部漢字 for nan,
zh-TW lyrics for MIR-1K, lowercased en for Jamendo). Output: dataset/manifests/
{train,dev,test}.jsonl with {id, audio, text, lang, source, split}.

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

# C-layer negative samples (TRAINING_PLAN.md §1.1/§3.3): clips whose target text is
# empty on purpose — pure 間奏/前奏/伴奏/噪音 — the anti-hallucination signal. Marked
# by a ``neg_`` id prefix. The first NEG_TEST (sorted by stem) are frozen as the
# hallucination eval set and never enter training; the next NEG_DEV go to dev; the
# rest become the 5–10% training replay. Empty text is the label here, so — unlike
# every other source — an empty ``transcription`` must be kept, not dropped.
NEG_TEST = 30
NEG_DEV = 10


def negative_split(rank: int) -> str:
    """Split for the ``rank``-th negative (sorted by stem): frozen eval slice first,
    then dev, the rest as training replay."""
    return "test" if rank < NEG_TEST else "dev" if rank < NEG_TEST + NEG_DEV else "train"


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
    mir_singer_split: dict[str, str] = {}
    neg_rows: list[dict] = []

    for row in _rows():
        fn = row["file_name"]  # chunks/<id>.wav
        stem = Path(fn).stem
        text = (row["transcription"] or "").strip()
        rec = {
            "id": stem,
            "audio": f"dataset/{fn}",
            "text": text,
            "lang": row["language"],
            "source": row["source_dataset_or_song_id"],
        }
        if not text:
            if stem.startswith("neg_"):
                # Empty target is the label, not a defect — this is a C-layer negative.
                rec["negative"] = True
                neg_rows.append(rec)
            else:
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
        elif stem.startswith("speech_"):
            # 說話回歸集(會議錄音):凍結的純說話評測,全進 test、永不訓練
            # (TRAINING_PLAN §1.2.5,防災難性遺忘)。eval_asr 以 speech_ 前綴讀取。
            split = "test"
            counts[("speech", split)] += 1
        else:
            dropped["unknown_prefix"] += 1
            continue

        rec["split"] = split
        splits[split].append(rec)

    # Split negatives deterministically by stem: the frozen eval slice first, then the
    # dev slice, the rest as training replay. No speaker to keep disjoint here.
    for rank, rec in enumerate(sorted(neg_rows, key=lambda r: r["id"])):
        split = negative_split(rank)
        rec["split"] = split
        counts[("negatives", split)] += 1
        splits[split].append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        path = OUT / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in splits[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- report ---
    print("=== manifest counts (source x split) ===")
    for src in ("nan", "mir1k", "jamendo", "speech", "negatives"):
        line = "  ".join(f"{sp}={counts[(src, sp)]:>6}" for sp in ("train", "dev", "test"))
        print(f"  {src:8} {line}")
    print("=== split totals ===")
    for split in ("train", "dev", "test"):
        print(f"  {split:6} {len(splits[split]):>6}")
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
