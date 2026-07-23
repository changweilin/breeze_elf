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

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
NAN = DATASET / "nan-tw"
OUT = DATASET / "manifests"

# MIR-1K held-out singers (alphabetically last two of the 19 — deterministic &
# reproducible; both have ~5 song prefixes so eval size is adequate).
MIR_DEV = "titon"
MIR_TEST = "yifen"


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
    train_stems = read_tsv_stems(NAN / "train.tsv")
    dev_stems = read_tsv_stems(NAN / "dev.tsv")
    test_stems = read_tsv_stems(NAN / "test.tsv")
    client = read_stem_to_client(NAN / "validated.tsv")
    held_spk = {client.get(s) for s in dev_stems} | {client.get(s) for s in test_stems}

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
            dropped["empty_text"] += 1
            continue

        if stem.startswith("cv_"):
            cs = cv_stem(stem)
            if cs in test_stems:
                split = "test"
            elif cs in dev_stems:
                split = "dev"
            elif cs in train_stems:
                split = "train"
            elif client.get(cs) in held_spk:
                dropped["nan_speaker_leak"] += 1
                continue
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
