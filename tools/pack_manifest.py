"""Pack short nan utterances into ~25s training windows.

Whisper always encodes a fixed 30s mel window, but nan clips average 2.44s
(median 2.27s) — so 92% of every encoder forward pass is padding. Packing
same-speaker clips back-to-back turns ~16k forward passes into ~1.3k for the
same audio (12x), and gives the decoder the surrounding context that the
duration breakdown shows it lacks (CER 0.709 at 1.5-3s vs 0.506 at 6s+).

Only the TRAIN split is packed. dev/test stay one-utterance-per-record so they
keep matching real inference. A fraction of single clips is kept unpacked so the
model still sees the short-utterance distribution it is evaluated on.

  .venv/Scripts/python.exe tools/pack_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "dataset" / "manifests"
JOIN = "，"  # natural boundary for concatenated utterances; stripped when scoring


def load_durations() -> dict[str, float]:
    dur = {}
    with open(ROOT / "dataset" / "metadata.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dur[Path(row["file_name"]).stem] = float(row["duration"])
    return dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=25.0, help="window seconds")
    ap.add_argument("--gap", type=float, default=0.3, help="silence between clips (s)")
    ap.add_argument("--unpacked_frac", type=float, default=0.15,
                    help="fraction of nan clips also kept as single utterances")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="train_packed.jsonl")
    args = ap.parse_args()

    dur = load_durations()
    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in open(MANIFESTS / "train.jsonl", encoding="utf-8")]
    nan = [r for r in rows if r["id"].startswith("cv_")]
    other = [r for r in rows if not r["id"].startswith("cv_")]

    by_spk: dict[str, list] = defaultdict(list)
    for r in nan:
        by_spk[r.get("speaker", "")].append(r)

    packed: list[dict] = []
    for spk, items in sorted(by_spk.items()):
        rng.shuffle(items)
        cur: list[dict] = []
        cur_dur = 0.0
        for r in items:
            d = dur.get(r["id"], 0.0)
            if cur and cur_dur + args.gap + d > args.target:
                packed.append(_emit(cur, cur_dur, spk, len(packed)))
                cur, cur_dur = [], 0.0
            cur.append(r)
            cur_dur += d + (args.gap if len(cur) > 1 else 0.0)
        if cur:
            packed.append(_emit(cur, cur_dur, spk, len(packed)))

    keep_n = int(len(nan) * args.unpacked_frac)
    unpacked = rng.sample(nan, keep_n) if keep_n else []

    out_rows = packed + unpacked + other
    rng.shuffle(out_rows)
    out_path = MANIFESTS / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pack_dur = sum(r["dur"] for r in packed)
    sizes = [r["n"] for r in packed]
    print(f"nan clips in      : {len(nan)}  ({sum(dur.get(r['id'],0) for r in nan)/3600:.2f} h)")
    print(f"packed windows    : {len(packed)}  ({pack_dur/3600:.2f} h)  "
          f"mean {pack_dur/max(1,len(packed)):.1f}s  "
          f"clips/window {sum(sizes)/max(1,len(sizes)):.1f}")
    print(f"unpacked kept     : {len(unpacked)}  ({args.unpacked_frac:.0%})")
    print(f"other (mir/jam)   : {len(other)}")
    print(f"TOTAL records     : {len(out_rows)}  (was {len(rows)}) "
          f"-> {len(rows)/max(1,len(out_rows)):.1f}x fewer forward passes")
    print(f"wrote -> {out_path}")
    return 0


def _emit(items: list[dict], total_dur: float, spk: str, idx: int) -> dict:
    return {
        "id": f"nanpack_{idx:05d}",
        "audios": [r["audio"] for r in items],
        "text": JOIN.join(r["text"] for r in items),
        "lang": "nan",
        "source": "nan_packed",
        "speaker": spk,
        "n": len(items),
        "dur": round(total_dur, 2),
    }


if __name__ == "__main__":
    sys.exit(main())
