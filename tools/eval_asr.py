"""Evaluate a faster-whisper / CTranslate2 ASR model on the test manifest.

Decoding mirrors production (breeze_elf/asr.py): beam_size=1, vad_filter=False,
condition_on_previous_text=False, task=transcribe, and OpenCC s→t on Chinese
output. Scoring normalisation (TRAINING_PLAN.md §P0.3): NFKC full/half-width
unify, strip punctuation, drop CJK whitespace, then CER for nan/zh, WER for en.
MER is reported alongside CER for nan (mixed 漢字/台羅 lines).

Usage (always via venv python, NOT uv run — see memory training-toolchain):
  .venv/Scripts/python.exe tools/eval_asr.py \
      --model models/breeze-asr-25-ct2 --sources nan,mir1k --tag breeze_baseline
  .venv/Scripts/python.exe tools/eval_asr.py \
      --model medium --sources jamendo --tag whisper_medium_baseline

Per-source lang/metric are fixed by the corpus:
  nan -> zh / CER (+MER, +s2t)   mir1k -> zh / CER (+s2t)   jamendo -> en / WER
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "dataset" / "manifests"
REPORTS = ROOT / "dataset" / "eval_reports"

# source -> (whisper language, metric, apply opencc s->t)
SOURCE_CFG = {
    "nan": ("zh", "cer", True),
    "mir1k": ("zh", "cer", True),
    "jamendo": ("en", "wer", False),
}
PREFIX = {"nan": "cv_", "mir1k": "mir1k_", "jamendo": "jamendo_"}

# Punctuation (ASCII + CJK) stripped before scoring.
_PUNCT = set(
    " \t\n\r"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "。，、！？；：「」『』（）《》〈〉【】〔〕…—～·．‧・“”‘’〜°％"
)


def make_converter():
    try:
        from opencc import OpenCC
    except Exception:
        return None
    for name in ("s2twp", "s2tw", "s2t"):
        try:
            return OpenCC(name)
        except Exception:
            continue
    return None


def normalize(text: str, *, cjk: bool) -> str:
    text = unicodedata.normalize("NFKC", text)
    if not cjk:
        text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    if cjk:
        # char-level: remove every whitespace so CER counts only content chars
        text = "".join(text.split())
    else:
        text = " ".join(text.split())
    return text


def load_manifest(split: str, source: str) -> list[dict]:
    path = MANIFESTS / f"{split}.jsonl"
    pre = PREFIX[source]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"].startswith(pre):
                rows.append(rec)
    return rows


def transcribe_one(model, audio_path: Path, language: str) -> str:
    samples, _sr = sf.read(str(audio_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    segments, _info = model.transcribe(
        samples,
        language=language,
        task="transcribe",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CT2 dir or whisper size")
    ap.add_argument("--sources", required=True, help="comma list: nan,mir1k,jamendo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True, help="report filename stem")
    ap.add_argument("--limit", type=int, default=0, help="cap clips per source (0=all)")
    ap.add_argument("--compute-type", default="float16")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    print(f"[load] {args.model} (cuda/{args.compute_type})", flush=True)
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    converter = make_converter()

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "split": args.split, "sources": {}}
    dump_dir = REPORTS / f"{args.tag}_preds"
    dump_dir.mkdir(exist_ok=True)

    for source in args.sources.split(","):
        source = source.strip()
        language, metric, use_cc = SOURCE_CFG[source]
        cjk = metric == "cer"
        rows = load_manifest(args.split, source)
        if args.limit:
            rows = rows[: args.limit]
        print(f"\n[{source}] {len(rows)} clips  lang={language} metric={metric}", flush=True)

        refs, hyps, pairs = [], [], []
        t0 = time.perf_counter()
        for i, rec in enumerate(rows, 1):
            raw = transcribe_one(model, ROOT / rec["audio"], language)
            if use_cc and converter is not None:
                raw = converter.convert(raw)
            ref = normalize(rec["text"], cjk=cjk)
            hyp = normalize(raw, cjk=cjk)
            if not ref:
                continue
            refs.append(ref)
            hyps.append(hyp)
            pairs.append({"id": rec["id"], "ref": rec["text"], "hyp": raw})
            if i % 250 == 0 or i == len(rows):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(rows)}  ({rate:.1f} clip/s)", flush=True)

        res = {"n": len(refs)}
        if metric == "cer":
            res["cer"] = round(jiwer.cer(refs, hyps), 5)
            res["mer"] = round(jiwer.mer(refs, hyps), 5)
        else:
            res["wer"] = round(jiwer.wer(refs, hyps), 5)
        report["sources"][source] = res
        print(f"  -> {res}", flush=True)

        with open(dump_dir / f"{source}.jsonl", "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    out = REPORTS / f"{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n=== REPORT {out} ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
