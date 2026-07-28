"""Evaluate a faster-whisper / CTranslate2 ASR model on the test manifest.

Reports all five of `TRAINING_PLAN.md §1.2`'s metrics in one run — CER/WER/MER,
幻覺率, 對齊誤差, RTF, 說話回歸 — because four of them cannot be inferred from the
fifth. CER in particular is blind to the two failures that hurt singing most: a
model that invents a line of lyrics over an interlude has no reference characters
to be wrong about, and a model that shifts every word boundary 300 ms late scores
exactly as it did before while the 簡譜 built on those boundaries falls apart.

Decoding mirrors production (breeze_elf/asr.py): beam_size=1, vad_filter=False,
condition_on_previous_text=False, word_timestamps=True, task=transcribe, and
OpenCC s→t on Chinese output. Scoring normalisation (TRAINING_PLAN.md §P0.3):
NFKC full/half-width unify, strip punctuation, drop CJK whitespace, then CER for
nan/zh, WER for en. MER is reported alongside CER for nan (mixed 漢字/台羅 lines).

Usage (always via venv python, NOT uv run — see memory training-toolchain):
  .venv/Scripts/python.exe tools/eval_asr.py \
      --model models/breeze-asr-25-ct2 --sources nan,mir1k,negative --tag breeze_baseline
  .venv/Scripts/python.exe tools/eval_asr.py \
      --model medium --sources jamendo --tag whisper_medium_baseline

Per-source lang/metric are fixed by the corpus:
  nan -> zh / CER (+MER, +s2t)   mir1k -> zh / CER (+s2t)   jamendo -> en / WER
  speech -> zh / CER (+s2t), the 說話回歸 set   negative -> 幻覺率, no CER

RTF is measured for every source. 對齊誤差 needs hand-annotated boundaries; add a
``words`` list to the manifest rows you have annotated (see
``tools/eval_metrics.py:reference_words``) and it is reported for those clips.
The 說話回歸 and 負樣本 sets are manifest rows like any other — 20 段會議錄音 with
``speech_`` ids, and the empty-target 間奏 chunks the dataset builder cuts from
mixed songs (``--negative_ratio``).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import (  # noqa: E402
    AlignmentCounter,
    HallucinationCounter,
    RtfCounter,
    reference_words,
)
from text_norm import strip_reference_gloss  # noqa: E402  (re-exported for rescore.py)

from breeze_elf.audio import calculate_rms  # noqa: E402
from breeze_elf.config import get_settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "dataset" / "manifests"
REPORTS = ROOT / "dataset" / "eval_reports"

# source -> (whisper language, metric, apply opencc s->t)
SOURCE_CFG = {
    "nan": ("zh", "cer", True),
    "mir1k": ("zh", "cer", True),
    "jamendo": ("en", "wer", False),
    # 說話回歸: read/meeting speech, the catastrophic-forgetting guard. Same
    # scoring as nan so a Breeze preset's two numbers are directly comparable.
    "speech": ("zh", "cer", True),
    # C 層: empty target over instrumental audio. Scored by 幻覺率, never by CER —
    # an empty reference makes CER either 0 or undefined, and neither is the
    # question being asked.
    "negative": ("zh", "hallucination", True),
}
PREFIX = {"nan": "cv_", "mir1k": "mir1k_", "jamendo": "jamendo_", "speech": "speech_"}

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
    """Manifest rows for one source.

    ``negative`` is not a prefix but a layer: the empty-target chunks, whichever
    corpus they were cut from. Every other source takes only rows that have a
    reference to score against, so a stray empty row cannot silently shrink a CER.
    """
    path = MANIFESTS / f"{split}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            has_text = bool((rec.get("text") or "").strip())
            if source == "negative":
                if not has_text:
                    rows.append(rec)
            elif rec["id"].startswith(PREFIX[source]) and has_text:
                rows.append(rec)
    return rows


def transcribe_one(
    model,
    audio_path: Path,
    language: str,
    beam: int = 1,
) -> dict:
    """One clip through the production decoding path, with everything the five
    metrics need: the joined text, the per-segment ``no_speech_prob`` the
    幻覺閘門 reads, the word timings 對齊誤差 compares, the clip's RMS (the gate's
    other input), and the decode time RTF divides."""
    samples, sample_rate = sf.read(str(audio_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    started = time.perf_counter()
    segments, _info = model.transcribe(
        samples,
        language=language,
        task="transcribe",
        beam_size=beam,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    segment_list = list(segments)  # the generator is lazy: decoding happens here
    decode_seconds = time.perf_counter() - started

    words: list[tuple[str, float, float]] = []
    for segment in segment_list:
        for word in getattr(segment, "words", None) or []:
            token = (getattr(word, "word", "") or "").strip()
            if token:
                words.append(
                    (token, float(getattr(word, "start", 0.0)), float(getattr(word, "end", 0.0)))
                )
    return {
        "text": " ".join(seg.text.strip() for seg in segment_list).strip(),
        "segments": [
            (seg.text.strip(), _optional_float(getattr(seg, "no_speech_prob", None)))
            for seg in segment_list
        ],
        "words": words,
        "rms": float(calculate_rms(samples)),
        "audio_seconds": len(samples) / float(sample_rate or 1),
        "decode_seconds": decode_seconds,
    }


def _optional_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CT2 dir or whisper size")
    ap.add_argument(
        "--sources", required=True, help="comma list: nan,mir1k,jamendo,speech,negative"
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True, help="report filename stem")
    ap.add_argument("--limit", type=int, default=0, help="cap clips per source (0=all)")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--beam", type=int, default=1, help="beam_size (production uses 1)")
    ap.add_argument(
        "--no-speech-threshold",
        type=float,
        default=settings.asr_no_speech_prob_threshold,
        help="幻覺閘門 no_speech_prob cut (default: the running config's value)",
    )
    ap.add_argument(
        "--rms-threshold",
        type=float,
        default=settings.asr_hallucination_rms_threshold,
        help="幻覺閘門 RMS cut (default: the running config's value). TRAINING_PLAN §2 "
        "wants both re-swept on separated vocals — they were tuned for speech.",
    )
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    print(f"[load] {args.model} (cuda/{args.compute_type})", flush=True)
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    converter = make_converter()

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "split": args.split,
        "beam_size": args.beam,
        "compute_type": args.compute_type,
        "gate": {
            "no_speech_prob_threshold": args.no_speech_threshold,
            "rms_threshold": args.rms_threshold,
        },
        "sources": {},
    }
    dump_dir = REPORTS / f"{args.tag}_preds"
    dump_dir.mkdir(exist_ok=True)

    for source in args.sources.split(","):
        source = source.strip()
        language, metric, use_cc = SOURCE_CFG[source]
        cjk = metric != "wer"
        rows = load_manifest(args.split, source)
        if args.limit:
            rows = rows[: args.limit]
        print(f"\n[{source}] {len(rows)} clips  lang={language} metric={metric}", flush=True)

        refs, hyps, pairs = [], [], []
        rtf = RtfCounter()
        alignment = AlignmentCounter()
        hallucination = HallucinationCounter()
        t0 = time.perf_counter()
        for i, rec in enumerate(rows, 1):
            decoded = transcribe_one(model, ROOT / rec["audio"], language, beam=args.beam)
            rtf.add(decoded["decode_seconds"], decoded["audio_seconds"])
            raw = decoded["text"]
            if use_cc and converter is not None:
                raw = converter.convert(raw)

            if metric == "hallucination":
                hallucination.add_clip(
                    decoded["segments"],
                    rms=decoded["rms"],
                    no_speech_threshold=args.no_speech_threshold,
                    rms_threshold=args.rms_threshold,
                )
                pairs.append({"id": rec["id"], "ref": "", "hyp": raw})
            else:
                ref_text = strip_reference_gloss(rec["text"]) if source == "nan" else rec["text"]
                ref = normalize(ref_text, cjk=cjk)
                hyp = normalize(raw, cjk=cjk)
                if ref:
                    refs.append(ref)
                    hyps.append(hyp)
                    pairs.append({"id": rec["id"], "ref": rec["text"], "hyp": raw})
                annotated = reference_words(rec)
                if annotated:
                    alignment.add(annotated, decoded["words"])
            if i % 250 == 0 or i == len(rows):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(rows)}  ({rate:.1f} clip/s)", flush=True)

        res: dict = {}
        if metric == "hallucination":
            res.update(hallucination.summary())
        else:
            res["n"] = len(refs)
            if metric == "cer":
                res["cer"] = round(jiwer.cer(refs, hyps), 5)
                res["mer"] = round(jiwer.mer(refs, hyps), 5)
            else:
                res["wer"] = round(jiwer.wer(refs, hyps), 5)
            if alignment.reference_tokens:
                res["alignment"] = alignment.summary()
        res["rtf"] = round(rtf.rtf, 4) if rtf.rtf is not None else None
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
    _print_missing(report)
    return 0


def _print_missing(report: dict) -> None:
    """Say which of the five numbers this run could not produce, and why.

    A missing metric that goes unmentioned reads as a passing one, which is how a
    fine-tune ships having never been measured on the thing it broke.
    """
    sources = report["sources"]
    missing = []
    if "negative" not in sources:
        missing.append("幻覺率: 沒跑 --sources negative(C 層負樣本)")
    if "speech" not in sources:
        missing.append("說話回歸: 沒跑 --sources speech(需 speech_ 開頭的會議錄音)")
    if not any("alignment" in value for value in sources.values()):
        missing.append("對齊誤差: manifest 沒有人工標註的 words 欄位")
    if missing:
        print("\n=== 未量到的指標 (TRAINING_PLAN §1.2) ===")
        for line in missing:
            print(f"  - {line}")


if __name__ == "__main__":
    sys.exit(main())
