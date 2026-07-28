"""Evaluate a faster-whisper / CTranslate2 ASR model on the test manifest.

Reports the five metrics of TRAINING_PLAN.md §1.2 in one pass, each only when the
data to compute it is present, so a plain run stays comparable to older reports:

  1. CER (zh) / WER (en) [+ MER for nan]  — asr sources (nan, mir1k, jamendo, speech)
  2. hallucination rate  — negative sources (empty-ref clips): non-empty rate,
     credit-string hit rate, and post-gate leak rate through ``should_drop_text``
     (the same gate as production).
  3. alignment error (median |Δ| ms of word start/end vs a human-labelled gold at
     dataset/manifests/align_gold.jsonl) — the metric the 簡譜 timeline lives on.
  4. RTF — decode seconds / audio seconds, always reported.
  5. speech regression — just an asr source ("speech_"): 會議錄音 CER against a
     frozen preset, to catch catastrophic forgetting.

Decoding mirrors production (breeze_elf/asr.py) by default: beam_size=1,
vad_filter=False, condition_on_previous_text=False, task=transcribe, OpenCC s→t on
Chinese output. Every knob in TRAINING_PLAN.md §2 is a CLI flag so the training-free
sweep runs one gain at a time without editing this file. Scoring normalisation
(§P0.3): NFKC full/half-width unify, strip punctuation, drop CJK whitespace.

Usage (always via venv python, NOT uv run — see memory training-toolchain):
  .venv/Scripts/python.exe tools/eval_asr.py \
      --model models/breeze-asr-25-ct2 --sources nan,mir1k --tag breeze_baseline
  # §2 sweep example — force-open condition_on_previous_text on the file path:
  .venv/Scripts/python.exe tools/eval_asr.py --model medium --sources jamendo \
      --condition-on-previous-text --temperature 0.0,0.2,0.4,0.6,0.8,1.0 \
      --tag whisper_condprev

Per-source lang/metric/kind are fixed by the corpus (see SOURCE_CFG).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # import breeze_elf.* when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import strip_reference_gloss  # noqa: E402  (re-exported for rescore.py)

from breeze_elf.hallucination import (  # noqa: E402
    is_credit_hallucination,
    should_drop_text,
)

MANIFESTS = ROOT / "dataset" / "manifests"
REPORTS = ROOT / "dataset" / "eval_reports"
ALIGN_GOLD = MANIFESTS / "align_gold.jsonl"

# source -> (whisper language, metric, apply opencc s->t, kind).
#   kind "asr"      -> CER/WER (+ RTF, + alignment when gold covers the clip)
#   kind "negative" -> hallucination stats (+ RTF); refs are empty by design
SOURCE_CFG = {
    "nan": ("zh", "cer", True, "asr"),
    "mir1k": ("zh", "cer", True, "asr"),
    "jamendo": ("en", "wer", False, "asr"),
    "speech": ("zh", "cer", True, "asr"),  # 說話回歸: 會議錄音 (§1.2.5)
    "negatives": ("zh", "hallucination", True, "negative"),  # C 層負樣本 (§1.1)
}
PREFIX = {
    "nan": "cv_",
    "mir1k": "mir1k_",
    "jamendo": "jamendo_",
    "speech": "speech_",
    "negatives": "neg_",
}

# Punctuation (ASCII + CJK) stripped before scoring.
_PUNCT = set(
    " \t\n\r"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "。，、！？；：「」『』（）《》〈〉【】〔〕…—～·．‧・“”‘’〜°％"
)


@dataclass(frozen=True)
class DecodeKnobs:
    """The faster-whisper decode parameters. Defaults mirror production so an
    unflagged run reproduces every previous report; each field is a §2 sweep knob."""

    beam: int = 1
    vad_filter: bool = False
    condition_on_previous_text: bool = False
    temperature: tuple[float, ...] | None = None  # None -> faster-whisper default ladder
    compression_ratio_threshold: float | None = None  # None -> not passed (model default)
    initial_prompt: str | None = None
    word_timestamps: bool = False


@dataclass
class Decoded:
    text: str
    words: list[dict] = field(default_factory=list)
    no_speech_prob: float | None = None
    rms: float = 0.0
    audio_seconds: float = 0.0
    decode_seconds: float = 0.0


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


def load_alignment_gold() -> dict[str, list[dict]]:
    """id -> [{word, start, end}] human-labelled word/char timings, if present.

    Optional file: without it the alignment metric is simply not reported (the
    other four still run), so the eval is useful before anyone hand-labels a clip."""
    if not ALIGN_GOLD.exists():
        return {}
    gold: dict[str, list[dict]] = {}
    with open(ALIGN_GOLD, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            words = rec.get("words") or []
            if rec.get("id") and words:
                gold[rec["id"]] = words
    return gold


def transcribe_one(model, audio_path: Path, language: str, knobs: DecodeKnobs) -> Decoded:
    import soundfile as sf

    samples, sr = sf.read(str(audio_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    audio_seconds = float(samples.shape[0]) / sr if sr else 0.0

    extra: dict = {}
    if knobs.temperature is not None:
        extra["temperature"] = knobs.temperature
    if knobs.compression_ratio_threshold is not None:
        extra["compression_ratio_threshold"] = knobs.compression_ratio_threshold

    started = time.perf_counter()
    segments, _info = model.transcribe(
        samples,
        language=language,
        task="transcribe",
        beam_size=knobs.beam,
        vad_filter=knobs.vad_filter,
        condition_on_previous_text=knobs.condition_on_previous_text,
        initial_prompt=knobs.initial_prompt,
        word_timestamps=knobs.word_timestamps,
        **extra,
    )
    seg_list = list(segments)  # generator: nothing decodes until drained
    decode_seconds = time.perf_counter() - started

    text = " ".join(seg.text.strip() for seg in seg_list).strip()
    words: list[dict] = []
    if knobs.word_timestamps:
        for seg in seg_list:
            for word in getattr(seg, "words", None) or []:
                try:
                    words.append(
                        {
                            "word": (getattr(word, "word", "") or "").strip(),
                            "start": float(word.start),
                            "end": float(word.end),
                        }
                    )
                except (TypeError, ValueError):
                    continue
    probs = [
        float(seg.no_speech_prob)
        for seg in seg_list
        if getattr(seg, "no_speech_prob", None) is not None
    ]
    rms = float((samples.astype("float64") ** 2).mean() ** 0.5) if samples.size else 0.0
    return Decoded(
        text=text,
        words=words,
        no_speech_prob=max(probs) if probs else None,
        rms=rms,
        audio_seconds=audio_seconds,
        decode_seconds=decode_seconds,
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def char_cer_mer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Character-level (CER, MER).

    ``jiwer.mer`` tokenises on whitespace; our CJK normalisation strips every space,
    so ``jiwer.mer`` would see one token per line and degenerate to a per-clip
    exact-match rate (~1.0 for any imperfect transcription). MER must therefore be
    derived from the character alignment: (S+D+I)/(H+S+D+I). ``.cer`` here is byte-for-
    byte the old ``jiwer.cer`` value, so CER stays comparable to earlier reports."""
    import jiwer

    chars = jiwer.process_characters(refs, hyps)
    edits = chars.substitutions + chars.deletions + chars.insertions
    denom = edits + chars.hits
    mer = edits / denom if denom else 0.0
    return round(chars.cer, 5), round(mer, 5)


def _norm_token(text: str) -> str:
    # cjk=True strips whitespace/punct but does NOT lowercase; ``.lower()`` makes
    # latin (jamendo) token matching case-insensitive like its WER scoring — a no-op
    # for CJK, so the 簡譜 timeline path is unaffected.
    return normalize(text, cjk=True).lower()


def alignment_error_ms(pred_words: list[dict], gold_words: list[dict]) -> float | None:
    """Median absolute error (ms) of matched word start/end times.

    Forward greedy match on normalised token text: pred insertions are skipped and
    gold-only tokens are ignored, so a stray extra word does not misalign the rest.
    Start and end errors are pooled. ``None`` when nothing matches."""
    errs: list[float] = []
    cursor = 0
    for gold in gold_words:
        target = _norm_token(gold.get("word", ""))
        if not target:
            continue
        probe = cursor
        while probe < len(pred_words) and _norm_token(pred_words[probe].get("word", "")) != target:
            probe += 1
        if probe >= len(pred_words):
            continue
        pred = pred_words[probe]
        try:
            errs.append(abs(float(pred["start"]) - float(gold["start"])) * 1000.0)
            errs.append(abs(float(pred["end"]) - float(gold["end"])) * 1000.0)
        except (KeyError, TypeError, ValueError):
            pass
        cursor = probe + 1
    if not errs:
        return None
    return round(_median(errs), 1)


def hallucination_stats(records: list[dict], *, drop_no_speech: float, drop_rms: float) -> dict:
    """Hallucination metric on empty-reference (C-layer) clips.

    * nonempty_rate       — how often the model invents any text at all
    * credit_rate         — how often it emits subtitle-credit boilerplate
    * post_gate_leak_rate — non-empty text that survives ``should_drop_text``: the
                            real leak, since the production gate would have dropped
                            the rest. This is the number to drive to zero.
    Each ``record`` is ``{hyp, no_speech_prob, rms}``."""
    n = len(records)
    if not n:
        return {"n": 0}
    nonempty = credit = leaked = 0
    for rec in records:
        hyp = (rec.get("hyp") or "").strip()
        if not hyp:
            continue
        nonempty += 1
        if is_credit_hallucination(hyp):
            credit += 1
        dropped = should_drop_text(
            hyp,
            rec.get("no_speech_prob"),
            rec.get("rms", 0.0),
            no_speech_threshold=drop_no_speech,
            rms_threshold=drop_rms,
        )
        if not dropped:
            leaked += 1
    return {
        "n": n,
        "nonempty_rate": round(nonempty / n, 5),
        "credit_rate": round(credit / n, 5),
        "post_gate_leak_rate": round(leaked / n, 5),
    }


def _rtf(decode_seconds: float, audio_seconds: float) -> float | None:
    return round(decode_seconds / audio_seconds, 4) if audio_seconds > 0 else None


def _parse_temperature(raw: str) -> tuple[float, ...] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    return tuple(float(part) for part in raw.split(",") if part.strip())


def _score_asr_source(model, rows, cfg, knobs, gold, dump_path: Path) -> dict:
    import jiwer

    language, metric, use_cc, _kind = cfg
    cjk = metric == "cer"
    converter = make_converter() if use_cc else None

    refs, hyps, pairs = [], [], []
    align_errs: list[float] = []
    audio_total = decode_total = 0.0
    for rec in rows:
        decoded = transcribe_one(model, ROOT / rec["audio"], language, knobs)
        audio_total += decoded.audio_seconds
        decode_total += decoded.decode_seconds
        raw = converter.convert(decoded.text) if converter is not None else decoded.text
        raw_ref = rec["text"]
        ref_text = strip_reference_gloss(raw_ref) if rec["id"].startswith("cv_") else raw_ref
        ref = normalize(ref_text, cjk=cjk)
        hyp = normalize(raw, cjk=cjk)
        gold_words = gold.get(rec["id"])
        if gold_words and decoded.words:
            err = alignment_error_ms(decoded.words, gold_words)
            if err is not None:
                align_errs.append(err)
        if not ref:
            continue
        refs.append(ref)
        hyps.append(hyp)
        pairs.append({"id": rec["id"], "ref": rec["text"], "hyp": raw})

    res: dict = {"n": len(refs)}
    if metric == "cer":
        res["cer"], res["mer"] = char_cer_mer(refs, hyps)
    else:
        res["wer"] = round(jiwer.wer(refs, hyps), 5)
    res["rtf"] = _rtf(decode_total, audio_total)
    if align_errs:
        res["align_err_ms"] = round(_median(align_errs), 1)
        res["align_n"] = len(align_errs)

    with open(dump_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return res


def _score_negative_source(model, rows, cfg, knobs, drop_no_speech, drop_rms, dump_path) -> dict:
    language = cfg[0]
    converter = make_converter() if cfg[2] else None
    records, dump = [], []
    audio_total = decode_total = 0.0
    for rec in rows:
        decoded = transcribe_one(model, ROOT / rec["audio"], language, knobs)
        audio_total += decoded.audio_seconds
        decode_total += decoded.decode_seconds
        raw = converter.convert(decoded.text) if converter is not None else decoded.text
        records.append(
            {
                "hyp": raw,
                "no_speech_prob": decoded.no_speech_prob,
                "rms": decoded.rms,
            }
        )
        dump.append(
            {"id": rec["id"], "ref": "", "hyp": raw, "no_speech_prob": decoded.no_speech_prob}
        )

    res = hallucination_stats(records, drop_no_speech=drop_no_speech, drop_rms=drop_rms)
    res["rtf"] = _rtf(decode_total, audio_total)
    with open(dump_path, "w", encoding="utf-8") as f:
        for d in dump:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CT2 dir or whisper size")
    ap.add_argument("--sources", required=True, help="comma list: " + ",".join(SOURCE_CFG))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True, help="report filename stem")
    ap.add_argument("--limit", type=int, default=0, help="cap clips per source (0=all)")
    ap.add_argument("--compute-type", default="float16", help="int8 | int8_float16 | float16")
    # §2 decode knobs — defaults mirror production.
    ap.add_argument("--beam", type=int, default=1, help="beam_size (production uses 1)")
    ap.add_argument("--vad-filter", action="store_true", help="enable faster-whisper VAD")
    ap.add_argument("--condition-on-previous-text", action="store_true")
    ap.add_argument("--temperature", default="", help="comma list e.g. 0.0,0.2 (blank=default)")
    ap.add_argument("--compression-ratio-threshold", type=float, default=None)
    ap.add_argument("--initial-prompt", default=None, help="e.g. 以下是歌詞…")
    ap.add_argument("--word-timestamps", action="store_true", help="force word timings on")
    # Hallucination-gate thresholds for the negative-source metric (§2: re-sweep for
    # separated vocals — the 0.6/0.02 defaults are tuned for speech, not singing).
    ap.add_argument("--drop-no-speech", type=float, default=0.6)
    ap.add_argument("--drop-rms", type=float, default=0.02)
    args = ap.parse_args()

    gold = load_alignment_gold()
    knobs = DecodeKnobs(
        beam=args.beam,
        vad_filter=args.vad_filter,
        condition_on_previous_text=args.condition_on_previous_text,
        temperature=_parse_temperature(args.temperature),
        compression_ratio_threshold=args.compression_ratio_threshold,
        initial_prompt=args.initial_prompt,
        # Alignment needs word timings; turn them on automatically when a gold set exists.
        word_timestamps=args.word_timestamps or bool(gold),
    )

    from faster_whisper import WhisperModel

    print(f"[load] {args.model} (cuda/{args.compute_type})", flush=True)
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)

    REPORTS.mkdir(parents=True, exist_ok=True)
    dump_dir = REPORTS / f"{args.tag}_preds"
    dump_dir.mkdir(exist_ok=True)
    report = {
        "model": args.model,
        "split": args.split,
        "decode": {
            "beam": knobs.beam,
            "vad_filter": knobs.vad_filter,
            "condition_on_previous_text": knobs.condition_on_previous_text,
            "temperature": list(knobs.temperature) if knobs.temperature else None,
            "compression_ratio_threshold": knobs.compression_ratio_threshold,
            "initial_prompt": knobs.initial_prompt,
            "compute_type": args.compute_type,
            "drop_no_speech": args.drop_no_speech,
            "drop_rms": args.drop_rms,
        },
        "sources": {},
    }

    for source in args.sources.split(","):
        source = source.strip()
        if source not in SOURCE_CFG:
            print(f"[skip] unknown source {source!r}", flush=True)
            continue
        cfg = SOURCE_CFG[source]
        rows = load_manifest(args.split, source)
        if args.limit:
            rows = rows[: args.limit]
        print(f"\n[{source}] {len(rows)} clips  lang={cfg[0]} kind={cfg[3]}", flush=True)
        if not rows:
            print("  (no clips in manifest — skipped)", flush=True)
            continue

        dump_path = dump_dir / f"{source}.jsonl"
        if cfg[3] == "negative":
            res = _score_negative_source(
                model, rows, cfg, knobs, args.drop_no_speech, args.drop_rms, dump_path
            )
        else:
            res = _score_asr_source(model, rows, cfg, knobs, gold, dump_path)
        report["sources"][source] = res
        print(f"  -> {res}", flush=True)

    out = REPORTS / f"{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n=== REPORT {out} ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
