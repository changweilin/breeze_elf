"""Recompute eval metrics from a saved prediction dump — no GPU, no re-decode.

The dumps in dataset/eval_reports/<tag>_preds/<source>.jsonl hold raw ref/hyp
text, so a scoring-rule change (e.g. stripping the 台羅 gloss from references)
can be applied retroactively to every model already evaluated. Only the
text-distance metrics: 幻覺率, 對齊誤差 and RTF depend on audio and timing the dump
does not carry, so those come from a fresh tools/eval_asr.py run.

  .venv/Scripts/python.exe tools/rescore.py --tags breeze_baseline,breeze_lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_asr import REPORTS, SOURCE_CFG, normalize, strip_reference_gloss  # noqa: E402


def score(tag: str, raw_refs: bool) -> dict:
    out: dict[str, dict] = {}
    for source, (_lang, metric, _cc) in SOURCE_CFG.items():
        # 幻覺率 is not a text-distance metric: its dump has empty references and
        # scoring it as CER would report a meaningless 0. It also cannot be
        # recomputed from text alone — the gate reads the clip's RMS and the
        # model's no_speech_prob — so re-run tools/eval_asr.py to change it.
        if metric == "hallucination":
            continue
        path = REPORTS / f"{tag}_preds" / f"{source}.jsonl"
        if not path.exists():
            continue
        cjk = metric == "cer"
        refs, hyps = [], []
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            text = rec["ref"]
            if source == "nan" and not raw_refs:
                text = strip_reference_gloss(text)
            ref = normalize(text, cjk=cjk)
            if not ref:
                continue
            refs.append(ref)
            hyps.append(normalize(rec["hyp"], cjk=cjk))
        res = {"n": len(refs)}
        if metric == "cer":
            res["cer"] = round(jiwer.cer(refs, hyps), 5)
            res["mer"] = round(jiwer.mer(refs, hyps), 5)
        else:
            res["wer"] = round(jiwer.wer(refs, hyps), 5)
        out[source] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma list of report tags")
    ap.add_argument("--raw_refs", action="store_true", help="score without gloss stripping")
    ap.add_argument("--write", action="store_true", help="overwrite <tag>.json reports")
    args = ap.parse_args()

    all_res = {}
    for tag in args.tags.split(","):
        tag = tag.strip()
        res = score(tag, args.raw_refs)
        all_res[tag] = res
        print(f"\n=== {tag} ===")
        for source, r in res.items():
            print(f"  {source:8} {r}")
        if args.write:
            out = REPORTS / f"{tag}.json"
            payload = json.load(open(out, encoding="utf-8")) if out.exists() else {}
            payload["sources"] = res
            payload["scoring"] = "raw_refs" if args.raw_refs else "gloss_stripped"
            json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"  -> wrote {out}")

    # relative deltas vs the first tag (treated as baseline)
    tags = list(all_res)
    if len(tags) > 1:
        base = all_res[tags[0]]
        print(f"\n=== relative vs {tags[0]} ===")
        for tag in tags[1:]:
            for source, r in all_res[tag].items():
                if source not in base:
                    continue
                key = "cer" if "cer" in r else "wer"
                b, v = base[source][key], r[key]
                print(f"  {tag:20} {source:8} {key} {b:.4f} -> {v:.4f}  ({(b-v)/b:+.1%} rel.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
