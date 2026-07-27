"""P3 deploy: merge a LoRA adapter into an fp16 base and convert to CTranslate2.

The adapter trained on an 8-bit base (train_lora.py); merging reloads the base
in fp16 so merge_and_unload produces clean fp16 weights, then
ct2-transformers-converter emits a faster-whisper-loadable CT2 dir. Output goes
to a NEW dir — never overwrites models/breeze-asr-25-ct2 (A/B; see memory
asr-model-switcher).

Run (venv python):
  .venv/Scripts/python.exe tools/merge_and_convert.py \
      --adapter models/lora/breeze-nan/adapter_best \
      --ct2_out models/breeze-asr-25-nan-ct2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
# faster-whisper reads these next to model.bin; ct2 converter copies them verbatim.
COPY_FILES = ["tokenizer.json", "preprocessor_config.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir (adapter_best)")
    ap.add_argument("--base", default=MODEL_ID, help="base model id or dir")
    ap.add_argument("--merged_out", default="", help="fp16 merged HF dir (default: <ct2_out>_hf)")
    ap.add_argument("--ct2_out", required=True, help="output CT2 dir")
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--keep_merged", action="store_true", help="don't delete the fp16 merged dir")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    merged_out = Path(args.merged_out or f"{args.ct2_out}_hf")
    ct2_out = Path(args.ct2_out)
    if ct2_out.exists():
        print(f"[abort] {ct2_out} exists — refusing to overwrite (A/B safety)")
        return 1

    # fp32 on CPU: some fp16 ops aren't implemented on CPU; CT2 quantizes to fp16 after.
    print(f"[merge] base={args.base} in fp32 (CPU merge) ...", flush=True)
    base = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32)
    print(f"[merge] applying adapter {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    merged_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_out, safe_serialization=True)
    try:
        proc = WhisperProcessor.from_pretrained(args.adapter)
    except Exception:
        proc = WhisperProcessor.from_pretrained(args.base)
    proc.save_pretrained(merged_out)
    print(f"[merge] merged HF -> {merged_out}", flush=True)

    copy_files = [f for f in COPY_FILES if (merged_out / f).exists()]
    converter = Path(sys.executable).parent / "ct2-transformers-converter.exe"
    conv_cmd = [str(converter)] if converter.exists() else \
        [sys.executable, "-m", "ctranslate2.converters.transformers"]
    cmd = [
        *conv_cmd,
        "--model", str(merged_out),
        "--output_dir", str(ct2_out),
        "--quantization", args.quantization,
    ]
    if copy_files:
        cmd += ["--copy_files", *copy_files]
    print(f"[ct2] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[ct2] converter failed")
        return r.returncode

    # faster-whisper needs tokenizer.json + preprocessor_config.json + vocabulary.json
    # next to model.bin. LoRA never touches the tokenizer/preprocessor, so any file the
    # converter didn't emit is byte-identical in the base CT2 dir — copy it in.
    base_ct2 = ROOT / "models" / "breeze-asr-25-ct2"
    for fname in ("tokenizer.json", "preprocessor_config.json", "vocabulary.json"):
        if not (ct2_out / fname).exists() and (base_ct2 / fname).exists():
            shutil.copy(base_ct2 / fname, ct2_out / fname)
            print(f"[ct2] copied {fname} from base CT2")

    if not args.keep_merged:
        shutil.rmtree(merged_out, ignore_errors=True)
        print(f"[cleanup] removed {merged_out}")
    print(f"[done] CT2 model -> {ct2_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
