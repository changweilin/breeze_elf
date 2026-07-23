"""LoRA post-training for Breeze-ASR-25 (nan + zh-TW singing) — TRAINING_PLAN §P1.

Recipe (RTX 3060 12GB): 8-bit base (bitsandbytes, more conservative than the
plan's fp16 base — leaves headroom, well-trodden HF PEFT-Whisper path) + LoRA
r=16 on q_proj/v_proj + gradient checkpointing + adamw_bnb_8bit, fp16 mixed
precision. Base stays frozen; only the adapter trains, so the CT2 merge in P3
reloads an fp16 base regardless of this dtype.

Data mix: all nan train + MIR-1K train oversampled to ~10:1 (nan:mir). MIR-1K
gets waveform augmentation (accompaniment mix-back at SNR 0-15dB using the
perfectly-aligned left channel, pitch ±2 st, tempo 0.9-1.1); SpecAugment is on
for every sample via model.config.

Early stopping uses eval_loss on a dev subset (robust for an unattended overnight
run — PEFT + predict_with_generate in-Trainer is fragile). Per-epoch checkpoints
are kept; dev/test CER selection happens post-hoc in P3 via the CT2 eval.

Run (venv python, NOT uv run — see memory training-toolchain):
  .venv/Scripts/python.exe tools/train_lora.py --output_dir models/lora/breeze-nan
OOM fallbacks: --r 8 ; --batch 1 ; --freeze_encoder
Smoke test:  --max_steps 4 --nan_cap 40 --mir_cap 20 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "dataset" / "manifests"
MIR_WAV = ROOT / "dataset" / "MIR-1K" / "Wavfile"
MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
SR = 16000


# ----------------------------- data ---------------------------------------
def read_manifest(split: str) -> list[dict]:
    rows = []
    with open(MANIFESTS / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mir_accompaniment_path(chunk_id: str) -> Path:
    # mir1k_abjones_1_01_0000 -> abjones_1_01
    name = re.sub(r"_\d+$", "", chunk_id[len("mir1k_"):])
    return MIR_WAV / f"{name}.wav"


class ASRDataset(Dataset):
    """Loads audio, applies MIR-1K waveform augmentation on train, returns
    {input_features, labels}. Augmentation is a no-op for nan and for dev."""

    def __init__(self, rows, feature_extractor, tokenizer, *, augment: bool, rng_seed: int = 0):
        self.rows = rows
        self.fe = feature_extractor
        self.tok = tokenizer
        self.augment = augment
        self._rng = random.Random(rng_seed)

    def __len__(self):
        return len(self.rows)

    def _load(self, rec) -> np.ndarray:
        audio, _ = sf.read(str(ROOT / rec["audio"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if self.augment and rec["id"].startswith(("mir1k_", "jamendo_")):
            audio = self._augment_singing(audio, rec["id"])
        return audio

    def _augment_singing(self, vocal: np.ndarray, chunk_id: str) -> np.ndarray:
        import librosa

        rng = self._rng
        # accompaniment mix-back (aligned left channel), SNR 0-15 dB — MIR-1K only
        # (Jamendo vocals are demucs-separated with no aligned accompaniment stem).
        if chunk_id.startswith("mir1k_") and rng.random() < 0.7:
            ap = mir_accompaniment_path(chunk_id)
            if ap.exists():
                try:
                    stereo, _ = sf.read(str(ap), dtype="float32")
                    accomp = stereo[:, 0] if stereo.ndim > 1 else stereo
                    n = min(len(vocal), len(accomp))
                    v, a = vocal[:n], accomp[:n]
                    pv = float(np.mean(v**2)) + 1e-9
                    pa = float(np.mean(a**2)) + 1e-9
                    snr = rng.uniform(0.0, 15.0)
                    scale = np.sqrt(pv / (pa * (10 ** (snr / 10.0))))
                    vocal = v + scale * a
                except Exception:
                    pass
        # pitch shift +/- 2 semitones
        if rng.random() < 0.5:
            steps = rng.uniform(-2.0, 2.0)
            vocal = librosa.effects.pitch_shift(vocal, sr=SR, n_steps=steps)
        # tempo 0.9-1.1
        if rng.random() < 0.5:
            rate = rng.uniform(0.9, 1.1)
            vocal = librosa.effects.time_stretch(vocal, rate=rate)
        peak = float(np.max(np.abs(vocal))) if vocal.size else 0.0
        if peak > 1.0:
            vocal = vocal / peak
        return vocal.astype(np.float32)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        audio = self._load(rec)
        feat = self.fe(audio, sampling_rate=SR).input_features[0]
        labels = self.tok(rec["text"]).input_ids
        return {"input_features": feat, "labels": labels}


class Collator:
    def __init__(self, processor):
        self.processor = processor
        # The tokenizer prepends the sot prefix [<|sot|> <|lang|> <|task|> <|notimestamps|>];
        # <|sot|> IS the model's decoder_start_token_id, which forward() re-adds via
        # shift_tokens_right. Strip it here or the start token gets duplicated in the
        # decoder inputs. NOTE: Whisper's bos_token_id is <|endoftext|> (50257), NOT the
        # sot token (50258) — the common recipe's bos check silently no-ops here.
        self.sot_id = processor.tokenizer.prefix_tokens[0]

    def __call__(self, features):
        fe = self.processor.feature_extractor
        tok = self.processor.tokenizer
        batch = fe.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = tok.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.sot_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


_PREFIX = {"nan": "cv_", "mir1k": "mir1k_", "jamendo": "jamendo_"}


def _pick(rows, source):
    return [r for r in rows if r["id"].startswith(_PREFIX[source])]


def build_train_rows(sources, ratio, nan_cap, mir_cap, jam_cap, seed) -> list[dict]:
    """P1: sources=[nan,mir1k] → all nan + MIR oversampled to nan:mir==ratio:1.
    P2: sources=[jamendo] → all jamendo, no oversampling."""
    rows = read_manifest("train")
    rng = random.Random(seed)
    nan = _pick(rows, "nan") if "nan" in sources else []
    mir = _pick(rows, "mir1k") if "mir1k" in sources else []
    jam = _pick(rows, "jamendo") if "jamendo" in sources else []
    for lst in (nan, mir, jam):
        rng.shuffle(lst)
    if nan_cap:
        nan = nan[:nan_cap]
    if mir_cap:
        mir = mir[:mir_cap]
    if jam_cap:
        jam = jam[:jam_cap]
    train = list(nan) + list(jam)
    if nan and mir:  # oversample the singing minority to the target ratio
        target_mir = max(len(mir), len(nan) // ratio)
        mir = [mir[i % len(mir)] for i in range(target_mir)]
    train += mir
    rng.shuffle(train)
    print(f"[data] nan={len(nan)} mir={len(mir)} jam={len(jam)} total={len(train)}")
    return train


def build_dev_rows(sources, nan_dev, seed) -> list[dict]:
    rows = read_manifest("dev")
    rng = random.Random(seed)
    dev = []
    if "nan" in sources:
        nan = _pick(rows, "nan")
        rng.shuffle(nan)
        dev += nan[:nan_dev]
    if "mir1k" in sources:
        dev += _pick(rows, "mir1k")
    if "jamendo" in sources:
        dev += _pick(rows, "jamendo")
    print(f"[data] dev subset: total={len(dev)}")
    return dev


# ----------------------------- model --------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="models/lora/breeze-nan")
    ap.add_argument("--model_id", default=MODEL_ID, help="P2: openai/whisper-medium")
    ap.add_argument("--language", default="chinese", help="P2: english")
    ap.add_argument("--sources", default="nan,mir1k", help="P2: jamendo")
    ap.add_argument("--jam_cap", type=int, default=0, help="smoke: cap jamendo train")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ratio", type=int, default=10)
    ap.add_argument("--nan_dev", type=int, default=400)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--nan_cap", type=int, default=0, help="smoke: cap nan train")
    ap.add_argument("--mir_cap", type=int, default=0, help="smoke: cap mir train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    processor = WhisperProcessor.from_pretrained(
        args.model_id, language=args.language, task="transcribe"
    )

    print(f"[model] loading {args.model_id} in 8-bit ...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # SpecAugment for every training sample
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.r,
        lora_alpha=2 * args.r,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora)
    if args.freeze_encoder:
        # OOM / anti-forgetting fallback: keep encoder LoRA frozen, train decoder only.
        frozen = 0
        for name, p in model.named_parameters():
            if p.requires_grad and ".encoder." in name:
                p.requires_grad = False
                frozen += 1
        print(f"[lora] freeze_encoder: froze {frozen} encoder adapter tensors")
    model.print_trainable_parameters()
    model.config.use_cache = False

    train_rows = build_train_rows(
        sources, args.ratio, args.nan_cap, args.mir_cap, args.jam_cap, args.seed
    )
    dev_rows = build_dev_rows(sources, args.nan_dev, args.seed)
    train_ds = ASRDataset(train_rows, processor.feature_extractor, processor.tokenizer,
                          augment=True, rng_seed=args.seed)
    dev_ds = ASRDataset(dev_rows, processor.feature_extractor, processor.tokenizer,
                        augment=False)
    collator = Collator(processor)

    steps_per_epoch = max(1, len(train_rows) // (args.batch * args.grad_accum))
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        logging_first_step=True,
        disable_tqdm=True,  # clean file logs for the long unattended run
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
    )
    print(f"[train] ~{steps_per_epoch} optim steps/epoch, effective batch {args.batch*args.grad_accum}", flush=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    out = Path(args.output_dir)
    model.save_pretrained(out / "adapter_best")
    processor.save_pretrained(out / "adapter_best")
    print(f"[done] adapter saved -> {out/'adapter_best'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
