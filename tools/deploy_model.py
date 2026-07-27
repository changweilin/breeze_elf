"""Deploy a post-trained ASR model into the app's model switcher.

One command from a LoRA adapter (or an already-converted CT2 dir) to a preset the
phone can tap in 模型與演算法:

    merge+convert  ->  verify files  ->  smoke load/transcribe  ->  register  ->  live switch check

Nothing is registered before it loads, so a broken conversion fails here instead of
at switch time on the phone. Re-deploying the same ``--id`` replaces its entry.

Run with the venv python directly (**not** ``uv run`` — that re-syncs and breaks the
training stack's pinned tokenizers; see memory training-toolchain):

    # already converted (fast path, no torch needed)
    .venv/Scripts/python.exe tools/deploy_model.py --id breeze-nan \
        --label "Breeze ASR 台語強化" --ct2-dir models/breeze-asr-25-nan-ct2 \
        --sample dataset/chunks/cv_common_voice_nan-tw_39640916_0000.wav

    # straight from a trained adapter (merges + converts first, needs torch/peft)
    .venv/Scripts/python.exe tools/deploy_model.py --id breeze-nan2 \
        --label "Breeze ASR 台語強化 v2" --adapter models/lora/breeze-nan2/adapter_best

    # list / remove
    .venv/Scripts/python.exe tools/deploy_model.py --list
    .venv/Scripts/python.exe tools/deploy_model.py --remove breeze-nan2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breeze_elf.config import get_settings  # noqa: E402
from breeze_elf.model_registry import (  # noqa: E402
    RegisteredModel,
    RegistryError,
    load_registry,
    registry_path,
    remove_entry,
    resolve_path,
    save_registry,
    upsert_entry,
    validate_entry,
    verify_ct2_dir,
)

MERGE_SCRIPT = ROOT / "tools" / "merge_and_convert.py"


def log(step: str, message: str) -> None:
    print(f"[{step}] {message}", flush=True)


def rel_to_root(path: Path) -> str:
    """Store paths relative to the project root when possible — the registry is
    read by a server that may run from a different CWD."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run_merge(adapter: Path, ct2_dir: Path, base: str, quantization: str) -> int:
    cmd = [
        sys.executable,
        str(MERGE_SCRIPT),
        "--adapter", str(adapter),
        "--ct2_out", str(ct2_dir),
        "--base", base,
        "--quantization", quantization,
    ]
    log("merge", " ".join(cmd))
    return subprocess.run(cmd).returncode


def smoke_test(ct2_dir: Path, device: str, sample: Path | None) -> str:
    """Load the CT2 dir with faster-whisper and, when a sample is given, transcribe
    it. Loading alone already catches a truncated/incomplete conversion."""
    from faster_whisper import WhisperModel

    log("smoke", f"loading {ct2_dir} on device={device} ...")
    started = time.monotonic()
    model = WhisperModel(str(ct2_dir), device=device)
    log("smoke", f"loaded in {time.monotonic() - started:.1f}s")
    if sample is None:
        return ""

    with wave.open(str(sample), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit(f"[smoke] sample must be 16-bit mono WAV: {sample}")
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    import numpy as np

    samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if rate != 16000:
        raise SystemExit(f"[smoke] sample must be 16 kHz (got {rate}): {sample}")
    segments, _info = model.transcribe(samples, beam_size=1)
    text = "".join(segment.text for segment in segments).strip()
    log("smoke", f"transcript: {text or '(empty)'}")
    if not text:
        raise SystemExit("[smoke] model produced an empty transcript — refusing to register")
    return text


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def verify_on_server(server: str, entry: RegisteredModel, restore: bool) -> None:
    """Switch a running server to the new preset and confirm it actually loaded —
    the end-to-end check that the phone will get the model it tapped."""
    base = server.rstrip("/")
    listing = http_json(f"{base}/api/asr/models")
    previous = listing["activeId"]
    ids = [model["id"] for model in listing["models"]]
    if entry.id not in ids:
        raise SystemExit(f"[verify] server does not list {entry.id!r} (sees {ids})")

    log("verify", f"switching to {entry.id} (was {previous}) ...")
    http_json(f"{base}/api/asr/model", {"id": entry.id})
    while True:
        status = http_json(f"{base}/api/asr/model/status")
        if status["switch"]["status"] != "loading":
            break
        time.sleep(2)
    if status["activeModel"] != entry.model:
        raise SystemExit(
            f"[verify] server loaded {status['activeModel']!r}, expected {entry.model!r} "
            f"(switch: {status['switch']})"
        )
    log("verify", f"active={status['activeId']} model={status['activeModel']} "
                  f"device={status['device']}/{status['computeType']}")
    if restore and previous != entry.id:
        log("verify", f"restoring previous model {previous} ...")
        http_json(f"{base}/api/asr/model", {"id": previous})


def cmd_list(path: Path) -> int:
    entries = load_registry(path)
    print(f"registry: {path}")
    if not entries:
        print("  (no deployed models)")
        return 0
    for entry in entries:
        target = resolve_path(entry.model)
        state = "ok" if entry.kind != "breeze" or target.is_dir() else "MISSING DIR"
        print(f"  {entry.id:<16} {entry.label:<28} {entry.model}  [{state}]")
        if entry.note:
            print(f"  {'':<16} note: {entry.note}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy a post-trained model into the switcher.")
    ap.add_argument("--id", help="preset id (lowercase, url-safe), e.g. breeze-nan")
    ap.add_argument("--label", help="switcher button label, e.g. 'Breeze ASR 台語強化'")
    ap.add_argument("--ct2-dir", help="already-converted CT2 dir (default: models/<id>-ct2)")
    ap.add_argument("--adapter", help="LoRA adapter dir to merge+convert first")
    ap.add_argument("--base", default="MediaTek-Research/Breeze-ASR-25", help="merge base model")
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--kind", default="breeze", choices=("breeze", "whisper"))
    ap.add_argument("--note", default="", help="free-text provenance note kept in the registry")
    ap.add_argument("--sample", help="16 kHz mono WAV transcribed as the smoke test")
    ap.add_argument("--device", default="auto", help="smoke-test device (auto|cuda|cpu)")
    ap.add_argument("--skip-smoke", action="store_true", help="register without loading the model")
    ap.add_argument("--server", help="running server to switch+verify, e.g. http://127.0.0.1:8788")
    ap.add_argument(
        "--keep-active",
        action="store_true",
        help="leave the server on the new model (default: restore the previous one)",
    )
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--remove", help="remove a preset id and exit")
    args = ap.parse_args()

    settings = get_settings()
    path = registry_path(settings)

    if args.list:
        return cmd_list(path)
    if args.remove:
        entries = load_registry(path)
        if all(entry.id != args.remove for entry in entries):
            print(f"[remove] no such preset: {args.remove}")
            return 1
        save_registry(path, remove_entry(entries, args.remove))
        log("remove", f"{args.remove} -> {path}")
        return 0

    if not args.id or not args.label:
        ap.error("--id and --label are required (or use --list / --remove)")

    ct2_dir = resolve_path(args.ct2_dir or f"models/{args.id}-ct2")
    try:
        entry = validate_entry(
            RegisteredModel(
                id=args.id,
                label=args.label,
                model=rel_to_root(ct2_dir) if args.kind == "breeze" else args.id,
                kind=args.kind,
                note=args.note,
            )
        )
    except RegistryError as exc:
        print(f"[abort] {exc}")
        return 1

    if args.adapter:
        if ct2_dir.exists():
            print(f"[abort] {ct2_dir} exists — pass --ct2-dir to register it, or pick a new --id")
            return 1
        code = run_merge(resolve_path(args.adapter), ct2_dir, args.base, args.quantization)
        if code != 0:
            print("[abort] merge/convert failed")
            return code

    missing = verify_ct2_dir(ct2_dir)
    if missing:
        print(f"[abort] {ct2_dir} is not a loadable CT2 model — missing {missing}")
        return 1
    log("verify", f"CT2 files present in {ct2_dir}")

    if not args.skip_smoke:
        sample = resolve_path(args.sample) if args.sample else None
        if sample is not None and not sample.is_file():
            print(f"[abort] sample not found: {sample}")
            return 1
        smoke_test(ct2_dir, args.device, sample)

    save_registry(path, upsert_entry(load_registry(path), entry))
    log("register", f"{entry.id} ({entry.label}) -> {entry.model} in {path}")

    if args.server:
        try:
            verify_on_server(args.server, entry, restore=not args.keep_active)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[abort] server at {args.server} unreachable: {exc}")
            return 1

    log("done", "the preset is live — reopen 模型與演算法 on the phone to see it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
