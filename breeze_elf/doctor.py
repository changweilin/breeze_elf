"""``breeze-elf doctor`` — one-shot environment self-check.

Digests the README's torch / CUDA / cuDNN install pitfalls into an actionable
report: it verifies the ASR runtime, GPU visibility (ctranslate2 for Whisper,
torch for the optional enhance/voice stages), the cuDNN DLL on Windows, the
opt-in extras, and whether each configured local model is actually on disk —
then prints a PASS/WARN/FAIL line per check with the exact fix.

Every probe is import-guarded and non-destructive: it never loads a model unless
``--load`` is passed, so it is safe to run on a broken environment (which is
precisely when it is useful).
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings

ROOT_DIR = Path(__file__).resolve().parent.parent

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
INFO = "INFO"

_ANSI = {
    OK: "\033[32m",
    WARN: "\033[33m",
    FAIL: "\033[31m",
    SKIP: "\033[90m",
    INFO: "\033[36m",
}
_RESET = "\033[0m"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""
    action: str = ""


def _module_version(dist_name: str) -> str | None:
    """The installed version of a *distribution* (e.g. ``faster-whisper``), or
    ``None`` when it is not installed. Resolves by distribution metadata, not by
    import, so hyphenated PyPI names work — the import module often differs from
    the package name (``faster-whisper`` → ``faster_whisper``)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - metadata lookup is best-effort
        return None


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _ctranslate2_cuda_count() -> int | None:
    """CUDA device count ctranslate2 (the Whisper runtime) can see, or ``None``
    when ctranslate2 is not importable."""
    if importlib.util.find_spec("ctranslate2") is None:
        return None
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # pragma: no cover - depends on local install
        return None


def check_python() -> Check:
    # The declared minimum is 3.10 (pyproject requires-python), so this is purely
    # informational — it reports the interpreter, it does not gate on it.
    version = platform.python_version()
    return Check("Python", OK, f"{version} on {platform.system()} {platform.release()}")


def check_asr_runtime(settings: Settings) -> Check:
    if settings.asr_provider == "mock":
        return Check("ASR runtime", SKIP, "BREEZE_ASR_PROVIDER=mock (no real ASR loaded)")
    fw = _module_version("faster-whisper")
    ct2 = _module_version("ctranslate2")
    if fw is None:
        return Check(
            "ASR runtime",
            FAIL,
            "faster-whisper is not installed",
            "Run `uv sync` (or set BREEZE_ASR_PROVIDER=mock for UI-only work).",
        )
    return Check("ASR runtime", OK, f"faster-whisper {fw}, ctranslate2 {ct2 or '?'}")


def check_gpu_ctranslate2(settings: Settings) -> Check:
    """Whether Whisper will run on the GPU. ctranslate2 is the only thing that
    matters for the base ASR path — torch is irrelevant to it."""
    if settings.asr_provider == "mock":
        return Check("ASR GPU (ctranslate2)", SKIP, "mock provider")
    wants_gpu = settings.asr_device in {"auto", "cuda"}
    count = _ctranslate2_cuda_count()
    if count is None:
        return Check(
            "ASR GPU (ctranslate2)", WARN, "ctranslate2 not importable - cannot probe CUDA"
        )
    if count > 0:
        return Check("ASR GPU (ctranslate2)", OK, f"{count} CUDA device(s) visible")
    if settings.asr_device == "cpu":
        return Check("ASR GPU (ctranslate2)", INFO, "BREEZE_ASR_DEVICE=cpu - GPU not requested")
    detail = "no CUDA device visible to ctranslate2"
    action = (
        "Whisper will fall back to CPU int8 (slow). Install the CUDA runtime "
        "(nvidia driver + CUDA/cuDNN) so ctranslate2 sees the GPU, or set "
        "BREEZE_ASR_DEVICE=cpu to silence this."
    )
    return Check("ASR GPU (ctranslate2)", WARN if wants_gpu else INFO, detail, action)


def check_cudnn(settings: Settings) -> Check:
    """The cuDNN DLL load test — the README's biggest Windows pitfall (Whisper on
    CUDA needs ``cudnn64_9.dll`` on the DLL path)."""
    if platform.system() != "Windows":
        return Check("cuDNN (Windows)", SKIP, "not Windows")
    if (
        settings.asr_provider == "mock"
        or settings.asr_device == "cpu"
        or (_ctranslate2_cuda_count() or 0) == 0
    ):
        return Check("cuDNN (Windows)", SKIP, "GPU not in use")
    import ctypes

    for dll in ("cudnn64_9.dll", "cudnn_ops64_9.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            return Check(
                "cuDNN (Windows)",
                WARN,
                f"{dll} could not be loaded",
                "Whisper-on-CUDA may fail. Ensure the cuDNN 9 DLLs are on PATH "
                "(a matching torch CUDA wheel ships them; see the README enhance note).",
            )
    return Check("cuDNN (Windows)", OK, "cudnn64_9.dll loadable")


def check_torch(settings: Settings) -> Check:
    """torch matters only for the optional enhance / OpenVoice stages, never for
    base ASR. Reports the README's `uv sync --extra enhance` CPU-torch trap."""
    needs_torch = (
        settings.enhance_live != "off"
        or settings.enhance_file != "off"
        or settings.voice_provider == "openvoice"
    )
    torch_version = _module_version("torch")
    if not needs_torch:
        status = INFO if torch_version else SKIP
        detail = f"torch {torch_version} (not required by current config)" if torch_version else (
            "not installed (not required by current config)"
        )
        return Check("torch (enhance/voice)", status, detail)
    if torch_version is None:
        return Check(
            "torch (enhance/voice)",
            FAIL,
            "enhance/openvoice is enabled but torch is not installed",
            "Install the CUDA wheel: `uv pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu124` - NOT `uv sync --extra enhance` "
            "(that pins the CPU-only torch and silently disables the GPU).",
        )
    cuda_ok = False
    torch_cuda = "?"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        torch_cuda = torch.version.cuda or "cpu-build"
    except Exception as exc:  # pragma: no cover - depends on local torch
        return Check("torch (enhance/voice)", WARN, f"torch {torch_version} import failed: {exc}")
    if settings.enhance_device != "cpu" and not cuda_ok:
        return Check(
            "torch (enhance/voice)",
            WARN,
            f"torch {torch_version} (CUDA build: {torch_cuda}) "
            "but torch.cuda.is_available() is False",
            "This is the CPU-only-torch trap. Reinstall from the CUDA wheel index "
            "(see README), or set BREEZE_ENHANCE_DEVICE=cpu.",
        )
    return Check("torch (enhance/voice)", OK, f"torch {torch_version}, CUDA {torch_cuda}")


def check_extras(settings: Settings) -> list[Check]:
    """Presence of the opt-in extras, only flagged when the config actually needs
    them (so a base install shows them as SKIP, not WARN)."""
    checks: list[Check] = []

    want_translate = settings.translate_provider == "nllb"
    spm = _module_version("sentencepiece")
    if want_translate and spm is None:
        checks.append(
            Check(
                "extra: sentencepiece",
                WARN,
                "BREEZE_TRANSLATE=nllb but sentencepiece is missing",
                "Install the translate extra: `uv sync --extra translate`.",
            )
        )
    else:
        checks.append(
            Check(
                "extra: sentencepiece",
                OK if spm else SKIP,
                spm or "not installed (translate off)",
            )
        )

    want_onnx = settings.diarize_enabled or settings.vad_detector == "silero"
    # Presence by import (matching diarize.py's find_spec), not by distribution
    # metadata: the onnxruntime-gpu wheel imports as module ``onnxruntime`` but is
    # distributed as ``onnxruntime-gpu``, so a metadata lookup would falsely miss it.
    onnx_present = importlib.util.find_spec("onnxruntime") is not None
    onnx_version = _module_version("onnxruntime") or _module_version("onnxruntime-gpu")
    if want_onnx and not onnx_present:
        checks.append(
            Check(
                "extra: onnxruntime",
                WARN,
                "diarization / silero VAD needs onnxruntime",
                "Install it: `uv sync --extra diarize` (also bundled with faster-whisper).",
            )
        )
    else:
        detail = onnx_version or ("present" if onnx_present else "not needed")
        checks.append(Check("extra: onnxruntime", OK if onnx_present else SKIP, detail))

    want_enhance = settings.enhance_live != "off" or settings.enhance_file != "off"
    if want_enhance:
        for pkg in ("deepfilternet", "demucs"):
            present = _module_version(pkg)
            checks.append(
                Check(
                    f"extra: {pkg}",
                    OK if present else WARN,
                    present or f"enhancement enabled but {pkg} missing",
                    "" if present else "Install the enhance extra (CUDA torch first - see README).",
                )
            )
    return checks


def check_models(settings: Settings) -> list[Check]:
    """Whether each configured local model directory / file is actually present,
    using the same availability logic the server uses at startup."""
    checks: list[Check] = []

    if settings.translate_provider == "nllb":
        from .translate import build_translator

        translator = build_translator(settings, base_dir=ROOT_DIR)
        model_dir = _resolve(settings.translate_model)
        if translator.available:
            checks.append(Check("model: NLLB translate", OK, str(model_dir)))
        else:
            checks.append(
                Check(
                    "model: NLLB translate",
                    WARN,
                    f"missing or incomplete at {model_dir}",
                    "Fetch the CT2-converted NLLB-200 model into that dir (offline-first; "
                    "see the README Translation section).",
                )
            )

    if settings.diarize_enabled:
        from .diarize import build_diarizer

        diarizer = build_diarizer(settings, base_dir=ROOT_DIR)
        model_path = _resolve(settings.diarize_model)
        if getattr(diarizer, "available", False):
            checks.append(Check("model: speaker embedding", OK, str(model_path)))
        else:
            checks.append(
                Check(
                    "model: speaker embedding",
                    WARN,
                    f"missing ONNX model at {model_path}",
                    "Supply a permissively-licensed speaker-embedding ONNX model "
                    "(see the README Diarization section).",
                )
            )

    # Deployed presets (tools/deploy_model.py). A registered model whose dir vanished
    # is silently hidden from the switcher — surface it here instead.
    from .model_registry import load_registry, registry_path, resolve_path, verify_ct2_dir

    presets_file = registry_path(settings)
    for entry in load_registry(presets_file):
        name = f"model: deployed {entry.id}"
        if entry.kind != "breeze":
            checks.append(Check(name, OK, f"{entry.label} -> {entry.model}"))
            continue
        model_dir = resolve_path(entry.model)
        missing = verify_ct2_dir(model_dir)
        if not missing:
            checks.append(Check(name, OK, f"{entry.label} -> {model_dir}"))
        else:
            checks.append(
                Check(
                    name,
                    WARN,
                    f"{model_dir} missing {', '.join(missing)} (hidden from the switcher)",
                    f"Re-run tools/deploy_model.py for this id, or remove it: "
                    f"deploy_model.py --remove {entry.id}",
                )
            )

    if settings.voice_provider == "openvoice":
        converter = _resolve(settings.voice_checkpoints_dir) / "converter" / "checkpoint.pth"
        if converter.is_file():
            checks.append(Check("model: OpenVoice converter", OK, str(converter)))
        else:
            checks.append(
                Check(
                    "model: OpenVoice converter",
                    WARN,
                    f"missing at {converter}",
                    "Download the OpenVoice v2 converter checkpoint "
                    "(see the README Voice section).",
                )
            )
    return checks


def check_model_load(settings: Settings) -> Check:
    """Actually load Whisper and report the device/compute type it resolved to.
    Slow and may download the model — only runs behind ``doctor --load``."""
    if settings.asr_provider == "mock":
        return Check("model load", SKIP, "mock provider")
    try:
        from .asr import FasterWhisperASR

        engine = FasterWhisperASR(settings.asr_model, settings.asr_device)
        engine.load()
    except Exception as exc:
        return Check(
            "model load",
            FAIL,
            f"{settings.asr_model!r} failed to load: {exc}",
            "See the ASR GPU / cuDNN checks above for the likely cause.",
        )
    return Check("model load", OK, f"{settings.asr_model} -> {engine.device}/{engine.compute_type}")


def collect_checks(settings: Settings, *, load_model: bool = False) -> list[Check]:
    checks: list[Check] = [
        check_python(),
        check_asr_runtime(settings),
        check_gpu_ctranslate2(settings),
        check_cudnn(settings),
        check_torch(settings),
    ]
    checks.extend(check_extras(settings))
    checks.extend(check_models(settings))
    if load_model:
        checks.append(check_model_load(settings))
    return checks


def _use_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _format_line(check: Check, color: bool) -> str:
    tag = f"[{check.status}]"
    if color:
        tag = f"{_ANSI.get(check.status, '')}{tag}{_RESET}"
    # ASCII-only scaffolding: doctor runs in exactly the legacy-codepage consoles
    # where a decorative em-dash would raise UnicodeEncodeError.
    line = f"  {tag:<7} {check.name}"
    if check.detail:
        line += f" - {check.detail}"
    if check.action:
        line += f"\n           -> {check.action}"
    return line


def run_doctor(load_model: bool = False) -> int:
    """Print the report; return an exit code (1 if any check FAILed, else 0)."""
    settings = get_settings()
    color = _use_color()
    checks = collect_checks(settings, load_model=load_model)

    print("Breeze Elf doctor")
    print(
        f"  config: model={settings.asr_model} device={settings.asr_device} "
        f"provider={settings.asr_provider} segmenter={settings.segmenter}/{settings.vad_detector}"
    )
    print()
    for check in checks:
        print(_format_line(check, color))

    fails = sum(1 for c in checks if c.status == FAIL)
    warns = sum(1 for c in checks if c.status == WARN)
    print()
    print(f"  {len(checks)} checks | {fails} fail | {warns} warn")
    return 1 if fails else 0
