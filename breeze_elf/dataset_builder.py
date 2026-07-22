"""Offline lyric/singing dataset builder for ASR post-training.

Builds a Hugging Face ``audiofolder``-compatible dataset (``metadata.csv`` +
``chunks/*.wav``) from public corpora and custom songs::

    uv run python build_lyric_dataset.py --source_type local \
        --input song.mp3 --lyrics song.lrc --target_lang nan
    uv run python build_lyric_dataset.py --source_type youtube \
        --input https://youtu.be/XXXX --lyrics lyrics.txt --target_lang zh-TW
    uv run python build_lyric_dataset.py --source_type hf_dataset \
        --dataset_name jamendolyrics --target_lang en
    uv run python build_lyric_dataset.py --source_type hf_dataset \
        --dataset_name mir1k --input D:/data/MIR-1K
    uv run python build_lyric_dataset.py --source_type hf_dataset \
        --dataset_name common_voice --input D:/data/cv-corpus-17.0/nan-tw --target_lang nan

Pipeline per song: (optional) Demucs vocal separation -> per-language text
normalization -> timestamps (LRC / dataset annotations, else MMS forced
alignment) -> line merging into 1-20 s chunks -> quality filter -> export.

Dependencies: the ``enhance`` extra (CUDA torch + demucs, already installed in
.venv) plus the ``dataset`` extra. Install the latter with ``uv pip install``,
not ``uv sync``, so the manually-installed CUDA torch wheel is not replaced.

Dataset acquisition notes (verified 2026-07):
- JamendoLyrics MultiLang: auto-cloned from GitHub (audio + line/word timings
  included; languages en/fr/de/es — use ``--target_lang en`` here).
- MIR-1K: mirlab.org is unreachable; download manually (mirlab mirror or
  Kaggle "datasets/elemento/mir-1k") and pass the extracted dir via --input.
  Right channel = clean vocals, so no separation is run.
- Common Voice (nan-tw): Mozilla pulled Common Voice off HF in 2025-10; get
  the tar from https://datacollective.mozilla.org, extract, pass via --input.
- TAT-Vol1: license-gated (NARLabs) — obtain it yourself, then feed the wav +
  transcript pairs through ``--source_type local`` one file at a time (or ask
  for a bulk ingestor once the corpus layout is known).
- DALI v2: annotations are public but audio must be re-fetched from mostly
  dead YouTube links; use ``--source_type youtube`` per song instead.

Forced alignment uses torchaudio's MMS_FA bundle (romanized via uroman): solid
for en/zh-TW/ko. uroman reads Han characters with Mandarin pronunciations, so
kanji-heavy ja lyrics and Han-character nan lyrics align roughly — prefer .lrc
timestamps for those; kana-only ja and Tai-lo (Latin) nan lyrics align fine.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_lyric_dataset")

TARGET_LANGS = ("en", "zh-TW", "nan", "ja", "ko")

CSV_FIELDS = ["file_name", "transcription", "duration", "language", "source_dataset_or_song_id"]

# uroman ISO-639-3 hints for MMS romanization. nan falls back to Mandarin
# readings for Han characters (see module docstring).
_UROMAN_LCODE = {"en": "eng", "zh-TW": "cmn", "nan": "nan", "ja": "jpn", "ko": "kor"}

_JAMENDO_REPO = "https://github.com/f90/jamendolyrics.git"


@dataclass
class LyricLine:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    start: float
    end: float
    text: str


@dataclass
class PipelineConfig:
    output_dir: Path
    sample_rate: int = 16000
    separate: str = "auto"  # auto | always | never
    separator_model: str = "htdemucs_ft"
    device: str = "auto"
    min_chunk: float = 1.0
    max_chunk: float = 20.0
    hard_min: float = 0.8
    hard_max: float = 25.0
    min_rms: float = 5e-4
    pad: float = 0.1
    lowercase: bool = False
    nan_mapping: Path | None = None
    limit: int | None = None


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_SECTION_TAG_LINE = re.compile(r"^\s*(\[[^\]]{0,60}\]|【[^】]{0,60}】)\s*$")
_INLINE_TAG = re.compile(r"\[[^\]]{0,60}\]|【[^】]{0,60}】")
# "大武(Tāi-bú)"-style Tai-lo reading annotations (Common Voice nan-tw, MOE
# dictionaries): the word is sung/spoken once, so the transcript keeps Han only.
_NAN_READING = re.compile(r"\(([A-Za-zÀ-ʯ̀-ͯ̍'’\- ·.,!?]+)\)")
_HAN_CHAR = re.compile(r"[一-鿿]")
_FURIGANA = re.compile(
    r"([一-鿿々々]+)[((]([぀-ゟ゠-ヿー]+)[))]"
)
_WS = re.compile(r"\s+")

_opencc_converter = None
_uroman_instance = None


def _opencc_s2twp():
    global _opencc_converter
    if _opencc_converter is None:
        import opencc

        _opencc_converter = opencc.OpenCC("s2twp")
    return _opencc_converter


def _decode_text(raw: bytes) -> str:
    """MIR-1K era corpora ship Big5 text files; try UTF-8 first, then cp950."""
    for encoding in ("utf-8-sig", "cp950"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_nan_mapping(path: Path) -> list[tuple[str, str]]:
    """User-supplied from,to CSV (e.g. derived from ChhoeTaigi's MOE hanzi
    tables) applied longest-first, since no off-the-shelf MOE normalizer exists."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or row[0].lower() in ("from", "來源"):
                continue
            if row[0]:
                pairs.append((row[0], row[1]))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def strip_furigana(text: str) -> str:
    return _FURIGANA.sub(r"\1", text)


def normalize_text(
    text: str,
    lang: str,
    *,
    nan_mapping: list[tuple[str, str]] | None = None,
    lowercase: bool = False,
) -> str:
    """One lyric line -> training transcription. Empty string means 'drop'."""
    text = _INLINE_TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    if lang == "zh-TW":
        text = _opencc_s2twp().convert(text)
    elif lang == "nan":
        if _HAN_CHAR.search(_NAN_READING.sub("", text)):
            text = _NAN_READING.sub("", text)  # pure-Tai-lo lines keep their parens
        for src, dst in nan_mapping or []:
            text = text.replace(src, dst)
    elif lang == "ja":
        text = strip_furigana(text)
    elif lang == "en" and lowercase:
        text = text.lower()
    return _WS.sub(" ", text).strip()


def is_section_tag_line(line: str) -> bool:
    return bool(_SECTION_TAG_LINE.match(line))


def lang_joiner(lang: str) -> str:
    return "" if lang in ("zh-TW", "ja") else " "


# ---------------------------------------------------------------------------
# LRC parsing
# ---------------------------------------------------------------------------

_LRC_TIME = re.compile(r"\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]")
_LRC_META = re.compile(r"^\s*\[(ti|ar|al|by|re|ve|offset|length|tool)\b[^\]]*\]\s*$", re.I)


def parse_lrc(text: str) -> list[tuple[float, str]]:
    """(seconds, raw line text) pairs, sorted; supports repeated timestamps."""
    entries: list[tuple[float, str]] = []
    for raw in text.splitlines():
        if _LRC_META.match(raw):
            continue
        stamps = list(_LRC_TIME.finditer(raw))
        if not stamps:
            continue
        content = _LRC_TIME.sub("", raw).strip()
        for m in stamps:
            seconds = int(m.group(1)) * 60 + float(m.group(2).replace(":", "."))
            entries.append((seconds, content))
    entries.sort(key=lambda e: e[0])
    return entries


def lrc_to_lines(
    entries: list[tuple[float, str]], audio_duration: float, max_line: float = 25.0
) -> list[LyricLine]:
    lines: list[LyricLine] = []
    for i, (start, text) in enumerate(entries):
        if not text or start >= audio_duration:
            continue
        end = entries[i + 1][0] if i + 1 < len(entries) else audio_duration
        end = min(end, start + max_line, audio_duration)
        if end > start:
            lines.append(LyricLine(start, end, text))
    return lines


# ---------------------------------------------------------------------------
# Chunking + quality filter
# ---------------------------------------------------------------------------


def merge_lines_to_chunks(
    lines: list[LyricLine],
    min_dur: float = 1.0,
    max_dur: float = 20.0,
    max_gap: float = 2.0,
    joiner: str = " ",
) -> list[Chunk]:
    """Greedy merge of consecutive short lines so chunks land in [min, max]."""
    chunks: list[Chunk] = []
    group: list[LyricLine] = []

    def flush() -> None:
        if not group:
            return
        text = joiner.join(line.text for line in group).strip()
        chunks.append(Chunk(group[0].start, group[-1].end, text))
        group.clear()

    for line in lines:
        if not line.text.strip():
            continue
        if group:
            duration = group[-1].end - group[0].start
            gap = line.start - group[-1].end
            if (
                duration < min_dur
                and gap <= max_gap
                and (line.end - group[0].start) <= max_dur
            ):
                group.append(line)
                continue
            flush()
        group.append(line)
    flush()
    return chunks


def passes_quality(samples: np.ndarray, sr: int, text: str, cfg: PipelineConfig) -> bool:
    duration = len(samples) / sr
    if duration < cfg.hard_min or duration > cfg.hard_max:
        return False
    if not text.strip():
        return False
    from breeze_elf.audio import calculate_rms

    return calculate_rms(samples) >= cfg.min_rms


# ---------------------------------------------------------------------------
# Audio IO
# ---------------------------------------------------------------------------


def resample(samples: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr:
        return samples
    try:
        import torch
        import torchaudio

        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
            return torchaudio.functional.resample(tensor, orig_sr, new_sr).numpy()
    except Exception:  # torch-less fallback: linear interp is fine for speech SRs
        count = int(round(len(samples) * new_sr / orig_sr))
        old_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=count, endpoint=False)
        return np.interp(new_x, old_x, samples).astype(np.float32)


def load_audio(path: Path, mono: bool = True) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    samples = data.mean(axis=1) if mono else data
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


def _cuda_gc() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Vocal separation (reuses the server's DemucsSeparator)
# ---------------------------------------------------------------------------


class VocalSeparator:
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._separator = None

    def separate(self, samples: np.ndarray, sr: int) -> np.ndarray:
        if self._separator is None:
            from breeze_elf.enhance import DemucsSeparator

            self._separator = DemucsSeparator(self._device, model_name=self._model_name)
            if not self._separator.available:
                raise RuntimeError(
                    "demucs/torch unavailable — install the 'enhance' extra "
                    "or pass --separate never"
                )
        return self._separator.separate_vocals(samples, sr)


# ---------------------------------------------------------------------------
# MMS forced alignment (raw-text lyrics only)
# ---------------------------------------------------------------------------


class MmsForcedAligner:
    """torchaudio MMS_FA: romanize each line with uroman, align the whole song
    once, then map word spans back to line time ranges. Emission is computed in
    30 s windows so a full song fits on the RTX 3060."""

    SAMPLE_RATE = 16000
    WINDOW_SECONDS = 30.0

    def __init__(self, device: str = "auto") -> None:
        self._device_pref = device
        self._model = None
        self._tokenizer = None
        self._aligner = None
        self.device = "unloaded"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchaudio

        want_cuda = self._device_pref in ("auto", "cuda") and torch.cuda.is_available()
        self.device = "cuda" if want_cuda else "cpu"
        bundle = torchaudio.pipelines.MMS_FA
        self._model = bundle.get_model(with_star=False).to(self.device).eval()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()
        logger.info("MMS_FA alignment model loaded on %s", self.device)

    def _romanize_words(self, text: str, lang: str) -> list[str]:
        global _uroman_instance
        if _uroman_instance is None:
            import uroman

            _uroman_instance = uroman.Uroman()
        romanized = _uroman_instance.romanize_string(text, lcode=_UROMAN_LCODE.get(lang))
        words = (re.sub(r"[^a-z']", "", word) for word in romanized.lower().split())
        return [w for w in words if w]

    def align_lines(
        self, samples_16k: np.ndarray, line_texts: list[str], lang: str
    ) -> list[tuple[float, float] | None] | None:
        """Per-line (start, end) seconds; None entries for unalignable lines;
        overall None when alignment itself failed (caller should skip the song)."""
        try:
            self._load()
            import torch

            words: list[str] = []
            line_word_spans: list[tuple[int, int]] = []
            for text in line_texts:
                line_words = self._romanize_words(text, lang)
                line_word_spans.append((len(words), len(words) + len(line_words)))
                words.extend(line_words)
            if not words:
                return [None] * len(line_texts)

            wav = torch.from_numpy(
                np.ascontiguousarray(samples_16k, dtype=np.float32)
            ).unsqueeze(0)
            window = int(self.WINDOW_SECONDS * self.SAMPLE_RATE)
            emissions = []
            with torch.inference_mode():
                for offset in range(0, wav.size(1), window):
                    piece = wav[:, offset : offset + window].to(self.device)
                    if piece.size(1) < 640:  # sub-frame tail; timing error < 40 ms
                        break
                    emission, _ = self._model(piece)
                    emissions.append(emission[0].cpu())
                if not emissions:
                    return None
                full_emission = torch.cat(emissions, dim=0)
                token_spans = self._aligner(full_emission, self._tokenizer(words))

            seconds_per_frame = wav.size(1) / full_emission.size(0) / self.SAMPLE_RATE
            results: list[tuple[float, float] | None] = []
            for first, last in line_word_spans:
                if last <= first:
                    results.append(None)
                    continue
                start = token_spans[first][0].start * seconds_per_frame
                end = token_spans[last - 1][-1].end * seconds_per_frame
                results.append((start, end))
            return results
        except Exception as exc:
            logger.warning("forced alignment failed: %s", exc)
            return None
        finally:
            _cuda_gc()


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "song"


_CHUNK_NAME = re.compile(r"^chunks/(.+)_\d{4}\.wav$")


class DatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.chunk_dir = output_dir / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = output_dir / "metadata.csv"
        self._existing: set[str] = set()
        self._sources: set[str] = set()
        if self.csv_path.exists():
            with open(self.csv_path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    name = row.get("file_name")
                    if not name:
                        continue
                    self._existing.add(name)
                    match = _CHUNK_NAME.match(name)
                    if match:
                        self._sources.add(match.group(1))
        self.written = 0
        self.skipped = 0

    def add_chunk(
        self, samples: np.ndarray, sr: int, text: str, lang: str, source_id: str, index: int
    ) -> bool:
        import soundfile as sf

        rel_name = f"chunks/{sanitize_id(source_id)}_{index:04d}.wav"
        if rel_name in self._existing:
            self.skipped += 1
            return False
        sf.write(str(self.output_dir / rel_name), samples, sr, subtype="PCM_16")
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "file_name": rel_name,
                    "transcription": text,
                    "duration": round(len(samples) / sr, 3),
                    "language": lang,
                    "source_dataset_or_song_id": source_id,
                }
            )
        self._existing.add(rel_name)
        self._sources.add(sanitize_id(source_id))
        self.written += 1
        return True

    def has_source(self, source_id: str) -> bool:
        """True when any chunk of this song is already exported — lets dataset
        re-runs skip the song before paying for separation again."""
        return sanitize_id(source_id) in self._sources


# ---------------------------------------------------------------------------
# Core per-song pipeline
# ---------------------------------------------------------------------------


def _should_separate(mode: str, is_mix: bool) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return is_mix


def process_song(
    writer: DatasetWriter,
    samples: np.ndarray,
    sr: int,
    *,
    lang: str,
    source_id: str,
    cfg: PipelineConfig,
    timed_lines: list[LyricLine] | None = None,
    raw_lines: list[str] | None = None,
    is_mix: bool = True,
    separator: VocalSeparator | None = None,
    aligner: MmsForcedAligner | None = None,
    nan_mapping: list[tuple[str, str]] | None = None,
) -> int:
    """Returns the number of chunks written for this song."""
    if samples.size == 0:
        logger.warning("%s: empty audio, skipped", source_id)
        return 0

    if _should_separate(cfg.separate, is_mix):
        if separator is None:
            raise RuntimeError("separation requested but no separator configured")
        logger.info("%s: separating vocals (%s)…", source_id, cfg.separator_model)
        samples = separator.separate(samples, sr)
        _cuda_gc()

    out = resample(samples, sr, cfg.sample_rate)
    out_sr = cfg.sample_rate
    duration = len(out) / out_sr

    def norm(text: str) -> str:
        return normalize_text(text, lang, nan_mapping=nan_mapping, lowercase=cfg.lowercase)

    if timed_lines is not None:
        lines = [
            LyricLine(line.start, line.end, norm(line.text))
            for line in timed_lines
            if norm(line.text)
        ]
    else:
        texts = [
            norm(line)
            for line in (raw_lines or [])
            if line.strip() and not is_section_tag_line(line)
        ]
        texts = [t for t in texts if t]
        if not texts:
            logger.warning("%s: no usable lyric lines, skipped", source_id)
            return 0
        if aligner is None:
            raise RuntimeError("raw-text lyrics need forced alignment but no aligner is set")
        align_input = out if out_sr == MmsForcedAligner.SAMPLE_RATE else resample(
            out, out_sr, MmsForcedAligner.SAMPLE_RATE
        )
        spans = aligner.align_lines(align_input, texts, lang)
        if spans is None:
            logger.warning("%s: alignment failed, song skipped", source_id)
            return 0
        lines = [
            LyricLine(span[0], span[1], text)
            for text, span in zip(texts, spans)
            if span is not None and span[1] > span[0]
        ]

    chunks = merge_lines_to_chunks(
        lines, cfg.min_chunk, cfg.max_chunk, joiner=lang_joiner(lang)
    )
    written = 0
    for index, chunk in enumerate(chunks):
        start = max(0.0, chunk.start - cfg.pad)
        end = min(duration, chunk.end + cfg.pad)
        piece = out[int(start * out_sr) : int(end * out_sr)]
        if not passes_quality(piece, out_sr, chunk.text, cfg):
            continue
        if writer.add_chunk(piece, out_sr, chunk.text, lang, source_id, index):
            written += 1
    logger.info("%s: %d/%d chunks kept", source_id, written, len(chunks))
    return written


# ---------------------------------------------------------------------------
# Sources: custom (local / youtube)
# ---------------------------------------------------------------------------


def _read_lyrics(
    path: Path, audio_duration: float
) -> tuple[list[LyricLine] | None, list[str] | None]:
    """(timed_lines, raw_lines) — exactly one is non-None."""
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".lrc" or _LRC_TIME.search(text):
        return lrc_to_lines(parse_lrc(text), audio_duration), None
    return None, text.splitlines()


def ingest_local_file(
    audio_path: Path,
    lyrics_path: Path,
    lang: str,
    song_id: str | None,
    writer: DatasetWriter,
    cfg: PipelineConfig,
    separator: VocalSeparator,
    aligner: MmsForcedAligner,
    nan_mapping: list[tuple[str, str]] | None,
) -> int:
    samples, sr = load_audio(audio_path)
    timed, raw = _read_lyrics(lyrics_path, len(samples) / sr)
    return process_song(
        writer,
        samples,
        sr,
        lang=lang,
        source_id=song_id or audio_path.stem,
        cfg=cfg,
        timed_lines=timed,
        raw_lines=raw,
        is_mix=True,
        separator=separator,
        aligner=aligner,
        nan_mapping=nan_mapping,
    )


def _ffmpeg_location() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # bundled static ffmpeg — this box has none on PATH

    return imageio_ffmpeg.get_ffmpeg_exe()


def download_youtube_audio(url: str, workdir: Path) -> Path:
    import yt_dlp

    workdir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "ffmpeg_location": _ffmpeg_location(),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    wav_path = workdir / f"{info['id']}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"yt-dlp did not produce {wav_path}")
    return wav_path


# ---------------------------------------------------------------------------
# Sources: public datasets
# ---------------------------------------------------------------------------

_JAMENDO_LANG = {"en": "english", "fr": "french", "de": "german", "es": "spanish"}


def _jamendo_language_matches(row_value: str, target_lang: str) -> bool:
    value = row_value.strip().lower()
    want = _JAMENDO_LANG.get(target_lang, target_lang.lower())
    return value in (want, target_lang.lower(), target_lang.split("-")[0].lower())


def _parse_jamendo_lines(path: Path) -> list[LyricLine]:
    lines: list[LyricLine] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                start, end = float(row[0]), float(row[1])
            except ValueError:
                continue  # header row
            text = row[2] if len(row) > 2 else ""
            if text.strip():
                lines.append(LyricLine(start, end, text))
    return lines


def ingest_jamendolyrics(
    root: Path | None,
    target_lang: str,
    writer: DatasetWriter,
    cfg: PipelineConfig,
    separator: VocalSeparator,
    nan_mapping: list[tuple[str, str]] | None,
) -> int:
    if target_lang not in _JAMENDO_LANG:
        raise SystemExit(
            f"JamendoLyrics covers en/fr/de/es only — no '{target_lang}' songs there."
        )
    if root is None:
        root = cfg.output_dir / "_cache" / "jamendolyrics"
    if not (root / "JamendoLyrics.csv").exists():
        logger.info("cloning JamendoLyrics into %s …", root)
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", _JAMENDO_REPO, str(root)], check=True
        )

    with open(root / "JamendoLyrics.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("JamendoLyrics.csv is empty — clone looks broken")
    lang_key = next((k for k in rows[0] if k and "lang" in k.lower()), None)
    file_key = next(
        (k for k in rows[0] if k and rows[0][k] and str(rows[0][k]).lower().endswith(".mp3")),
        None,
    )
    if lang_key is None:
        raise SystemExit(f"no language column in JamendoLyrics.csv: {list(rows[0])}")

    total = 0
    processed = 0
    for row in rows:
        if not _jamendo_language_matches(row.get(lang_key, ""), target_lang):
            continue
        if cfg.limit is not None and processed >= cfg.limit:
            break
        stem = (
            Path(row[file_key]).stem
            if file_key and row.get(file_key)
            else Path(next((v for v in row.values() if str(v).endswith(".mp3")), "")).stem
        )
        if not stem:
            continue
        mp3 = root / "mp3" / f"{stem}.mp3"
        lines_csv = root / "annotations" / "lines" / f"{stem}.csv"
        if not mp3.exists() or not lines_csv.exists():
            logger.warning("jamendo %s: missing mp3 or line annotations, skipped", stem)
            continue
        timed = _parse_jamendo_lines(lines_csv)
        if not timed:
            logger.warning("jamendo %s: no parsable lines, skipped", stem)
            continue
        if writer.has_source(f"jamendo_{stem}"):
            logger.info("jamendo %s: already exported, skipped", stem)
            continue
        processed += 1
        try:
            samples, sr = load_audio(mp3)
            total += process_song(
                writer,
                samples,
                sr,
                lang=target_lang,
                source_id=f"jamendo_{stem}",
                cfg=cfg,
                timed_lines=timed,
                is_mix=True,
                separator=separator,
                nan_mapping=nan_mapping,
            )
        except Exception as exc:
            logger.error("jamendo %s failed: %s", stem, exc)
    return total


def ingest_mir1k(
    root: Path | None,
    writer: DatasetWriter,
    cfg: PipelineConfig,
    nan_mapping: list[tuple[str, str]] | None,
) -> int:
    if root is None or not root.exists():
        raise SystemExit(
            "MIR-1K must be downloaded manually (mirlab.org is unreachable; try the "
            "Kaggle mirror 'elemento/mir-1k'), then pass the extracted folder via "
            "--input. Expected layout: <root>/Wavfile/*.wav (+ optional Lyrics/*.txt)."
        )
    wav_dir = next(
        (d for d in (root / "Wavfile", root / "MIR-1K" / "Wavfile", root) if d.is_dir()),
        None,
    )
    wavs = sorted(wav_dir.glob("*.wav")) if wav_dir else []
    if not wavs:
        raise SystemExit(f"no .wav files under {root} — is this the extracted MIR-1K?")
    lyrics_dirs = [d for d in (root / "Lyrics", wav_dir.parent / "Lyrics") if d.is_dir()]

    import soundfile as sf

    total = 0
    processed = 0
    missing_lyrics = 0
    for wav in wavs:
        if cfg.limit is not None and processed >= cfg.limit:
            break
        # Per-clip text only — song-level lyrics can't be trusted for a 4-13 s clip.
        lyric_text = None
        for directory in lyrics_dirs:
            candidate = directory / f"{wav.stem}.txt"
            if candidate.exists():
                lyric_text = _decode_text(candidate.read_bytes())
                break
        if lyric_text is None or not lyric_text.strip():
            missing_lyrics += 1
            continue
        if writer.has_source(f"mir1k_{wav.stem}"):
            continue
        processed += 1
        try:
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            vocals = data[:, 1] if data.shape[1] >= 2 else data[:, 0]  # R = clean vocals
            text = " ".join(lyric_text.replace("_", " ").split())  # _ delimits phrases
            timed = [LyricLine(0.0, len(vocals) / sr, text)]
            total += process_song(
                writer,
                np.ascontiguousarray(vocals),
                int(sr),
                lang="zh-TW",
                source_id=f"mir1k_{wav.stem}",
                cfg=cfg,
                timed_lines=timed,
                is_mix=False,
                nan_mapping=nan_mapping,
            )
        except Exception as exc:
            logger.error("mir1k %s failed: %s", wav.stem, exc)
    if missing_lyrics:
        logger.warning(
            "mir1k: %d clips skipped (no per-clip lyric .txt found)", missing_lyrics
        )
    return total


def ingest_common_voice_dir(
    root: Path | None,
    lang: str,
    writer: DatasetWriter,
    cfg: PipelineConfig,
    nan_mapping: list[tuple[str, str]] | None,
) -> int:
    if root is None or not root.exists():
        raise SystemExit(
            "Common Voice left Hugging Face in 2025-10; download the language tar from "
            "https://datacollective.mozilla.org (e.g. nan-tw), extract it, and pass the "
            "folder containing validated.tsv + clips/ via --input."
        )
    if (root / "validated.tsv").exists():
        tsvs = [root / "validated.tsv"]
    else:  # newer Data Collective drops ship pre-split train/dev/test instead
        tsvs = [root / n for n in ("train.tsv", "dev.tsv", "test.tsv") if (root / n).exists()]
    clips = root / "clips"
    if not tsvs or not clips.is_dir():
        raise SystemExit(f"{root} lacks validated/train tsv + clips/ — not a Common Voice extract")

    rows: list[dict] = []
    for tsv in tsvs:
        with open(tsv, encoding="utf-8", newline="") as fh:
            rows.extend(csv.DictReader(fh, delimiter="\t"))

    total = 0
    processed = 0
    for row in rows:
        if cfg.limit is not None and processed >= cfg.limit:
            break
        clip = clips / row.get("path", "")
        sentence = (row.get("sentence") or "").strip()
        if not sentence or not clip.exists():
            continue
        if writer.has_source(f"cv_{clip.stem}"):
            continue
        processed += 1
        try:
            samples, sr = load_audio(clip)
            timed = [LyricLine(0.0, len(samples) / sr, sentence)]
            total += process_song(
                writer,
                samples,
                sr,
                lang=lang,
                source_id=f"cv_{clip.stem}",
                cfg=cfg,
                timed_lines=timed,
                is_mix=False,  # read speech — never separate
                nan_mapping=nan_mapping,
            )
        except Exception as exc:
            logger.error("common_voice %s failed: %s", clip.name, exc)
    return total


_DATASET_ALIASES = {
    "jamendolyrics": "jamendo",
    "jamendo": "jamendo",
    "mir1k": "mir1k",
    "mir-1k": "mir1k",
    "common_voice": "common_voice",
    "common-voice": "common_voice",
    "cv": "common_voice",
    "dali": "dali",
    "dali-v2": "dali",
    "musdb18": "musdb18",
    "musdb18-lyrics": "musdb18",
    "tat": "tat",
    "tat-vol1": "tat",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_lyric_dataset",
        description="Build an HF audiofolder dataset (metadata.csv + chunks/) "
        "from public corpora, YouTube, or local song files.",
    )
    parser.add_argument("--source_type", required=True, choices=["hf_dataset", "youtube", "local"])
    parser.add_argument("--dataset_name", help="jamendolyrics | mir1k | common_voice | …")
    parser.add_argument("--input", help="YouTube URL, audio file, or dataset root dir")
    parser.add_argument("--lyrics", help=".lrc (pre-timed) or plain .txt lyrics file")
    parser.add_argument("--target_lang", choices=list(TARGET_LANGS) + ["fr", "de", "es"])
    parser.add_argument("--song_id", help="stable id for --source_type youtube/local")
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--separate", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--separator_model", default="htdemucs_ft")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--min_chunk", type=float, default=1.0)
    parser.add_argument("--max_chunk", type=float, default=20.0)
    parser.add_argument("--min_rms", type=float, default=5e-4)
    parser.add_argument("--nan_mapping", help="from,to CSV for MOE hanzi normalization")
    parser.add_argument("--lowercase", action="store_true", help="lower-case English text")
    parser.add_argument("--limit", type=int, help="max songs/clips to process")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    cfg = PipelineConfig(
        output_dir=Path(args.output_dir),
        sample_rate=args.sample_rate,
        separate=args.separate,
        separator_model=args.separator_model,
        device=args.device,
        min_chunk=args.min_chunk,
        max_chunk=args.max_chunk,
        min_rms=args.min_rms,
        lowercase=args.lowercase,
        limit=args.limit,
    )
    nan_mapping = load_nan_mapping(Path(args.nan_mapping)) if args.nan_mapping else None
    writer = DatasetWriter(cfg.output_dir)
    separator = VocalSeparator(cfg.separator_model, cfg.device)
    aligner = MmsForcedAligner(cfg.device)

    if args.source_type in ("local", "youtube"):
        if not args.input or not args.lyrics:
            raise SystemExit("--input and --lyrics are required for local/youtube sources")
        if not args.target_lang:
            raise SystemExit("--target_lang is required for local/youtube sources")
        if args.source_type == "youtube":
            audio_path = download_youtube_audio(args.input, cfg.output_dir / "_cache" / "yt")
        else:
            audio_path = Path(args.input)
            if not audio_path.exists():
                raise SystemExit(f"audio file not found: {audio_path}")
        written = ingest_local_file(
            audio_path,
            Path(args.lyrics),
            args.target_lang,
            args.song_id,
            writer,
            cfg,
            separator,
            aligner,
            nan_mapping,
        )
    else:
        if not args.dataset_name:
            raise SystemExit("--dataset_name is required for --source_type hf_dataset")
        kind = _DATASET_ALIASES.get(args.dataset_name.lower())
        root = Path(args.input) if args.input else None
        if kind == "jamendo":
            written = ingest_jamendolyrics(
                root, args.target_lang or "en", writer, cfg, separator, nan_mapping
            )
        elif kind == "mir1k":
            written = ingest_mir1k(root, writer, cfg, nan_mapping)
        elif kind == "common_voice":
            written = ingest_common_voice_dir(
                root, args.target_lang or "nan", writer, cfg, nan_mapping
            )
        elif kind == "dali":
            raise SystemExit(
                "DALI v2 ships annotations only; its audio must be re-fetched from "
                "YouTube ids that are now mostly dead. Use --source_type youtube per "
                "song you can still reach instead."
            )
        elif kind == "musdb18":
            raise SystemExit(
                "MUSDB18 audio is access-gated on Zenodo (request required) and stored "
                "as .mp4 stems. Decode the vocal stems to wav yourself, then feed each "
                "through --source_type local with the lyrics-extension timings."
            )
        elif kind == "tat":
            raise SystemExit(
                "TAT-Vol1 is license-gated (NARLabs) and cannot be auto-downloaded. "
                "Once obtained, ingest wav+transcript pairs via --source_type local."
            )
        else:
            raise SystemExit(f"unknown dataset '{args.dataset_name}'")

    logger.info(
        "done: %d chunks written, %d duplicates skipped -> %s",
        writer.written,
        writer.skipped,
        writer.csv_path,
    )
    return 0 if written > 0 else 1
