import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_manifests as mm  # noqa: E402


class NegativeSplitTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(mm.negative_split(0), "test")
        self.assertEqual(mm.negative_split(mm.NEG_TEST - 1), "test")
        self.assertEqual(mm.negative_split(mm.NEG_TEST), "dev")
        self.assertEqual(mm.negative_split(mm.NEG_TEST + mm.NEG_DEV - 1), "dev")
        self.assertEqual(mm.negative_split(mm.NEG_TEST + mm.NEG_DEV), "train")


class ManifestBuildTests(unittest.TestCase):
    def _patch(self, name, value):
        original = getattr(mm, name)
        self.addCleanup(setattr, mm, name, original)
        setattr(mm, name, value)

    def _build(self, rows, out_dir):
        # No dataset on disk: stub the CV tsv readers and feed synthetic metadata rows.
        self._patch("read_tsv_stems", lambda path: set())
        self._patch("read_stem_to_client", lambda path: {})
        self._patch("_rows", lambda: rows)
        self._patch("OUT", out_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mm.main(), 0)

    @staticmethod
    def _read(out_dir, split):
        path = out_dir / f"{split}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def _rows_fixture(self):
        return [
            # Two curated negatives (one whitespace-only) — kept, target empty.
            {
                "file_name": "chunks/neg_001.wav",
                "transcription": "",
                "duration": "8",
                "language": "zh",
                "source_dataset_or_song_id": "neg_batch",
            },
            {
                "file_name": "chunks/neg_002.wav",
                "transcription": "  ",
                "duration": "8",
                "language": "zh",
                "source_dataset_or_song_id": "neg_batch",
            },
            # A normal nan clip (kept) and a NON-negative empty clip (dropped).
            {
                "file_name": "chunks/cv_common_voice_nan-tw_1_0000.wav",
                "transcription": "你好",
                "duration": "3",
                "language": "nan",
                "source_dataset_or_song_id": "cv_common_voice_nan-tw_1",
            },
            {
                "file_name": "chunks/cv_common_voice_nan-tw_9_0000.wav",
                "transcription": "",
                "duration": "3",
                "language": "nan",
                "source_dataset_or_song_id": "cv_common_voice_nan-tw_9",
            },
            {
                "file_name": "chunks/mir1k_alice_1.wav",
                "transcription": "歌詞",
                "duration": "10",
                "language": "zh",
                "source_dataset_or_song_id": "MIR-1K",
            },
            {
                "file_name": "chunks/jamendo_song1.wav",
                "transcription": "lyric",
                "duration": "10",
                "language": "en",
                "source_dataset_or_song_id": "jamendo_song1",
            },
            # Frozen speech-regression clip (會議錄音) — must be routed, not dropped.
            {
                "file_name": "chunks/speech_meeting1.wav",
                "transcription": "會議記錄",
                "duration": "12",
                "language": "zh",
                "source_dataset_or_song_id": "meeting",
            },
        ]

    def test_negatives_kept_and_marked(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._build(self._rows_fixture(), out)
            negatives = [r for r in self._read(out, "test") if r.get("negative")]
            self.assertEqual({r["id"] for r in negatives}, {"neg_001", "neg_002"})
            for rec in negatives:
                self.assertEqual(rec["text"], "")
                self.assertTrue(rec["negative"])
                self.assertEqual(rec["split"], "test")  # both ranks < NEG_TEST

    def test_speech_regression_routed_to_test(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._build(self._rows_fixture(), out)
            test_ids = {rec["id"] for rec in self._read(out, "test")}
            self.assertIn("speech_meeting1", test_ids)  # not dropped as unknown_prefix
            train_ids = {rec["id"] for rec in self._read(out, "train")}
            self.assertNotIn("speech_meeting1", train_ids)  # frozen: never trains

    def test_non_negative_empty_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._build(self._rows_fixture(), out)
            all_ids = {
                rec["id"]
                for split in ("train", "dev", "test")
                for rec in self._read(out, split)
            }
            self.assertNotIn("cv_common_voice_nan-tw_9_0000", all_ids)  # empty, not neg_
            self.assertIn("cv_common_voice_nan-tw_1_0000", all_ids)  # has text

    def test_normal_sources_unaffected_have_no_negative_flag(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._build(self._rows_fixture(), out)
            train_ids = {rec["id"] for rec in self._read(out, "train")}
            self.assertIn("cv_common_voice_nan-tw_1_0000", train_ids)
            self.assertIn("mir1k_alice_1", train_ids)
            for rec in self._read(out, "train"):
                self.assertNotIn("negative", rec)


if __name__ == "__main__":
    unittest.main()
