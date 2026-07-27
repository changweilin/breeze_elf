import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from breeze_elf.storage import save_transcript


class StorageTests(unittest.TestCase):
    def test_save_transcript_writes_utf8_text_with_safe_filename(self):
        created_at = datetime(2026, 5, 26, 15, 30, 45, tzinfo=UTC)

        with tempfile.TemporaryDirectory() as tmp:
            stored = save_transcript(
                "  Hello remote\nstorage  ",
                tmp,
                title="../../Meeting Notes",
                now=created_at,
            )

            self.assertEqual(
                stored.filename,
                "breeze-elf-20260526-153045-meeting-notes.txt",
            )
            self.assertEqual(stored.id, "breeze-elf-20260526-153045-meeting-notes")
            self.assertEqual(stored.created_at, "2026-05-26T15:30:45+00:00")
            self.assertEqual(
                (Path(tmp) / stored.filename).read_text(encoding="utf-8"),
                "Hello remote\nstorage\n",
            )

    def test_save_transcript_allocates_unique_filenames(self):
        created_at = datetime(2026, 5, 26, 15, 30, 45, tzinfo=UTC)

        with tempfile.TemporaryDirectory() as tmp:
            first = save_transcript("one", tmp, title="note", now=created_at)
            second = save_transcript("two", tmp, title="note", now=created_at)

            self.assertEqual(first.filename, "breeze-elf-20260526-153045-note.txt")
            self.assertEqual(second.filename, "breeze-elf-20260526-153045-note-2.txt")

    def test_save_transcript_rejects_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_transcript(" \n\t ", tmp)

    def test_save_transcript_bundles_structured_metadata_and_audio(self):
        created_at = datetime(2026, 5, 26, 15, 30, 45, tzinfo=UTC)
        structured = {
            "title": "note",
            "sampleRate": 16000,
            "blocks": [
                {
                    "text": "天氣",
                    "characters": [
                        {"char": "天", "startSeconds": 0.0, "jianpu": "3"},
                        {"char": "氣", "startSeconds": 0.2, "jianpu": "5"},
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            stored = save_transcript(
                "天氣很好",
                tmp,
                title="note",
                now=created_at,
                structured=structured,
                audio=b"RIFFfake-wav-bytes",
            )

            directory = Path(tmp)
            self.assertTrue(stored.filename.endswith(".txt"))
            self.assertEqual(stored.json_filename, f"{stored.id}.json")
            self.assertEqual(stored.audio_filename, f"{stored.id}.wav")

            self.assertEqual(
                (directory / stored.filename).read_text(encoding="utf-8"),
                "天氣很好\n",
            )
            self.assertEqual(
                (directory / stored.audio_filename).read_bytes(),
                b"RIFFfake-wav-bytes",
            )

            document = json.loads((directory / stored.json_filename).read_text(encoding="utf-8"))
            self.assertEqual(document["createdAt"], "2026-05-26T15:30:45+00:00")
            self.assertEqual(document["blocks"][0]["characters"][0]["jianpu"], "3")
            self.assertEqual(document["sampleRate"], 16000)

    def test_save_transcript_without_extras_writes_only_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            stored = save_transcript("只有文字", tmp)
            self.assertIsNone(stored.json_filename)
            self.assertIsNone(stored.audio_filename)
            self.assertIsNone(stored.csv_filename)
            self.assertEqual(list(Path(tmp).glob("*")), [Path(tmp) / stored.filename])

    def test_save_transcript_writes_pitch_csv_sibling(self):
        csv = "time_seconds,hz,intensity,text\n0.020,220.0,0.05,天\n0.040,,0.01,\n"
        with tempfile.TemporaryDirectory() as tmp:
            stored = save_transcript("天氣", tmp, title="note", pitch_csv=csv)
            self.assertEqual(stored.csv_filename, f"{stored.id}.csv")
            self.assertEqual(
                (Path(tmp) / stored.csv_filename).read_text(encoding="utf-8"),
                csv,
            )


if __name__ == "__main__":
    unittest.main()
