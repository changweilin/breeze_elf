import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from breeze_elf.storage import save_transcript


class StorageTests(unittest.TestCase):
    def test_save_transcript_writes_utf8_text_with_safe_filename(self):
        created_at = datetime(2026, 5, 26, 15, 30, 45, tzinfo=timezone.utc)

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
        created_at = datetime(2026, 5, 26, 15, 30, 45, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp:
            first = save_transcript("one", tmp, title="note", now=created_at)
            second = save_transcript("two", tmp, title="note", now=created_at)

            self.assertEqual(first.filename, "breeze-elf-20260526-153045-note.txt")
            self.assertEqual(second.filename, "breeze-elf-20260526-153045-note-2.txt")

    def test_save_transcript_rejects_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_transcript(" \n\t ", tmp)


if __name__ == "__main__":
    unittest.main()
