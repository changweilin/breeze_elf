"""Manifest building, with the C 層負樣本 that used to be thrown away.

`tools/` is a script directory, not a package, so it is put on the path the same
way `tools/rescore.py` does it.
"""

import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import make_manifests  # noqa: E402

from breeze_elf.dataset_builder import CSV_FIELDS  # noqa: E402

CV_ROWS = [
    # (stem, client, text) — client "spk_test" is the held-out test voice.
    ("common_voice_nan-tw_1", "spk_test", "公館(Kong-kuán)"),
    ("common_voice_nan-tw_2", "spk_train", "食飽未"),
    ("common_voice_nan-tw_3", "spk_train", ""),  # metadata hole, not an interlude
]


class ManifestTests(unittest.TestCase):
    def build(self, extra_rows=()):
        """Run make_manifests against a throwaway dataset; return the split rows."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            nan = dataset / "nan-tw"
            nan.mkdir(parents=True)

            def write_tsv(path, stems):
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    writer = csv.writer(fh, delimiter="\t")
                    writer.writerow(["client_id", "path"])
                    for stem, client in stems:
                        writer.writerow([client, f"{stem}.mp3"])

            write_tsv(nan / "test.tsv", [(CV_ROWS[0][0], CV_ROWS[0][1])])
            write_tsv(nan / "validated.tsv", [(stem, client) for stem, client, _ in CV_ROWS])

            rows = [
                {
                    "file_name": f"chunks/cv_{stem}_0000.wav",
                    "transcription": text,
                    "duration": "3.0",
                    "language": "nan",
                    "source_dataset_or_song_id": "common_voice",
                }
                for stem, _client, text in CV_ROWS
            ] + list(extra_rows)
            with open(dataset / "metadata.csv", "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            original = (make_manifests.DATASET, make_manifests.NAN, make_manifests.OUT)
            make_manifests.DATASET = dataset
            make_manifests.NAN = nan
            make_manifests.OUT = dataset / "manifests"
            make_manifests._ROWS_CACHE = None
            try:
                with redirect_stdout(StringIO()) as out:
                    make_manifests.main()
                splits = {
                    split: [
                        json.loads(line)
                        for line in (make_manifests.OUT / f"{split}.jsonl").read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    for split in ("train", "dev", "test")
                }
                return splits, out.getvalue()
            finally:
                make_manifests.DATASET, make_manifests.NAN, make_manifests.OUT = original
                make_manifests._ROWS_CACHE = None

    def jamendo_row(self, song, index, text):
        return {
            "file_name": f"chunks/jamendo_{song}_{index:04d}.wav",
            "transcription": text,
            "duration": "6.0",
            "language": "en",
            "source_dataset_or_song_id": song,
        }

    def test_lyric_rows_are_labelled_and_gloss_stripped(self):
        splits, _ = self.build()
        test_row = splits["test"][0]
        self.assertEqual(test_row["layer"], "lyric")
        self.assertEqual(test_row["text"], "公館")

    def test_empty_common_voice_row_is_still_dropped(self):
        # Nothing musical is happening in it — there is no "stay silent over the
        # interlude" lesson in a hole in Common Voice's metadata.
        splits, output = self.build()
        ids = [rec["id"] for rows in splits.values() for rec in rows]
        self.assertNotIn("cv_common_voice_nan-tw_3_0000", ids)
        self.assertIn("empty_text", output)

    def test_instrumental_gap_becomes_a_negative_instead_of_being_dropped(self):
        extra = [
            self.jamendo_row("song_a", 0, "first line"),
            self.jamendo_row("song_a", 1, ""),  # 間奏
        ]
        splits, output = self.build(extra)
        rows = [
            rec
            for group in splits.values()
            for rec in group
            if rec["id"].startswith("jamendo")
        ]
        by_layer = {rec["layer"] for rec in rows}
        self.assertEqual(by_layer, {"lyric", "negative"})
        negative = next(rec for rec in rows if rec["layer"] == "negative")
        self.assertEqual(negative["text"], "")
        # It lands in its song's split, so a negative never leaks across the boundary.
        lyric = next(rec for rec in rows if rec["layer"] == "lyric")
        self.assertEqual(negative["split"], lyric["split"])
        self.assertIn("負樣本", output)

    def test_report_flags_a_dataset_with_no_negatives_at_all(self):
        _splits, output = self.build()
        self.assertIn("none — rebuild mixed songs", output)


if __name__ == "__main__":
    unittest.main()
