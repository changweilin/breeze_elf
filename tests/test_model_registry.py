import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from breeze_elf.asr_models import deployed_asr_models, find_asr_model, list_asr_models
from breeze_elf.config import get_settings
from breeze_elf.model_registry import (
    RegisteredModel,
    RegistryError,
    load_registry,
    registry_path,
    remove_entry,
    save_registry,
    upsert_entry,
    validate_entry,
    verify_ct2_dir,
)


def _entry(preset_id="breeze-nan", label="台語強化", model="models/nan-ct2", **kwargs):
    return RegisteredModel(id=preset_id, label=label, model=model, **kwargs)


class RegistryFileTests(unittest.TestCase):
    def test_missing_file_is_empty_registry(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_registry(Path(d) / "presets.json"), [])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "presets.json"
            entries = [_entry(note="LoRA r16")]
            save_registry(path, entries)
            self.assertEqual(load_registry(path), entries)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 1)

    def test_malformed_file_degrades_to_empty(self):
        # The switcher must survive a hand-edited / half-written registry.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "presets.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertLogs("breeze_elf.model_registry", level="WARNING"):
                self.assertEqual(load_registry(path), [])

    def test_invalid_entries_are_dropped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "presets.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "presets": [
                            {"id": "Bad Id", "label": "x", "model": "m"},
                            {"id": "no-label", "model": "m"},
                            {"id": "good", "label": "Good", "model": "models/good-ct2"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertLogs("breeze_elf.model_registry", level="WARNING"):
                entries = load_registry(path)
            self.assertEqual([e.id for e in entries], ["good"])

    def test_duplicate_ids_keep_the_first(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "presets.json"
            path.write_text(
                json.dumps(
                    {
                        "presets": [
                            {"id": "dup", "label": "First", "model": "a"},
                            {"id": "dup", "label": "Second", "model": "b"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            entries = load_registry(path)
            self.assertEqual([(e.id, e.label) for e in entries], [("dup", "First")])


class RegistryMutationTests(unittest.TestCase):
    def test_upsert_replaces_in_place(self):
        entries = [_entry("a", "A", "models/a"), _entry("b", "B", "models/b")]
        updated = upsert_entry(entries, _entry("a", "A v2", "models/a2"))
        self.assertEqual([e.id for e in updated], ["a", "b"])
        self.assertEqual(updated[0].label, "A v2")

    def test_upsert_appends_new(self):
        updated = upsert_entry([_entry("a", "A", "models/a")], _entry("c", "C", "models/c"))
        self.assertEqual([e.id for e in updated], ["a", "c"])

    def test_remove(self):
        entries = [_entry("a", "A", "models/a"), _entry("b", "B", "models/b")]
        self.assertEqual([e.id for e in remove_entry(entries, "a")], ["b"])

    def test_validate_rejects_bad_input(self):
        for bad in (
            _entry("Upper-Case"),
            _entry("has space"),
            _entry(label="  "),
            _entry(model=""),
            _entry(kind="magic"),
        ):
            with self.assertRaises(RegistryError):
                validate_entry(bad)


class CT2VerificationTests(unittest.TestCase):
    def test_missing_dir_reports_every_file(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(verify_ct2_dir(Path(d) / "nope")), 3)

    def test_partial_dir_reports_the_gap(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.bin").write_bytes(b"x")
            self.assertEqual(
                verify_ct2_dir(Path(d)), ["tokenizer.json", "preprocessor_config.json"]
            )

    def test_complete_dir_passes(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("model.bin", "tokenizer.json", "preprocessor_config.json"):
                (Path(d) / name).write_bytes(b"x")
            self.assertEqual(verify_ct2_dir(Path(d)), [])


class DeployedPresetsInSwitcherTests(unittest.TestCase):
    """The deploy pipeline's payoff: a registered model shows up as a switchable
    preset with no code change."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / "presets.json"
        self.model_dir = self.root / "deployed-ct2"
        self.model_dir.mkdir()
        self.settings = replace(
            get_settings(),
            asr_breeze_nan_model="models/__no_such_nan_ct2__",
            asr_presets_file=str(self.registry),
        )

    def _register(self, **kwargs):
        save_registry(registry_path(self.settings), [_entry(**kwargs)])

    def test_registered_model_becomes_a_preset(self):
        self._register(preset_id="nan-v2", label="台語 v2", model=str(self.model_dir))
        opts = list_asr_models(self.settings, "medium")
        ids = [o.id for o in opts]
        self.assertEqual(ids, ["breeze", "medium", "large-v3", "nan-v2"])
        self.assertEqual(find_asr_model(self.settings, "medium", "nan-v2").model,
                         str(self.model_dir))

    def test_missing_dir_is_hidden(self):
        # Same gate as the builtin A/B preset: a dangling dir would make
        # faster-whisper reach for HuggingFace at switch time.
        self._register(preset_id="gone", model=str(self.root / "not-there"))
        self.assertEqual(deployed_asr_models(self.settings), [])

    def test_whisper_kind_needs_no_dir(self):
        self._register(preset_id="small", label="Whisper small", model="small", kind="whisper")
        ids = [o.id for o in list_asr_models(self.settings, "medium")]
        self.assertIn("small", ids)

    def test_deployed_entry_cannot_shadow_a_builtin(self):
        self._register(preset_id="breeze", label="Hijacked", model=str(self.model_dir))
        opts = list_asr_models(self.settings, "medium")
        self.assertEqual([o.id for o in opts], ["breeze", "medium", "large-v3"])
        self.assertEqual(opts[0].label, "Breeze ASR")

    def test_active_registered_model_adds_no_current_entry(self):
        self._register(preset_id="nan-v2", model=str(self.model_dir))
        ids = [o.id for o in list_asr_models(self.settings, str(self.model_dir))]
        self.assertNotIn("current", ids)


if __name__ == "__main__":
    unittest.main()
