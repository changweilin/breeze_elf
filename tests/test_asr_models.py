import tempfile
import unittest
from dataclasses import replace

from breeze_elf.asr_models import (
    active_model_id,
    builtin_asr_models,
    find_asr_model,
    list_asr_models,
)
from breeze_elf.config import get_settings


class ASRModelRegistryTests(unittest.TestCase):
    def setUp(self):
        # Point the A/B nan preset at a path that does not exist so the base preset
        # list is deterministic regardless of what CT2 dirs happen to be on disk.
        self.settings = replace(
            get_settings(), asr_breeze_nan_model="models/__no_such_nan_ct2__"
        )

    def test_builtin_presets_present_and_ordered(self):
        ids = [opt.id for opt in builtin_asr_models(self.settings)]
        self.assertEqual(ids, ["breeze", "medium", "large-v3"])

    def test_breeze_nan_preset_surfaced_only_when_dir_exists(self):
        with tempfile.TemporaryDirectory() as d:
            settings = replace(self.settings, asr_breeze_nan_model=d)
            opts = builtin_asr_models(settings)
            ids = [o.id for o in opts]
            self.assertEqual(ids, ["breeze", "breeze-nan", "medium", "large-v3"])
            nan = next(o for o in opts if o.id == "breeze-nan")
            self.assertEqual(nan.model, d)
            self.assertEqual(nan.kind, "breeze")

    def test_breeze_resolves_to_configured_local_dir(self):
        settings = replace(self.settings, asr_breeze_model="models/custom-breeze-ct2")
        breeze = next(o for o in builtin_asr_models(settings) if o.id == "breeze")
        self.assertEqual(breeze.model, "models/custom-breeze-ct2")
        self.assertEqual(breeze.kind, "breeze")

    def test_active_id_matches_builtin_model(self):
        self.assertEqual(active_model_id(self.settings, "medium"), "medium")
        self.assertEqual(active_model_id(self.settings, "large-v3"), "large-v3")

    def test_active_id_for_breeze_path(self):
        settings = replace(self.settings, asr_breeze_model="models/breeze-ct2")
        self.assertEqual(active_model_id(settings, "models/breeze-ct2"), "breeze")

    def test_unlisted_running_model_gets_current_entry(self):
        options = list_asr_models(self.settings, "small")
        current = options[-1]
        self.assertEqual(current.id, "current")
        self.assertEqual(current.model, "small")
        self.assertEqual(active_model_id(self.settings, "small"), "current")

    def test_builtin_model_adds_no_current_entry(self):
        ids = [opt.id for opt in list_asr_models(self.settings, "medium")]
        self.assertNotIn("current", ids)

    def test_find_by_id(self):
        self.assertEqual(find_asr_model(self.settings, "medium", "large-v3").model, "large-v3")
        self.assertIsNone(find_asr_model(self.settings, "medium", "nope"))


if __name__ == "__main__":
    unittest.main()
