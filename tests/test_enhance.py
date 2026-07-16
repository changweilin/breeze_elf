import os
import unittest
from unittest.mock import patch

import numpy as np

from breeze_elf.config import get_settings
from breeze_elf.enhance import (
    DeepFilterEnhancer,
    DemucsSeparator,
    NullEnhancer,
    build_enhancer,
    build_separator,
)


class EnhanceConfigTests(unittest.TestCase):
    def test_enhance_env_is_read_and_validated(self):
        with patch.dict(
            os.environ,
            {
                "BREEZE_ENHANCE_LIVE": "deepfilter",
                "BREEZE_ENHANCE_FILE": "off",
                "BREEZE_ENHANCE_DEVICE": "cuda",
            },
        ):
            settings = get_settings()
        self.assertEqual(settings.enhance_live, "deepfilter")
        self.assertEqual(settings.enhance_file, "off")
        self.assertEqual(settings.enhance_device, "cuda")

    def test_unknown_enhance_choice_falls_back_to_off(self):
        with patch.dict(os.environ, {"BREEZE_ENHANCE_LIVE": "wavenet"}):
            settings = get_settings()
        self.assertEqual(settings.enhance_live, "off")

    def test_unknown_device_falls_back_to_auto(self):
        with patch.dict(os.environ, {"BREEZE_ENHANCE_DEVICE": "tpu"}):
            settings = get_settings()
        self.assertEqual(settings.enhance_device, "auto")


class BuildEnhancerTests(unittest.TestCase):
    def test_off_builds_null_enhancer(self):
        with patch.dict(os.environ, {"BREEZE_ENHANCE_LIVE": "off"}):
            settings = get_settings()
        enhancer = build_enhancer("live", settings)
        self.assertIsInstance(enhancer, NullEnhancer)
        self.assertEqual(enhancer.name, "off")

    def test_deepfilter_choice_builds_deepfilter_enhancer(self):
        with patch.dict(os.environ, {"BREEZE_ENHANCE_FILE": "deepfilter"}):
            settings = get_settings()
        enhancer = build_enhancer("file", settings)
        self.assertIsInstance(enhancer, DeepFilterEnhancer)
        self.assertEqual(enhancer.name, "deepfilter")

    def test_live_and_file_modes_are_independent(self):
        with patch.dict(
            os.environ,
            {"BREEZE_ENHANCE_LIVE": "deepfilter", "BREEZE_ENHANCE_FILE": "off"},
        ):
            settings = get_settings()
        self.assertIsInstance(build_enhancer("live", settings), DeepFilterEnhancer)
        self.assertIsInstance(build_enhancer("file", settings), NullEnhancer)


class NullEnhancerTests(unittest.TestCase):
    def test_passthrough_returns_input_unchanged(self):
        enhancer = NullEnhancer()
        samples = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
        out = enhancer.enhance(samples, 16_000)
        self.assertTrue(enhancer.ready)
        np.testing.assert_array_equal(out, samples)


class DeepFilterFallbackTests(unittest.TestCase):
    def test_enhance_falls_back_to_raw_audio_when_model_unavailable(self):
        # Without the optional extra, load() marks itself failed and enhance()
        # must return the input untouched rather than raising.
        enhancer = DeepFilterEnhancer("cpu")
        enhancer._failed = True  # simulate a failed load without importing torch
        samples = np.ones(64, dtype=np.float32) * 0.25
        out = enhancer.enhance(samples, 16_000)
        np.testing.assert_array_equal(out, samples)
        self.assertFalse(enhancer.ready)

    def test_empty_input_is_returned_immediately(self):
        enhancer = DeepFilterEnhancer("cpu")
        empty = np.empty(0, dtype=np.float32)
        np.testing.assert_array_equal(enhancer.enhance(empty, 16_000), empty)


class SeparatorTests(unittest.TestCase):
    def test_build_separator_reports_availability_matching_import(self):
        import importlib.util

        separator = build_separator(get_settings())
        self.assertIsInstance(separator, DemucsSeparator)
        expected = (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("demucs") is not None
        )
        self.assertEqual(separator.available, expected)
        self.assertEqual(separator.device, "unloaded")


if __name__ == "__main__":
    unittest.main()
