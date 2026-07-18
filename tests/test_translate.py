import dataclasses
import os
import unittest
from unittest import mock

from breeze_elf.config import get_settings
from breeze_elf.translate import (
    NllbTranslator,
    NullTranslator,
    build_translator,
    to_flores,
)


class _FakeResult:
    def __init__(self, hypothesis):
        self.hypotheses = [hypothesis]


class _FakeTranslator:
    """Records the exact token sequence it is handed so the recipe can be locked,
    and echoes the source pieces back (prefixed with the target-language token, and
    — for one variant — a trailing ``</s>``) as a stand-in translation."""

    def __init__(self, append_eos=False):
        self.calls = []
        self.append_eos = append_eos

    def translate_batch(self, source, target_prefix=None, **kwargs):
        self.calls.append({"source": source, "target_prefix": target_prefix, "kwargs": kwargs})
        pieces = source[0][1:-1]  # strip the leading src_lang and trailing </s>
        tgt = target_prefix[0][0]
        hyp = [tgt, *pieces] + (["</s>"] if self.append_eos else [])
        return [_FakeResult(hyp)]


class _FakeSp:
    def encode(self, text, out_type=str):
        assert out_type is str
        return ["▁he", "llo"]

    def decode(self, tokens):
        return " ".join(tokens)


def _loaded_translator(fake_translator):
    t = NllbTranslator("nonexistent-model-dir")
    # Bypass load() (no ctranslate2/sentencepiece needed) by pre-seeding internals.
    t._translator = fake_translator
    t._sp = _FakeSp()
    return t


class FloresMappingTests(unittest.TestCase):
    def test_common_codes(self):
        self.assertEqual(to_flores("zh"), "zho_Hant")
        self.assertEqual(to_flores("en"), "eng_Latn")
        self.assertEqual(to_flores("JA"), "jpn_Jpan")
        self.assertEqual(to_flores("ko"), "kor_Hang")

    def test_auto_and_empty_are_none(self):
        self.assertIsNone(to_flores("auto"))
        self.assertIsNone(to_flores(""))
        self.assertIsNone(to_flores(None))
        self.assertIsNone(to_flores("qq"))  # unknown

    def test_already_flores_passthrough(self):
        self.assertEqual(to_flores("zho_Hant"), "zho_Hant")


class NllbRecipeTests(unittest.TestCase):
    """Lock the sentencepiece-only NLLB recipe: [src_lang] + pieces + </s> as the
    source, target_prefix=[[tgt_lang]], and the leading tgt token stripped back off."""

    def test_source_token_shape_is_exact(self):
        fake = _FakeTranslator()
        out = _loaded_translator(fake).translate("hello", "en", "zh")
        self.assertEqual(
            fake.calls[0]["source"], [["eng_Latn", "▁he", "llo", "</s>"]]
        )
        self.assertEqual(fake.calls[0]["target_prefix"], [["zho_Hant"]])
        # Decoded output has the tgt-lang prefix removed.
        self.assertEqual(out, "▁he llo")

    def test_trailing_eos_is_stripped(self):
        out = _loaded_translator(_FakeTranslator(append_eos=True)).translate("hello", "en", "zh")
        self.assertEqual(out, "▁he llo")  # no </s> in the decoded text

    def test_same_language_is_noop(self):
        fake = _FakeTranslator()
        self.assertEqual(_loaded_translator(fake).translate("hi", "en", "en"), "")
        self.assertEqual(fake.calls, [])  # never touched the model

    def test_auto_source_is_noop(self):
        fake = _FakeTranslator()
        self.assertEqual(_loaded_translator(fake).translate("hi", "auto", "zh"), "")
        self.assertEqual(fake.calls, [])

    def test_empty_text_is_noop(self):
        fake = _FakeTranslator()
        self.assertEqual(_loaded_translator(fake).translate("   ", "en", "zh"), "")
        self.assertEqual(fake.calls, [])

    def test_inference_failure_returns_empty(self):
        class _Boom:
            def translate_batch(self, *a, **k):
                raise RuntimeError("oom")

        self.assertEqual(_loaded_translator(_Boom()).translate("hello", "en", "zh"), "")


class TranslatorCapabilityTests(unittest.TestCase):
    def test_null_translator_is_unavailable_and_silent(self):
        t = NullTranslator()
        self.assertFalse(t.available)
        self.assertEqual(t.translate("hello", "en", "zh"), "")

    def test_available_false_without_model_dir(self):
        self.assertFalse(NllbTranslator("definitely/not/here").available)

    def test_resolve_spm_finds_model_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            spm = Path(tmp) / "sentencepiece.bpe.model"
            spm.write_bytes(b"stub")
            self.assertEqual(NllbTranslator(tmp)._resolve_spm(), spm)

    def test_resolve_spm_none_when_absent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(NllbTranslator(tmp)._resolve_spm())


class BuildTranslatorTests(unittest.TestCase):
    def test_off_returns_null(self):
        settings = dataclasses.replace(get_settings(), translate_provider="off")
        self.assertIsInstance(build_translator(settings), NullTranslator)

    def test_nllb_returns_translator(self):
        settings = dataclasses.replace(
            get_settings(), translate_provider="nllb", translate_model="models/x"
        )
        translator = build_translator(settings)
        self.assertIsInstance(translator, NllbTranslator)


class TranslateConfigTests(unittest.TestCase):
    def test_provider_env(self):
        with mock.patch.dict(os.environ, {"BREEZE_TRANSLATE": "nllb"}, clear=True):
            self.assertEqual(get_settings().translate_provider, "nllb")

    def test_invalid_provider_defaults_off(self):
        with mock.patch.dict(os.environ, {"BREEZE_TRANSLATE": "google"}, clear=True):
            self.assertEqual(get_settings().translate_provider, "off")

    def test_target_and_beam_env(self):
        env = {"BREEZE_TRANSLATE_TARGET": "en", "BREEZE_TRANSLATE_BEAM": "4"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = get_settings()
            self.assertEqual(settings.translate_target, "en")
            self.assertEqual(settings.translate_beam, 4)


if __name__ == "__main__":
    unittest.main()
