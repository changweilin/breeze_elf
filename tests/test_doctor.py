import contextlib
import io
import unittest
from dataclasses import replace
from unittest.mock import patch

from breeze_elf import doctor
from breeze_elf.config import get_settings


def _settings(**overrides):
    base = replace(
        get_settings(),
        asr_provider="mock",
        translate_provider="off",
        diarize_enabled=False,
        voice_provider="mock",
        enhance_live="off",
        enhance_file="off",
    )
    return replace(base, **overrides)


class DoctorTests(unittest.TestCase):
    def test_python_check_is_ok(self):
        self.assertEqual(doctor.check_python().status, doctor.OK)

    def test_mock_provider_skips_runtime_and_gpu_and_cudnn(self):
        settings = _settings()
        self.assertEqual(doctor.check_asr_runtime(settings).status, doctor.SKIP)
        self.assertEqual(doctor.check_gpu_ctranslate2(settings).status, doctor.SKIP)
        self.assertEqual(doctor.check_cudnn(settings).status, doctor.SKIP)

    def test_missing_asr_runtime_fails_for_real_provider(self):
        settings = _settings(asr_provider="faster-whisper")
        with patch.object(doctor, "_module_version", lambda name: None):
            check = doctor.check_asr_runtime(settings)
        self.assertEqual(check.status, doctor.FAIL)
        self.assertTrue(check.action)

    def test_missing_translate_model_warns(self):
        settings = _settings(
            translate_provider="nllb", translate_model="models/definitely-not-here"
        )
        by_name = {c.name: c for c in doctor.check_models(settings)}
        self.assertIn("model: NLLB translate", by_name)
        self.assertEqual(by_name["model: NLLB translate"].status, doctor.WARN)

    def test_collect_checks_has_no_fail_for_mock(self):
        settings = _settings()
        self.assertFalse(any(c.status == doctor.FAIL for c in doctor.collect_checks(settings)))

    def test_run_doctor_returns_zero_for_mock(self):
        with patch.object(doctor, "get_settings", _settings):
            self.assertEqual(doctor.run_doctor(), 0)

    def test_run_doctor_returns_one_when_a_check_fails(self):
        failing = [doctor.Check("boom", doctor.FAIL, "forced")]
        with patch.object(doctor, "get_settings", _settings), patch.object(
            doctor, "collect_checks", lambda *a, **k: failing
        ):
            self.assertEqual(doctor.run_doctor(), 1)

    def test_run_doctor_output_is_ascii(self):
        # Contract: doctor output must stay ASCII so it can't raise UnicodeEncodeError
        # on a legacy-codepage (cp950/cp1252) Windows console — the exact environment
        # this diagnostic exists to inspect.
        buffer = io.StringIO()
        with patch.object(doctor, "get_settings", _settings), contextlib.redirect_stdout(buffer):
            doctor.run_doctor()
        output = buffer.getvalue()
        self.assertTrue(output)
        output.encode("ascii")  # raises if any non-ASCII char leaked into the report

    def test_deployed_preset_with_cjk_label_stays_ascii(self):
        # A deployed preset can carry a CJK label ("Breeze ASR 台語強化 v2") + note.
        # It must not break the ASCII contract — and this must hold independent of a
        # real models/presets.json, which CI lacks (so the end-to-end test above is a
        # trivial pass there and would not catch this regression).
        from breeze_elf.model_registry import RegisteredModel

        cjk = [
            RegisteredModel(
                id="breeze-nan-v2",
                label="Breeze ASR 台語強化 v2",
                model="models/breeze-asr-25-nan-v2-ct2",
                kind="whisper",
                note="音準",
            )
        ]
        settings = _settings()
        with patch("breeze_elf.model_registry.load_registry", lambda *a, **k: cjk):
            lines = [
                doctor._format_line(check, color=False)
                for check in doctor.check_models(settings)
            ]
        "\n".join(lines).encode("ascii")  # raises if the CJK label leaked verbatim
        self.assertTrue(any("breeze-nan-v2" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
