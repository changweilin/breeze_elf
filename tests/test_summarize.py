import json
import os
import unittest
import urllib.error
from unittest import mock
from unittest.mock import patch

from breeze_elf import main
from breeze_elf.config import get_settings
from breeze_elf.summarize import (
    ExtractiveSummarizer,
    NullSummarizer,
    OllamaSummarizer,
    build_summarizer,
)

try:
    from starlette.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None

# Five sentences; only 天氣真好 is off-topic. The 第三季-heavy sentences carry the
# most recurring content, so a 2-sentence extract should prefer them over 天氣真好.
_TEXT = (
    "今天開會討論新產品的上市時程。"
    "行銷團隊認為第三季最合適,因為競品也在第三季推出。"
    "工程團隊擔心第三季時程太趕,建議延到第四季。"
    "最後決定以第三季為目標,但保留第四季作為備案。"
    "天氣真好。"
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ExtractiveSummarizerTests(unittest.TestCase):
    def setUp(self):
        self.summarizer = ExtractiveSummarizer()

    def test_selects_informative_sentences_over_offtopic(self):
        summary = self.summarizer.summarize(_TEXT, max_sentences=2)
        self.assertIn("第三季", summary)
        self.assertNotIn("天氣真好", summary)  # off-topic, low frequency score

    def test_preserves_reading_order(self):
        summary = self.summarizer.summarize(_TEXT, max_sentences=3)
        picked = [s for s in ("上市時程", "最合適", "作為備案") if s in summary]
        positions = [summary.index(s) for s in picked]
        self.assertEqual(positions, sorted(positions))

    def test_short_text_returned_whole(self):
        self.assertEqual(self.summarizer.summarize("只有一句話。", max_sentences=5), "只有一句話。")

    def test_empty_text_is_empty(self):
        self.assertEqual(self.summarizer.summarize("   ", max_sentences=5), "")

    def test_is_deterministic(self):
        a = self.summarizer.summarize(_TEXT, max_sentences=2)
        b = self.summarizer.summarize(_TEXT, max_sentences=2)
        self.assertEqual(a, b)


class OllamaSummarizerTests(unittest.TestCase):
    def test_returns_model_response_on_success(self):
        fake = _FakeResponse({"response": "重點一\n重點二"})
        with patch("breeze_elf.summarize.urllib.request.urlopen", return_value=fake):
            out = OllamaSummarizer("http://x", "m").summarize(_TEXT, max_sentences=3)
        self.assertEqual(out, "重點一\n重點二")

    def test_falls_back_to_extractive_when_unreachable(self):
        with patch(
            "breeze_elf.summarize.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            out = OllamaSummarizer("http://x", "m").summarize(_TEXT, max_sentences=2)
        self.assertEqual(out, ExtractiveSummarizer().summarize(_TEXT, max_sentences=2))

    def test_empty_model_response_falls_back(self):
        with patch(
            "breeze_elf.summarize.urllib.request.urlopen",
            return_value=_FakeResponse({"response": "   "}),
        ):
            out = OllamaSummarizer("http://x", "m").summarize(_TEXT, max_sentences=2)
        self.assertTrue(out)  # non-empty extractive fallback, never blank

    def test_non_object_json_body_falls_back_without_raising(self):
        # A valid-JSON but non-object body (proxy/other service) must degrade to
        # extractive, not raise AttributeError out of the "never raises" contract.
        with patch(
            "breeze_elf.summarize.urllib.request.urlopen",
            return_value=_FakeResponse([1, 2, 3]),
        ):
            out = OllamaSummarizer("http://x", "m").summarize(_TEXT, max_sentences=2)
        self.assertEqual(out, ExtractiveSummarizer().summarize(_TEXT, max_sentences=2))


class BuildSummarizerTests(unittest.TestCase):
    def test_off_is_disabled(self):
        summ = build_summarizer("off")
        self.assertIsInstance(summ, NullSummarizer)
        self.assertFalse(summ.available)

    def test_default_is_extractive(self):
        self.assertEqual(build_summarizer("extractive").name, "extractive")
        self.assertEqual(build_summarizer("").name, "extractive")

    def test_ollama_selected(self):
        self.assertEqual(build_summarizer("ollama").name, "ollama")


class SummaryConfigTests(unittest.TestCase):
    def test_provider_env(self):
        with mock.patch.dict(os.environ, {"BREEZE_SUMMARY_PROVIDER": "ollama"}, clear=True):
            self.assertEqual(get_settings().summary_provider, "ollama")

    def test_invalid_provider_defaults_to_extractive(self):
        with mock.patch.dict(os.environ, {"BREEZE_SUMMARY_PROVIDER": "bogus"}, clear=True):
            self.assertEqual(get_settings().summary_provider, "extractive")


@unittest.skipIf(TestClient is None, "starlette TestClient unavailable")
class SummaryEndpointTests(unittest.TestCase):
    def test_summarize_returns_summary(self):
        with patch.object(main, "summarizer", ExtractiveSummarizer()):
            client = TestClient(main.app)
            resp = client.post("/api/summary", json={"text": _TEXT, "maxSentences": 2})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["provider"], "extractive")
        self.assertIn("第三季", body["summary"])

    def test_empty_text_is_400(self):
        with patch.object(main, "summarizer", ExtractiveSummarizer()):
            client = TestClient(main.app)
            resp = client.post("/api/summary", json={"text": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_disabled_provider_is_503(self):
        with patch.object(main, "summarizer", NullSummarizer()):
            client = TestClient(main.app)
            resp = client.post("/api/summary", json={"text": _TEXT})
        self.assertEqual(resp.status_code, 503)

    def test_input_is_capped(self):
        # A huge transcript must be truncated to summary_max_chars before work.
        captured = {}

        class _Spy(ExtractiveSummarizer):
            def summarize(self, text, *, max_sentences=5):
                captured["len"] = len(text)
                return "ok"

        with patch.object(main, "summarizer", _Spy()):
            client = TestClient(main.app)
            resp = client.post("/api/summary", json={"text": "字" * 50000})
        self.assertEqual(resp.status_code, 200)
        self.assertLessEqual(captured["len"], main.settings.summary_max_chars)


if __name__ == "__main__":
    unittest.main()
