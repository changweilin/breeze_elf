import unittest

from breeze_elf.protocol import (
    GlossaryEntry,
    ProtocolError,
    StartMessage,
    StopMessage,
    parse_client_text,
    server_event,
)


class ProtocolTests(unittest.TestCase):
    def test_parse_start(self):
        message = parse_client_text(
            '{"type":"start","sampleRate":16000,"language":"zh","chunkMs":1000}'
        )
        self.assertEqual(
            message,
            StartMessage(
                sample_rate=16000, language="zh", chunk_ms=1000, languages=("zh",)
            ),
        )
        self.assertEqual(message.mode, "live")
        # Translation defaults off with no target.
        self.assertFalse(message.translate)
        self.assertEqual(message.translate_target, "")

    def test_parse_start_reads_translate_flags(self):
        message = parse_client_text(
            '{"type":"start","sampleRate":16000,"chunkMs":1000,"language":"en",'
            '"translate":true,"translateTarget":"ZH"}'
        )
        self.assertTrue(message.translate)
        self.assertEqual(message.translate_target, "zh")  # lower-cased

    def test_parse_start_languages_dedupes_lowercases_and_caps_at_four(self):
        message = parse_client_text(
            '{"type":"start","sampleRate":16000,"chunkMs":250,'
            '"languages":["ZH","en","zh","ja","ko","es"]}'
        )
        self.assertEqual(message.languages, ("zh", "en", "ja", "ko"))
        # The primary/back-compat single language mirrors the first entry.
        self.assertEqual(message.language, "zh")

    def test_parse_start_languages_fall_back_to_single_language(self):
        message = parse_client_text(
            '{"type":"start","sampleRate":16000,"language":"auto","chunkMs":250,"mode":"file"}'
        )
        self.assertEqual(message.languages, ("auto",))

    def test_parse_start_glossary_filters_invalid_entries(self):
        message = parse_client_text(
            '{"type":"start","sampleRate":16000,"chunkMs":250,"glossary":['
            '{"from":"機器學習","to":"Mike 學習"},'
            '{"from":"同","to":"同"},'          # no-op, dropped
            '{"from":"","to":"x"},'            # empty source, dropped
            '"nonsense"]}'                      # not an object, dropped
        )
        self.assertEqual(
            message.glossary, (GlossaryEntry(source="機器學習", target="Mike 學習"),)
        )

    def test_parse_start_file_mode(self):
        message = parse_client_text(
            '{"type":"start","sampleRate":16000,"language":"zh","chunkMs":250,"mode":"file"}'
        )
        self.assertEqual(message.mode, "file")

    def test_parse_start_unknown_mode_falls_back_to_live(self):
        message = parse_client_text(
            '{"type":"start","sampleRate":16000,"language":"zh","chunkMs":250,"mode":"bogus"}'
        )
        self.assertEqual(message.mode, "live")

    def test_rejects_wrong_sample_rate(self):
        with self.assertRaises(ProtocolError):
            parse_client_text('{"type":"start","sampleRate":48000}')

    def test_parse_stop(self):
        message = parse_client_text('{"type":"stop","reason":"button"}')
        self.assertEqual(message, StopMessage(reason="button"))

    def test_server_event(self):
        self.assertEqual(
            server_event("ready", sampleRate=16000),
            {"type": "ready", "sampleRate": 16000},
        )


if __name__ == "__main__":
    unittest.main()
