import unittest

from breeze_elf.protocol import (
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
        self.assertEqual(message, StartMessage(sample_rate=16000, language="zh", chunk_ms=1000))
        self.assertEqual(message.mode, "live")

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
