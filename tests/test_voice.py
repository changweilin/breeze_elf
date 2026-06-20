import base64
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from breeze_elf import main
from breeze_elf.voice import MockVoiceEngine, _pitch_shift
from breeze_elf.voice_storage import (
    decode_wav,
    delete_voice,
    encode_wav,
    list_voices,
    load_embedding,
    save_voice,
    update_voice,
)


def _tone(hz: float, sample_rate: int = 16_000, seconds: float = 1.0) -> np.ndarray:
    axis = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (0.4 * np.sin(2 * np.pi * hz * axis)).astype(np.float32)


def _median_hz(samples: np.ndarray, sample_rate: int = 16_000) -> float:
    from breeze_elf.audio import summarize_pitch

    return summarize_pitch(samples, sample_rate).median_hz or 0.0


class WavIoTests(unittest.TestCase):
    def test_wav_round_trip_preserves_samples(self):
        samples = _tone(220.0)
        decoded, rate = decode_wav(encode_wav(samples, 16_000))
        self.assertEqual(rate, 16_000)
        self.assertEqual(decoded.shape, samples.shape)
        # 16-bit quantization keeps the waveform within ~1 LSB.
        self.assertLess(float(np.max(np.abs(decoded - samples))), 2e-4)

    def test_decode_rejects_non_pcm16(self):
        with self.assertRaises(ValueError):
            decode_wav(b"not a wav file")


class MockEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = MockVoiceEngine(sample_rate=16_000, warmup_seconds=0.0)
        self.engine.load()

    def test_load_reports_monotonic_progress_to_completion(self):
        events = []
        MockVoiceEngine(warmup_seconds=0.0).load(lambda fraction, stage: events.append(fraction))
        self.assertTrue(events)
        self.assertEqual(events[-1], 1.0)
        self.assertEqual(events, sorted(events))

    def test_pitch_shift_raises_pitch_toward_target(self):
        low = _tone(120.0)
        shifted = _pitch_shift(low, 7.0)
        self.assertEqual(shifted.shape, low.shape)
        self.assertGreater(_median_hz(shifted), _median_hz(low) + 20)

    def test_convert_moves_source_pitch_toward_saved_voice(self):
        target = _tone(210.0)
        embedding = self.engine.extract_embedding(target, 16_000)
        source = _tone(120.0)
        result = self.engine.convert(source, 16_000, embedding)

        self.assertEqual(result.sample_rate, 16_000)
        self.assertTrue(np.isfinite(result.samples).all())
        # The converted clip should sit much closer to A's pitch than B's did.
        self.assertGreater(_median_hz(result.samples), 180.0)

    def test_synthesize_scales_length_with_text_and_is_audible(self):
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        short = self.engine.synthesize("你好", "zh", embedding)
        long = self.engine.synthesize("你好世界一二三四", "zh", embedding)

        self.assertEqual(short.sample_rate, 16_000)
        self.assertGreater(long.samples.size, short.samples.size)
        self.assertGreater(float(np.max(np.abs(short.samples))), 0.1)

    def test_embedding_is_json_with_pitch(self):
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        payload = json.loads(embedding.decode("utf-8"))
        self.assertEqual(payload["kind"], "mock")
        self.assertGreater(payload["medianHz"], 150.0)


class VoiceStorageTests(unittest.TestCase):
    def test_save_list_update_delete_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            stored = save_voice(
                "阿福",
                b'{"medianHz": 200}',
                tmp,
                sample_audio=b"RIFFfake",
                sample_rate=16_000,
                duration_seconds=1.5,
                favorite=False,
            )
            self.assertTrue(stored.id.startswith("voice-"))
            self.assertTrue(stored.has_sample)
            self.assertEqual(load_embedding(stored.id, tmp), b'{"medianHz": 200}')

            updated = update_voice(stored.id, tmp, name="阿福2", favorite=True)
            self.assertEqual(updated.name, "阿福2")
            self.assertTrue(updated.favorite)

            voices = list_voices(tmp)
            self.assertEqual(len(voices), 1)
            self.assertEqual(voices[0].name, "阿福2")

            # camelCase public payload matches the rest of the API contract.
            public = voices[0].to_public()
            self.assertIn("createdAt", public)
            self.assertIn("durationSeconds", public)
            self.assertTrue(public["favorite"])

            self.assertTrue(delete_voice(stored.id, tmp))
            self.assertEqual(list_voices(tmp), [])
            with self.assertRaises(FileNotFoundError):
                load_embedding(stored.id, tmp)

    def test_favorites_sort_before_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_voice("plain", b"x", tmp, favorite=False)
            fav = save_voice("starred", b"y", tmp, favorite=True)
            voices = list_voices(tmp)
            self.assertEqual(voices[0].id, fav.id)

    def test_rejects_path_traversal_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_embedding("../secret", tmp)


class VoiceApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(
            main, "settings", replace(main.settings, voice_storage_dir=self._tmp.name)
        )
        self._patch.start()
        self._engine_patch = patch.object(
            main, "voice_engine", MockVoiceEngine(sample_rate=16_000, warmup_seconds=0.0)
        )
        self._engine_patch.start()

    async def asyncTearDown(self):
        self._engine_patch.stop()
        self._patch.stop()
        self._tmp.cleanup()

    def _wav_b64(self, hz: float) -> str:
        return base64.b64encode(encode_wav(_tone(hz), 16_000)).decode("ascii")

    async def test_create_convert_tts_flow(self):
        create = await main.create_saved_voice(
            main.VoiceCreateRequest(name="阿福", audioBase64=self._wav_b64(210.0), favorite=True)
        )
        voice = json.loads(create.body)["voice"]
        self.assertTrue(voice["favorite"])
        self.assertAlmostEqual(voice["durationSeconds"], 1.0, places=2)
        voice_id = voice["id"]

        convert = await main.convert_voice(
            main.VoiceConvertRequest(voiceId=voice_id, audioBase64=self._wav_b64(120.0))
        )
        convert_data = json.loads(convert.body)
        self.assertTrue(convert_data["ok"])
        decoded, rate = decode_wav(base64.b64decode(convert_data["audioBase64"]))
        self.assertEqual(rate, 16_000)
        self.assertGreater(decoded.size, 0)

        tts = await main.voice_tts(
            main.VoiceTtsRequest(voiceId=voice_id, text="你好世界", language="zh")
        )
        tts_data = json.loads(tts.body)
        self.assertTrue(tts_data["ok"])
        self.assertGreater(tts_data["durationSeconds"], 0.0)

    async def test_convert_unknown_voice_returns_404(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            await main.convert_voice(
                main.VoiceConvertRequest(voiceId="voice-missing", audioBase64=self._wav_b64(120.0))
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_status_reports_voice_count(self):
        await main.create_saved_voice(
            main.VoiceCreateRequest(name="a", audioBase64=self._wav_b64(200.0))
        )
        status = json.loads((await main.voice_status()).body)
        self.assertTrue(status["ok"])
        self.assertEqual(status["voiceCount"], 1)
        self.assertEqual(status["backend"], "mock")


class VoiceLoadStateTests(unittest.TestCase):
    def test_begin_guards_against_concurrent_loads(self):
        loader = main.VoiceLoadState()
        self.assertTrue(loader.begin())
        self.assertFalse(loader.begin())
        loader.report(0.5, "halfway")
        self.assertEqual(loader.snapshot()["progress"], 0.5)
        loader.finish()
        snapshot = loader.snapshot()
        self.assertEqual(snapshot["status"], "ready")
        self.assertEqual(snapshot["progress"], 1.0)

    def test_fail_records_error(self):
        loader = main.VoiceLoadState()
        loader.begin()
        loader.fail("boom")
        snapshot = loader.snapshot()
        self.assertEqual(snapshot["status"], "error")
        self.assertEqual(snapshot["error"], "boom")


if __name__ == "__main__":
    unittest.main()
