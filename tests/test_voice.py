import asyncio
import base64
import json
import pathlib
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from breeze_elf import main
from breeze_elf.voice import (
    MockVoiceEngine,
    _breath_sound,
    _correct_song_register,
    _estimate_median_hz,
    _fit_duration,
    _fold_hz_into_range,
    _glide_semitones,
    _pitch_shift,
    _resample_rate,
    _resolve_song_notes,
    _sing_contour_os,
    _song_base,
    _trim_silence,
)
from breeze_elf.voice_storage import (
    decode_wav,
    delete_voice,
    encode_wav,
    list_voices,
    load_embedding,
    save_voice,
    save_voice_output,
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
        # Force the synthetic fallback so synth tests stay deterministic and do
        # not shell out to the OS text-to-speech voice.
        self.engine = MockVoiceEngine(sample_rate=16_000, warmup_seconds=0.0, use_os_tts=False)
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

    def test_synthesize_song_follows_jianpu_pitch(self):
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        notes = [
            {"char": "do", "jianpu": "1", "durationSeconds": 0.4},
            {"char": "rest", "jianpu": "0", "durationSeconds": 0.2},
            {"char": "sol", "jianpu": "5", "durationSeconds": 0.4},
        ]
        result = self.engine.synthesize_song(notes, 220.0, embedding)
        self.assertEqual(result.sample_rate, 16_000)
        # 1 (220 Hz) + rest + 5 (sol, ~330 Hz) => roughly a second of audio.
        self.assertGreater(result.samples.size / result.sample_rate, 0.8)
        self.assertGreater(float(np.max(np.abs(result.samples))), 0.1)

    def test_extract_embedding_includes_timbre_bands(self):
        payload = json.loads(self.engine.extract_embedding(_tone(200.0), 16_000).decode())
        self.assertEqual(payload["version"], 2)
        self.assertIsInstance(payload["bands"], list)
        self.assertEqual(len(payload["bands"]), 28)

    def test_synthesize_speed_shortens_and_lengthens_without_pitch_change(self):
        embedding = self.engine.extract_embedding(_tone(180.0), 16_000)
        normal = self.engine.synthesize("你好世界一二三", "zh", embedding)
        fast = self.engine.synthesize("你好世界一二三", "zh", embedding, speed=2.0)
        slow = self.engine.synthesize("你好世界一二三", "zh", embedding, speed=0.5)
        self.assertLess(fast.samples.size, normal.samples.size)
        self.assertGreater(slow.samples.size, normal.samples.size)
        # Tempo only — the median pitch is unchanged by the speed multiplier.
        self.assertAlmostEqual(_median_hz(fast.samples), _median_hz(normal.samples), delta=20.0)

    def test_synthesize_base_hz_sets_voice_register(self):
        embedding = self.engine.extract_embedding(_tone(150.0), 16_000)
        low = self.engine.synthesize("啊啊啊啊", "zh", embedding, base_hz=110.0)
        high = self.engine.synthesize("啊啊啊啊", "zh", embedding, base_hz=260.0)
        self.assertGreater(_median_hz(high.samples), _median_hz(low.samples) + 20)

    def test_synthesize_song_speed_shortens_audio(self):
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        notes = [
            {"char": "do", "jianpu": "1", "durationSeconds": 0.5},
            {"char": "sol", "jianpu": "5", "durationSeconds": 0.5},
        ]
        normal = self.engine.synthesize_song(notes, 220.0, embedding)
        fast = self.engine.synthesize_song(notes, 220.0, embedding, speed=2.0)
        self.assertLess(fast.samples.size, normal.samples.size)

    def test_synthesize_song_duration_matches_note_lengths(self):
        # The sung clip lines up in time with the notes: total length ≈ the sum of
        # the note durations (no min-length floor / inter-note gaps stretching it).
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        notes = [
            {"char": "do", "jianpu": "1", "durationSeconds": 0.5},
            {"char": "mi", "jianpu": "3", "durationSeconds": 0.3},
            {"char": "sol", "jianpu": "5", "durationSeconds": 0.4},
        ]
        result = self.engine.synthesize_song(notes, 220.0, embedding)
        self.assertAlmostEqual(result.samples.size / result.sample_rate, 1.2, delta=0.03)

    def test_synthesize_song_trims_head_and_tail_silence(self):
        # Leading / trailing rests (head / tail 靜默) are dropped, inner timing kept.
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        notes = [
            {"char": "-", "jianpu": "0", "durationSeconds": 0.5},
            {"char": "do", "jianpu": "1", "durationSeconds": 0.4},
            {"char": "-", "jianpu": "0", "durationSeconds": 0.5},
        ]
        result = self.engine.synthesize_song(notes, 220.0, embedding)
        self.assertAlmostEqual(result.samples.size / result.sample_rate, 0.4, delta=0.05)

    def test_fold_hz_into_range_keeps_pitch_class(self):
        self.assertAlmostEqual(_fold_hz_into_range(500.0), 250.0, delta=0.1)  # octave down
        self.assertAlmostEqual(_fold_hz_into_range(40.0), 80.0, delta=0.1)  # octave up

    def test_correct_song_register_aligns_octave_to_target(self):
        # Original median 150, target 300 → shift the whole melody up one octave,
        # keeping intervals (150→300, 180→360).
        notes = [
            {"kind": "voiced", "contour": [150.0], "freq": 150.0},
            {"kind": "voiced", "contour": [180.0], "freq": 180.0},
        ]
        _correct_song_register(notes, 150.0, 300.0)
        self.assertAlmostEqual(notes[0]["contour"][0], 300.0, delta=2.0)
        self.assertAlmostEqual(notes[1]["contour"][0], 360.0, delta=2.0)

    def test_correct_song_register_small_diff_no_shift(self):
        # A modest median difference must NOT transpose (that caused 太高/小孩聲).
        notes = [{"kind": "voiced", "contour": [200.0], "freq": 200.0}]
        _correct_song_register(notes, 200.0, 230.0)
        self.assertAlmostEqual(notes[0]["contour"][0], 200.0, delta=1.0)

    def test_correct_song_register_folds_out_of_range_note(self):
        # An out-of-range note is folded by octaves (in tune), not clamped.
        notes = [{"kind": "voiced", "contour": [520.0], "freq": 520.0}]
        _correct_song_register(notes, 250.0, 250.0)
        self.assertAlmostEqual(notes[0]["contour"][0], 260.0, delta=1.0)

    def test_trim_edge_rests_removes_only_head_and_tail(self):
        from breeze_elf.voice import _trim_edge_rests

        notes = [
            {"kind": "rest"},
            {"kind": "voiced"},
            {"kind": "rest"},
            {"kind": "voiced"},
            {"kind": "rest"},
        ]
        trimmed = _trim_edge_rests(notes)
        self.assertEqual([note["kind"] for note in trimmed], ["voiced", "rest", "voiced"])

    def test_sing_contour_os_flattens_lexical_tone_to_steady_note(self):
        # A spoken syllable with a big falling tone (260 → 150 Hz) should, when
        # sung as a steady 簡譜 degree, come out roughly flat — the per-window
        # retune levels the Chinese tone instead of letting it inflate the 高低落差.
        rate = 22_050
        count = int(rate * 0.4)
        axis = np.arange(count) / rate
        f0 = np.linspace(260.0, 150.0, count)
        syllable = (0.5 * np.sin(2 * np.pi * np.cumsum(f0) / rate)).astype(np.float32)
        source = _estimate_median_hz(syllable, rate)

        sung = _sing_contour_os(syllable, source, [200.0], count, rate)
        half = sung.size // 2
        first = _estimate_median_hz(sung[:half], rate)
        second = _estimate_median_hz(sung[half:], rate)
        # The input's two halves differ by ~50 Hz; the steady note's must be far
        # smaller, and the whole note sits near the 200 Hz target.
        self.assertLess(abs(first - second), 25.0)
        self.assertAlmostEqual(_estimate_median_hz(sung, rate), 200.0, delta=20.0)

    def test_sing_contour_os_follows_rising_contour(self):
        rate = 22_050
        count = int(rate * 0.4)
        syllable = _tone(200.0, rate, 0.4)
        sung = _sing_contour_os(syllable, 200.0, [170.0, 250.0], count, rate)
        half = sung.size // 2
        self.assertLess(
            _estimate_median_hz(sung[:half], rate), _estimate_median_hz(sung[half:], rate)
        )

    def test_glide_semitones_reads_arrows_and_endpoints(self):
        self.assertEqual(_glide_semitones({"jianpu": "3↗5"}), (4.0, 7.0))
        self.assertEqual(_glide_semitones({"jianpuStart": "1", "jianpuEnd": "5"}), (0.0, 7.0))
        self.assertIsNone(_glide_semitones({"jianpu": "3"}))  # steady note, not a glide

    def test_resolve_song_notes_builds_contour_kind_and_breath(self):
        # A 簡譜 glide is a hold → slide → hold portamento: it starts on the first
        # degree, holds, then settles on the second.
        glide = _resolve_song_notes([{"char": "啦", "jianpu": "1↗5", "durationSeconds": 0.5}], 220.0, False)
        self.assertEqual(glide[0]["kind"], "voiced")
        self.assertGreater(len(glide[0]["contour"]), 3)
        self.assertAlmostEqual(glide[0]["contour"][0], 220.0, delta=2.0)
        self.assertAlmostEqual(glide[0]["contour"][1], 220.0, delta=2.0)  # held start
        self.assertAlmostEqual(glide[0]["contour"][-1], 220.0 * 2 ** (7 / 12), delta=2.0)

    def test_glide_position_places_the_slide_late(self):
        # With glideMid late (0.8) the note holds the start most of the way, then
        # slides near the end — the contour's first half stays on the start pitch.
        notes = _resolve_song_notes(
            [{"char": "啦", "jianpu": "1↗5", "durationSeconds": 0.5, "glideMid": 0.8}], 220.0, False
        )
        contour = notes[0]["contour"]
        mid = contour[len(contour) // 2]
        self.assertAlmostEqual(mid, 220.0, delta=4.0)  # still on the start pitch mid-way
        self.assertAlmostEqual(contour[-1], 220.0 * 2 ** (7 / 12), delta=2.0)

        measured = _resolve_song_notes(
            [{"char": "啦", "contour": [200, 260, 320], "durationSeconds": 0.5}], 0.0, True
        )
        self.assertEqual(measured[0]["contour"], [200.0, 260.0, 320.0])

        breath = _resolve_song_notes([{"kind": "breath", "intensity": 0.2, "durationSeconds": 0.2}], 0.0, True)
        self.assertEqual(breath[0]["kind"], "breath")
        rest = _resolve_song_notes([{"char": "-", "jianpu": "0", "durationSeconds": 0.2}], 220.0, False)
        self.assertEqual(rest[0]["kind"], "rest")

    def test_synthesize_song_follows_rising_contour(self):
        # A rising measured contour must actually rise in pitch (抑揚頓挫), not sit
        # flat like a single-pitch 簡譜 note.
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        notes = [{"char": "啦", "kind": "voiced", "contour": [180, 240, 320], "durationSeconds": 0.9}]
        result = self.engine.synthesize_song(notes, 0.0, embedding, True)
        samples = result.samples
        third = samples.size // 3
        head = _median_hz(samples[:third])
        tail = _median_hz(samples[-third:])
        self.assertGreater(tail, head + 20)

    def test_synthesize_song_renders_breath_note_audibly(self):
        embedding = self.engine.extract_embedding(_tone(200.0), 16_000)
        notes = [
            {"char": "啦", "jianpu": "1", "durationSeconds": 0.4},
            {"char": "h", "kind": "breath", "intensity": 0.3, "durationSeconds": 0.3},
        ]
        result = self.engine.synthesize_song(notes, 220.0, embedding, False)
        self.assertGreater(float(np.max(np.abs(result.samples))), 0.05)

    def test_breath_sound_is_noisy_and_unpitched(self):
        breath = _breath_sound(8_000, 0.3, 16_000)
        self.assertEqual(breath.size, 8_000)
        self.assertGreater(float(np.max(np.abs(breath))), 0.05)

    def test_synthesize_song_uses_target_pitch_when_no_tonic(self):
        # With tonic_hz<=0 the mock falls back to the saved voice's median pitch.
        embedding = self.engine.extract_embedding(_tone(300.0), 16_000)
        result = self.engine.synthesize_song(
            [{"char": "la", "jianpu": "1", "durationSeconds": 0.5}], 0.0, embedding
        )
        self.assertGreater(_median_hz(result.samples), 240.0)

    def test_fit_duration_resizes_without_changing_pitch(self):
        tone = _tone(220.0, 16_000, 1.0)
        shorter = _fit_duration(tone, 8_000)
        longer = _fit_duration(tone, 24_000)
        self.assertEqual(shorter.size, 8_000)
        self.assertEqual(longer.size, 24_000)
        # Time-stretching changes length but keeps the pitch (unlike resampling).
        self.assertAlmostEqual(_median_hz(shorter), 220.0, delta=20.0)
        self.assertAlmostEqual(_median_hz(longer), 220.0, delta=20.0)

    def test_resample_rate_preserves_pitch_and_duration(self):
        tone = _tone(220.0, 16_000, 1.0)
        out = _resample_rate(tone, 16_000, 24_000)
        self.assertAlmostEqual(out.size / 24_000, 1.0, places=2)
        self.assertAlmostEqual(_median_hz(out, 24_000), 220.0, delta=20.0)

    def test_trim_silence_drops_padding(self):
        rate = 16_000
        voiced = _tone(220.0, rate, 0.3)
        pad = np.zeros(int(rate * 0.2), dtype=np.float32)
        padded = np.concatenate([pad, voiced, pad]).astype(np.float32)
        trimmed = _trim_silence(padded)
        # The padding is gone and roughly the voiced span remains.
        self.assertLess(trimmed.size, padded.size)
        self.assertGreater(trimmed.size, voiced.size * 0.8)
        self.assertLess(trimmed.size, voiced.size * 1.2)

    def test_song_base_falls_back_to_synth_without_os_tts(self):
        # The base track shared by both engines (mock plays it; OpenVoice re-voices
        # it) uses the synthetic voice at synth_rate when no OS voice is requested.
        notes = [{"char": "a", "freq": 220.0, "duration": 0.4}]
        samples, rate, is_real = _song_base(notes, use_os_tts=False, synth_rate=22_050)
        self.assertFalse(is_real)
        self.assertEqual(rate, 22_050)
        self.assertGreater(samples.size, 0)
        self.assertTrue(np.isfinite(samples).all())

    def test_synthesize_song_uses_real_voice_when_os_tts_available(self):
        from breeze_elf.voice import _system_tts

        if _system_tts("你好") is None:
            self.skipTest("no OS text-to-speech voice on this host")
        engine = MockVoiceEngine(sample_rate=16_000, warmup_seconds=0.0, use_os_tts=True)
        engine.load()
        embedding = engine.extract_embedding(_tone(200.0), 16_000)
        notes = [
            {"char": "小", "jianpu": "1", "durationSeconds": 0.5},
            {"char": "星", "jianpu": "5", "durationSeconds": 0.5},
            {"char": "-", "jianpu": "0", "durationSeconds": 0.3},  # rest
            {"char": "星", "jianpu": "5", "durationSeconds": 0.5},
        ]
        result = engine.synthesize_song(notes, 220.0, embedding)
        # ~1.8s of melody + small gaps, real audio, no NaNs.
        self.assertGreater(result.samples.size / result.sample_rate, 1.2)
        self.assertTrue(np.isfinite(result.samples).all())
        self.assertGreater(float(np.max(np.abs(result.samples))), 0.1)

    def test_os_tts_produces_real_speech_when_available(self):
        from breeze_elf.voice import _system_tts

        spoken = _system_tts("你好世界,測試語音。")
        if spoken is None:
            self.skipTest("no OS text-to-speech voice on this host")
        samples, rate = spoken
        # Real speech is far longer than the few hundred ms a beep would last.
        self.assertGreater(samples.size / rate, 0.8)
        self.assertGreater(float(np.max(np.abs(samples))), 0.1)

        engine = MockVoiceEngine(sample_rate=16_000, warmup_seconds=0.0, use_os_tts=True)
        engine.load()
        embedding = engine.extract_embedding(_tone(150.0), 16_000)
        result = engine.synthesize("你好,這是一段比較長的測試語音。", "zh", embedding)
        self.assertGreater(result.samples.size / result.sample_rate, 0.8)
        self.assertTrue(np.isfinite(result.samples).all())


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

    def test_save_voice_output_writes_wav_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = encode_wav(_tone(200.0), 16_000)
            stored = save_voice_output(wav, tmp, kind="tts", voice_id="voice-x", text="你好")
            self.assertTrue(stored.id.startswith("tts-"))
            self.assertEqual(stored.kind, "tts")
            self.assertEqual(stored.size_bytes, len(wav))
            wav_path = pathlib.Path(tmp) / stored.filename
            self.assertTrue(wav_path.exists())
            sidecar = json.loads((pathlib.Path(tmp) / f"{stored.id}.json").read_text("utf-8"))
            self.assertEqual(sidecar["text"], "你好")
            self.assertEqual(sidecar["voice_id"], "voice-x")

    def test_save_voice_output_defaults_unknown_kind_to_convert(self):
        with tempfile.TemporaryDirectory() as tmp:
            stored = save_voice_output(encode_wav(_tone(200.0), 16_000), tmp, kind="weird")
            self.assertEqual(stored.kind, "convert")
            self.assertTrue(stored.id.startswith("convert-"))

    def test_save_voice_output_keeps_known_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("sing", "recording", "tts", "convert", "convert-source"):
                stored = save_voice_output(encode_wav(_tone(200.0), 16_000), tmp, kind=kind)
                self.assertEqual(stored.kind, kind)
                self.assertTrue(stored.id.startswith(f"{kind}-"))


class VoiceApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(
            main,
            "settings",
            replace(
                main.settings,
                voice_storage_dir=self._tmp.name,
                voice_output_dir=self._tmp.name,
            ),
        )
        self._patch.start()
        self._engine_patch = patch.object(
            main,
            "voice_engine",
            MockVoiceEngine(sample_rate=16_000, warmup_seconds=0.0, use_os_tts=False),
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

    async def test_sing_returns_audio_from_jianpu(self):
        vid = json.loads(
            (
                await main.create_saved_voice(
                    main.VoiceCreateRequest(name="歌手", audioBase64=self._wav_b64(220.0))
                )
            ).body
        )["voice"]["id"]
        sing = await main.voice_sing(
            main.VoiceSingRequest(
                voiceId=vid,
                tonicHz=220.0,
                notes=[
                    main.VoiceSingNote(char="小", jianpu="1", durationSeconds=0.4),
                    main.VoiceSingNote(char="星", jianpu="2", durationSeconds=0.4),
                ],
            )
        )
        data = json.loads(sing.body)
        self.assertTrue(data["ok"])
        self.assertGreater(data["durationSeconds"], 0.0)

    async def test_sing_rejects_when_no_singable_notes(self):
        from fastapi import HTTPException

        vid = json.loads(
            (
                await main.create_saved_voice(
                    main.VoiceCreateRequest(name="x", audioBase64=self._wav_b64(220.0))
                )
            ).body
        )["voice"]["id"]
        with self.assertRaises(HTTPException) as ctx:
            await main.voice_sing(
                main.VoiceSingRequest(
                    voiceId=vid, notes=[main.VoiceSingNote(char="-", jianpu="0")]
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_save_output_persists_generated_clip(self):
        wav_b64 = base64.b64encode(encode_wav(_tone(180.0), 16_000)).decode("ascii")
        saved = await main.save_voice_output_endpoint(
            main.VoiceOutputSaveRequest(audioBase64=wav_b64, kind="tts", text="你好")
        )
        data = json.loads(saved.body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["kind"], "tts")
        self.assertTrue((pathlib.Path(self._tmp.name) / data["filename"]).exists())

    async def test_transcript_analyze_recomputes_pitch_with_global_tonic(self):
        sample_rate = 16_000

        def syllable(hz: float, seconds: float = 0.4) -> np.ndarray:
            axis = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
            tone = 0.4 * np.sin(2 * np.pi * hz * axis) + 0.2 * np.sin(2 * np.pi * 2 * hz * axis)
            return tone.astype(np.float32)

        gap = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
        pieces, characters, start = [], [], 0.0
        for hz in (220.0, 330.0, 440.0):  # 1, 5, then 4 (octave up) against tonic 330
            pieces.append(syllable(hz))
            characters.append(
                {"char": "音", "startSeconds": round(start, 3), "endSeconds": round(start + 0.4, 3)}
            )
            start += 0.5
            pieces.append(gap)
        audio_b64 = base64.b64encode(
            encode_wav(np.concatenate(pieces), sample_rate)
        ).decode("ascii")
        block = main.TranscriptBlock(
            text="音音音", startSeconds=0.0, endSeconds=round(start, 3), characters=characters
        )

        started = await main.analyze_transcript(
            main.TranscriptAnalyzeRequest(
                audioBase64=audio_b64, sampleRate=sample_rate, blocks=[block]
            )
        )
        self.assertTrue(json.loads(started.body)["ok"])

        result = None
        for _ in range(200):
            status = json.loads((await main.analyze_transcript_status()).body)
            if status["status"] == "done":
                result = status["result"]
                break
            if status["status"] == "error":
                self.fail(status["error"])
            await asyncio.sleep(0.02)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["tonicHz"], 330.0, delta=4.0)
        block = result["blocks"][0]
        chars = block["characters"]
        self.assertEqual(len(chars), 3)
        self.assertAlmostEqual(chars[0]["hz"], 220.0, delta=4.0)
        self.assertAlmostEqual(chars[2]["hz"], 440.0, delta=4.0)
        # 330 Hz is the tonic → 簡譜 "1" (do).
        self.assertEqual(chars[1]["jianpu"], "1")

        # 基頻分析: spectrogram carries aligned f0 / intensity / text per time bin.
        spectro = block["spectrogram"]
        bins = spectro["timeBins"]
        for field in ("f0", "intensity", "times", "text"):
            self.assertEqual(len(spectro[field]), bins)
        self.assertIn("音", spectro["text"])  # time bins tagged with the sounding 文字
        self.assertTrue(any(hz for hz in spectro["f0"]))

    async def test_save_output_rejects_empty_audio(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            await main.save_voice_output_endpoint(
                main.VoiceOutputSaveRequest(audioBase64="!!!notbase64!!!", kind="convert")
            )
        self.assertEqual(ctx.exception.status_code, 400)

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
