import asyncio
import base64
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from breeze_elf import main
from breeze_elf.asr import ASRResult, MockASR, WordTiming
from breeze_elf.asr_queue import QueuedASRResult
from breeze_elf.audio import (
    AudioUtteranceBuffer,
    AudioWindow,
    AudioWindowBuffer,
    calculate_rms,
    estimate_noise_floor,
    summarize_pitch,
)
from breeze_elf.main import (
    StreamState,
    _character_payloads,
    _dedupe_overlap_cap,
    _handle_audio_payload,
    _handle_text_message,
    _novel_text,
    _padded_char_spans,
    _should_drop_asr_result,
)


class ImmediateASRQueue:
    backend = "test"
    device = "cpu"
    queue_depth = 0

    async def transcribe(self, samples, sample_rate, language, *, languages=(), prompt_terms=()):
        del samples, sample_rate, languages, prompt_terms
        return QueuedASRResult(
            result=ASRResult(
                text="最後一句",
                language=language,
                duration_ms=0,
                backend=self.backend,
                device=self.device,
            ),
            queue_wait_ms=0,
            queue_depth=0,
        )


class RecordingASRQueue:
    backend = "test"
    device = "cpu"
    queue_depth = 0

    def __init__(self):
        self.received_rms = 0.0

    async def transcribe(self, samples, sample_rate, language, *, languages=(), prompt_terms=()):
        del sample_rate, languages, prompt_terms
        self.received_rms = calculate_rms(samples)
        return QueuedASRResult(
            result=ASRResult(
                text="",
                language=language,
                duration_ms=0,
                backend=self.backend,
                device=self.device,
            ),
            queue_wait_ms=0,
            queue_depth=0,
        )


class MainTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_transcript_endpoint_saves_to_configured_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(main, "settings", replace(main.settings, remote_storage_dir=tmp)):
                response = await main.create_remote_transcript(
                    main.TranscriptSaveRequest(text="遠端儲存測試")
                )

            data = json.loads(response.body)

            self.assertTrue(data["ok"])
            self.assertTrue(data["filename"].endswith(".txt"))
            self.assertEqual(
                (Path(tmp) / data["filename"]).read_text(encoding="utf-8"),
                "遠端儲存測試\n",
            )

    async def test_remote_transcript_endpoint_bundles_jianpu_and_audio(self):
        audio = b"RIFFfake-wav-payload"
        payload = main.TranscriptSaveRequest(
            text="天氣很好",
            title="note",
            sampleRate=16000,
            blocks=[
                {
                    "text": "天氣很好",
                    "startSeconds": 0.0,
                    "endSeconds": 0.6,
                    "characters": [
                        {"char": "天", "startSeconds": 0.0, "hz": 200.0, "jianpu": "3"},
                        {"char": "氣", "startSeconds": 0.2, "hz": 260.0, "jianpu": "5"},
                    ],
                }
            ],
            audioBase64=base64.b64encode(audio).decode("ascii"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(main, "settings", replace(main.settings, remote_storage_dir=tmp)):
                response = await main.create_remote_transcript(payload)

            data = json.loads(response.body)
            directory = Path(tmp)

            self.assertTrue(data["ok"])
            self.assertTrue(data["filename"].endswith(".txt"))
            self.assertTrue(data["jsonFilename"].endswith(".json"))
            self.assertTrue(data["audioFilename"].endswith(".wav"))
            self.assertEqual((directory / data["audioFilename"]).read_bytes(), audio)

            document = json.loads((directory / data["jsonFilename"]).read_text(encoding="utf-8"))
            self.assertEqual(document["blocks"][0]["characters"][1]["jianpu"], "5")
            self.assertEqual(document["sampleRate"], 16000)

    async def test_audio_payload_reports_backpressure_when_queue_drops_window(self):
        state = StreamState(
            started=True,
            segmenter=AudioWindowBuffer(
                sample_rate=4,
                window_seconds=1.0,
                overlap_seconds=0.0,
                rms_threshold=0.0,
            ),
            queue=asyncio.Queue(maxsize=1),
        )
        events = []

        async def send_json(payload):
            events.append(payload)

        payload = np.full(8, 1000, dtype="<i2").tobytes()
        await _handle_audio_payload(payload, state, send_json)

        self.assertEqual(state.dropped_windows, 1)
        self.assertEqual(state.queue.qsize(), 1)
        self.assertTrue(any(event.get("backpressure") for event in events))

    async def test_audio_payload_rejects_audio_before_start(self):
        events = []

        async def send_json(payload):
            events.append(payload)

        await _handle_audio_payload(b"1234", StreamState(), send_json)

        self.assertEqual(events[0]["type"], "error")
        self.assertIn("before start", events[0]["message"])

    async def test_stop_flushes_active_utterance_before_reporting_stopped(self):
        events = []

        async def send_json(payload):
            events.append(payload)

        state = StreamState(
            started=True,
            segmenter=AudioUtteranceBuffer(
                sample_rate=10,
                frame_ms=100,
                pre_roll_ms=0,
                end_silence_ms=500,
                rms_threshold=0.01,
            ),
            queue=asyncio.Queue(maxsize=4),
        )
        state.processor_task = asyncio.create_task(
            main._process_windows(state, send_json, ImmediateASRQueue(), ("zh",), ())
        )
        state.segmenter.append_pcm16(np.array([1000, 1000, 1000], dtype="<i2").tobytes())

        should_stop = await _handle_text_message(
            '{"type":"stop","reason":"test"}',
            state,
            send_json,
            ImmediateASRQueue(),
        )

        self.assertTrue(should_stop)
        final_index = next(index for index, event in enumerate(events) if event["type"] == "final")
        stopped_index = next(
            index for index, event in enumerate(events) if event.get("stopped")
        )
        self.assertLess(final_index, stopped_index)
        self.assertEqual(events[final_index]["text"], "最後一句")
        self.assertEqual(events[final_index]["startSeconds"], 0.0)
        self.assertEqual(events[final_index]["endSeconds"], 0.3)
        self.assertIsNone(events[final_index]["pitch"]["medianHz"])
        self.assertEqual(events[final_index]["pitch"]["points"], [])
        self.assertEqual(events[stopped_index]["reason"], "test")

    async def test_process_windows_prepares_asr_audio_without_changing_window_stats(self):
        sample_rate = 16_000
        time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (0.02 * np.sin(2 * np.pi * 220.0 * time_axis)).astype(np.float32)
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=samples,
            rms=calculate_rms(samples),
            is_speech=True,
            kind="utterance",
        )
        state = StreamState(started=True, queue=asyncio.Queue(maxsize=1))
        await state.queue.put(window)
        state.stop_event.set()
        events = []

        async def send_json(payload):
            events.append(payload)

        asr_queue = RecordingASRQueue()
        with patch.object(
            main,
            "settings",
            replace(main.settings, audio_preprocess="natural", sample_rate=sample_rate),
        ):
            await main._process_windows(state, send_json, asr_queue, ("zh",), ())

        self.assertGreater(asr_queue.received_rms, window.rms)
        self.assertTrue(any(event.get("rms") == round(window.rms, 5) for event in events))

    async def test_process_windows_applies_glossary_corrections(self):
        sample_rate = 16_000
        time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (0.05 * np.sin(2 * np.pi * 220.0 * time_axis)).astype(np.float32)
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=samples,
            rms=calculate_rms(samples),
            is_speech=True,
            kind="utterance",
        )

        class _FixedTextQueue(ImmediateASRQueue):
            async def transcribe(
                self, samples, sample_rate, language, *, languages=(), prompt_terms=()
            ):
                del samples, sample_rate, languages, prompt_terms
                return QueuedASRResult(
                    result=ASRResult(
                        text="機器學習很有趣",
                        language=language,
                        duration_ms=0,
                        backend=self.backend,
                        device=self.device,
                    ),
                    queue_wait_ms=0,
                    queue_depth=0,
                )

        state = StreamState(started=True, queue=asyncio.Queue(maxsize=1))
        await state.queue.put(window)
        state.stop_event.set()
        events = []

        async def send_json(payload):
            events.append(payload)

        glossary = (main.GlossaryEntry(source="機器學習", target="Mike 學習"),)
        await main._process_windows(state, send_json, _FixedTextQueue(), ("zh",), glossary)

        final = next(event for event in events if event["type"] == "final")
        self.assertEqual(final["text"], "Mike 學習很有趣")
        self.assertEqual(final["transcript"], "Mike 學習很有趣")

    async def test_process_windows_attaches_translation_and_speaker_on_final(self):
        from breeze_elf.diarize import OnlineSpeakerClusterer

        sample_rate = 16_000
        time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (0.05 * np.sin(2 * np.pi * 220.0 * time_axis)).astype(np.float32)
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=samples,
            rms=calculate_rms(samples),
            is_speech=True,
            kind="utterance",
        )

        class _FixedTextQueue(ImmediateASRQueue):
            async def transcribe(
                self, samples, sample_rate, language, *, languages=(), prompt_terms=()
            ):
                del samples, sample_rate, languages, prompt_terms
                return QueuedASRResult(
                    result=ASRResult(
                        text="你好世界", language=language, duration_ms=0,
                        backend=self.backend, device=self.device,
                    ),
                    queue_wait_ms=0, queue_depth=0,
                )

        class _FakeTranslator:
            available = True

            def translate(self, text, src, tgt):
                return f"[{tgt}]{text}"

        class _FakeEmbedder:
            def embed(self, samples, sample_rate):
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        state = StreamState(started=True, queue=asyncio.Queue(maxsize=1))
        state.clusterer = OnlineSpeakerClusterer(max_speakers=4, threshold=0.75)
        await state.queue.put(window)
        state.stop_event.set()
        events = []

        async def send_json(payload):
            events.append(payload)

        with patch.object(main, "translator", _FakeTranslator()), patch.object(
            main, "diarizer", _FakeEmbedder()
        ):
            await main._process_windows(
                state, send_json, _FixedTextQueue(), ("zh",), (),
                translate=True, translate_target="en",
            )

        final = next(event for event in events if event["type"] == "final")
        self.assertEqual(final["translation"], "[en]你好世界")
        self.assertEqual(final["speaker"], 0)  # first speaker, 0-based

    async def test_process_windows_no_translation_or_speaker_when_disabled(self):
        tone = 0.05 * np.sin(2 * np.pi * 220.0 * np.arange(16_000) / 16_000)
        window = AudioWindow(
            index=0, start_seconds=0.0, end_seconds=1.0,
            samples=tone.astype(np.float32),
            rms=0.03, is_speech=True, kind="utterance",
        )

        class _FixedTextQueue(ImmediateASRQueue):
            async def transcribe(
                self, samples, sample_rate, language, *, languages=(), prompt_terms=()
            ):
                del samples, sample_rate, languages, prompt_terms
                return QueuedASRResult(
                    result=ASRResult(text="嗨", language=language, duration_ms=0,
                                     backend=self.backend, device=self.device),
                    queue_wait_ms=0, queue_depth=0,
                )

        state = StreamState(started=True, queue=asyncio.Queue(maxsize=1))  # no clusterer
        await state.queue.put(window)
        state.stop_event.set()
        events = []

        async def send_json(payload):
            events.append(payload)

        await main._process_windows(state, send_json, _FixedTextQueue(), ("zh",), ())
        final = next(event for event in events if event["type"] == "final")
        self.assertIsNone(final["translation"])  # translate defaulted off
        self.assertIsNone(final["speaker"])  # no clusterer built


class CharacterPayloadTests(unittest.TestCase):
    def test_character_payloads_attach_jianpu_and_absolute_timing(self):
        sample_rate = 16_000
        seconds = 0.6
        time_axis = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
        samples = (0.4 * np.sin(2 * np.pi * 220.0 * time_axis)).astype(np.float32)
        window = AudioWindow(
            index=0,
            start_seconds=2.0,
            end_seconds=2.0 + seconds,
            samples=samples,
            rms=calculate_rms(samples),
            is_speech=True,
            kind="utterance",
        )
        words = MockASR().transcribe(samples, sample_rate, "zh").words
        summary = summarize_pitch(samples, sample_rate)

        characters = _character_payloads(window, words, sample_rate, summary)

        self.assertEqual(len(characters), len(words))
        self.assertTrue(all(set(c) >= {"char", "startSeconds", "jianpu", "hz"} for c in characters))
        # absolute timing is offset by the window start
        self.assertGreaterEqual(characters[0]["startSeconds"], 2.0)
        self.assertLessEqual(characters[-1]["endSeconds"], 2.0 + seconds + 1e-6)
        # a steady tone resolves to the tonic (1) for every voiced character
        voiced = [c["jianpu"] for c in characters if c["hz"]]
        self.assertTrue(voiced)
        self.assertTrue(all(jianpu == "1" for jianpu in voiced))

    def test_character_payloads_mark_glide_and_attach_analysis_fields(self):
        sample_rate = 16_000
        seconds = 0.5
        time_axis = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
        rate = (300.0 - 200.0) / seconds
        phase = 2 * np.pi * (200.0 * time_axis + 0.5 * rate * time_axis * time_axis)
        samples = (0.4 * np.sin(phase)).astype(np.float32)
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=seconds,
            samples=samples,
            rms=calculate_rms(samples),
            is_speech=True,
            kind="utterance",
        )
        words = (WordTiming(word="滑", start=0.0, end=seconds),)
        summary = summarize_pitch(samples, sample_rate)

        characters = _character_payloads(window, words, sample_rate, summary)

        self.assertEqual(len(characters), 1)
        character = characters[0]
        self.assertTrue(character["isGlide"])
        self.assertIn("↗", character["jianpu"])
        self.assertLess(character["startHz"], character["endHz"])
        self.assertIn("centsOff", character)
        self.assertGreater(character["intensity"], 0.0)
        expected_fields = {
            "char",
            "startSeconds",
            "jianpu",
            "hz",
            "centsOff",
            "intensityStart",
            "intensityEnd",
        }
        self.assertTrue(set(character) >= expected_fields)

    def test_character_payloads_empty_without_words(self):
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=np.zeros(16_000, dtype=np.float32),
            rms=0.0,
            is_speech=True,
        )
        summary = summarize_pitch(window.samples, 16_000)
        self.assertEqual(_character_payloads(window, (), 16_000, summary), [])


class PaddedCharSpanTests(unittest.TestCase):
    def test_spans_match_fixed_padding_without_audio(self):
        block = {
            "startSeconds": 0.0,
            "endSeconds": 1.0,
            "characters": [{"char": "好", "startSeconds": 0.4, "endSeconds": 0.6}],
        }
        spans = _padded_char_spans(block)
        self.assertEqual(len(spans), 1)
        _, start, end = spans[0]
        # fixed attack 40ms / release 90ms (a lone 字 keeps the full pad)
        self.assertAlmostEqual(start, 0.36, places=3)
        self.assertAlmostEqual(end, 0.69, places=3)

    def test_spans_grow_through_voiceless_onset_with_audio(self):
        sample_rate = 16_000
        silence = np.full(3_200, 0.002, dtype=np.float32)  # 0.0–0.2s room tone
        onset = np.full(1_600, 0.012, dtype=np.float32)  # 0.2–0.3s unvoiced consonant
        voiced = np.full(3_200, 0.3, dtype=np.float32)  # 0.3–0.5s loud vowel
        tail = np.full(1_600, 0.002, dtype=np.float32)  # 0.5–0.6s silence
        samples = np.concatenate([silence, onset, voiced, tail])
        block = {
            "startSeconds": 0.0,
            "endSeconds": 0.6,
            "characters": [{"char": "西", "startSeconds": 0.3, "endSeconds": 0.5}],
        }
        floor = estimate_noise_floor(samples, sample_rate)

        _, start, end = _padded_char_spans(block, samples, sample_rate, floor)[0]

        # fixed attack alone would only reach 0.26s; content growth reaches 0.2s.
        self.assertAlmostEqual(start, 0.2, delta=0.02)
        self.assertLess(start, 0.26)


class NovelTextTests(unittest.TestCase):
    def test_novel_text_removes_overlap(self):
        self.assertEqual(_novel_text("今天 天氣很好", "天氣很好 我們出門"), "我們出門")

    def test_novel_text_ignores_duplicate_tail(self):
        self.assertEqual(_novel_text("今天 天氣很好", "天氣很好"), "")

    def test_novel_text_ignores_punctuation_and_spacing_when_deduping(self):
        self.assertEqual(_novel_text("今天，天氣很好。", "今天 天氣很好"), "")

    def test_novel_text_removes_overlap_with_spacing_difference(self):
        self.assertEqual(_novel_text("今天，天氣很好", "天氣 很好 我們出門"), "我們出門")

    def test_novel_text_keeps_repeated_lyric_for_disjoint_utterance(self):
        # A disjoint VAD utterance (cap 0) that repeats an earlier 歌詞 line must be
        # kept — a sung chorus is a real repeat, not a window-overlap artefact.
        self.assertEqual(_novel_text("好想你", "好想你", max_overlap_chars=0), " 好想你")
        self.assertEqual(
            _novel_text("天氣很好 我們出門", "天氣很好", max_overlap_chars=0),
            " 天氣很好",
        )

    def test_novel_text_trims_force_split_boundary_overlap(self):
        # A >max_segment force-split re-includes the pre-roll, so the continuation
        # duplicates the previous segment's last few 字 — trim just that boundary.
        self.assertEqual(
            _novel_text("今天天氣很好", "很好我們出門去玩", max_overlap_chars=6),
            "我們出門去玩",
        )


class DedupeOverlapCapTests(unittest.TestCase):
    def _window(self, kind, start, end):
        return AudioWindow(
            index=0,
            start_seconds=start,
            end_seconds=end,
            samples=np.zeros(1, dtype=np.float32),
            rms=0.0,
            is_speech=True,
            kind=kind,
        )

    def test_window_segmenter_uses_full_dedupe(self):
        self.assertIsNone(_dedupe_overlap_cap(self._window("window", 2.0, 4.0), 2.5))

    def test_disjoint_utterance_disables_dedupe(self):
        # Starts after the previous segment ended → real repeat, keep everything.
        self.assertEqual(_dedupe_overlap_cap(self._window("utterance", 5.0, 7.0), 4.0), 0)
        self.assertEqual(_dedupe_overlap_cap(self._window("utterance", 0.0, 2.0), None), 0)

    def test_overlapping_continuation_trims_boundary(self):
        # An overlapping continuation (force-split fallback / window-mode carry)
        # starts before the previous segment ended → trim just that boundary.
        self.assertEqual(_dedupe_overlap_cap(self._window("utterance", 11.7, 23.0), 12.0), 6)


class SilenceHallucinationTests(unittest.TestCase):
    def test_drops_low_energy_high_no_speech_result(self):
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=np.zeros(16000, dtype=np.float32),
            rms=0.01,
            is_speech=True,
        )
        result = ASRResult(
            text="這是一段幻覺文字",
            language="zh",
            duration_ms=1,
            backend="test",
            device="cpu",
            no_speech_prob=0.8,
        )

        self.assertTrue(_should_drop_asr_result(window, result))

    def test_drops_common_sponsor_hallucination_when_low_energy(self):
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=np.zeros(16000, dtype=np.float32),
            rms=0.01,
            is_speech=True,
        )
        result = ASRResult(
            text="請不吝點讚訂閱轉發打賞支持明鏡與點點欄目",
            language="zh",
            duration_ms=1,
            backend="test",
            device="cpu",
            no_speech_prob=0.1,
        )

        self.assertTrue(_should_drop_asr_result(window, result))

    def test_keeps_normal_speech(self):
        window = AudioWindow(
            index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            samples=np.ones(16000, dtype=np.float32) * 0.05,
            rms=0.05,
            is_speech=True,
        )
        result = ASRResult(
            text="今天下午三點開會",
            language="zh",
            duration_ms=1,
            backend="test",
            device="cpu",
            no_speech_prob=0.1,
        )

        self.assertFalse(_should_drop_asr_result(window, result))


class StaticAssetsTests(unittest.TestCase):
    def test_web_dir_points_to_existing_static_assets(self):
        self.assertTrue((Path(main.WEB_DIR) / "index.html").is_file())

    def test_root_static_assets_are_whitelisted_and_present(self):
        for asset_name in main.ROOT_STATIC_MEDIA_TYPES:
            self.assertTrue((Path(main.WEB_DIR) / asset_name).is_file(), asset_name)


if __name__ == "__main__":
    unittest.main()
