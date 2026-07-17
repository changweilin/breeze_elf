import os
import unittest
from unittest import mock

import numpy as np

from breeze_elf.audio import (
    _SILERO_CHUNK_SAMPLES,
    _SILERO_CONTEXT_SAMPLES,
    _SILERO_STATE,
    AudioUtteranceBuffer,
    RmsSpeechDetector,
    SileroSpeechDetector,
    make_speech_detector,
)
from breeze_elf.config import get_settings


class ScriptedDetector:
    """Returns a fixed booleans sequence regardless of audio, so a test can prove
    the *detector* (not RMS energy) drives segmentation."""

    name = "scripted"

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self._index = 0

    def is_speech(self, frame):
        if self._index < len(self._decisions):
            value = self._decisions[self._index]
            self._index += 1
            return value
        return False


class _FakeSileroSession:
    """Stand-in for the onnxruntime session: records each input and the LSTM state
    it was handed, returns the next scripted probability, and *evolves* the state
    (h+1, c+1) so a test can prove the detector carries returned state forward
    across chunks instead of resetting it to zeros every call."""

    def __init__(self, probs):
        self._probs = list(probs)
        self.calls = []       # model inputs, per call
        self.state_in = []    # (h, c) the detector handed us, per call

    def run(self, _outputs, feeds):
        model_input = feeds["input"]
        assert model_input.shape == (1, _SILERO_CHUNK_SAMPLES + _SILERO_CONTEXT_SAMPLES)
        assert feeds["h"].shape == _SILERO_STATE
        assert feeds["c"].shape == _SILERO_STATE
        self.calls.append(model_input.copy())
        self.state_in.append((feeds["h"].copy(), feeds["c"].copy()))
        prob = self._probs.pop(0)
        return np.array([[prob]], dtype=np.float32), feeds["h"] + 1.0, feeds["c"] + 1.0


def _stub_silero(probs, **kwargs):
    """A SileroSpeechDetector wired to a fake session (real model never loaded)."""
    detector = SileroSpeechDetector(
        rms_threshold=kwargs.pop("rms_threshold", 0.01),
        model_path="__no_such_silero_model__.onnx",
        **kwargs,
    )
    assert not detector.available  # bogus path => real load failed, as intended
    detector._session = _FakeSileroSession(probs)
    detector.available = True
    detector._degraded = False
    return detector


class SpeechDetectorSegmentationTests(unittest.TestCase):
    def test_injected_detector_drives_segmentation_over_energy(self):
        # All-zero audio (RMS == 0, energy VAD would never fire); the scripted
        # detector alone decides where the utterance is.
        detector = ScriptedDetector([False, True, True, True, False, False])
        buffer = AudioUtteranceBuffer(
            sample_rate=10,
            frame_ms=100,  # 1 sample / frame at 10 Hz
            pre_roll_ms=0,
            end_silence_ms=200,  # 2 silent frames end the segment
            rms_threshold=0.01,
            speech_detector=detector,
        )
        payload = np.zeros(6, dtype="<i2").tobytes()

        windows = buffer.append_pcm16(payload)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].kind, "utterance")
        self.assertEqual(windows[0].start_seconds, 0.1)  # segment starts at frame 1
        self.assertEqual(windows[0].samples.size, 5)  # frames 1..5 (incl. trailing silence)

    def test_default_detector_is_rms(self):
        buffer = AudioUtteranceBuffer(sample_rate=10, frame_ms=100, rms_threshold=0.01)
        self.assertEqual(buffer.detector_name, "rms")

    def test_rms_detector_matches_threshold(self):
        detector = RmsSpeechDetector(0.02)
        self.assertTrue(detector.is_speech(np.full(160, 0.5, dtype=np.float32)))
        self.assertFalse(detector.is_speech(np.zeros(160, dtype=np.float32)))


class SileroStreamingGlueTests(unittest.TestCase):
    def test_hysteresis_holds_decision_between_thresholds(self):
        detector = _stub_silero([0.9, 0.4, 0.2, 0.4], speech_threshold=0.5, neg_threshold=0.35)
        frame = (np.arange(_SILERO_CHUNK_SAMPLES, dtype=np.float32) / _SILERO_CHUNK_SAMPLES)

        self.assertTrue(detector.is_speech(frame))   # 0.9 >= 0.5 -> speech
        self.assertTrue(detector.is_speech(frame))   # 0.4 in band -> hold True
        self.assertFalse(detector.is_speech(frame))  # 0.2 <  0.35 -> silence
        self.assertFalse(detector.is_speech(frame))  # 0.4 in band -> hold False

    def test_context_carries_previous_chunk_tail(self):
        detector = _stub_silero([0.1, 0.1], speech_threshold=0.5, neg_threshold=0.35)
        frame = (np.arange(_SILERO_CHUNK_SAMPLES, dtype=np.float32) + 1.0)

        detector.is_speech(frame)
        detector.is_speech(frame)

        session = detector._session
        self.assertEqual(len(session.calls), 2)
        # First chunk's context is zeros; the second reuses the first chunk's tail.
        np.testing.assert_array_equal(session.calls[0][0, :_SILERO_CONTEXT_SAMPLES], 0.0)
        np.testing.assert_array_equal(
            session.calls[1][0, :_SILERO_CONTEXT_SAMPLES], frame[-_SILERO_CONTEXT_SAMPLES:]
        )

    def test_lstm_state_persists_across_chunks(self):
        # The streaming adaptation: the h/c the model returns must be fed into the
        # next call (not reset to zeros as faster-whisper's batch __call__ does).
        detector = _stub_silero([0.1, 0.1], speech_threshold=0.5, neg_threshold=0.35)
        frame = np.zeros(_SILERO_CHUNK_SAMPLES, dtype=np.float32)

        detector.is_speech(frame)
        detector.is_speech(frame)

        session = detector._session
        first_h, first_c = session.state_in[0]
        second_h, second_c = session.state_in[1]
        np.testing.assert_array_equal(first_h, 0.0)  # initial state is zeros
        np.testing.assert_array_equal(first_c, 0.0)
        np.testing.assert_array_equal(second_h, 1.0)  # carried the model's returned h (0+1)
        np.testing.assert_array_equal(second_c, 1.0)

    def test_multiple_chunks_per_frame_use_max_probability(self):
        # Loud chunk FIRST so 'max over the frame' and 'last chunk wins' disagree:
        # max(0.9, 0.2) -> speech, whereas last-chunk (0.2) would be silence.
        detector = _stub_silero([0.9, 0.2], speech_threshold=0.5, neg_threshold=0.35)
        frame = np.zeros(_SILERO_CHUNK_SAMPLES * 2, dtype=np.float32)

        self.assertTrue(detector.is_speech(frame))
        self.assertEqual(len(detector._session.calls), 2)

    def test_threshold_boundaries_are_inclusive_and_exclusive(self):
        # p == speech_threshold counts as speech (>=); p just under neg is silence (<).
        detector = _stub_silero([0.5, 0.349], speech_threshold=0.5, neg_threshold=0.35)
        frame = np.zeros(_SILERO_CHUNK_SAMPLES, dtype=np.float32)

        self.assertTrue(detector.is_speech(frame))   # 0.5 >= 0.5
        self.assertFalse(detector.is_speech(frame))  # 0.349 < 0.35

    def test_subchunk_frames_buffer_until_aligned(self):
        detector = _stub_silero([0.9], speech_threshold=0.5, neg_threshold=0.35)
        partial = np.zeros(300, dtype=np.float32)

        self.assertFalse(detector.is_speech(partial))  # 300 < 512, no chunk yet -> hold
        self.assertEqual(len(detector._session.calls), 0)
        self.assertTrue(detector.is_speech(partial))   # 600 >= 512, one chunk fires
        self.assertEqual(len(detector._session.calls), 1)


class SpeechDetectorFactoryTests(unittest.TestCase):
    def test_rms_request_returns_rms(self):
        self.assertEqual(make_speech_detector("rms", rms_threshold=0.01).name, "rms")

    def test_silero_missing_model_falls_back_to_rms(self):
        detector = make_speech_detector(
            "silero", rms_threshold=0.01, model_path="__no_such_silero_model__.onnx"
        )
        self.assertEqual(detector.name, "rms")

    def test_runtime_inference_failure_degrades_to_rms(self):
        class _BoomSession:
            def run(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        detector = _stub_silero([], speech_threshold=0.5)
        detector._session = _BoomSession()
        loud = np.full(_SILERO_CHUNK_SAMPLES, 0.5, dtype=np.float32)
        # First inference raises -> permanent RMS fallback, still returns a decision.
        self.assertTrue(detector.is_speech(loud))
        self.assertTrue(detector._degraded)
        self.assertFalse(detector.is_speech(np.zeros(_SILERO_CHUNK_SAMPLES, dtype=np.float32)))


class SileroRealModelSmokeTest(unittest.TestCase):
    def test_bundled_model_loads_and_runs(self):
        detector = make_speech_detector("silero", rms_threshold=0.01)
        if detector.name != "silero":
            self.skipTest("silero model / onnxruntime not available in this env")
        rng = np.random.default_rng(0)
        for _ in range(3):
            frame = rng.standard_normal(1600).astype(np.float32) * 0.1
            self.assertIsInstance(detector.is_speech(frame), bool)


class VadConfigResolutionTests(unittest.TestCase):
    # clear=True makes these hermetic: get_settings only reads BREEZE_* (defaulting
    # when absent), so ambient BREEZE_* env can't flip an assertion.
    def test_segmenter_silero_is_alias_for_vad_plus_silero(self):
        with mock.patch.dict(os.environ, {"BREEZE_SEGMENTER": "silero"}, clear=True):
            settings = get_settings()
        self.assertEqual(settings.segmenter, "vad")
        self.assertEqual(settings.vad_detector, "silero")

    def test_segmenter_silero_alias_beats_explicit_rms_detector(self):
        # Documented precedence: BREEZE_SEGMENTER=silero forces silero even if
        # BREEZE_VAD_DETECTOR contradicts it.
        with mock.patch.dict(
            os.environ, {"BREEZE_SEGMENTER": "silero", "BREEZE_VAD_DETECTOR": "rms"}, clear=True
        ):
            settings = get_settings()
        self.assertEqual(settings.segmenter, "vad")
        self.assertEqual(settings.vad_detector, "silero")

    def test_vad_detector_env_selects_silero(self):
        with mock.patch.dict(os.environ, {"BREEZE_VAD_DETECTOR": "silero"}, clear=True):
            settings = get_settings()
        self.assertEqual(settings.vad_detector, "silero")
        self.assertEqual(settings.segmenter, "vad")

    def test_invalid_vad_detector_falls_back_to_rms(self):
        with mock.patch.dict(os.environ, {"BREEZE_VAD_DETECTOR": "bogus"}, clear=True):
            settings = get_settings()
        self.assertEqual(settings.vad_detector, "rms")

    def test_thresholds_are_clamped_and_ordered(self):
        with mock.patch.dict(
            os.environ,
            {"BREEZE_VAD_SPEECH_THRESHOLD": "1.5", "BREEZE_VAD_NEG_THRESHOLD": "0.9"},
            clear=True,
        ):
            settings = get_settings()
        self.assertEqual(settings.vad_speech_threshold, 1.0)  # clamped into [0, 1]
        self.assertLessEqual(settings.vad_neg_threshold, settings.vad_speech_threshold)

    def test_negative_threshold_clamps_to_zero(self):
        with mock.patch.dict(
            os.environ, {"BREEZE_VAD_SPEECH_THRESHOLD": "-0.5"}, clear=True
        ):
            settings = get_settings()
        self.assertEqual(settings.vad_speech_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
