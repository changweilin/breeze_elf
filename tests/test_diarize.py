import dataclasses
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from breeze_elf.config import get_settings
from breeze_elf.diarize import (
    NullDiarizer,
    OnlineSpeakerClusterer,
    OnnxSpeakerEmbedder,
    _unit_normalize,
    build_clusterer,
    build_diarizer,
    log_mel_fbank,
)


class UnitNormalizeTests(unittest.TestCase):
    def test_unit_norm(self):
        out = _unit_normalize(np.array([3.0, 4.0]))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)

    def test_rejects_non_finite(self):
        self.assertIsNone(_unit_normalize(np.array([np.nan, 1.0])))
        self.assertIsNone(_unit_normalize(np.array([np.inf, 0.0])))

    def test_rejects_zero_vector(self):
        self.assertIsNone(_unit_normalize(np.zeros(4)))


class OnlineSpeakerClustererTests(unittest.TestCase):
    def test_first_utterance_is_speaker_zero(self):
        cl = OnlineSpeakerClusterer(max_speakers=4, threshold=0.75)
        self.assertEqual(cl.assign(np.array([1.0, 0.0, 0.0])), 0)  # 0-based

    def test_same_voice_reuses_label(self):
        cl = OnlineSpeakerClusterer(max_speakers=4, threshold=0.75)
        cl.assign(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(cl.assign(np.array([2.0, 0.0, 0.0])), 0)  # same direction
        self.assertEqual(cl.speaker_count, 1)

    def test_distinct_voice_makes_new_speaker(self):
        cl = OnlineSpeakerClusterer(max_speakers=4, threshold=0.75)
        cl.assign(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(cl.assign(np.array([0.0, 1.0, 0.0])), 1)  # orthogonal → new
        self.assertEqual(cl.speaker_count, 2)

    def test_max_speakers_cap_folds_into_nearest(self):
        cl = OnlineSpeakerClusterer(max_speakers=2, threshold=0.9)
        cl.assign(np.array([1.0, 0.0, 0.0]))
        cl.assign(np.array([0.0, 1.0, 0.0]))
        # A third distinct voice can't open a new cluster; it joins the nearest.
        label = cl.assign(np.array([0.0, 0.0, 1.0]))
        self.assertIn(label, (0, 1))
        self.assertEqual(cl.speaker_count, 2)

    def test_non_finite_embedding_returns_none(self):
        cl = OnlineSpeakerClusterer()
        self.assertIsNone(cl.assign(np.array([np.nan, 1.0, 0.0])))
        self.assertEqual(cl.speaker_count, 0)  # centroid never polluted

    def test_build_clusterer_uses_settings(self):
        settings = dataclasses.replace(
            get_settings(), diarize_max_speakers=3, diarize_threshold=0.6
        )
        cl = build_clusterer(settings)
        self.assertEqual(cl.max_speakers, 3)
        self.assertEqual(cl.threshold, 0.6)


class LogMelFbankTests(unittest.TestCase):
    def test_shape_and_finiteness(self):
        feats = log_mel_fbank(np.random.randn(16000).astype(np.float32), 16000, n_mels=80)
        self.assertEqual(feats.shape[1], 80)
        self.assertGreater(feats.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_empty_for_short_audio(self):
        self.assertEqual(log_mel_fbank(np.zeros(50, dtype=np.float32), 16000).shape, (0, 80))

    def test_resamples_non_16k(self):
        # 8 kHz for 1 s → resampled to ~16 k before framing, so frames are produced.
        feats = log_mel_fbank(np.random.randn(8000).astype(np.float32), 8000)
        self.assertGreater(feats.shape[0], 0)


class _FakeSession:
    """Stand-in onnxruntime session. Records the feature shapes it is fed and can
    reject the first layout to exercise the (frames,mels)→(mels,frames) fallback."""

    def __init__(self, fail_first=False, always_fail=False):
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.seen_shapes = []

    def run(self, output_names, feeds):
        array = next(iter(feeds.values()))
        self.seen_shapes.append(array.shape)
        if self.always_fail or (self.fail_first and len(self.seen_shapes) == 1):
            raise RuntimeError("shape mismatch")
        return [np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)]


def _embedder_with_session(session):
    emb = OnnxSpeakerEmbedder(Path("nonexistent.onnx"))
    emb._session = session  # bypass load()
    emb._input_name = "feats"
    return emb


class OnnxSpeakerEmbedderTests(unittest.TestCase):
    def test_available_false_without_model_file(self):
        self.assertFalse(OnnxSpeakerEmbedder(Path("nope.onnx")).available)

    def test_embed_none_when_session_unavailable(self):
        emb = OnnxSpeakerEmbedder(Path("nope.onnx"))
        emb._failed = True  # simulate a failed/absent load
        self.assertIsNone(emb.embed(np.random.randn(16000).astype(np.float32), 16000))

    def test_embed_returns_flat_vector(self):
        emb = _embedder_with_session(_FakeSession())
        out = emb.embed(np.random.randn(16000).astype(np.float32), 16000)
        self.assertIsNotNone(out)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape[0], 4)

    def test_layout_fallback_on_first_failure(self):
        session = _FakeSession(fail_first=True)
        out = _embedder_with_session(session).embed(
            np.random.randn(16000).astype(np.float32), 16000
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(session.seen_shapes), 2)  # tried both layouts

    def test_total_inference_failure_returns_none(self):
        out = _embedder_with_session(_FakeSession(always_fail=True)).embed(
            np.random.randn(16000).astype(np.float32), 16000
        )
        self.assertIsNone(out)

    def test_empty_audio_returns_none(self):
        self.assertIsNone(_embedder_with_session(_FakeSession()).embed(np.empty(0), 16000))


class BuildDiarizerTests(unittest.TestCase):
    def test_off_returns_null(self):
        settings = dataclasses.replace(get_settings(), diarize_enabled=False)
        self.assertIsInstance(build_diarizer(settings), NullDiarizer)

    def test_enabled_but_missing_model_returns_null(self):
        settings = dataclasses.replace(
            get_settings(), diarize_enabled=True, diarize_model="nonexistent.onnx"
        )
        self.assertIsInstance(build_diarizer(settings), NullDiarizer)


class DiarizeConfigTests(unittest.TestCase):
    def test_enable_env(self):
        with mock.patch.dict(os.environ, {"BREEZE_DIARIZE": "on"}, clear=True):
            self.assertTrue(get_settings().diarize_enabled)

    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(get_settings().diarize_enabled)

    def test_threshold_clamped(self):
        with mock.patch.dict(os.environ, {"BREEZE_DIARIZE_THRESHOLD": "9"}, clear=True):
            self.assertEqual(get_settings().diarize_threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
