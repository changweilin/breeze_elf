import unittest

import numpy as np

from breeze_elf.audio import (
    AudioUtteranceBuffer,
    AudioWindowBuffer,
    analyze_segment,
    calculate_rms,
    clean_f0_track,
    estimate_noise_floor,
    _extend_attack_release,
    extend_voiced_span,
    hz_to_jianpu,
    jianpu_glide,
    jianpu_to_semitones,
    pcm16le_to_float32,
    pitch_cents_off,
    prepare_asr_audio,
    summarize_pitch,
)


def _chirp(sample_rate, seconds, f0, f1, amplitude=0.4):
    """A linear frequency sweep, used to model a 滑音 (portamento)."""
    time_axis = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    rate = (f1 - f0) / seconds
    phase = 2 * np.pi * (f0 * time_axis + 0.5 * rate * time_axis * time_axis)
    return (amplitude * np.sin(phase)).astype(np.float32)


class AudioTests(unittest.TestCase):
    def test_pcm16le_to_float32(self):
        payload = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        samples = pcm16le_to_float32(payload)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(float(samples[0]), -1.0, places=5)
        self.assertAlmostEqual(float(samples[1]), 0.0, places=5)
        self.assertAlmostEqual(float(samples[2]), 32767 / 32768, places=5)

    def test_prepare_asr_audio_off_preserves_samples(self):
        samples = np.array([-0.25, 0.0, 0.25], dtype=np.float32)

        prepared = prepare_asr_audio(samples, 16_000, profile="off")

        self.assertEqual(prepared.dtype, np.float32)
        np.testing.assert_allclose(prepared, samples)

    def test_prepare_asr_audio_normalizes_with_headroom(self):
        sample_rate = 16_000
        time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (0.02 * np.sin(2 * np.pi * 220.0 * time_axis)).astype(np.float32)

        prepared = prepare_asr_audio(samples, sample_rate, profile="natural")

        self.assertGreater(calculate_rms(prepared), calculate_rms(samples))
        self.assertLessEqual(float(np.max(np.abs(prepared))), 0.98)

    def test_prepare_asr_audio_speech_profile_reduces_noise_floor(self):
        sample_rate = 16_000
        duration = 2.0
        time_axis = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
        noise = 0.006 * np.sin(2 * np.pi * 900.0 * time_axis)
        speech = 0.08 * np.sin(2 * np.pi * 220.0 * time_axis)
        samples = noise.copy()
        speech_start = round(sample_rate * 0.5)
        speech_end = round(sample_rate * 1.5)
        samples[speech_start:speech_end] += speech[speech_start:speech_end]
        samples = samples.astype(np.float32)

        prepared = prepare_asr_audio(samples, sample_rate, profile="speech")

        before_ratio = calculate_rms(samples[:speech_start]) / calculate_rms(
            samples[speech_start:speech_end]
        )
        after_ratio = calculate_rms(prepared[:speech_start]) / calculate_rms(
            prepared[speech_start:speech_end]
        )
        self.assertLess(after_ratio, before_ratio * 0.85)

    def test_summarize_pitch_detects_sine_frequency(self):
        sample_rate = 16_000
        seconds = 0.6
        frequency = 220.0
        time_axis = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
        samples = (np.sin(2 * np.pi * frequency * time_axis) * 0.4).astype(np.float32)

        summary = summarize_pitch(samples, sample_rate)

        self.assertIsNotNone(summary.median_hz)
        self.assertAlmostEqual(summary.median_hz, frequency, delta=4.0)
        self.assertGreater(summary.voiced_ratio, 0.9)
        self.assertGreater(len(summary.points), 0)

    def test_hz_to_jianpu_maps_scale_degrees_relative_to_tonic(self):
        tonic = 220.0
        self.assertEqual(hz_to_jianpu(tonic, tonic), "1")
        self.assertEqual(hz_to_jianpu(tonic * 2 ** (2 / 12), tonic), "2")
        self.assertEqual(hz_to_jianpu(tonic * 2 ** (4 / 12), tonic), "3")
        self.assertEqual(hz_to_jianpu(tonic * 2 ** (7 / 12), tonic), "5")
        self.assertEqual(hz_to_jianpu(tonic * 2 ** (1 / 12), tonic), "#1")

    def test_hz_to_jianpu_marks_octaves_with_combining_dots(self):
        tonic = 220.0
        self.assertEqual(hz_to_jianpu(tonic * 2, tonic), "1̇")
        self.assertEqual(hz_to_jianpu(tonic / 2, tonic), "1̣")
        self.assertEqual(hz_to_jianpu(tonic * 4, tonic), "1̇̇")

    def test_hz_to_jianpu_returns_empty_without_pitch_or_tonic(self):
        self.assertEqual(hz_to_jianpu(None, 220.0), "")
        self.assertEqual(hz_to_jianpu(220.0, None), "")
        self.assertEqual(hz_to_jianpu(220.0, 0.0), "")

    def test_jianpu_glide_collapses_steady_note_to_single_degree(self):
        tonic = 220.0
        # a small wobble still lands on the same degree -> not a glide
        self.assertEqual(jianpu_glide(220.0, 223.0, tonic), "1")
        self.assertEqual(jianpu_glide(220.0, 220.0, tonic), "1")

    def test_jianpu_glide_marks_rising_and_falling_slides(self):
        tonic = 220.0
        rising = jianpu_glide(tonic, tonic * 2 ** (7 / 12), tonic)
        falling = jianpu_glide(tonic * 2 ** (7 / 12), tonic, tonic)
        self.assertEqual(rising, "1↗5")
        self.assertEqual(falling, "5↘1")

    def test_pitch_cents_off_measures_distance_from_scale_degree(self):
        tonic = 220.0
        self.assertAlmostEqual(pitch_cents_off(tonic, tonic), 0.0, delta=0.01)
        self.assertAlmostEqual(pitch_cents_off(tonic * 2 ** (0.25 / 12), tonic), 25.0, delta=1.0)
        self.assertAlmostEqual(pitch_cents_off(tonic * 2 ** (-0.3 / 12), tonic), -30.0, delta=1.0)
        self.assertIsNone(pitch_cents_off(None, tonic))
        self.assertIsNone(pitch_cents_off(220.0, 0.0))

    def test_analyze_segment_tracks_slide_edges_and_intensity(self):
        sample_rate = 16_000
        samples = _chirp(sample_rate, 0.5, 200.0, 300.0)

        analysis = analyze_segment(samples, sample_rate)

        self.assertIsNotNone(analysis.start_hz)
        self.assertIsNotNone(analysis.end_hz)
        self.assertLess(analysis.start_hz, analysis.end_hz)
        self.assertGreater(analysis.end_hz - analysis.start_hz, 40.0)
        self.assertGreater(analysis.intensity, 0.0)

    def test_analyze_segment_handles_empty_segment(self):
        analysis = analyze_segment(np.empty(0, dtype=np.float32), 16_000)
        self.assertIsNone(analysis.median_hz)
        self.assertEqual(analysis.intensity, 0.0)

    def test_summarize_pitch_returns_empty_for_silence(self):
        summary = summarize_pitch(np.zeros(16_000, dtype=np.float32), 16_000)

        self.assertIsNone(summary.median_hz)
        self.assertEqual(summary.voiced_ratio, 0.0)
        self.assertEqual(summary.points, ())

    def test_compute_spectrogram_returns_aligned_series(self):
        from breeze_elf.audio import compute_spectrogram

        sample_rate = 16_000
        axis = np.arange(round(sample_rate * 0.8), dtype=np.float64) / sample_rate
        tone = (0.4 * np.sin(2 * np.pi * 220 * axis)).astype(np.float32)
        spectro = compute_spectrogram(tone, sample_rate)

        self.assertIsNotNone(spectro)
        bins = spectro["timeBins"]
        # f0 / intensity / times all share the spectrogram's time axis.
        self.assertEqual(len(spectro["f0"]), bins)
        self.assertEqual(len(spectro["intensity"]), bins)
        self.assertEqual(len(spectro["times"]), bins)
        self.assertEqual(len(spectro["magnitudes"]), 0 if bins == 0 else len(spectro["magnitudes"]))
        # times are non-decreasing; the voiced f0 tracks 220 Hz.
        self.assertTrue(all(b >= a for a, b in zip(spectro["times"], spectro["times"][1:])))
        voiced = [hz for hz in spectro["f0"] if hz]
        self.assertTrue(voiced)
        self.assertAlmostEqual(float(np.median(voiced)), 220.0, delta=4.0)

    def test_compute_spectrogram_marks_silence_unvoiced(self):
        from breeze_elf.audio import compute_spectrogram

        spectro = compute_spectrogram(np.zeros(8_000, dtype=np.float32), 16_000)
        self.assertIsNotNone(spectro)
        self.assertTrue(all(hz is None for hz in spectro["f0"]))

    @staticmethod
    def _uniform_times(f0, step=0.02):
        return [round(i * step, 3) for i in range(len(f0))]

    def test_clean_f0_drops_floating_erratic_blob(self):
        # A voiced run that zig-zags by ~an octave with silence on both sides — a
        # classic YIN artifact on a breath/noise burst — is removed entirely.
        f0 = [None] * 3 + [200.0, 400.0, 205.0, 380.0, 210.0] + [None] * 3
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertTrue(all(hz is None for hz in cleaned[3:8]))
        # The surrounding silence is untouched (still None), nothing else changed.
        self.assertEqual(cleaned, [None] * len(f0))

    def test_clean_f0_keeps_glide_between_two_notes(self):
        # 平穩音高 → 滑音 → 平穩音高: the unstable middle is anchored both sides, kept.
        f0 = [220.0] * 5 + [233.0, 247.0, 262.0, 277.0] + [294.0] * 5
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, f0)

    def test_clean_f0_keeps_isolated_monotonic_glide(self):
        # A one-directional sweep with silence on both sides is still a 滑音, not
        # noise — the smoothness guard keeps it even without a held neighbour.
        f0 = [None] * 2 + [200.0, 220.0, 240.0, 262.0, 285.0, 311.0] + [None] * 2
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, f0)

    def test_clean_f0_keeps_isolated_vibrato(self):
        # 抖音: ±1 semitone wobble (spread ~2 st) sits under the drastic gate, so an
        # isolated vibrato note survives even with silence on both sides.
        center = 262.0
        wobble = [center * (2.0 ** (depth / 12.0)) for depth in (0, 1, 0, -1, 0, 1, 0, -1)]
        f0 = [None] * 2 + wobble + [None] * 2
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, f0)

    def test_clean_f0_keeps_quiet_steady_note(self):
        # A steady isolated note that merely misses the tight stability window must
        # never be deleted: its narrow spread fails the drastic gate.
        f0 = [None] * 2 + [262.0, 263.0, 261.5, 262.5, 262.0, 263.0] + [None] * 2
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, f0)

    def test_clean_f0_bridges_interior_octave_spike(self):
        # A one-bin octave jump up-then-back-down in the middle of a held note is a
        # YIN artifact: remove it and connect 前後 (both anchors are 262 Hz).
        f0 = [262.0] * 5 + [524.0] + [262.0] * 5
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, [262.0] * 11)

    def test_clean_f0_bridges_interior_valley(self):
        # 反方向: a drastic drop-then-return dip is bridged just the same.
        f0 = [392.0] * 5 + [196.0] + [392.0] * 5
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, [392.0] * 11)

    def test_clean_f0_bridges_multibin_spike(self):
        # A non-harmonic spike (a fifth up, not an octave) so the fill's bridge — not
        # the octave snapper — is what removes it.
        f0 = [262.0] * 4 + [393.0, 396.0] + [262.0] * 4
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, [262.0] * 10)

    def test_clean_f0_holds_stable_pitch_over_one_sided_artifact(self):
        # An erratic (non-harmonic) blob adjacent to a note on one side (silence on
        # the other) is not cut to None — it is filled by holding the note's pitch.
        f0 = [None] * 2 + [200.0, 500.0, 210.0, 480.0] + [330.0] * 4
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, [None] * 2 + [330.0] * 8)

    def test_clean_f0_holds_leading_octave_error(self):
        # A leading octave-error bin (silence before, note after) is replaced with
        # the following note's pitch rather than deleted.
        f0 = [None] * 2 + [524.0] + [262.0] * 6
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, [None] * 2 + [262.0] * 7)

    def test_clean_f0_keeps_held_note_leap(self):
        # A real melodic leap up to a *held* note (a plateau longer than the spike
        # cap) is not a return-spike and must survive untouched.
        f0 = [262.0] * 4 + [392.0] * 8 + [262.0] * 4
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, f0)

    def test_clean_f0_bridge_leaves_glide_untouched(self):
        # A one-directional 滑音 never clears both anchors, so bridging skips it.
        f0 = [220.0] * 4 + [233.0, 247.0, 262.0, 277.0, 294.0] + [311.0] * 4
        cleaned = clean_f0_track(f0, self._uniform_times(f0))
        self.assertEqual(cleaned, f0)

    def test_clean_f0_snaps_harmonic_plateau_to_surrounding_pitch(self):
        # A held note mis-tracked an octave (×2) up for several bins registers as a
        # "stable" plateau the fill pass can't see; the octave snapper folds it back
        # onto the surrounding pitch. Realistic ~75 ms bins.
        times = [round(i * 0.075, 3) for i in range(20)]
        f0 = [110.0] * 8 + [220.0] * 4 + [110.0] * 8
        cleaned = clean_f0_track(f0, times)
        self.assertEqual(cleaned, [110.0] * 20)

    def test_clean_f0_snaps_third_harmonic_plateau(self):
        times = [round(i * 0.075, 3) for i in range(19)]
        f0 = [130.0] * 8 + [390.0] * 3 + [130.0] * 8  # ×3 harmonic error
        cleaned = clean_f0_track(f0, times)
        self.assertEqual(cleaned, [130.0] * 19)

    def test_clean_f0_keeps_real_interval_leap(self):
        # A genuine melodic leap (a perfect fifth up, held) is NOT an octave/harmonic
        # ratio of the surrounding note, so the snapper must leave it untouched.
        times = [round(i * 0.075, 3) for i in range(20)]
        hi = round(220.0 * 2 ** (7 / 12.0), 1)  # perfect fifth
        f0 = [220.0] * 8 + [hi] * 4 + [220.0] * 8
        cleaned = clean_f0_track(f0, times)
        self.assertEqual(cleaned, f0)

    def test_clean_f0_noop_on_degenerate_input(self):
        self.assertEqual(clean_f0_track([]), [])
        self.assertEqual(clean_f0_track([None, None]), [None, None])
        self.assertEqual(clean_f0_track([None] * 6), [None] * 6)

    def test_extend_attack_release_fills_edges_and_stops_at_noise(self):
        # Note at bins 3-5. The bins right at the edges (2, 6) carry real note energy
        # (attack/release); the outer bins sit at the noise floor and must be left.
        f0 = [None, None, None, 220.0, 220.0, 220.0, None, None, None]
        intensity = [0.05, 0.05, 0.15, 0.20, 0.20, 0.20, 0.15, 0.05, 0.05]
        out = _extend_attack_release(f0, intensity)
        self.assertEqual(out, [None, None, 220.0, 220.0, 220.0, 220.0, 220.0, None, None])

    def test_extend_attack_release_caps_reach(self):
        # A long above-gate ramp before the onset is only followed for max_bins.
        f0 = [None] * 8 + [200.0] * 3
        intensity = [0.02, 0.02] + [0.30] * 6 + [0.40] * 3
        out = _extend_attack_release(f0, intensity, max_bins=3)
        self.assertEqual(out[5:8], [200.0, 200.0, 200.0])  # 3 bins of attack held
        self.assertTrue(all(v is None for v in out[:5]))  # nothing beyond the cap

    def test_extend_attack_release_ignores_pure_noise_neighbours(self):
        # Every non-note bin is at the floor → nothing is added (勿補太多抓到 noise).
        f0 = [None, None, 262.0, 262.0, 262.0, None, None]
        intensity = [0.04, 0.04, 0.22, 0.22, 0.22, 0.04, 0.04]
        self.assertEqual(_extend_attack_release(f0, intensity), f0)

    def test_compute_spectrogram_clean_f0_toggle_is_passthrough(self):
        from breeze_elf.audio import compute_spectrogram

        sample_rate = 16_000
        axis = np.arange(round(sample_rate * 0.8), dtype=np.float64) / sample_rate
        tone = (0.4 * np.sin(2 * np.pi * 220 * axis)).astype(np.float32)
        # A clean sustained tone has no floating erratic blob, so cleaning is a
        # no-op: both paths yield the same f0 track.
        raw = compute_spectrogram(tone, sample_rate, clean_f0=False)
        cleaned = compute_spectrogram(tone, sample_rate, clean_f0=True)
        self.assertEqual(raw["f0"], cleaned["f0"])

    def test_summarize_pitch_is_robust_to_harmonics(self):
        # A harmonic-rich tone (strong 2nd/3rd harmonics) is exactly what makes
        # plain autocorrelation report an octave error; YIN must track the true
        # fundamental, not a harmonic or sub-harmonic.
        sample_rate = 16_000
        time_axis = np.arange(round(sample_rate * 0.6), dtype=np.float64) / sample_rate
        for f0 in (110.0, 147.0, 220.0, 262.0):
            tone = (
                0.6 * np.sin(2 * np.pi * f0 * time_axis)
                + 0.9 * np.sin(2 * np.pi * 2 * f0 * time_axis)
                + 0.7 * np.sin(2 * np.pi * 3 * f0 * time_axis)
            ).astype(np.float32)
            tone *= 0.3 / float(np.max(np.abs(tone)))
            summary = summarize_pitch(tone, sample_rate)
            self.assertIsNotNone(summary.median_hz)
            self.assertAlmostEqual(summary.median_hz, f0, delta=3.0)

    def test_silence_gate(self):
        buffer = AudioWindowBuffer(
            sample_rate=4,
            window_seconds=1.0,
            overlap_seconds=0.25,
            rms_threshold=0.1,
        )
        payload = np.zeros(4, dtype="<i2").tobytes()
        windows = buffer.append_pcm16(payload)
        self.assertEqual(len(windows), 1)
        self.assertFalse(windows[0].is_speech)
        self.assertEqual(calculate_rms(windows[0].samples), 0.0)

    def test_emits_overlapping_windows(self):
        buffer = AudioWindowBuffer(
            sample_rate=4,
            window_seconds=1.0,
            overlap_seconds=0.25,
            rms_threshold=0.0,
        )
        payload = np.full(7, 1000, dtype="<i2").tobytes()
        windows = buffer.append_pcm16(payload)
        self.assertEqual([window.index for window in windows], [0, 1])
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[1].start_seconds, 0.75)
        self.assertAlmostEqual(buffer.buffered_seconds, 0.25)

    def test_ring_buffer_preserves_overlapping_samples_after_wrap(self):
        buffer = AudioWindowBuffer(
            sample_rate=4,
            window_seconds=1.0,
            overlap_seconds=0.5,
            rms_threshold=0.0,
        )

        first = buffer.append_pcm16(np.array([1, 2, 3, 4], dtype="<i2").tobytes())[0]
        second = buffer.append_pcm16(np.array([5, 6], dtype="<i2").tobytes())[0]
        third = buffer.append_pcm16(np.array([7, 8], dtype="<i2").tobytes())[0]
        fourth = buffer.append_pcm16(np.array([9, 10], dtype="<i2").tobytes())[0]

        scale = 1 / 32768
        np.testing.assert_allclose(first.samples, np.array([1, 2, 3, 4], dtype=np.float32) * scale)
        np.testing.assert_allclose(second.samples, np.array([3, 4, 5, 6], dtype=np.float32) * scale)
        np.testing.assert_allclose(third.samples, np.array([5, 6, 7, 8], dtype=np.float32) * scale)
        np.testing.assert_allclose(
            fourth.samples,
            np.array([7, 8, 9, 10], dtype=np.float32) * scale,
        )
        self.assertAlmostEqual(buffer.buffered_seconds, 0.5)

    def test_utterance_buffer_emits_speech_after_trailing_silence(self):
        buffer = AudioUtteranceBuffer(
            sample_rate=10,
            frame_ms=100,
            pre_roll_ms=200,
            end_silence_ms=200,
            rms_threshold=0.01,
        )
        payload = np.array([0, 0, 1000, 1000, 1000, 0, 0, 0], dtype="<i2").tobytes()

        windows = buffer.append_pcm16(payload)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].kind, "utterance")
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].end_seconds, 0.7)
        self.assertEqual(windows[0].samples.size, 7)
        self.assertTrue(windows[0].is_speech)

    def test_utterance_buffer_force_split_cuts_at_quiet_dip(self):
        # A continuous phrase longer than the cap is split at the quiet dip in its
        # tail region (a syllable gap), not hard-cut through the sound, and the
        # remainder carries over so no samples are lost.
        sample_rate = 1_000
        buffer = AudioUtteranceBuffer(
            sample_rate=sample_rate,
            frame_ms=100,  # 100-sample frames
            pre_roll_ms=0,
            end_silence_ms=10_000,  # never ends on silence here
            max_segment_seconds=1.0,  # cap at 10 frames
            rms_threshold=0.05,
        )
        pcm = np.full(1_000, 10_000, dtype="<i2")
        pcm[800:900] = 100  # a quiet dip in the tail region

        windows = buffer.append_pcm16(pcm.tobytes())

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_seconds, 0.0)
        cut = windows[0].samples.size
        self.assertGreaterEqual(cut, 800)  # cut lands inside the quiet dip [800, 900)
        self.assertLess(cut, 900)
        self.assertLess(calculate_rms(windows[0].samples[-20:]), 0.05)  # ends quiet, not mid-note

        tail = buffer.flush()
        self.assertEqual(len(tail), 1)
        self.assertEqual(cut + tail[0].samples.size, 1_000)  # contiguous carry, no samples lost
        self.assertAlmostEqual(tail[0].start_seconds, cut / 1_000, places=6)

    def test_utterance_buffer_flushes_active_speech(self):
        buffer = AudioUtteranceBuffer(
            sample_rate=10,
            frame_ms=100,
            pre_roll_ms=100,
            end_silence_ms=300,
            rms_threshold=0.01,
        )

        self.assertEqual(buffer.append_pcm16(np.array([0, 1000, 1000], dtype="<i2").tobytes()), [])
        windows = buffer.flush()

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].samples.size, 3)


class BoundaryExtensionTests(unittest.TestCase):
    def test_estimate_noise_floor_picks_quiet_frames(self):
        sample_rate = 16_000
        quiet = np.full(8_000, 0.003, dtype=np.float32)
        loud = np.full(8_000, 0.2, dtype=np.float32)
        samples = np.concatenate([quiet, loud])

        floor = estimate_noise_floor(samples, sample_rate)

        self.assertAlmostEqual(floor, 0.003, delta=0.001)

    def test_extend_voiced_span_grows_through_subthreshold_onset(self):
        # silence (room tone) | unvoiced onset (sub-threshold, above floor) | loud
        # voiced | silence — the span must reach back over the onset but stop at the
        # silence on both sides.
        sample_rate = 16_000
        silence = np.full(1_600, 0.002, dtype=np.float32)
        onset = np.full(1_600, 0.012, dtype=np.float32)
        voiced = np.full(3_200, 0.3, dtype=np.float32)
        tail = np.full(1_600, 0.002, dtype=np.float32)
        samples = np.concatenate([silence, onset, voiced, tail])

        start, end = extend_voiced_span(
            samples,
            sample_rate,
            3_200,
            6_400,
            floor_sample=0,
            ceil_sample=samples.size,
            noise_floor=0.002,
        )

        self.assertEqual(start, 1_600)  # grew back to the onset, stopped at silence
        self.assertEqual(end, 6_400)  # trailing silence → no growth

    def test_extend_voiced_span_respects_neighbour_bounds(self):
        sample_rate = 16_000
        samples = np.full(8_000, 0.05, dtype=np.float32)  # all above the floor

        start, end = extend_voiced_span(
            samples,
            sample_rate,
            3_000,
            5_000,
            floor_sample=2_000,
            ceil_sample=6_000,
            noise_floor=0.002,
        )

        self.assertGreaterEqual(start, 2_000)
        self.assertLessEqual(end, 6_000)


class JianpuParseTests(unittest.TestCase):
    def test_round_trips_hz_to_jianpu_to_semitones(self):
        tonic = 220.0
        for semitones in (-13, -12, -1, 0, 1, 2, 4, 5, 7, 9, 11, 12, 13):
            token = hz_to_jianpu(tonic * 2 ** (semitones / 12.0), tonic)
            self.assertEqual(jianpu_to_semitones(token), float(semitones))

    def test_ascii_octave_marks(self):
        self.assertEqual(jianpu_to_semitones("1'"), 12.0)
        self.assertEqual(jianpu_to_semitones("1,"), -12.0)
        self.assertEqual(jianpu_to_semitones("#4"), 6.0)

    def test_glide_uses_leading_degree(self):
        self.assertEqual(jianpu_to_semitones("3↗5"), 4.0)
        self.assertEqual(jianpu_to_semitones("5↘1"), 7.0)

    def test_rests_and_garbage_return_none(self):
        for token in ("", "0", "-", None, "x"):
            self.assertIsNone(jianpu_to_semitones(token))


if __name__ == "__main__":
    unittest.main()
