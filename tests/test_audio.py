import unittest

import numpy as np

from breeze_elf.audio import (
    AudioUtteranceBuffer,
    AudioWindowBuffer,
    _extend_attack_release,
    _meter_from_accents,
    analyze_segment,
    bar_before_flags,
    calculate_rms,
    clean_f0_track,
    estimate_meter,
    estimate_noise_floor,
    estimate_tempo,
    estimate_tonic,
    extend_voiced_span,
    hz_to_jianpu,
    jianpu_glide,
    jianpu_to_semitones,
    pcm16le_to_float32,
    pitch_cents_off,
    prepare_asr_audio,
    quantize_beats,
    score_intonation,
    summarize_pitch,
    tuning_label,
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


def _melody(tonic_hz, degrees, duration=0.4, detune_cents=0.0):
    """(hz, seconds) notes at the given semitone offsets above ``tonic_hz``."""
    return [
        (tonic_hz * 2 ** ((semitones + detune_cents / 100.0) / 12.0), duration)
        for semitones in degrees
    ]


# Real tunes, as semitones above their tonic. Each one's *median* pitch sits
# well away from its tonic — writing 簡譜 against the median is the bug these
# tests exist to catch. 小蜜蜂 dwells on the third and fifth and only touches
# the tonic at the cadence; 五聲音階 is the pentatonic case Chinese songs hit.
_TWINKLE = (0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0)
_LITTLE_BEE = (4, 2, 0, 2, 4, 4, 4, 2, 2, 2, 4, 7, 7, 4, 2, 0)
_PENTATONIC = (0, 2, 4, 7, 9, 12, 9, 7, 4, 2, 0, 4, 7, 4, 2, 0)
# A natural-minor tune resolving on its own root (semitones above that root).
_MINOR_TUNE = (0, 3, 7, 3, 0, -2, 0, 3, 5, 7, 5, 3, 0, -2, 0)


class TonicEstimateTests(unittest.TestCase):
    def _assert_tonic(self, notes, expected_hz, *, delta_semitones=0.1):
        estimate = estimate_tonic(notes)
        self.assertEqual(estimate.source, "key")
        self.assertAlmostEqual(12 * np.log2(estimate.hz / expected_hz), 0.0, delta=delta_semitones)
        return estimate

    def test_finds_the_root_not_the_median_pitch(self):
        notes = _melody(261.63, _TWINKLE)
        estimate = self._assert_tonic(notes, 261.63)
        self.assertGreater(estimate.confidence, 0.95)
        # The median pitch is 4.5 semitones above the tonic — the old 主音.
        median = float(np.median([hz for hz, _ in notes]))
        self.assertGreater(abs(12 * np.log2(median / estimate.hz)), 3.0)

    def test_resolves_a_tune_that_dwells_on_the_fifth(self):
        # Histogram alone reads this as G major; the cadence on the tonic is
        # what makes it C.
        self._assert_tonic(_melody(261.63, _LITTLE_BEE), 261.63)

    def test_pentatonic_tune(self):
        self._assert_tonic(_melody(261.63, _PENTATONIC), 261.63)

    def test_every_note_lands_on_a_plain_degree(self):
        notes = _melody(261.63, _TWINKLE)
        estimate = estimate_tonic(notes)
        for hz, _ in notes:
            self.assertNotIn("#", hz_to_jianpu(hz, estimate.hz))

    def test_minor_key_is_written_in_its_relative_major(self):
        # A minor → 簡譜 convention writes 1 = C (three semitones up), so the
        # melody centres on 6 rather than 1.
        estimate = self._assert_tonic(_melody(220.0, _MINOR_TUNE), 220.0 * 2 ** (3 / 12.0))
        self.assertEqual(hz_to_jianpu(220.0, estimate.hz), "6" + "̣")

    def test_tracks_a_recording_tuned_off_the_grid(self):
        # A whole performance sung 30 cents sharp still resolves to one key,
        # shifted by that same 30 cents rather than smeared across two.
        estimate = estimate_tonic(_melody(261.63, _TWINKLE, detune_cents=30.0))
        self.assertEqual(estimate.source, "key")
        self.assertAlmostEqual(12 * np.log2(estimate.hz / 261.63), 0.3, delta=0.1)

    def test_long_notes_outweigh_passing_ones(self):
        # Sustained scale tones plus fleeting chromatic passing notes: the key
        # must come from the held pitches, not from a raw note count.
        notes = _melody(261.63, _TWINKLE, duration=0.8)
        notes[-1:] = _melody(261.63, (1, 3, 6, 8, 10, 1, 6, 8, 0), duration=0.02)
        self._assert_tonic(notes, 261.63)

    def test_falls_back_to_median_on_thin_evidence(self):
        notes = _melody(261.63, (0, 4, 7))
        estimate = estimate_tonic(notes)
        self.assertEqual(estimate.source, "median")
        self.assertEqual(estimate.confidence, 0.0)
        self.assertAlmostEqual(estimate.hz, float(np.median([hz for hz, _ in notes])), places=3)

    def test_falls_back_when_nothing_fits_a_key(self):
        # A chromatic run is in no key; forcing one onto it would be worse than
        # the median, so the estimator declines.
        estimate = estimate_tonic(_melody(261.63, tuple(range(12)) * 2))
        self.assertEqual(estimate.source, "median")

    def test_survives_realistic_pitch_jitter(self):
        # Real singing (and YIN's own error) never lands dead on the grid.
        rng = np.random.default_rng(7)
        notes = [
            (hz * 2 ** (rng.uniform(-0.35, 0.35) / 12.0), duration + rng.uniform(-0.1, 0.1))
            for hz, duration in _melody(261.63, _TWINKLE + _LITTLE_BEE)
        ]
        self._assert_tonic(notes, 261.63, delta_semitones=0.3)

    def test_a_sustained_tone_has_no_key(self):
        # One pitch class fits every scale containing it, so it would score a
        # perfect confidence; without a scale in evidence, keep the median.
        estimate = estimate_tonic(_melody(220.0, (0,) * 12))
        self.assertEqual(estimate.source, "median")
        self.assertAlmostEqual(estimate.hz, 220.0, places=3)

    def test_no_notes(self):
        self.assertEqual(estimate_tonic([]).source, "empty")
        self.assertIsNone(estimate_tonic([(None, 0.4), (0.0, 0.4)]).hz)

    def test_tonic_sits_in_the_melody_octave(self):
        # Same key an octave apart: the tonic follows the melody so the 簡譜
        # keeps plain degrees instead of a pile of octave dots.
        self._assert_tonic(_melody(130.81, _TWINKLE), 130.81)
        self._assert_tonic(_melody(523.25, _TWINKLE), 523.25)


def _sung_notes(bpm, degrees, sample_rate=16_000, tonic_hz=261.63):
    """A steady tune: one note per beat, each with an attack and a release so the
    onset detector has something to find (no percussion, like real singing)."""
    beat = 60.0 / bpm
    out = []
    for semitones in degrees:
        count = int(sample_rate * beat)
        axis = np.arange(count) / sample_rate
        envelope = np.minimum(1.0, np.arange(count) / (0.02 * sample_rate)) * np.minimum(
            1.0, (count - np.arange(count)) / (0.05 * sample_rate)
        )
        hz = tonic_hz * 2 ** (semitones / 12.0)
        out.append((0.4 * np.sin(2 * np.pi * hz * axis) * envelope).astype(np.float32))
    return np.concatenate(out)


class TempoEstimateTests(unittest.TestCase):
    def test_recovers_the_tempo_of_a_steady_tune(self):
        for bpm in (75, 90, 120, 140):
            estimate = estimate_tempo(_sung_notes(bpm, _TWINKLE * 3), 16_000)
            self.assertIsNotNone(estimate.bpm, f"{bpm} BPM not detected")
            self.assertAlmostEqual(estimate.bpm, bpm, delta=1.0)
            self.assertAlmostEqual(estimate.beat_seconds, 60.0 / bpm, delta=0.01)
            self.assertGreater(estimate.confidence, 0.8)

    def test_a_tempo_whose_period_falls_between_frames(self):
        # 160 BPM is 37.5 frames at a 10 ms hop. Rounding that to a whole frame
        # is what makes an estimator report the song at half speed.
        estimate = estimate_tempo(_sung_notes(160, _TWINKLE * 3), 16_000)
        self.assertAlmostEqual(estimate.bpm, 160.0, delta=3.0)

    def test_free_tempo_audio_reports_no_beat(self):
        # Random syllable lengths — speech, or rubato singing. Quantising note
        # values against an invented pulse is worse than showing none.
        rng = np.random.default_rng(7)
        sample_rate = 16_000
        parts = []
        for _ in range(40):
            count = int(rng.uniform(0.15, 0.5) * sample_rate)
            axis = np.arange(count) / sample_rate
            parts.append(
                (0.3 * np.sin(2 * np.pi * rng.uniform(120, 240) * axis) * np.hanning(count)).astype(
                    np.float32
                )
            )
            parts.append(np.zeros(int(rng.uniform(0.05, 0.5) * sample_rate), dtype=np.float32))
        estimate = estimate_tempo(np.concatenate(parts), sample_rate)
        self.assertIsNone(estimate.bpm)
        self.assertIsNone(estimate.beat_seconds)

    def test_silence_and_tiny_input(self):
        self.assertIsNone(estimate_tempo(np.zeros(16_000, dtype=np.float32), 16_000).bpm)
        self.assertIsNone(estimate_tempo(np.zeros(64, dtype=np.float32), 16_000).bpm)

    def test_phase_points_at_a_note_onset(self):
        estimate = estimate_tempo(_sung_notes(120, _TWINKLE * 3), 16_000)
        # Notes start every 0.5 s from 0, so the downbeat must land on a
        # multiple of the beat (within a couple of frames).
        offset = estimate.phase_seconds % estimate.beat_seconds
        self.assertLess(min(offset, estimate.beat_seconds - offset), 0.05)


class QuantizeBeatsTests(unittest.TestCase):
    def test_snaps_to_note_values(self):
        beat = 0.5  # 120 BPM
        cases = {
            0.12: (0.25, "十六分音符"),
            0.26: (0.5, "八分音符"),
            0.38: (0.75, "附點八分音符"),
            0.52: (1.0, "四分音符"),
            0.78: (1.5, "附點四分音符"),
            1.03: (2.0, "二分音符"),
            2.1: (4.0, "全音符"),
        }
        for seconds, expected in cases.items():
            self.assertEqual(quantize_beats(seconds, beat), expected, f"{seconds}s")

    def test_compared_in_log_space_so_short_notes_are_not_swallowed(self):
        # 0.3 beats is nearer 0.25 than 0.5 proportionally, even though the
        # linear distance to each is the same.
        self.assertEqual(quantize_beats(0.15, 0.5)[0], 0.25)

    def test_no_beat_means_no_note_value(self):
        self.assertEqual(quantize_beats(0.5, None), (None, ""))
        self.assertEqual(quantize_beats(None, 0.5), (None, ""))
        self.assertEqual(quantize_beats(0.0, 0.5), (None, ""))


class IntonationScoreTests(unittest.TestCase):
    def test_on_the_grid_scores_full_marks(self):
        score = score_intonation([(0.0, 0.5)] * 8)
        self.assertEqual(score.score, 100.0)
        self.assertEqual(score.in_tune_ratio, 1.0)
        self.assertEqual(score.off_pitch_count, 0)
        self.assertEqual(score.counted, 8)

    def test_score_falls_to_zero_between_two_semitones(self):
        self.assertAlmostEqual(score_intonation([(50.0, 0.5)]).score, 0.0)
        self.assertAlmostEqual(score_intonation([(25.0, 0.5)]).score, 50.0)
        self.assertAlmostEqual(score_intonation([(-25.0, 0.5)]).score, 50.0)

    def test_held_notes_weigh_more_than_passing_ones(self):
        # One sour note held for two seconds against eight clean tenth-second
        # ones: the score must follow what is actually heard.
        held = score_intonation([(45.0, 2.0)] + [(0.0, 0.1)] * 8)
        passing = score_intonation([(45.0, 0.1)] + [(0.0, 2.0)] * 8)
        # 2 s at quality 0.1 against 0.8 s at quality 1.0 → 36; flip the
        # durations and the same two pitches score 99.
        self.assertLess(held.score, 40.0)
        self.assertGreater(passing.score, 95.0)
        self.assertGreater(passing.score - held.score, 55.0)

    def test_counts_the_notes_worth_looking_at(self):
        score = score_intonation([(5.0, 0.5), (30.0, 0.5), (44.0, 0.5), (-40.0, 0.5)])
        self.assertEqual(score.off_pitch_count, 2)
        self.assertAlmostEqual(score.in_tune_ratio, 0.25)
        self.assertAlmostEqual(score.median_abs_cents, 35.0)

    def test_nothing_measurable(self):
        empty = score_intonation([(None, 0.5), (float("nan"), 0.5)])
        self.assertIsNone(empty.score)
        self.assertEqual(empty.counted, 0)
        self.assertIsNone(score_intonation([]).score)

    def test_zero_durations_fall_back_to_counting_notes(self):
        score = score_intonation([(0.0, 0.0), (50.0, 0.0)])
        self.assertAlmostEqual(score.score, 50.0)

    def test_tuning_labels(self):
        self.assertEqual(tuning_label(0.0), "準")
        self.assertEqual(tuning_label(-20.0), "準")
        self.assertEqual(tuning_label(30.0), "略偏")
        self.assertEqual(tuning_label(-45.0), "走音")
        self.assertEqual(tuning_label(None), "")


class MeterTests(unittest.TestCase):
    def test_quadruple_accent_reads_as_common_time(self):
        accents = np.tile([1.0, 0.2, 0.2, 0.2], 8)
        meter = _meter_from_accents(accents)
        self.assertEqual(meter.beats_per_measure, 4)
        self.assertEqual(meter.downbeat_offset, 0)
        self.assertGreater(meter.confidence, 0.12)

    def test_triple_accent_reads_as_waltz(self):
        accents = np.tile([1.0, 0.2, 0.2], 10)
        meter = _meter_from_accents(accents)
        self.assertEqual(meter.beats_per_measure, 3)
        self.assertGreater(meter.confidence, 0.12)

    def test_duple_accent_prefers_two_over_four(self):
        # A strong beat every 2 correlates at 4 too; the tighter fold must win.
        accents = np.tile([1.0, 0.2], 12)
        meter = _meter_from_accents(accents)
        self.assertEqual(meter.beats_per_measure, 2)

    def test_downbeat_offset_tracks_the_loud_beat(self):
        accents = np.tile([0.2, 0.2, 1.0, 0.2], 8)
        meter = _meter_from_accents(accents)
        self.assertEqual(meter.beats_per_measure, 4)
        self.assertEqual(meter.downbeat_offset, 2)

    def test_flat_accents_assume_common_time_without_confidence(self):
        meter = _meter_from_accents(np.full(16, 0.5))
        self.assertEqual(meter.beats_per_measure, 4)
        self.assertEqual(meter.confidence, 0.0)

    def test_too_few_beats_default_and_empty_gives_none(self):
        short = _meter_from_accents(np.array([1.0, 0.2, 0.2, 0.2]))
        self.assertEqual(short.beats_per_measure, 4)
        self.assertEqual(short.confidence, 0.0)
        self.assertIsNone(_meter_from_accents(np.zeros(0)).beats_per_measure)

    def test_estimate_meter_without_a_beat_returns_none(self):
        silence = np.zeros(1_000, dtype=np.float32)
        meter = estimate_meter(silence, 16_000, estimate_tempo(silence, 16_000))
        self.assertIsNone(meter.beats_per_measure)


class BarLineTests(unittest.TestCase):
    def test_marks_the_first_note_of_each_new_measure(self):
        starts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        flags = bar_before_flags(starts, measure_seconds=2.0, origin_seconds=0.0)
        self.assertEqual(flags, [False, False, False, False, True, False])

    def test_first_note_never_carries_a_bar_line(self):
        flags = bar_before_flags([2.0, 4.0], measure_seconds=2.0, origin_seconds=0.0)
        self.assertEqual(flags, [False, True])

    def test_origin_shifts_the_measure_boundaries(self):
        starts = [0.4, 0.6, 2.4, 2.6]
        flags = bar_before_flags(starts, measure_seconds=2.0, origin_seconds=0.5)
        self.assertEqual(flags, [False, True, False, True])

    def test_no_metre_means_no_bar_lines(self):
        self.assertEqual(
            bar_before_flags([0.0, 1.0], measure_seconds=None, origin_seconds=0.0),
            [False, False],
        )


if __name__ == "__main__":
    unittest.main()
