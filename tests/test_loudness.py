"""Loudness measurement, gain staging, and the timeline truncation that
stops a consumed window from playing twice.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.render.loudness import (
    SILENCE,
    gain_for_target,
    limit,
    master_chain,
    measure_lufs,
    normalize_to,
    true_peak_db,
)
from app.render.timeline import Clip, Timeline

SR = 44100


def _sine(seconds=3.0, freq=1000.0, amplitude=1.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    mono = (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)
    return np.repeat(mono[np.newaxis, :], 2, axis=0)


def _noise(seconds=3.0, amplitude=0.1, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    mono = (rng.standard_normal(int(seconds * sr)) * amplitude).astype(np.float32)
    return np.repeat(mono[np.newaxis, :], 2, axis=0)


# ---------------------------------------------------------- measurement ----

def test_louder_signals_measure_higher():
    quiet = measure_lufs(_sine(amplitude=0.1), SR)
    loud = measure_lufs(_sine(amplitude=0.9), SR)
    assert loud > quiet


def test_gain_maps_one_to_one_onto_lufs():
    """The property the whole of gain staging rests on.

    If a 6dB gain did not move the measurement by exactly 6 LU, matching two
    tracks' loudness would not converge.
    """
    base_audio = _sine(amplitude=0.5)
    base = measure_lufs(base_audio, SR)

    for db in (-30.0, -20.0, -10.0, -6.0, 6.0):
        scaled = base_audio * (10.0 ** (db / 20.0))
        assert measure_lufs(scaled, SR) - base == pytest.approx(db, abs=0.05)


def test_silence_and_empty_return_negative_infinity():
    """Callers treat a silent segment as "nothing to match" rather than guard
    every call site."""
    assert measure_lufs(np.zeros((2, SR)), SR) == SILENCE
    assert measure_lufs(np.zeros((2, 0)), SR) == SILENCE


def test_short_signals_are_still_measurable():
    """Shorter than one 400ms gating block must not raise."""
    assert np.isfinite(measure_lufs(_sine(seconds=0.1), SR))


def test_measurement_is_independent_of_duration():
    short = measure_lufs(_noise(seconds=2.0), SR)
    long = measure_lufs(_noise(seconds=8.0), SR)
    assert short == pytest.approx(long, abs=1.0)


def test_full_scale_sine_lands_near_zero_lufs():
    """A sanity anchor against the BS.1770 reference."""
    assert measure_lufs(_sine(amplitude=1.0), SR) == pytest.approx(-0.25, abs=1.0)


# --------------------------------------------------------- normalisation ---

def test_two_signals_far_apart_normalize_to_the_same_level():
    quiet, loud = _sine(amplitude=0.03), _sine(amplitude=0.9)
    assert measure_lufs(loud, SR) - measure_lufs(quiet, SR) > 12.0

    quiet_out, _ = normalize_to(quiet, SR, -14.0)
    loud_out, _ = normalize_to(loud, SR, -14.0)

    assert measure_lufs(quiet_out, SR) == pytest.approx(-14.0, abs=0.5)
    assert measure_lufs(loud_out, SR) == pytest.approx(-14.0, abs=0.5)
    assert abs(measure_lufs(quiet_out, SR) - measure_lufs(loud_out, SR)) < 0.5


def test_normalization_reports_the_gain_it_applied():
    audio = _sine(amplitude=0.1)
    out, applied = normalize_to(audio, SR, -14.0)
    assert measure_lufs(out, SR) - measure_lufs(audio, SR) == pytest.approx(applied, abs=0.1)


def test_gain_is_clamped_so_near_silence_is_not_amplified_into_noise():
    """A near-silent intro would otherwise be handed 40dB."""
    assert gain_for_target(-90.0, -14.0, max_gain_db=18.0) == pytest.approx(
        10.0 ** (18.0 / 20.0))


def test_silent_audio_is_left_alone():
    silent = np.zeros((2, SR), dtype=np.float32)
    out, applied = normalize_to(silent, SR, -14.0)
    assert applied == pytest.approx(0.0)
    assert not np.any(out)


# --------------------------------------------------------------- master ----

def test_master_chain_hits_the_target_and_respects_the_ceiling():
    audio = _noise(seconds=5.0, amplitude=0.05)
    out, report = master_chain(audio, SR, target_lufs=-14.0, ceiling_db=-1.0)

    assert measure_lufs(out, SR) == pytest.approx(-14.0, abs=1.5)
    assert true_peak_db(out) <= -1.0 + 0.5
    assert report["output_lufs"] is not None


def test_limiter_catches_a_transient_without_pulling_the_mix_down():
    """Peak normalisation would drop the whole set to fit one spike."""
    audio = _noise(seconds=4.0, amplitude=0.1)
    audio[..., SR: SR + 200] = 0.99                    # one loud transient

    limited = limit(audio, SR, ceiling_db=-1.0)
    assert true_peak_db(limited) <= -1.0 + 0.5

    body = slice(2 * SR, 3 * SR)
    assert measure_lufs(limited[..., body], SR) == pytest.approx(
        measure_lufs(audio[..., body], SR), abs=1.5
    ), "the body of the mix should be largely untouched"


def test_master_chain_handles_silence():
    out, report = master_chain(np.zeros((2, SR), dtype=np.float32), SR)
    assert out.shape[-1] == SR
    assert report["gain_db"] == pytest.approx(0.0)


# ----------------------------------------------------------- truncation ----

def _clip(start, seconds, value=0.5):
    return Clip(audio=np.full((2, int(seconds * SR)), value, dtype=np.float32),
                start=start)


def test_truncate_shortens_a_clip_that_spans_the_cut():
    timeline = Timeline(sample_rate=SR, channels=2)
    clip = timeline.add(_clip(0.0, 4.0))

    assert timeline.truncate_at(2.0) == 1
    assert clip.audio.shape[-1] == pytest.approx(2.0 * SR, abs=2)


def test_truncate_leaves_earlier_clips_alone():
    timeline = Timeline(sample_rate=SR, channels=2)
    clip = timeline.add(_clip(0.0, 1.0))
    original = clip.audio.shape[-1]

    assert timeline.truncate_at(3.0) == 0
    assert clip.audio.shape[-1] == original


def test_truncate_drops_a_fade_that_would_land_in_discarded_audio():
    timeline = Timeline(sample_rate=SR, channels=2)
    clip = timeline.add(Clip(audio=np.ones((2, 4 * SR), dtype=np.float32),
                             start=0.0, fade_out=1.0))
    timeline.truncate_at(2.0)
    assert clip.fade_out == 0.0


def test_consumed_window_is_not_left_behind():
    """The regression this whole change exists for.

    consume_tail renders a window and must remove it. When those were two
    steps, the processed copy was added back on top of the original and every
    transition played the outgoing track twice.
    """
    from app.render.mix import consume_tail

    timeline = Timeline(sample_rate=SR, channels=2)
    timeline.add(_clip(0.0, 4.0, value=0.5))

    tail = consume_tail(timeline, 2.0, 1.0, SR)
    assert float(np.max(np.abs(tail))) == pytest.approx(0.5, abs=1e-3)

    # Put the processed window back, exactly as the renderer does.
    timeline.add(Clip(audio=tail, start=2.0))
    rendered = timeline.render()

    window = rendered[..., int(2.1 * SR): int(2.9 * SR)]
    assert float(np.max(np.abs(window))) == pytest.approx(0.5, abs=1e-3), (
        "the window must appear once, not summed with the original"
    )


# ----------------------------------------------------------- level ride ----

def _loud_patch(seconds=12.0, patch_at=(5.0, 7.0), boost_db=9.0, sr=SR):
    """Steady noise with a loud stretch in the middle, like a hot transition."""
    audio = _noise(seconds=seconds, amplitude=0.08, sr=sr)
    lo, hi = int(patch_at[0] * sr), int(patch_at[1] * sr)
    audio[..., lo:hi] *= 10.0 ** (boost_db / 20.0)
    return audio


def _short_term_db(audio, sr=SR, window=1.0):
    step = int(window * sr)
    mono = audio.mean(axis=0)
    blocks = [mono[i:i + step] for i in range(0, len(mono) - step, step)]
    return np.array([20 * np.log10(max(np.sqrt(np.mean(b ** 2)), 1e-9)) for b in blocks])


def test_level_ride_flattens_a_hot_patch():
    """The remaining cause of transition spikes, measured directly.

    Two beat-matched tracks are correlated, so a blend sums in amplitude and
    runs hot even with equal-power fades and every effect switched off.
    """
    from app.render.loudness import level_ride

    audio = _loud_patch()
    before = _short_term_db(audio)
    ridden, applied = level_ride(audio, SR, target_lufs=-14.0)
    after = _short_term_db(ridden)

    assert applied > 1.0, "a 9dB excursion should trigger a correction"
    assert np.ptp(after) < np.ptp(before), "spread must narrow"
    assert np.ptp(after) < np.ptp(before) - 2.0


def test_level_ride_leaves_already_even_audio_alone():
    """Musical dynamics inside the tolerance band must survive."""
    from app.render.loudness import level_ride

    steady, _ = normalize_to(_noise(seconds=12.0), SR, -14.0)
    ridden, applied = level_ride(steady, SR, target_lufs=-14.0, tolerance_lu=1.5)

    assert applied < 1.5
    assert measure_lufs(ridden, SR) == pytest.approx(measure_lufs(steady, SR), abs=1.0)


def test_level_ride_correction_is_bounded():
    """A very loud patch must not be crushed to nothing."""
    from app.render.loudness import level_ride

    audio = _loud_patch(boost_db=24.0)
    _, applied = level_ride(audio, SR, target_lufs=-14.0, max_cut_db=8.0)
    assert applied <= 8.0 + 1e-6


def test_level_ride_skips_audio_too_short_to_measure():
    from app.render.loudness import level_ride

    short = _noise(seconds=0.5)
    ridden, applied = level_ride(short, SR)
    assert applied == 0.0
    assert np.allclose(ridden, short)


def test_master_chain_reports_the_ride_it_applied():
    audio = _loud_patch()
    _, report = master_chain(audio, SR, target_lufs=-14.0)
    assert "level_ride_db" in report
    assert report["level_ride_db"] > 0.0


def test_level_ride_can_be_switched_off():
    audio = _loud_patch()
    with_ride, _ = master_chain(audio, SR, ride_levels=True)
    without, report = master_chain(audio, SR, ride_levels=False)
    assert report["level_ride_db"] == 0.0
    assert np.ptp(_short_term_db(with_ride)) < np.ptp(_short_term_db(without))
