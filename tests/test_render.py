"""Render primitives and a full mashup, checked against the tempo it claims."""
from __future__ import annotations

import numpy as np
import pytest

from app.render.engine import (
    apply_fades,
    bar_grid,
    db_to_gain,
    get_stretcher,
    latest_start_for_bars,
    normalize_peak,
    slice_bars,
    snap_to_bar,
    to_stereo,
)
from app.render.timeline import Clip, Timeline
from tests.test_scorer import make_analysis

SR = 22050


def _tone(seconds=2.0, freq=220.0, sample_rate=SR, channels=2):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.repeat(wave[np.newaxis, :], channels, axis=0)


# ------------------------------------------------------------ primitives ---

def test_db_to_gain_reference_points():
    assert db_to_gain(0.0) == pytest.approx(1.0)
    assert db_to_gain(-6.0) == pytest.approx(0.5012, abs=1e-3)
    assert db_to_gain(6.0) == pytest.approx(1.995, abs=1e-3)


def test_to_stereo_upmixes_and_downmixes():
    assert to_stereo(np.zeros((1, 100))).shape == (2, 100)
    assert to_stereo(np.zeros(100)).shape == (2, 100)
    assert to_stereo(np.zeros((4, 100))).shape == (2, 100)


def test_fades_are_applied_at_both_ends():
    audio = np.ones((2, SR), dtype=np.float32)
    faded = apply_fades(audio, SR, fade_in=0.1, fade_out=0.1)
    assert faded[0, 0] < 0.01
    assert faded[0, -1] < 0.05
    assert faded[0, SR // 2] == pytest.approx(1.0, abs=1e-3), "the middle must be untouched"


def test_equal_power_crossfade_holds_its_level():
    """Linear fades dip in the middle; equal-power ones do not."""
    length = 1000
    rising = apply_fades(np.ones((1, length), dtype=np.float32), length, 1.0, 0.0)[0]
    falling = apply_fades(np.ones((1, length), dtype=np.float32), length, 0.0, 1.0)[0]
    power = rising ** 2 + falling ** 2
    assert np.allclose(power, 1.0, atol=0.02)


def test_normalize_peak_hits_the_headroom_target():
    audio = (np.random.default_rng(0).standard_normal((2, 1000)) * 0.1).astype(np.float32)
    out = normalize_peak(audio, headroom_db=-1.0)
    assert float(np.max(np.abs(out))) == pytest.approx(db_to_gain(-1.0), abs=1e-4)


def test_normalize_peak_handles_silence():
    silent = np.zeros((2, 100), dtype=np.float32)
    assert not np.any(np.isnan(normalize_peak(silent)))


# ----------------------------------------------------------- bar slicing ---

def test_bar_grid_uses_downbeats():
    analysis = make_analysis("a", bpm=128.0)
    grid = bar_grid(analysis)
    assert grid.size > 1
    assert float(np.median(np.diff(grid))) == pytest.approx(4 * 60.0 / 128.0, rel=1e-6)


def test_snap_to_bar_directions():
    analysis = make_analysis("a", bpm=128.0)
    bar = 4 * 60.0 / 128.0
    assert snap_to_bar(analysis, bar * 2 + 0.1, "prev") == pytest.approx(bar * 2, abs=1e-6)
    assert snap_to_bar(analysis, bar * 2 + 0.1, "next") == pytest.approx(bar * 3, abs=1e-6)


def test_slice_bars_returns_the_requested_length():
    """Bar-aligned slicing is what keeps layered tracks phase-locked."""
    analysis = make_analysis("a", bpm=128.0)
    bar = 4 * 60.0 / 128.0
    audio = _tone(seconds=analysis.duration)

    sliced, start, bars = slice_bars(audio, analysis, 10.0, 8, SR)
    assert bars == 8
    assert sliced.shape[-1] == pytest.approx(8 * bar * SR, rel=0.02)
    assert start == pytest.approx(snap_to_bar(analysis, 10.0, "next"), abs=1e-6)


def test_latest_start_for_bars_leaves_room():
    """Guards a real bug: a late section silently produced a short mashup."""
    analysis = make_analysis("a", bpm=128.0)
    bar = 4 * 60.0 / 128.0
    latest = latest_start_for_bars(analysis, 32)
    assert latest + 32 * bar <= analysis.duration + bar
    assert latest_start_for_bars(analysis, 8) > latest, "fewer bars allows a later start"


# -------------------------------------------------------------- stretch ----

def test_stretcher_changes_duration_in_the_right_direction():
    """The regression that matters: rate > 1 must SHORTEN the audio.

    Inverting this once slowed the instrumental to 120 BPM instead of speeding
    it to 128, so the mashup's two layers were never actually aligned.
    """
    stretcher = get_stretcher("librosa")
    audio = _tone(seconds=4.0)
    original = audio.shape[-1]

    faster = stretcher.stretch(audio, 1.25, SR)
    slower = stretcher.stretch(audio, 0.8, SR)

    assert faster.shape[-1] < original, "rate > 1 must make audio shorter"
    assert slower.shape[-1] > original, "rate < 1 must make audio longer"
    assert faster.shape[-1] == pytest.approx(original / 1.25, rel=0.05)


def test_stretch_of_one_is_a_no_op():
    stretcher = get_stretcher("librosa")
    audio = _tone(seconds=1.0)
    assert stretcher.stretch(audio, 1.0, SR).shape == audio.shape


def test_pitch_shift_preserves_duration():
    stretcher = get_stretcher("librosa")
    audio = _tone(seconds=1.0)
    shifted = stretcher.pitch_shift(audio, 2.0, SR)
    assert shifted.shape[-1] == pytest.approx(audio.shape[-1], rel=0.02)


def test_pitch_shift_actually_moves_the_pitch():
    import librosa

    stretcher = get_stretcher("librosa")
    audio = _tone(seconds=1.5, freq=220.0)
    shifted = stretcher.pitch_shift(audio, 12.0, SR)   # one octave up

    def dominant(signal):
        spectrum = np.abs(np.fft.rfft(signal[0]))
        return float(np.fft.rfftfreq(signal.shape[-1], 1 / SR)[int(np.argmax(spectrum))])

    assert dominant(shifted) == pytest.approx(dominant(audio) * 2, rel=0.12)


# ------------------------------------------------------------- timeline ----

def test_timeline_places_clips_at_their_start_time():
    timeline = Timeline(sample_rate=SR, channels=2)
    timeline.add(Clip(audio=_tone(1.0), start=2.0, name="late"))
    out = timeline.render()

    assert out.shape[0] == 2
    assert float(np.max(np.abs(out[:, : int(1.9 * SR)]))) < 1e-6, "must be silent before the clip"
    assert float(np.max(np.abs(out[:, int(2.1 * SR): int(2.9 * SR)]))) > 0.1


def test_timeline_sums_overlapping_clips():
    timeline = Timeline(sample_rate=SR, channels=2)
    timeline.add(Clip(audio=np.ones((2, SR), dtype=np.float32) * 0.3, start=0.0))
    timeline.add(Clip(audio=np.ones((2, SR), dtype=np.float32) * 0.3, start=0.0))
    assert float(np.max(timeline.render())) == pytest.approx(0.6, abs=1e-3)


def test_timeline_applies_clip_gain():
    timeline = Timeline(sample_rate=SR, channels=2)
    timeline.add(Clip(audio=np.ones((2, SR), dtype=np.float32), start=0.0, gain_db=-6.0))
    assert float(np.max(timeline.render())) == pytest.approx(0.5012, abs=1e-3)


def test_timeline_tracklist_is_sorted_with_timecodes():
    timeline = Timeline(sample_rate=SR)
    timeline.add(Clip(audio=_tone(1.0), start=0.0))
    timeline.mark(65.0, "transition", "second")
    timeline.mark(5.0, "track_in", "first")

    events = timeline.tracklist()
    assert [e["label"] for e in events] == ["first", "second"]
    assert events[0]["timecode"] == "00:05.00"
    assert events[1]["timecode"] == "01:05.00"


def test_empty_timeline_renders_without_error():
    assert Timeline(sample_rate=SR).render().shape[0] == 2


# ------------------------------------------------------- full integration --

@pytest.mark.slow
def test_mashup_renders_at_the_tempo_it_claims(long_fixture_tracks):
    """End to end: the rendered audio must actually be at the target tempo.

    This is the check that catches an inverted stretch ratio, which leaves the
    two layers sounding plausible in isolation but never locked together.
    """
    import librosa
    import soundfile as sf

    from app.render.mashup import build_mashup

    track_a, track_b = long_fixture_tracks
    result = build_mashup(track_a, track_b, bars=16, out_name="pytest_mashup")

    audio, sample_rate = sf.read(result.files["wav"])
    assert audio.shape[0] / sample_rate == pytest.approx(16 * 4 * 60.0 / 128.0, rel=0.06)
    assert float(np.max(np.abs(audio))) <= 1.0, "must not clip"

    mono = audio.mean(axis=1).astype("float32")
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sample_rate, trim=False)
    detected = 60.0 / float(np.median(np.diff(librosa.frames_to_time(beats, sr=sample_rate))))
    target = result.tracklist["target_bpm"]

    # Allow the usual metrical-level ambiguity, but the pulse must be a clean
    # multiple of the target -- 120 against a 128 target would fail here.
    errors = [abs(detected * m - target) / target for m in (0.5, 1.0, 2.0, 4.0)]
    assert min(errors) < 0.03, f"detected {detected:.2f} is not a multiple of target {target:.2f}"


@pytest.mark.slow
def test_mashup_writes_a_usable_tracklist(long_fixture_tracks):
    from app.render.mashup import build_mashup

    track_a, track_b = long_fixture_tracks
    result = build_mashup(track_a, track_b, bars=8, out_name="pytest_tracklist")

    tracklist = result.tracklist
    assert tracklist["mode"] == "mashup"
    assert len(tracklist["tracks"]) == 2
    assert {t["role"] for t in tracklist["tracks"]} == {"vocal", "instrumental"}
    assert tracklist["events"], "the tracklist must carry timestamped events"
    assert all("timecode" in e for e in tracklist["events"])
    assert tracklist["match"]["breakdown"]["reasons"]


@pytest.mark.slow
def test_transitions_do_not_spike_in_loudness(long_fixture_tracks):
    """The regression this guards is one the user heard before any test did.

    consume_tail renders a window of the timeline and must also remove it. When
    rendering and removing were separate steps they drifted apart: the
    processed window was added back on top of the original, so the outgoing
    track played twice through every transition. Measured spikes ran +10 to
    +17dB above steady state.
    """
    import json

    import soundfile as sf

    from app.render.mix import build_mix

    track_a, track_b = long_fixture_tracks
    result = build_mix([track_a, track_b], target_minutes=3, blend="classic",
                       out_name="pytest_levels", hype="off", fx_style="club")

    audio, sample_rate = sf.read(result.files["wav"])
    mono = audio.mean(axis=1)

    window = sample_rate // 4
    rms = np.array([
        np.sqrt(np.mean(mono[i:i + window] ** 2))
        for i in range(0, len(mono) - window, window)
    ])
    times = np.arange(len(rms)) * window / sample_rate
    steady = float(np.median(rms[rms > 1e-5]))
    assert steady > 0

    transitions = [e["time"] for e in result.tracklist["events"]
                   if e["kind"] == "transition"]
    assert transitions, "a two-track set must contain a transition"

    for at in transitions:
        mask = (times >= at - 2.0) & (times <= at + 8.0)
        if not mask.any():
            continue
        spike_db = 20 * np.log10(float(rms[mask].max()) / steady)
        assert spike_db < 6.0, (
            f"transition at {at:.0f}s is {spike_db:+.1f}dB above steady state; "
            "the outgoing track is probably being summed with itself"
        )


@pytest.mark.slow
def test_mix_is_mastered_to_the_configured_target(long_fixture_tracks):
    """Levels should land on the loudness target, under the peak ceiling."""
    import soundfile as sf

    from app.config import get_tuning
    from app.render.loudness import measure_lufs, true_peak_db
    from app.render.mix import build_mix

    tuning = get_tuning()
    track_a, track_b = long_fixture_tracks
    result = build_mix([track_a, track_b], target_minutes=3, blend="classic",
                       out_name="pytest_master", hype="off", fx_style="clean")

    audio, sample_rate = sf.read(result.files["wav"])
    audio = audio.T

    assert measure_lufs(audio, sample_rate) == pytest.approx(
        tuning.render.target_lufs, abs=2.0)
    assert true_peak_db(audio) <= tuning.render.true_peak_ceiling_db + 0.5
