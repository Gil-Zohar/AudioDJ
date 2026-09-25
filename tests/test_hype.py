"""Hype layer: sample synthesis, classification, and where elements land."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import TuningConfig
from app.render.hype import (
    DENSITY,
    NullTtsProvider,
    apply_hype,
    classify,
    collect_cues,
    generate_default_samples,
    get_tts_provider,
    load_samples,
    make_airhorn,
    make_impact,
    make_riser,
    make_sweep,
)
from app.render.timeline import Timeline

SR = 44100


def _brightness(audio: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """Mean spectral centroid over the first and last fifths."""
    import librosa

    mono = audio.mean(axis=0).astype("float32")
    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)[0]
    fifth = max(1, len(centroid) // 5)
    return float(centroid[:fifth].mean()), float(centroid[-fifth:].mean())


def _loudness(audio: np.ndarray) -> tuple[float, float]:
    mono = np.abs(audio.mean(axis=0))
    fifth = max(1, len(mono) // 5)
    return float(mono[:fifth].mean()), float(mono[-fifth:].mean())


# ------------------------------------------------------------- synthesis ---

@pytest.mark.parametrize("generator", [make_riser, make_impact, make_airhorn, make_sweep])
def test_generated_samples_are_sane_audio(generator):
    audio = generator(SR)
    assert audio.shape[0] == 2, "must be stereo"
    assert audio.shape[-1] > SR * 0.5
    assert np.all(np.isfinite(audio))
    assert float(np.max(np.abs(audio))) <= 1.0, "must not clip"
    assert float(np.max(np.abs(audio))) > 0.5, "must not be near-silent"


def test_riser_actually_rises():
    """A riser must brighten AND get louder.

    Ramping the volume of flat white noise leaves the spectrum unchanged, and
    the build does not read as rising at all -- which is exactly what the first
    implementation did.
    """
    audio = make_riser(SR)
    start_bright, end_bright = _brightness(audio)
    start_loud, end_loud = _loudness(audio)

    assert end_bright > start_bright * 1.5, "a riser must sweep upward in pitch"
    assert end_loud > start_loud, "a riser must build in volume"


def test_sweep_falls():
    """The downward counterpart, for coming out of a drop."""
    start_bright, end_bright = _brightness(make_sweep(SR))
    assert end_bright < start_bright * 0.7


def test_impact_darkens_as_it_rings():
    """A real impact is a bright crack decaying into a low boom."""
    audio = make_impact(SR)
    start_bright, end_bright = _brightness(audio)
    start_loud, end_loud = _loudness(audio)

    assert end_bright < start_bright, "the crack must decay faster than the boom"
    assert end_loud < start_loud, "an impact decays"


def test_generators_are_deterministic():
    assert np.array_equal(make_riser(SR), make_riser(SR))


def test_generate_default_samples_writes_files(tmp_path):
    written = generate_default_samples(tmp_path, SR)
    assert set(written) == {"riser", "impact", "airhorn", "sweep"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 1000


def test_generation_is_skipped_when_files_exist(tmp_path):
    first = generate_default_samples(tmp_path, SR)
    stamps = {k: p.stat().st_mtime_ns for k, p in first.items()}
    second = generate_default_samples(tmp_path, SR)
    assert all(second[k].stat().st_mtime_ns == stamps[k] for k in stamps)


# --------------------------------------------------------- classification ---

@pytest.mark.parametrize("filename,expected", [
    ("riser_01.wav", "riser"),
    ("airhorn.wav", "airhorn"),
    ("impact-hit.wav", "impact"),
    ("shout_hey.wav", "shout"),
    ("BUILD_up_long.wav", "riser"),
    ("downlifter_2.wav", "sweep"),
    ("boom_low.wav", "impact"),
    ("vox_tag.wav", "shout"),
])
def test_sample_names_are_classified(filename, expected):
    assert classify(filename) == expected


def test_unrecognised_names_are_ignored():
    """Better to skip a file than guess wrong and fire an airhorn as a riser."""
    assert classify("random_noise.wav") is None
    assert classify("track01.wav") is None


# -------------------------------------------------------------- library ----

def test_missing_samples_fall_back_to_generated(tmp_path):
    """The feature must work before the user owns any samples."""
    library = load_samples(tmp_path / "empty", tmp_path / "generated", SR)

    assert library.generated is True
    assert library.count() >= 4
    for kind in ("riser", "impact", "airhorn", "sweep"):
        assert library.by_kind[kind], f"{kind} should have a generated default"


def test_user_samples_are_loaded(tmp_path):
    import soundfile as sf

    samples = tmp_path / "samples"
    samples.mkdir()
    sf.write(samples / "riser_mine.wav",
             np.zeros((SR, 2), dtype=np.float32), SR)

    library = load_samples(samples, tmp_path / "generated", SR)
    names = [s.name for s in library.by_kind["riser"]]
    assert "riser_mine" in names


def test_a_corrupt_sample_does_not_break_the_mix(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "riser_broken.wav").write_bytes(b"not audio at all")

    library = load_samples(samples, tmp_path / "generated", SR)
    assert library.count() >= 4, "should fall back rather than raise"


# ------------------------------------------------------------ placement ----

def _timeline_with_cues(times, kind="drop"):
    timeline = Timeline(sample_rate=SR, channels=2)
    # A bed of audio so the timeline has a length for clips to sit inside.
    timeline.add(_silence(max(times) + 30.0))
    for t in times:
        timeline.mark(t, "cue", kind)
    return timeline


def _silence(seconds):
    from app.render.timeline import Clip

    return Clip(audio=np.zeros((2, int(seconds * SR)), dtype=np.float32), start=0.0)


def test_collect_cues_reads_both_cue_and_transition_events():
    timeline = Timeline(sample_rate=SR)
    timeline.mark(30.0, "cue", "drop")
    timeline.mark(10.0, "transition", "bass swap")
    timeline.mark(20.0, "track_in", "some track")

    cues = collect_cues(timeline)
    assert [c.kind for c in cues] == ["transition", "drop"], "must be time-ordered"
    assert not any(c.kind == "track_in" for c in cues)


def test_hype_off_places_nothing():
    timeline = _timeline_with_cues([40.0, 80.0])
    assert apply_hype(timeline, 128.0, "off", TuningConfig()) == 0


def test_heavy_places_more_than_light():
    tuning = TuningConfig()
    times = [40.0, 80.0, 120.0, 160.0]

    light = _timeline_with_cues(times)
    heavy = _timeline_with_cues(times)
    light_count = apply_hype(light, 128.0, "light", tuning, seed=1)
    heavy_count = apply_hype(heavy, 128.0, "heavy", tuning, seed=1)

    assert heavy_count >= light_count
    assert heavy_count > 0


def test_riser_ends_on_the_cue_not_after_it():
    """The whole point: the build has to resolve exactly where the drop lands."""
    tuning = TuningConfig()
    cue_time = 60.0
    timeline = _timeline_with_cues([cue_time])
    apply_hype(timeline, 128.0, "heavy", tuning, seed=3)

    risers = [c for c in timeline.clips if "riser" in (c.name or "")]
    assert risers, "a drop should get a riser"

    riser = risers[0]
    end = riser.start + riser.audio.shape[-1] / SR
    assert end == pytest.approx(cue_time, abs=0.05), (
        f"riser ends at {end:.2f}s but the cue is at {cue_time:.2f}s"
    )
    assert riser.start < cue_time


def test_impact_lands_on_the_cue():
    timeline = _timeline_with_cues([60.0])
    apply_hype(timeline, 128.0, "heavy", TuningConfig(), seed=3)

    impacts = [c for c in timeline.clips if "impact" in (c.name or "")]
    assert impacts
    assert impacts[0].start == pytest.approx(60.0, abs=0.01)


def test_hype_is_not_stacked_on_itself():
    """Cues packed together must not produce a pile-up."""
    tuning = TuningConfig()
    timeline = _timeline_with_cues([60.0, 61.0, 62.0, 63.0])
    apply_hype(timeline, 128.0, "light", tuning, seed=5)

    marks = sorted(e.time for e in timeline.events if e.kind == "hype")
    gaps = np.diff(marks) if len(marks) > 1 else np.array([999.0])
    assert all(g >= DENSITY["light"]["min_gap"] - 1e-6 for g in gaps)


def test_nothing_is_placed_before_the_track_starts():
    """A cue in the first bar has nothing to build into."""
    timeline = _timeline_with_cues([0.4])
    apply_hype(timeline, 128.0, "heavy", TuningConfig(), seed=7)
    assert all(c.start >= 0 for c in timeline.clips)


def test_placement_is_reproducible_with_a_seed():
    tuning = TuningConfig()
    times = [40.0, 80.0, 120.0]

    first = _timeline_with_cues(times)
    second = _timeline_with_cues(times)
    assert apply_hype(first, 128.0, "light", tuning, seed=11) == \
           apply_hype(second, 128.0, "light", tuning, seed=11)
    assert [c.start for c in first.clips] == [c.start for c in second.clips]


def test_hype_clips_are_tagged_for_the_tracklist():
    timeline = _timeline_with_cues([60.0])
    apply_hype(timeline, 128.0, "heavy", TuningConfig(), seed=3)

    hype_clips = [c for c in timeline.clips if c.stem == "hype"]
    assert hype_clips
    assert any(e.kind == "hype" for e in timeline.events)


def test_hype_respects_the_configured_gain():
    tuning = TuningConfig()
    tuning.hype.gain_db = -12.0
    timeline = _timeline_with_cues([60.0])
    apply_hype(timeline, 128.0, "heavy", tuning, seed=3)

    hype_clips = [c for c in timeline.clips if c.stem == "hype"]
    assert all(c.gain_db <= -11.0 for c in hype_clips)


# ------------------------------------------------------------------ tts ----

def test_null_tts_is_the_default_and_says_nothing():
    """No voice ships by default: a Hebrew voice means a big model or a
    network call at render time, and neither belongs in the core path."""
    provider = get_tts_provider("none")
    assert isinstance(provider, NullTtsProvider)
    assert provider.available() is False
    assert provider.say("יאללה", "he") is None


def test_unknown_tts_provider_falls_back_quietly():
    assert get_tts_provider("something-made-up").available() is False
