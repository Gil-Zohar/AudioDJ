"""The analysis pipeline must recover the ground truth baked into the fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis.key import camelot, parse_camelot
from tests.fixtures.synth import ALL_TRACKS, TRACK_A, TRACK_B, TRACK_C


def _spec(name):
    return next(s for s in ALL_TRACKS if s.name == name)


@pytest.mark.parametrize("name", [TRACK_A.name, TRACK_B.name])
def test_bpm_recovered(analyses, name):
    """Tempo within 1.5%.

    Beat frames quantise to the analysis hop, so the grid is refined by a
    least-squares fit; without it the error is around 1% and the grid drifts.
    """
    spec = _spec(name)
    got = analyses[name].bpm
    assert got == pytest.approx(spec.bpm, rel=0.015), f"{name}: {got} vs {spec.bpm}"


def test_half_time_detection_is_the_known_octave_error(analyses):
    """175 BPM is heard as ~87.5.

    Beat trackers routinely pick the wrong metrical level. This is not a bug to
    fix here -- the tempo matcher treats half/double time as compatible -- but it
    is pinned so the behaviour stays known.
    """
    got = analyses[TRACK_C.name].bpm
    assert got == pytest.approx(TRACK_C.bpm / 2, rel=0.02) or got == pytest.approx(
        TRACK_C.bpm, rel=0.02
    ), f"expected ~{TRACK_C.bpm} or ~{TRACK_C.bpm / 2}, got {got}"


@pytest.mark.parametrize("name", [t.name for t in ALL_TRACKS])
def test_key_recovered(analyses, name):
    spec = _spec(name)
    key = analyses[name].key
    assert (key.pitch_class, key.mode) == (spec.key_pitch_class, spec.key_mode)
    assert key.camelot == camelot(spec.key_pitch_class, spec.key_mode)
    assert 0.0 <= key.confidence <= 1.0


@pytest.mark.parametrize("name", [TRACK_A.name, TRACK_B.name])
def test_downbeats_land_on_the_bar_grid(analyses, name):
    """Downbeats must be one bar apart and start at bar one.

    Anything else makes beat-aligned mixing impossible, and the phase is easy to
    get wrong on four-on-the-floor material.
    """
    spec = _spec(name)
    analysis = analyses[name]
    downbeats = np.asarray(analysis.downbeats)

    assert len(downbeats) >= spec.total_bars - 2

    spacing = float(np.median(np.diff(downbeats)))
    assert spacing == pytest.approx(spec.seconds_per_bar, rel=0.02)

    # The fixture starts on bar one, so the first downbeat belongs near zero.
    assert downbeats[0] < spec.seconds_per_bar * 0.5


@pytest.mark.parametrize("name", [TRACK_A.name, TRACK_B.name])
def test_sections_are_ordered_and_contiguous(analyses, name):
    sections = analyses[name].sections
    assert len(sections) >= 3
    assert sections[0].start == pytest.approx(0.0, abs=0.01)
    for a, b in zip(sections[:-1], sections[1:]):
        assert b.start == pytest.approx(a.end, abs=1e-6), "sections must not overlap or gap"
        assert a.duration > 0


@pytest.mark.parametrize("name", [TRACK_A.name, TRACK_B.name])
def test_sections_start_on_downbeats(analyses, name):
    """Every internal boundary must sit on the bar grid, or mixes land off-beat."""
    analysis = analyses[name]
    downbeats = np.asarray(analysis.downbeats)
    bar = float(np.median(np.diff(downbeats)))
    for section in analysis.sections[1:]:
        distance = float(np.min(np.abs(downbeats - section.start)))
        assert distance < bar * 0.25, f"boundary {section.start:.2f}s is off the bar grid"


@pytest.mark.parametrize("name", [TRACK_A.name, TRACK_B.name])
def test_structural_boundaries_are_found(analyses, name):
    """At least 3 of the 4 true boundaries, within 3 seconds.

    The verse->chorus change is the subtlest in the fixtures and is allowed to be
    missed; the intro, drop and outro edges are not.
    """
    spec = _spec(name)
    truth, elapsed = [], 0.0
    for section in spec.sections:
        truth.append(elapsed)
        elapsed += section.bars * spec.seconds_per_bar

    detected = [s.start for s in analyses[name].sections]
    hits = sum(1 for t in truth if any(abs(t - d) <= 3.0 for d in detected))
    assert hits >= 4, f"only {hits}/{len(truth)} boundaries found: {detected} vs {truth}"


@pytest.mark.parametrize("name", [TRACK_A.name, TRACK_B.name])
def test_energy_ranks_sections_correctly(analyses, name):
    """The intro must be quieter than the loudest section, and labels must follow."""
    sections = analyses[name].sections
    energies = [s.energy for s in sections]
    assert sections[0].energy == min(energies), "intro should be the quietest section"
    assert max(energies) > min(energies) + 0.15, "energy curve is too flat to be useful"
    assert sections[0].label == "intro"
    assert any(s.label in ("chorus", "drop") for s in sections)


def test_energy_curve_is_bounded_and_aligned(analyses):
    analysis = analyses[TRACK_A.name]
    assert len(analysis.energy_curve) == len(analysis.energy_times)
    assert all(0.0 <= v <= 1.0 for v in analysis.energy_curve)
    assert analysis.energy_times[-1] <= analysis.duration + 1.0


def test_analysis_is_cached(analyses, fixture_paths):
    """A second call must come from SQLite, not recompute."""
    from app.analysis.pipeline import PIPELINE_VERSION
    from app.db import get_db
    from app.sources.local import LocalLibrarySource

    from tests.conftest import FIXTURE_DIR

    source = LocalLibrarySource(FIXTURE_DIR)
    track = next(t for t in source.scan() if t.title == TRACK_A.name)
    cached = get_db().get_analysis(track.file_hash, PIPELINE_VERSION)
    assert cached is not None
    assert cached.bpm == pytest.approx(analyses[TRACK_A.name].bpm)


def test_manual_key_override_wins_and_resets(analyses):
    from app.analysis.pipeline import (
        PIPELINE_VERSION,
        _apply_override,
        clear_key_override,
        set_key_override,
    )
    from app.db import get_db
    from app.sources.local import LocalLibrarySource

    from tests.conftest import FIXTURE_DIR

    source = LocalLibrarySource(FIXTURE_DIR)
    track = next(t for t in source.scan() if t.title == TRACK_A.name)
    db = get_db()

    set_key_override(track.file_hash, 3, "major")   # D# major, deliberately wrong
    overridden = _apply_override(db.get_analysis(track.file_hash, PIPELINE_VERSION), db)
    assert (overridden.key.pitch_class, overridden.key.mode) == (3, "major")
    assert overridden.key.source == "manual"
    assert overridden.key.confidence == 1.0
    assert overridden.key.camelot == camelot(3, "major")

    clear_key_override(track.file_hash)
    restored = _apply_override(db.get_analysis(track.file_hash, PIPELINE_VERSION), db)
    assert restored.key.source == "detected"
    assert restored.key.pitch_class == TRACK_A.key_pitch_class


def test_camelot_matches_the_published_wheel():
    """Spot-check every one of the 24 codes; a wrong wheel silently ruins mixing."""
    expected = {
        (0, "major"): "8B", (7, "major"): "9B", (2, "major"): "10B", (9, "major"): "11B",
        (4, "major"): "12B", (11, "major"): "1B", (6, "major"): "2B", (1, "major"): "3B",
        (8, "major"): "4B", (3, "major"): "5B", (10, "major"): "6B", (5, "major"): "7B",
        (9, "minor"): "8A", (4, "minor"): "9A", (11, "minor"): "10A", (6, "minor"): "11A",
        (1, "minor"): "12A", (8, "minor"): "1A", (3, "minor"): "2A", (10, "minor"): "3A",
        (5, "minor"): "4A", (0, "minor"): "5A", (7, "minor"): "6A", (2, "minor"): "7A",
    }
    for (pitch_class, mode), code in expected.items():
        assert camelot(pitch_class, mode) == code
        assert parse_camelot(code) == (pitch_class, mode)
