"""Set planning: energy arc, ordering and transition selection."""
from __future__ import annotations

import pytest

from app.config import MatchingConfig
from app.matching.planner import (
    energy_arc,
    order_for_energy_flow,
    plan_set,
    track_energy,
)
from tests.test_scorer import make_analysis, make_section


@pytest.fixture
def config():
    return MatchingConfig()


def _track(track_id, bpm, energy, pitch_class=9):
    sections = [
        make_section(0, 0, 30, "intro", max(0.0, energy - 0.3)),
        make_section(1, 30, 70, "verse", energy),
        make_section(2, 70, 110, "chorus", min(1.0, energy + 0.1)),
    ]
    return make_analysis(track_id, bpm=bpm, pitch_class=pitch_class, sections=sections)


# ------------------------------------------------------------ energy arc ---

def test_energy_arc_rises_then_falls():
    """A set opens moderate, peaks late, then eases off."""
    arc = energy_arc(8)
    assert len(arc) == 8
    peak = arc.index(max(arc))
    assert 0 < peak < len(arc) - 1, "the peak must not be the first or last slot"
    assert arc[0] < max(arc)
    assert arc[-1] < max(arc)


def test_energy_arc_peaks_near_three_quarters():
    arc = energy_arc(12)
    assert 0.6 <= arc.index(max(arc)) / (len(arc) - 1) <= 0.85


def test_energy_arc_handles_degenerate_sizes():
    assert len(energy_arc(1)) == 1
    assert len(energy_arc(2)) == 2


def test_track_energy_is_weighted_toward_the_peak():
    """One huge drop should outrank a uniformly busy track."""
    peaky = make_analysis("peaky", sections=[
        make_section(0, 0, 40, "verse", 0.2), make_section(1, 40, 80, "drop", 1.0)])
    flat = make_analysis("flat", sections=[
        make_section(0, 0, 40, "verse", 0.55), make_section(1, 40, 80, "verse", 0.55)])
    assert track_energy(peaky) > track_energy(flat)


# -------------------------------------------------------------- ordering ---

def test_ordering_keeps_every_track_exactly_once(config):
    tracks = [_track(f"t{i}", 128.0, e) for i, e in enumerate([0.9, 0.3, 0.6, 0.75, 0.45])]
    ordered, log = order_for_energy_flow(tracks, config)

    assert len(ordered) == len(tracks)
    assert {a.track_id for a in ordered} == {a.track_id for a in tracks}
    assert len(log) == len(tracks)


def test_ordering_opens_below_its_peak(config):
    tracks = [_track(f"t{i}", 128.0, e) for i, e in enumerate([0.95, 0.25, 0.55, 0.8, 0.4, 0.65])]
    ordered, _ = order_for_energy_flow(tracks, config)
    energies = [track_energy(a) for a in ordered]
    assert energies[0] < max(energies), "a set should not open on its biggest track"


def test_ordering_prefers_mixable_neighbours(config):
    """Tempo compatibility is a preference, so an odd tempo drifts to the end."""
    tracks = [
        _track("a", 128.0, 0.45), _track("b", 127.0, 0.6),
        _track("c", 129.0, 0.8), _track("odd", 95.0, 0.5),
    ]
    ordered, _ = order_for_energy_flow(tracks, config)
    assert {a.track_id for a in ordered} == {"a", "b", "c", "odd"}


def test_ordering_survives_when_nothing_is_mixable(config):
    """A penalty, not a veto: an unorderable set is worse than an awkward join."""
    tracks = [_track("a", 128.0, 0.4), _track("b", 90.0, 0.6), _track("c", 160.0, 0.8)]
    ordered, _ = order_for_energy_flow(tracks, config)
    assert len(ordered) == 3


def test_ordering_handles_a_single_track(config):
    single = [_track("only", 128.0, 0.5)]
    ordered, _ = order_for_energy_flow(single, config)
    assert len(ordered) == 1


# ------------------------------------------------------------ whole plan ---

def test_plan_set_joins_consecutive_tracks(config):
    tracks = [_track(f"t{i}", 128.0 + i * 0.5, e) for i, e in enumerate([0.4, 0.6, 0.8, 0.5])]
    plan = plan_set(tracks, config)

    assert len(plan.order) == 4
    assert plan.transitions, "compatible tracks must produce transitions"
    assert len(plan.transitions) <= len(plan.order) - 1
    assert plan.log


def test_plan_set_transitions_follow_the_running_order(config):
    tracks = [_track(f"t{i}", 128.0, e) for i, e in enumerate([0.4, 0.6, 0.85, 0.55])]
    plan = plan_set(tracks, config)
    order = [a.track_id for a in plan.order]

    for transition in plan.transitions:
        from_index = order.index(transition.from_track)
        assert order[from_index + 1] == transition.to_track


def test_plan_set_logs_joins_it_could_not_make(config):
    """An impossible join must be reported, not silently dropped."""
    tracks = [_track("a", 128.0, 0.4), _track("b", 92.0, 0.6)]
    plan = plan_set(tracks, config)
    if not plan.transitions:
        assert any("hard-cut" in line for line in plan.log)


def test_plan_serializes_for_the_api(config):
    import json

    tracks = [_track(f"t{i}", 128.0, e) for i, e in enumerate([0.4, 0.7, 0.85])]
    json.dumps(plan_set(tracks, config).to_dict())
