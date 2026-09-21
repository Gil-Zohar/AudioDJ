"""Tempo, key, chroma and energy compatibility primitives.

These are the rules that decide what may be mixed with what, so they are tested
directly rather than only through the scorer.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis.key import camelot
from app.matching.compat import (
    camelot_distance,
    camelot_score,
    chroma_similarity,
    energy_compat,
    key_compat,
    tempo_compat,
)


# ---------------------------------------------------------------- tempo ----

def test_identical_tempo_scores_perfectly():
    match = tempo_compat(128.0, 128.0)
    assert match.score == pytest.approx(1.0)
    assert match.ratio == 1.0
    assert match.stretch == pytest.approx(1.0)
    assert match.stretch_percent == pytest.approx(0.0)


def test_tempo_within_tolerance_is_accepted_and_scored_by_distance():
    close = tempo_compat(128.0, 126.0, tolerance=0.08)
    far = tempo_compat(128.0, 120.0, tolerance=0.08)
    assert close.compatible and far.compatible
    assert close.score > far.score, "a smaller stretch must score higher"


def test_tempo_outside_tolerance_is_rejected():
    match = tempo_compat(128.0, 100.0, tolerance=0.08, allow_half_double=True)
    assert not match.compatible
    assert match.score == 0.0


def test_stretch_direction_speeds_up_the_slower_track():
    """Guards a real bug: the ratio was once inverted, slowing B instead.

    B at 124 must be sped UP to reach A at 128, so stretch > 1.
    """
    match = tempo_compat(128.0, 124.0)
    assert match.stretch > 1.0
    assert match.stretch == pytest.approx(128.0 / 124.0, rel=1e-6)
    assert match.stretch_percent > 0

    reverse = tempo_compat(124.0, 128.0)
    assert reverse.stretch < 1.0, "a faster B must be slowed down"


def test_half_and_double_time_are_found():
    """175 and 87.5 BPM share a pulse; refusing that loses most cross-genre mixes."""
    match = tempo_compat(175.0, 87.5)
    assert match.compatible
    assert match.ratio == 2.0
    assert match.label == "double-time"
    assert match.stretch == pytest.approx(1.0), "a clean octave needs no stretching"

    other = tempo_compat(87.5, 175.0)
    assert other.ratio == 0.5
    assert other.label == "half-time"


def test_half_double_can_be_disabled():
    assert not tempo_compat(175.0, 87.5, allow_half_double=False).compatible


def test_exact_level_beats_half_time_at_equal_deviation():
    """A straight match should outrank a metrical-level jump."""
    same = tempo_compat(128.0, 128.0)
    doubled = tempo_compat(128.0, 64.0)
    assert doubled.compatible
    assert same.score > doubled.score


def test_three_two_is_off_by_default_and_opt_in():
    assert not tempo_compat(120.0, 80.0, allow_three_two=False).compatible
    assert tempo_compat(120.0, 80.0, allow_three_two=True).compatible


def test_invalid_tempo_is_handled():
    assert not tempo_compat(0.0, 128.0).compatible
    assert not tempo_compat(128.0, -5.0).compatible


# ------------------------------------------------------------------ key ----

def test_camelot_distance_follows_harmonic_mixing_rules():
    assert camelot_distance("8A", "8A") == 0          # same key
    assert camelot_distance("8A", "9A") == 1          # one hour round the wheel
    assert camelot_distance("8A", "7A") == 1
    assert camelot_distance("8A", "8B") == 1          # relative major/minor
    assert camelot_distance("12A", "1A") == 1, "the wheel must wrap"
    assert camelot_distance("8A", "2A") == 6          # opposite side


def test_camelot_score_decreases_with_distance():
    scores = [camelot_score(d) for d in range(7)]
    assert scores[0] == 1.0
    assert all(a >= b for a, b in zip(scores, scores[1:])), "must be non-increasing"
    assert scores[6] >= 0.0


def test_same_key_needs_no_shift():
    match = key_compat(9, "minor", 9, "minor")
    assert match.score == pytest.approx(1.0)
    assert match.semitones == 0
    assert match.distance == 0


def test_pitch_shift_is_used_to_reach_a_compatible_key():
    """A clashing key within reach should be fixed by shifting, not rejected.

    A minor (8A) against G# minor (1A) is five hours apart -- unusable as-is --
    but one semitone up lands G# minor exactly on A minor.
    """
    unshifted = camelot_distance(camelot(9, "minor"), camelot(8, "minor"))
    assert unshifted >= 4, "premise: these keys must genuinely clash"

    match = key_compat(9, "minor", 8, "minor", max_shift=2)
    assert match.semitones == 1
    assert match.distance == 0
    assert match.score > 0.8


def test_pitch_shift_respects_the_limit():
    for shift_limit in (0, 1, 2):
        match = key_compat(0, "major", 6, "major", max_shift=shift_limit)
        assert abs(match.semitones) <= shift_limit


def test_smaller_shifts_are_preferred_when_equally_compatible():
    match = key_compat(9, "minor", 9, "minor", max_shift=2)
    assert match.semitones == 0, "no shift should be chosen over an equivalent shift"


def test_low_key_confidence_pulls_the_score_toward_neutral():
    """Mizrahi/maqam material defeats 12-TET key detection.

    A weak estimate must not be trusted enough to veto a good match or wave
    through a clash, so the term fades toward 0.5.
    """
    confident = key_compat(9, "minor", 2, "minor", confidence_a=1.0, confidence_b=1.0)
    unsure = key_compat(9, "minor", 2, "minor", confidence_a=0.05, confidence_b=0.05)

    assert unsure.confidence_weight < confident.confidence_weight
    assert abs(unsure.score - 0.5) < abs(confident.score - 0.5)
    assert "low key confidence" in unsure.label


def test_key_match_reports_the_codes_it_used():
    match = key_compat(0, "major", 9, "minor")
    assert match.camelot_a == camelot(0, "major")
    assert match.camelot_b == camelot(9, "minor")
    assert match.camelot_b_shifted.endswith("A")


# --------------------------------------------------------------- chroma ----

def test_identical_chroma_is_fully_similar():
    vector = np.array([0.3, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.25, 0.0, 0.15, 0.0, 0.0])
    assert chroma_similarity(vector, vector) == pytest.approx(1.0)


def test_chroma_rotation_matches_the_pitch_shift():
    """Rotating by the chosen shift must realign a transposed copy."""
    vector = np.array([0.4, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    shifted = np.roll(vector, 3)

    assert chroma_similarity(vector, shifted, semitones=-3) == pytest.approx(1.0)
    assert chroma_similarity(vector, shifted, semitones=0) < 0.9


def test_disjoint_chroma_scores_low():
    a = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0])
    b = np.array([0, 1.0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 1.0, 0])
    assert chroma_similarity(a, b) < 0.3


def test_malformed_chroma_returns_zero():
    assert chroma_similarity([1, 2, 3], [1, 2, 3]) == 0.0
    assert chroma_similarity(np.zeros(12), np.zeros(12)) == 0.0


# --------------------------------------------------------------- energy ----

def test_energy_match_rewards_similar_levels():
    assert energy_compat(0.5, 0.5, "match").score == pytest.approx(1.0)
    assert energy_compat(0.5, 0.55, "match").score > energy_compat(0.5, 0.7, "match").score


def test_energy_build_rewards_a_deliberate_step_up():
    """A build is the goal before a drop, not a failed match."""
    on_target = energy_compat(0.4, 0.65, "build", build_target=0.25)
    flat = energy_compat(0.4, 0.4, "build", build_target=0.25)
    assert on_target.score == pytest.approx(1.0)
    assert on_target.score > flat.score
    assert on_target.delta == pytest.approx(0.25)


def test_energy_drop_rewards_a_step_down():
    assert energy_compat(0.8, 0.55, "drop", build_target=0.25).score == pytest.approx(1.0)


def test_energy_scores_stay_bounded():
    for intent in ("match", "build", "drop"):
        for a in (0.0, 0.5, 1.0):
            for b in (0.0, 0.5, 1.0):
                assert 0.0 <= energy_compat(a, b, intent).score <= 1.0


# ---------------------------------------------------------- set tempo ------

def test_fold_tempo_brings_octaves_together():
    """174 and 87 BPM share a pulse; folding is what makes that visible."""
    from app.matching.compat import fold_tempo

    assert fold_tempo(174.0)[0] == pytest.approx(87.0)
    assert fold_tempo(87.0)[0] == pytest.approx(87.0)
    assert fold_tempo(60.0)[0] == pytest.approx(120.0)
    assert fold_tempo(128.0)[0] == pytest.approx(128.0)


def test_fold_tempo_reports_the_ratio_used():
    from app.matching.compat import fold_tempo

    folded, ratio = fold_tempo(180.0)
    assert folded == pytest.approx(90.0)
    assert ratio == pytest.approx(0.5)
    assert folded == pytest.approx(180.0 * ratio)


def test_fold_tempo_handles_nonsense():
    from app.matching.compat import fold_tempo

    assert fold_tempo(0.0) == (0.0, 1.0)
    assert fold_tempo(-5.0) == (0.0, 1.0)


def test_choose_set_tempo_picks_the_densest_cluster():
    """The raw median is the obvious choice and a bad one.

    A real library mix spanned 86-182 BPM; the median landed at 117, which
    nothing could reach, and every track got stretched ~35% into mush.
    """
    from app.matching.compat import choose_set_tempo

    bpms = [129.67, 124.58, 120.0, 114.05, 110.41, 91.19, 182.0, 86.5]
    chosen = choose_set_tempo(bpms, tolerance=0.08)

    within = [b for b in bpms if abs(b - chosen) / chosen <= 0.10]
    assert len(within) >= 4, "the target must be reachable by most of the set"
    assert 100 <= chosen <= 140


def test_choose_set_tempo_ignores_a_lone_outlier():
    from app.matching.compat import choose_set_tempo

    chosen = choose_set_tempo([128.0, 127.0, 129.0, 128.5, 60.0], tolerance=0.08)
    assert chosen == pytest.approx(128.0, abs=2.0)


def test_choose_set_tempo_handles_empty_input():
    from app.matching.compat import choose_set_tempo

    assert choose_set_tempo([]) == 0.0
    assert choose_set_tempo([0.0, -1.0]) == 0.0


def test_stretch_to_target_uses_half_time_rather_than_mangling():
    """182 BPM joining a 95 BPM set is a 4% nudge at half time, not 48%."""
    from app.matching.compat import stretch_to_target

    rate, deviation = stretch_to_target(182.0, 95.0)

    assert deviation < 0.06, "half time should make this nearly free"
    assert rate == pytest.approx(95.0 / 91.0, rel=0.02)


def test_stretch_to_target_prefers_no_stretch_over_a_worse_fold():
    """Folding must not be applied blindly.

    Searching metrical levels has to keep whichever needs least stretching;
    forcing a tempo into a fixed window can pick a worse ratio than leaving it
    alone.
    """
    from app.matching.compat import stretch_to_target

    # Exactly double: reinterpreting the level costs nothing at all.
    rate, deviation = stretch_to_target(60.0, 120.0)
    assert rate == pytest.approx(1.0)
    assert deviation == pytest.approx(0.0)


def test_stretch_deviation_is_symmetric():
    """Speeding up and slowing down by the same factor are equally severe."""
    from app.matching.compat import stretch_to_target

    _, faster = stretch_to_target(100.0, 110.0)
    _, slower = stretch_to_target(110.0, 100.0)
    assert faster == pytest.approx(slower, rel=0.02)


def test_stretch_to_target_is_identity_at_the_same_tempo():
    from app.matching.compat import stretch_to_target

    rate, deviation = stretch_to_target(128.0, 128.0)
    assert rate == pytest.approx(1.0)
    assert deviation == pytest.approx(0.0)


def test_stretch_to_target_reports_genuinely_unreachable_tempos():
    """Some tempos simply cannot be reconciled, and must say so."""
    from app.matching.compat import stretch_to_target

    _, deviation = stretch_to_target(86.5, 123.5)
    assert deviation > 0.12, "this pair must exceed a sane stretch limit"
