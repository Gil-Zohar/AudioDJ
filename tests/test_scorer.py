"""Section-pair scoring: weighting, vetoes, ranking and explainability."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import MatchingConfig
from app.matching.scorer import rank_pairs, score_pair
from app.models import Analysis, KeyEstimate, Section


def make_section(index=0, start=0.0, end=30.0, label="chorus", energy=0.5, chroma=None):
    if chroma is None:
        chroma = [0.3, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.25, 0.0, 0.15, 0.0, 0.0]
    return Section(index=index, start=start, end=end, label=label,
                   energy=energy, chroma=list(chroma))


def make_analysis(track_id="t", bpm=128.0, pitch_class=9, mode="minor",
                  confidence=0.9, sections=None):
    from app.analysis.key import camelot

    return Analysis(
        track_id=track_id, file_hash=f"hash_{track_id}", version=1,
        duration=180.0, sample_rate=22050, bpm=bpm, beat_tracker="test",
        beats=list(np.arange(0, 180, 60.0 / bpm)),
        downbeats=list(np.arange(0, 180, 4 * 60.0 / bpm)),
        key=KeyEstimate(pitch_class=pitch_class, mode=mode, confidence=confidence,
                        camelot=camelot(pitch_class, mode)),
        sections=sections if sections is not None else [make_section()],
    )


@pytest.fixture
def config():
    return MatchingConfig()


def test_breakdown_contains_every_term(config):
    a, b = make_analysis("a"), make_analysis("b", bpm=126.0, pitch_class=4)
    breakdown = score_pair(a, a.sections[0], b, b.sections[0], config)

    assert 0.0 <= breakdown.total <= 1.0
    assert breakdown.tempo is not None and breakdown.key is not None
    assert 0.0 <= breakdown.chroma <= 1.0
    assert breakdown.energy is not None
    assert breakdown.reasons, "a decision must come with its reasoning"


def test_contributions_sum_to_the_total(config):
    """The total must be exactly the weighted sum, or the breakdown lies."""
    a, b = make_analysis("a"), make_analysis("b", bpm=127.0, pitch_class=4)
    breakdown = score_pair(a, a.sections[0], b, b.sections[0], config)
    assert sum(breakdown.contributions().values()) == pytest.approx(breakdown.total, abs=1e-9)


def test_weights_are_normalized_before_use():
    """Weights that do not sum to 1 must be normalized, not applied raw."""
    config = MatchingConfig()
    config.weights.tempo = 3.0
    config.weights.key = 3.0
    config.weights.chroma = 3.0
    config.weights.energy = 3.0

    a, b = make_analysis("a"), make_analysis("b")
    breakdown = score_pair(a, a.sections[0], b, b.sections[0], config)
    assert breakdown.total <= 1.0
    assert sum(vars(breakdown.weights).values()) == pytest.approx(1.0)


def test_incompatible_tempo_is_a_hard_veto(config):
    """No amount of harmonic agreement rescues a mix you cannot beat-match."""
    a = make_analysis("a", bpm=128.0, pitch_class=9)
    b = make_analysis("b", bpm=100.0, pitch_class=9)   # same key, impossible tempo

    breakdown = score_pair(a, a.sections[0], b, b.sections[0], config)
    assert not breakdown.viable
    assert breakdown.total == 0.0
    assert "tempo incompatible" in (breakdown.rejected or "")


def test_score_below_minimum_is_rejected(config):
    config.min_score = 0.99
    a, b = make_analysis("a"), make_analysis("b", bpm=126.0, pitch_class=3)
    breakdown = score_pair(a, a.sections[0], b, b.sections[0], config)
    assert not breakdown.viable
    assert "below minimum" in (breakdown.rejected or "")


def test_better_match_scores_higher(config):
    """Same key and tempo must beat a distant key and a big stretch."""
    a = make_analysis("a", bpm=128.0, pitch_class=9, mode="minor")
    ideal = make_analysis("ideal", bpm=128.0, pitch_class=9, mode="minor")
    poor = make_analysis("poor", bpm=119.0, pitch_class=3, mode="major",
                         sections=[make_section(chroma=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])])

    good = score_pair(a, a.sections[0], ideal, ideal.sections[0], config)
    bad = score_pair(a, a.sections[0], poor, poor.sections[0], config)
    assert good.total > bad.total


def test_chroma_uses_the_shift_the_key_term_chose(config):
    """The chroma term must measure the audio as it will actually be played."""
    base = [0.4, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
    a = make_analysis("a", pitch_class=0, mode="major",
                      sections=[make_section(chroma=base)])
    b = make_analysis("b", pitch_class=2, mode="major",
                      sections=[make_section(chroma=list(np.roll(base, 2)))])

    breakdown = score_pair(a, a.sections[0], b, b.sections[0], config)
    if breakdown.key.semitones == -2:
        assert breakdown.chroma > 0.95, "rotation should realign the transposed copy"


def test_scoring_is_deterministic(config):
    a, b = make_analysis("a"), make_analysis("b", bpm=126.0)
    first = score_pair(a, a.sections[0], b, b.sections[0], config)
    second = score_pair(a, a.sections[0], b, b.sections[0], config)
    assert first.total == second.total
    assert first.reasons == second.reasons


def test_reasons_mention_the_decisive_facts(config):
    a, b = make_analysis("a", bpm=128.0), make_analysis("b", bpm=124.0, pitch_class=4)
    text = " ".join(score_pair(a, a.sections[0], b, b.sections[0], config).reasons).lower()
    for token in ("tempo", "key", "chroma", "energy"):
        assert token in text


def test_rejected_pair_explains_itself(config):
    a = make_analysis("a", bpm=128.0)
    b = make_analysis("b", bpm=100.0)
    breakdown = score_pair(a, a.sections[0], b, b.sections[0], config)
    assert any("REJECTED" in r for r in breakdown.reasons)


# ---------------------------------------------------------------- ranking ---

def _multi_section_analysis(track_id, bpm, pitch_class):
    sections = [
        make_section(0, 0, 30, "intro", 0.2),
        make_section(1, 30, 70, "verse", 0.5),
        make_section(2, 70, 110, "chorus", 0.8),
    ]
    return make_analysis(track_id, bpm=bpm, pitch_class=pitch_class, sections=sections)


def test_rank_pairs_is_sorted_best_first(config):
    a = _multi_section_analysis("a", 128.0, 9)
    b = _multi_section_analysis("b", 127.0, 9)
    ranked = rank_pairs(a, b, config)
    assert ranked
    scores = [p.score for p in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_pairs_respects_top_n(config):
    a = _multi_section_analysis("a", 128.0, 9)
    b = _multi_section_analysis("b", 127.0, 9)
    assert len(rank_pairs(a, b, config, top_n=2)) <= 2


def test_rank_pairs_skips_sections_that_are_too_short(config):
    a = make_analysis("a", sections=[make_section(0, 0, 3.0)])     # 3s section
    b = make_analysis("b")
    assert rank_pairs(a, b, config, min_section_seconds=8.0) == []


def test_rank_pairs_returns_nothing_when_tempo_is_impossible(config):
    a = _multi_section_analysis("a", 128.0, 9)
    b = _multi_section_analysis("b", 100.0, 9)
    assert rank_pairs(a, b, config) == []


def test_breakdown_serializes_for_the_api(config):
    a, b = make_analysis("a"), make_analysis("b", bpm=126.0)
    payload = score_pair(a, a.sections[0], b, b.sections[0], config).to_dict()
    assert {"total", "terms", "contributions", "reasons"} <= set(payload)
    assert {"tempo", "key", "chroma", "energy"} == set(payload["terms"])
    import json

    json.dumps(payload)   # must be JSON-serializable for the UI
