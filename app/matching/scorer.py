"""Section-pair scoring.

Pure like `compat`: takes analysis values in, gives a score and a written
justification out. Every pair the renderer uses carries its breakdown, so a mix
that sounds wrong can be traced to the decision that caused it rather than
guessed at.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from app.config import EnergyConfig, MatchingConfig, ScoringWeights
from app.matching.compat import (
    EnergyMatch,
    KeyMatch,
    TempoMatch,
    chroma_similarity,
    energy_compat,
    key_compat,
    tempo_compat,
)
from app.models import Analysis, Section


@dataclass
class ScoreBreakdown:
    """Why a pair scored what it did."""

    total: float
    tempo: TempoMatch
    key: KeyMatch
    chroma: float
    energy: EnergyMatch
    weights: ScoringWeights
    reasons: list[str] = field(default_factory=list)
    rejected: Optional[str] = None

    @property
    def viable(self) -> bool:
        return self.rejected is None

    def contributions(self) -> dict[str, float]:
        """Weighted contribution of each term, for display."""
        return {
            "tempo": self.tempo.score * self.weights.tempo,
            "key": self.key.score * self.weights.key,
            "chroma": self.chroma * self.weights.chroma,
            "energy": self.energy.score * self.weights.energy,
        }

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "rejected": self.rejected,
            "reasons": self.reasons,
            "terms": {
                "tempo": {**asdict(self.tempo), "weight": self.weights.tempo},
                "key": {**asdict(self.key), "weight": self.weights.key},
                "chroma": {"score": round(self.chroma, 4), "weight": self.weights.chroma},
                "energy": {**asdict(self.energy), "weight": self.weights.energy},
            },
            "contributions": {k: round(v, 4) for k, v in self.contributions().items()},
        }


@dataclass
class SectionPair:
    """A candidate transition from one track's section into another's."""

    track_a_id: str
    track_b_id: str
    section_a: Section
    section_b: Section
    breakdown: ScoreBreakdown

    @property
    def score(self) -> float:
        return self.breakdown.total

    def to_dict(self) -> dict:
        return {
            "track_a": self.track_a_id,
            "track_b": self.track_b_id,
            "section_a": {
                "index": self.section_a.index, "label": self.section_a.label,
                "start": round(self.section_a.start, 2), "end": round(self.section_a.end, 2),
                "energy": round(self.section_a.energy, 3),
            },
            "section_b": {
                "index": self.section_b.index, "label": self.section_b.label,
                "start": round(self.section_b.start, 2), "end": round(self.section_b.end, 2),
                "energy": round(self.section_b.energy, 3),
            },
            "score": round(self.score, 4),
            "breakdown": self.breakdown.to_dict(),
        }


def score_pair(
    analysis_a: Analysis,
    section_a: Section,
    analysis_b: Analysis,
    section_b: Section,
    config: MatchingConfig,
    intent: str = "match",
) -> ScoreBreakdown:
    """Score moving from `section_a` into `section_b`.

    Terms are combined as a weighted sum of four normalized scores. An
    incompatible tempo is a hard veto rather than a low score: no amount of
    harmonic agreement rescues a mix you cannot beat-match.
    """
    weights = config.weights.normalized()

    tempo = tempo_compat(
        analysis_a.bpm, analysis_b.bpm,
        tolerance=config.tempo_tolerance,
        allow_half_double=config.allow_half_double,
        allow_three_two=config.allow_three_two,
    )

    key = key_compat(
        analysis_a.key.pitch_class, analysis_a.key.mode,
        analysis_b.key.pitch_class, analysis_b.key.mode,
        max_shift=config.max_pitch_shift,
        confidence_a=analysis_a.key.confidence,
        confidence_b=analysis_b.key.confidence,
        confidence_floor=config.key_confidence_floor,
    )

    # Rotate B's chroma by the shift the key term actually chose, so this
    # measures the audio as it will be played, not as it sits on disk.
    chroma = chroma_similarity(section_a.chroma, section_b.chroma, key.semitones)

    energy = energy_compat(
        section_a.energy, section_b.energy,
        intent=intent,
        match_tolerance=config.energy.match_tolerance,
        build_target=config.energy.build_target,
    )

    total = (
        tempo.score * weights.tempo
        + key.score * weights.key
        + chroma * weights.chroma
        + energy.score * weights.energy
    )

    breakdown = ScoreBreakdown(
        total=float(total), tempo=tempo, key=key, chroma=float(chroma),
        energy=energy, weights=weights,
    )

    if not tempo.compatible:
        breakdown.rejected = (
            f"tempo incompatible: {analysis_a.bpm:.1f} vs {analysis_b.bpm:.1f} BPM "
            f"exceeds +/-{config.tempo_tolerance:.0%} at every metrical level tried"
        )
        breakdown.total = 0.0
    elif total < config.min_score:
        breakdown.rejected = f"score {total:.3f} below minimum {config.min_score:.2f}"

    breakdown.reasons = _explain(breakdown, section_a, section_b)
    return breakdown


def _explain(breakdown: ScoreBreakdown, section_a: Section, section_b: Section) -> list[str]:
    """Human-readable account of the decision, logged and shown in the UI."""
    tempo, key = breakdown.tempo, breakdown.key
    contributions = breakdown.contributions()

    reasons = [
        f"{section_a.label} -> {section_b.label}",
        f"tempo: {tempo.label}, stretch {tempo.stretch_percent:+.1f}% to "
        f"{tempo.target_bpm:.1f} BPM (score {tempo.score:.2f}, "
        f"contributes {contributions['tempo']:.3f})",
        f"key: {key.camelot_a} vs {key.camelot_b} -> {key.camelot_b_shifted}, "
        f"{key.label} (score {key.score:.2f}, contributes {contributions['key']:.3f})",
        f"chroma similarity {breakdown.chroma:.2f} at {key.semitones:+d} semitones "
        f"(contributes {contributions['chroma']:.3f})",
        f"{breakdown.energy.label} (score {breakdown.energy.score:.2f}, "
        f"contributes {contributions['energy']:.3f})",
    ]
    if breakdown.rejected:
        reasons.append(f"REJECTED: {breakdown.rejected}")
    return reasons


def rank_pairs(
    analysis_a: Analysis,
    analysis_b: Analysis,
    config: MatchingConfig,
    intent: str = "match",
    top_n: Optional[int] = None,
    min_section_seconds: float = 8.0,
) -> list[SectionPair]:
    """Score every section of A against every section of B, best first.

    Ties break on section length: given equal scores, a longer section gives the
    renderer more room to work a transition.
    """
    top_n = top_n or config.top_n
    pairs: list[SectionPair] = []

    for section_a in analysis_a.sections:
        if section_a.duration < min_section_seconds:
            continue
        for section_b in analysis_b.sections:
            if section_b.duration < min_section_seconds:
                continue
            breakdown = score_pair(analysis_a, section_a, analysis_b, section_b, config, intent)
            if not breakdown.viable:
                continue
            pairs.append(
                SectionPair(
                    track_a_id=analysis_a.track_id, track_b_id=analysis_b.track_id,
                    section_a=section_a, section_b=section_b, breakdown=breakdown,
                )
            )

    pairs.sort(key=lambda p: (p.score, min(p.section_a.duration, p.section_b.duration)), reverse=True)
    return pairs[:top_n]


def best_pair(
    analysis_a: Analysis,
    analysis_b: Analysis,
    config: MatchingConfig,
    intent: str = "match",
    min_section_seconds: float = 8.0,
) -> Optional[SectionPair]:
    ranked = rank_pairs(analysis_a, analysis_b, config, intent, top_n=1,
                        min_section_seconds=min_section_seconds)
    return ranked[0] if ranked else None
