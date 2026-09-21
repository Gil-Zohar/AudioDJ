"""Set planning: what order to play tracks in, and where to join them.

A DJ set is not a playlist. It opens at moderate energy, climbs to a peak around
three-quarters of the way through, then eases off. This module orders a chosen
set of tracks to follow that arc while keeping consecutive tracks mixable, then
picks the specific section pair to transition on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import MatchingConfig
from app.matching.compat import tempo_compat
from app.matching.scorer import SectionPair, best_pair
from app.models import Analysis


@dataclass
class PlannedTransition:
    """One join in the set."""

    from_track: str
    to_track: str
    pair: SectionPair
    intent: str

    def to_dict(self) -> dict:
        return {
            "from": self.from_track, "to": self.to_track,
            "intent": self.intent, **self.pair.to_dict(),
        }


@dataclass
class SetPlan:
    order: list[Analysis] = field(default_factory=list)
    transitions: list[PlannedTransition] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "order": [a.track_id for a in self.order],
            "transitions": [t.to_dict() for t in self.transitions],
            "log": self.log,
        }


def track_energy(analysis: Analysis) -> float:
    """One number for how hard a track hits overall.

    Weighted toward its peak: what a crowd remembers is the biggest moment, not
    the average, so a track with one huge drop outranks a uniformly busy one.
    """
    if not analysis.sections:
        return float(np.mean(analysis.energy_curve)) if analysis.energy_curve else 0.5
    energies = np.array([s.energy for s in analysis.sections])
    return float(0.6 * energies.max() + 0.4 * energies.mean())


def energy_arc(count: int) -> list[float]:
    """Target energy per slot: open moderate, peak near the end, then ease off."""
    if count <= 1:
        return [0.6]
    peak_at = 0.75
    arc = []
    for i in range(count):
        position = i / (count - 1)
        if position <= peak_at:
            value = 0.45 + (0.95 - 0.45) * (position / peak_at)
        else:
            tail = (position - peak_at) / (1.0 - peak_at)
            value = 0.95 - (0.95 - 0.6) * tail
        arc.append(value)
    return arc


def order_for_energy_flow(
    analyses: list[Analysis], config: MatchingConfig
) -> tuple[list[Analysis], list[str]]:
    """Greedily order tracks to follow the energy arc, keeping neighbours mixable.

    Tempo compatibility is treated as a strong preference rather than a hard
    rule: a set that cannot be ordered at all is worse than one with a single
    awkward join, so an incompatible neighbour is penalised, not forbidden.
    """
    if len(analyses) <= 1:
        return list(analyses), []

    remaining = list(analyses)
    targets = energy_arc(len(analyses))
    ordered: list[Analysis] = []
    log: list[str] = []

    for slot, target in enumerate(targets):
        best, best_cost = None, float("inf")
        for candidate in remaining:
            cost = abs(track_energy(candidate) - target)
            if ordered:
                tempo = tempo_compat(
                    ordered[-1].bpm, candidate.bpm,
                    tolerance=config.tempo_tolerance,
                    allow_half_double=config.allow_half_double,
                    allow_three_two=config.allow_three_two,
                )
                # Penalty, not veto: keeps a set orderable when no candidate fits.
                cost += 0.0 if tempo.compatible else 0.6
                cost += (1.0 - tempo.score) * 0.25
            if cost < best_cost:
                best, best_cost = candidate, cost

        assert best is not None
        remaining.remove(best)
        ordered.append(best)
        log.append(
            f"slot {slot + 1}: {best.track_id[:8]} "
            f"energy {track_energy(best):.2f} (target {target:.2f}), "
            f"{best.bpm:.1f} BPM"
        )

    return ordered, log


def plan_set(
    analyses: list[Analysis], config: MatchingConfig, min_section_seconds: float = 8.0
) -> SetPlan:
    """Order the tracks, then choose where each join happens."""
    ordered, log = order_for_energy_flow(analyses, config)
    plan = SetPlan(order=ordered, log=list(log))

    for index in range(len(ordered) - 1):
        current, following = ordered[index], ordered[index + 1]
        # Climbing toward the peak we want a build; past it, a clean match.
        intent = "build" if index < len(ordered) * 0.75 else "match"

        pair = best_pair(current, following, config, intent, min_section_seconds)
        if pair is None:
            # Retry with the opposite intent before giving up on the join.
            fallback = "match" if intent == "build" else "build"
            pair = best_pair(current, following, config, fallback, min_section_seconds)
            if pair is not None:
                intent = fallback

        if pair is None:
            plan.log.append(
                f"no viable transition {current.track_id[:8]} -> {following.track_id[:8]}; "
                "will hard-cut"
            )
            continue

        plan.transitions.append(
            PlannedTransition(current.track_id, following.track_id, pair, intent)
        )
        plan.log.append(
            f"{current.track_id[:8]} -> {following.track_id[:8]} "
            f"({intent}): {pair.section_a.label}@{pair.section_a.start:.0f}s -> "
            f"{pair.section_b.label}@{pair.section_b.start:.0f}s score {pair.score:.3f}"
        )

    return plan


def pick_mashup_sections(
    analysis_vocal: Analysis,
    analysis_instrumental: Analysis,
    config: MatchingConfig,
    min_section_seconds: float = 8.0,
) -> Optional[SectionPair]:
    """Choose which part of A to sing over which part of B.

    Biased toward taking the vocal from a chorus or drop -- that is where the
    hook lives, and a mashup built on a verse rarely lands.
    """
    ranked = []
    for section_a in analysis_vocal.sections:
        if section_a.duration < min_section_seconds:
            continue
        for section_b in analysis_instrumental.sections:
            if section_b.duration < min_section_seconds:
                continue
            from app.matching.scorer import score_pair

            breakdown = score_pair(
                analysis_vocal, section_a, analysis_instrumental, section_b, config, "match"
            )
            if not breakdown.viable:
                continue

            bonus = 0.0
            if section_a.label in ("chorus", "drop"):
                bonus += 0.12
            if section_b.label in ("chorus", "drop", "verse"):
                bonus += 0.05

            ranked.append((breakdown.total + bonus, section_a, section_b, breakdown))

    if not ranked:
        return None

    ranked.sort(key=lambda r: r[0], reverse=True)
    _, section_a, section_b, breakdown = ranked[0]
    return SectionPair(
        track_a_id=analysis_vocal.track_id, track_b_id=analysis_instrumental.track_id,
        section_a=section_a, section_b=section_b, breakdown=breakdown,
    )
