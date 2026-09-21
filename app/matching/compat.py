"""Pure compatibility primitives for the matching engine.

Deliberately free of audio I/O, librosa and any model: everything here takes
plain floats, strings and 12-dim chroma vectors. That keeps the rules deciding
what mixes with what fast to test and easy to reason about in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.key import camelot

# Ratios worth trying when two tracks sit at different metrical levels. A 174 BPM
# drum & bass track and an 87 BPM hip-hop track share a pulse; refusing to look
# at half and double time would throw away most cross-genre mixes.
HALF_DOUBLE_RATIOS = (2.0, 0.5)
THREE_TWO_RATIOS = (1.5, 2.0 / 3.0)

# Half/double time is musically a bigger leap than a straight tempo nudge, so a
# match found there starts slightly behind an exact-level one.
RATIO_PENALTY = {1.0: 1.0, 2.0: 0.92, 0.5: 0.92, 1.5: 0.82, 2.0 / 3.0: 0.82}

RATIO_LABELS = {
    1.0: "same tempo",
    2.0: "double-time",
    0.5: "half-time",
    1.5: "3:2 polyrhythm",
    2.0 / 3.0: "2:3 polyrhythm",
}


@dataclass(frozen=True)
class TempoMatch:
    """How, and how well, two tempos can be reconciled."""

    score: float            # 0..1
    ratio: float            # multiply B's BPM by this to reach A's metrical level
    stretch: float          # factor B's audio must be stretched by
    stretch_percent: float  # signed, for display
    target_bpm: float
    label: str

    @property
    def compatible(self) -> bool:
        return self.score > 0.0


def tempo_compat(
    bpm_a: float,
    bpm_b: float,
    tolerance: float = 0.08,
    allow_half_double: bool = True,
    allow_three_two: bool = False,
) -> TempoMatch:
    """Best way to reconcile two tempos, searching metrical levels.

    `tolerance` is the fraction of stretch considered acceptable (0.08 = +/-8%).
    Score falls linearly from 1.0 at a perfect match to 0.0 at the tolerance
    edge, because stretching degrades audio roughly in proportion to how far it
    is pushed.
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return TempoMatch(0.0, 1.0, 1.0, 0.0, max(bpm_a, 0.0), "invalid tempo")

    ratios = [1.0]
    if allow_half_double:
        ratios.extend(HALF_DOUBLE_RATIOS)
    if allow_three_two:
        ratios.extend(THREE_TWO_RATIOS)

    best = TempoMatch(0.0, 1.0, 1.0, 0.0, bpm_a, "incompatible")
    for ratio in ratios:
        effective = bpm_b * ratio
        deviation = abs(effective - bpm_a) / bpm_a
        if deviation > tolerance:
            continue

        raw = 1.0 - deviation / tolerance if tolerance > 0 else 1.0
        score = raw * RATIO_PENALTY.get(ratio, 0.8)
        if score <= best.score:
            continue

        stretch = bpm_a / effective          # >1 speeds B up
        best = TempoMatch(
            score=float(np.clip(score, 0.0, 1.0)),
            ratio=ratio,
            stretch=stretch,
            stretch_percent=(stretch - 1.0) * 100.0,
            target_bpm=bpm_a,
            label=RATIO_LABELS.get(ratio, f"{ratio:.2f}x"),
        )
    return best


# A set has to share one clock. These fold tempos onto a common metrical level
# and pick a target the most tracks can actually reach.
CANONICAL_LOW = 82.0
CANONICAL_HIGH = 164.0


def fold_tempo(bpm: float, low: float = CANONICAL_LOW,
               high: float = CANONICAL_HIGH) -> tuple[float, float]:
    """Halve or double a tempo into one canonical octave.

    174 BPM drum & bass and 87 BPM hip-hop share a pulse; comparing the raw
    numbers says they are 2:1 apart and unmixable. Returns (folded, ratio),
    where ratio is what the original was multiplied by.
    """
    if bpm <= 0:
        return 0.0, 1.0
    folded, ratio = float(bpm), 1.0
    for _ in range(4):
        if folded < low:
            folded, ratio = folded * 2.0, ratio * 2.0
        elif folded > high:
            folded, ratio = folded / 2.0, ratio / 2.0
        else:
            break
    return folded, ratio


def choose_set_tempo(bpms: list[float], tolerance: float = 0.08) -> float:
    """Pick the tempo the largest number of tracks can reach within tolerance.

    The median of raw BPMs is the obvious choice and a bad one: a set spanning
    86 to 182 BPM gets a median nothing can reach, and every track ends up
    stretched 30% into mush. Folding first, then taking the densest cluster,
    finds a tempo that actually works for most of the set.
    """
    folded = [fold_tempo(b)[0] for b in bpms if b > 0]
    if not folded:
        return 0.0

    best_tempo, best_count = folded[0], -1
    for candidate in folded:
        count = sum(1 for f in folded if abs(f - candidate) / candidate <= tolerance)
        if count > best_count:
            best_tempo, best_count = candidate, count

    # Centre on the cluster rather than on whichever member was tested first.
    cluster = [f for f in folded if abs(f - best_tempo) / best_tempo <= tolerance]
    return float(np.median(cluster)) if cluster else float(best_tempo)


def stretch_to_target(bpm: float, target_bpm: float) -> tuple[float, float]:
    """(stretch rate, deviation) to play `bpm` at `target_bpm`.

    Searches metrical levels and keeps whichever needs least stretching: a
    182 BPM track joining a 95 BPM set is a 4% nudge at half time, not an
    impossible 48% one. Folding blindly into a fixed window is not enough --
    it can pick a worse ratio than leaving the tempo alone.

    Deviation is symmetric, so halving and doubling the speed are judged as
    equally severe. Using |rate - 1| would rate a 2x speed-up as twice as bad
    as a 2x slow-down, which is not how it sounds.
    """
    if bpm <= 0 or target_bpm <= 0:
        return 1.0, 1.0

    best_rate, best_deviation = 1.0, float("inf")
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        effective = bpm * ratio
        if effective <= 0:
            continue
        rate = target_bpm / effective
        deviation = max(rate, 1.0 / rate) - 1.0
        if deviation < best_deviation:
            best_rate, best_deviation = rate, deviation

    return best_rate, best_deviation


def camelot_distance(code_a: str, code_b: str) -> int:
    """Steps between two Camelot codes under standard harmonic-mixing rules.

    0 = same key, 1 = a classic compatible move (one hour around the wheel, or
    the relative major/minor), rising from there. DJs treat 0 and 1 as safe.
    """
    number_a, letter_a = int(code_a[:-1]), code_a[-1].upper()
    number_b, letter_b = int(code_b[:-1]), code_b[-1].upper()

    hours = min((number_a - number_b) % 12, (number_b - number_a) % 12)

    if letter_a == letter_b:
        return hours
    if number_a == number_b:
        return 1                    # relative major/minor: always compatible
    return hours + 2                # both a mode flip and a move around: harsher


def camelot_score(distance: int) -> float:
    """Map wheel distance onto 0..1."""
    table = {0: 1.0, 1: 0.85, 2: 0.55, 3: 0.3}
    if distance in table:
        return table[distance]
    return max(0.0, 0.2 - 0.05 * (distance - 4))


@dataclass(frozen=True)
class KeyMatch:
    score: float
    semitones: int              # pitch shift to apply to B
    distance: int               # resulting Camelot distance
    camelot_a: str
    camelot_b: str
    camelot_b_shifted: str
    confidence_weight: float
    label: str


def key_compat(
    pitch_class_a: int,
    mode_a: str,
    pitch_class_b: int,
    mode_b: str,
    max_shift: int = 2,
    confidence_a: float = 1.0,
    confidence_b: float = 1.0,
    confidence_floor: float = 0.35,
) -> KeyMatch:
    """Best key relationship, allowing B to be pitch-shifted up to `max_shift`.

    When either key estimate is weak the score is pulled toward neutral rather
    than trusted. Mizrahi and maqam material does not sit in 12-TET major/minor,
    so a confident-looking wrong key would otherwise veto good musical matches
    or wave through clashing ones.
    """
    code_a = camelot(pitch_class_a, mode_a)
    code_b = camelot(pitch_class_b, mode_b)

    best_shift, best_distance, best_raw = 0, camelot_distance(code_a, code_b), -1.0
    for shift in range(-max_shift, max_shift + 1):
        shifted_pc = (pitch_class_b + shift) % 12
        distance = camelot_distance(code_a, camelot(shifted_pc, mode_b))
        # Every semitone of shift costs a little: it audibly colours vocals.
        raw = camelot_score(distance) - 0.04 * abs(shift)
        if raw > best_raw:
            best_shift, best_distance, best_raw = shift, distance, raw

    confidence = min(confidence_a, confidence_b)
    # At or above the floor we trust the estimate fully; below it we fade the
    # term toward neutral (0.5) in proportion to how weak it is.
    if confidence_floor > 0:
        weight = float(np.clip(confidence / confidence_floor, 0.0, 1.0))
    else:
        weight = 1.0
    score = weight * float(np.clip(best_raw, 0.0, 1.0)) + (1.0 - weight) * 0.5

    return KeyMatch(
        score=float(np.clip(score, 0.0, 1.0)),
        semitones=best_shift,
        distance=best_distance,
        camelot_a=code_a,
        camelot_b=code_b,
        camelot_b_shifted=camelot((pitch_class_b + best_shift) % 12, mode_b),
        confidence_weight=weight,
        label=_key_label(best_distance, best_shift, weight),
    )


def _key_label(distance: int, shift: int, weight: float) -> str:
    base = {0: "same key", 1: "harmonically compatible"}.get(
        distance, f"{distance} steps apart on the wheel"
    )
    if shift:
        base += f", shift {shift:+d} semitone{'s' if abs(shift) != 1 else ''}"
    if weight < 1.0:
        base += " (low key confidence, de-weighted)"
    return base


def chroma_similarity(chroma_a, chroma_b, semitones: int = 0) -> float:
    """Cosine similarity of two 12-dim chroma vectors, B rotated by `semitones`.

    The rotation must match the pitch shift chosen by `key_compat`, otherwise
    this measures the similarity of audio that will never actually be played.
    This is the "this part sounds like that part" term.
    """
    a = np.asarray(chroma_a, dtype=float).reshape(-1)
    b = np.asarray(chroma_b, dtype=float).reshape(-1)
    if a.size != 12 or b.size != 12:
        return 0.0

    b = np.roll(b, semitones)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


@dataclass(frozen=True)
class EnergyMatch:
    score: float
    delta: float
    intent: str
    label: str


def energy_compat(
    energy_a: float,
    energy_b: float,
    intent: str = "match",
    match_tolerance: float = 0.15,
    build_target: float = 0.25,
) -> EnergyMatch:
    """Score an energy relationship against what the planner asked for.

    'match' keeps the floor steady across a transition; 'build' deliberately
    steps up into the next track. A build is not a failed match -- it is the
    thing you want before a drop -- so it is scored against its own target.
    """
    delta = energy_b - energy_a

    if intent == "build":
        error = abs(delta - build_target)
        score = 1.0 - error / max(build_target, 1e-6)
        label = f"energy build {delta:+.2f}"
    elif intent == "drop":
        error = abs(delta + build_target)
        score = 1.0 - error / max(build_target, 1e-6)
        label = f"energy drop {delta:+.2f}"
    else:
        score = 1.0 - abs(delta) / max(match_tolerance, 1e-6)
        label = f"energy match {delta:+.2f}"

    return EnergyMatch(
        score=float(np.clip(score, 0.0, 1.0)),
        delta=float(delta),
        intent=intent,
        label=label,
    )
