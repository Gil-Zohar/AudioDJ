"""Key detection (Krumhansl-Schmuckler) and Camelot-wheel mapping.

Key detection on Mizrahi / maqam material is genuinely unreliable -- those scales
do not sit in 12-TET major/minor -- so every estimate carries a confidence and the
matching engine shrinks the key term when confidence is low. The UI can pin a key
manually, which replaces the estimate with source="manual" and confidence 1.0.
"""
from __future__ import annotations

import numpy as np

from app.models import PITCH_NAMES, KeyEstimate

# Krumhansl-Kessler probe-tone profiles.
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Temperley's revision - generally better on popular music.
TEMPERLEY_MAJOR = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
)
TEMPERLEY_MINOR = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
)

PROFILES = {
    "krumhansl": (KRUMHANSL_MAJOR, KRUMHANSL_MINOR),
    "temperley": (TEMPERLEY_MAJOR, TEMPERLEY_MINOR),
}


def camelot(pitch_class: int, mode: str) -> str:
    """Map a key to its Camelot code (C major -> 8B, A minor -> 8A).

    The wheel is the circle of fifths, so the code number advances by one per
    seven semitones; minor keys borrow their relative major's number.
    """
    pc = pitch_class % 12
    if mode == "minor":
        pc = (pc + 3) % 12          # relative major
    number = ((pc * 7) % 12 + 7) % 12 + 1
    return f"{number}{'A' if mode == 'minor' else 'B'}"


def parse_camelot(code: str) -> tuple[int, str]:
    """Inverse of `camelot`: '8A' -> (9, 'minor')."""
    code = code.strip().upper()
    number, letter = int(code[:-1]), code[-1]
    mode = "minor" if letter == "A" else "major"
    # invert number = ((pc*7)%12 + 7)%12 + 1
    target = (number - 1 - 7) % 12
    for pc in range(12):
        if (pc * 7) % 12 == target:
            major_pc = pc
            break
    pitch_class = (major_pc - 3) % 12 if mode == "minor" else major_pc
    return pitch_class, mode


def _correlate(chroma_mean: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Pearson correlation of the chroma vector against all 12 rotations."""
    scores = np.zeros(12)
    c = chroma_mean - chroma_mean.mean()
    c_norm = np.linalg.norm(c)
    for pc in range(12):
        p = np.roll(profile, pc)
        p = p - p.mean()
        denom = c_norm * np.linalg.norm(p)
        scores[pc] = float(np.dot(c, p) / denom) if denom > 0 else 0.0
    return scores


def estimate_key(chroma_mean: np.ndarray, profile_name: str = "temperley") -> KeyEstimate:
    """Best key plus a confidence derived from its margin over the runner-up.

    The runner-up deliberately excludes the relative major/minor of the winner:
    those two always score alike and counting them would make every confident
    estimate look uncertain.
    """
    major_profile, minor_profile = PROFILES.get(profile_name, PROFILES["temperley"])
    chroma_mean = np.asarray(chroma_mean, dtype=float).reshape(12)

    major_scores = _correlate(chroma_mean, major_profile)
    minor_scores = _correlate(chroma_mean, minor_profile)

    all_scores = np.concatenate([major_scores, minor_scores])
    best_idx = int(np.argmax(all_scores))
    best_pc = best_idx % 12
    best_mode = "major" if best_idx < 12 else "minor"

    relative_idx = (best_pc + 9) % 12 + 12 if best_mode == "major" else (best_pc + 3) % 12
    masked = all_scores.copy()
    masked[best_idx] = -np.inf
    masked[relative_idx] = -np.inf
    runner_up = float(np.max(masked))

    best_score = float(all_scores[best_idx])
    spread = float(np.max(all_scores) - np.min(all_scores))
    confidence = (best_score - runner_up) / spread if spread > 0 else 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return KeyEstimate(
        pitch_class=best_pc,
        mode=best_mode,
        confidence=confidence,
        camelot=camelot(best_pc, best_mode),
        source="detected",
    )


def manual_key(pitch_class: int, mode: str) -> KeyEstimate:
    return KeyEstimate(
        pitch_class=pitch_class % 12,
        mode=mode,
        confidence=1.0,
        camelot=camelot(pitch_class, mode),
        source="manual",
    )


def key_name(pitch_class: int, mode: str) -> str:
    return f"{PITCH_NAMES[pitch_class % 12]} {mode}"
