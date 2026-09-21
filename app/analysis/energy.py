"""Energy curve: how hard the track is hitting, moment to moment.

Drives section labelling, set ordering, transition choice and hype placement,
so it is computed once and cached with the rest of the analysis.
"""
from __future__ import annotations

import numpy as np


def energy_curve(
    y: np.ndarray, sr: int, hop_length: int, smooth_seconds: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, values 0..1).

    RMS alone rates a loud sustained pad as highly as a busy drop, so it is
    combined with spectral flux (how much is *changing*) and percussive energy.
    """
    import librosa

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    n = min(len(rms), len(onset_env))
    rms, onset_env = rms[:n], onset_env[:n]

    def norm(a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return a
        lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
        if hi - lo < 1e-9:
            return np.zeros_like(a)
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    combined = 0.6 * norm(rms) + 0.4 * norm(onset_env)

    frames_per_second = sr / hop_length
    window = max(1, int(smooth_seconds * frames_per_second))
    if window > 1 and combined.size > window:
        kernel = np.ones(window) / window
        combined = np.convolve(combined, kernel, mode="same")

    combined = np.clip(combined, 0.0, 1.0)
    times = librosa.frames_to_time(np.arange(len(combined)), sr=sr, hop_length=hop_length)
    return times, combined


def energy_between(times: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    """Mean energy over [start, end)."""
    if len(times) == 0:
        return 0.0
    mask = (times >= start) & (times < end)
    if not mask.any():
        idx = int(np.clip(np.searchsorted(times, start), 0, len(values) - 1))
        return float(values[idx])
    return float(values[mask].mean())
