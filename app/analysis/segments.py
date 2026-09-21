"""Structural segmentation: split a track into intro/verse/chorus/drop/outro.

Boundaries come from beat-synchronous harmony + timbre + loudness features run
through contiguity-constrained agglomerative clustering, then snapped to the
downbeat grid -- a section that does not start on a downbeat is useless for
mixing, so snapping is a correctness requirement, not a nicety.

Labels are assigned from relative energy, position, and how often a section's
harmony recurs: a repeated peak is a chorus, a lone peak is a drop.
"""
from __future__ import annotations

import numpy as np

from app.analysis.energy import energy_between
from app.models import Section


def _zscore(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (a - mu) / sd


def _beat_sync_features(y: np.ndarray, sr: int, hop_length: int, beat_frames: np.ndarray):
    """Stack chroma, MFCC and loudness, beat-synchronised and z-scored.

    Loudness is repeated so it carries real weight: verse/chorus/drop often share
    harmony and timbre and differ mainly in how hard they hit.
    """
    import librosa
    import scipy.ndimage as ndi

    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    rms_db = librosa.amplitude_to_db(
        librosa.feature.rms(y=y, hop_length=hop_length), ref=np.max
    )

    if beat_frames.size >= 2:
        sync = lambda a: librosa.util.sync(a, beat_frames, aggregate=np.median)
    else:
        sync = lambda a: a

    chroma_sync, mfcc_sync, rms_sync = sync(chroma), sync(mfcc), sync(rms_db)

    features = np.vstack([
        _zscore(chroma_sync),
        _zscore(mfcc_sync),
        np.repeat(_zscore(rms_sync), 6, axis=0),
    ])
    # Smooth over ~1 bar so single odd beats do not create boundaries.
    features = ndi.uniform_filter1d(features, size=4, axis=1)
    return features, chroma_sync


def _snap_to_grid(times: list[float], downbeats: np.ndarray, max_distance: float) -> list[float]:
    """Snap each boundary to the nearest downbeat within `max_distance`."""
    if downbeats.size == 0:
        return times
    snapped = []
    for t in times:
        idx = int(np.argmin(np.abs(downbeats - t)))
        candidate = float(downbeats[idx])
        snapped.append(candidate if abs(candidate - t) <= max_distance else float(t))
    return snapped


def segment_track(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    beats: list[float],
    downbeats: list[float],
    energy_times: np.ndarray,
    energy_values: np.ndarray,
    duration: float,
    min_section_seconds: float = 4.0,
    max_sections: int = 14,
    min_section_bars: int = 4,
) -> list[Section]:
    import librosa

    beat_array = np.asarray(beats, dtype=float)
    downbeat_array = np.asarray(downbeats, dtype=float)

    if beat_array.size < 8 or duration <= 0:
        return [_whole_track_section(duration, energy_times, energy_values)]

    beat_frames = librosa.time_to_frames(beat_array, sr=sr, hop_length=hop_length)
    features, chroma_sync = _beat_sync_features(y, sr, hop_length, beat_frames)

    # Aim for a section roughly every 15s, which matches how pop/dance music is
    # actually built (8- and 16-bar phrases).
    n_segments = int(np.clip(round(duration / 15.0), 3, max_sections))
    n_segments = min(n_segments, features.shape[1] - 1)
    if n_segments < 2:
        return [_whole_track_section(duration, energy_times, energy_values)]

    boundary_beats = librosa.segment.agglomerative(features, n_segments)
    boundary_times = [float(beat_array[min(b, len(beat_array) - 1)]) for b in boundary_beats]

    # One bar of slack is enough to correct detector jitter without inventing
    # boundaries that were never there.
    bar_seconds = float(np.median(np.diff(downbeat_array))) if downbeat_array.size > 2 else 2.0
    boundary_times = _snap_to_grid(boundary_times, downbeat_array, max_distance=bar_seconds)

    # Minimum length is musical, not arbitrary: a 4-bar intro is a real section
    # and a seconds-only floor would delete it on fast tracks.
    min_len = max(min_section_bars * bar_seconds, min_section_seconds)
    # Never let the floor swallow the whole track.
    min_len = min(min_len, duration / 3.0)

    edges = sorted({0.0, *boundary_times, float(duration)})
    edges = _enforce_min_length(edges, min_len, duration)

    sections: list[Section] = []
    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        sections.append(
            Section(
                index=i,
                start=start,
                end=end,
                label="verse",  # replaced by _label_sections
                energy=energy_between(energy_times, energy_values, start, end),
                chroma=[float(v) for v in _section_chroma(chroma_sync, beat_array, start, end)],
                start_beat=int(np.searchsorted(beat_array, start)),
                start_downbeat=int(np.searchsorted(downbeat_array, start))
                if downbeat_array.size else 0,
            )
        )

    return _label_sections(sections)


def _enforce_min_length(edges: list[float], min_len: float, duration: float) -> list[float]:
    """Drop boundaries that would create a section shorter than `min_len`."""
    if len(edges) <= 2:
        return edges
    kept = [edges[0]]
    for edge in edges[1:-1]:
        if edge - kept[-1] >= min_len and duration - edge >= min_len:
            kept.append(edge)
    kept.append(edges[-1])
    return kept


def _section_chroma(chroma_sync: np.ndarray, beats: np.ndarray, start: float, end: float) -> np.ndarray:
    """Mean 12-dim chroma over a section, L1-normalized."""
    lo = int(np.searchsorted(beats, start))
    hi = max(int(np.searchsorted(beats, end)), lo + 1)
    if chroma_sync.shape[1] <= lo:
        return np.ones(12) / 12
    window = chroma_sync[:, lo:min(hi, chroma_sync.shape[1])]
    if window.size == 0:
        return np.ones(12) / 12
    vec = window.mean(axis=1)
    total = float(vec.sum())
    return vec / total if total > 0 else np.ones(12) / 12


def _whole_track_section(duration, energy_times, energy_values) -> Section:
    return Section(
        index=0, start=0.0, end=float(duration), label="verse",
        energy=energy_between(energy_times, energy_values, 0.0, duration),
        chroma=[1 / 12] * 12,
    )


def _label_sections(sections: list[Section]) -> list[Section]:
    """Assign structural labels from energy, position and harmonic recurrence."""
    if not sections:
        return sections
    if len(sections) == 1:
        sections[0].label = "verse"
        return sections

    energies = np.array([s.energy for s in sections])
    median_energy = float(np.median(energies))
    high_threshold = float(np.percentile(energies, 70)) if len(energies) > 2 else median_energy
    peak_energy = float(energies.max())

    chromas = np.array([s.chroma for s in sections])
    recurrence = np.zeros(len(sections))
    for i in range(len(sections)):
        for j in range(len(sections)):
            if i == j:
                continue
            a, b = chromas[i], chromas[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 0 and float(np.dot(a, b)) / denom > 0.92:
                recurrence[i] += 1

    for i, section in enumerate(sections):
        is_first, is_last = i == 0, i == len(sections) - 1

        if is_first and section.energy <= median_energy:
            section.label = "intro"
        elif is_last and section.energy <= median_energy:
            section.label = "outro"
        elif section.energy >= high_threshold:
            if section.energy >= peak_energy * 0.97 and recurrence[i] == 0:
                section.label = "drop"
            else:
                section.label = "chorus"
        elif section.energy < median_energy * 0.75:
            section.label = "bridge"
        else:
            section.label = "verse"

    return sections
