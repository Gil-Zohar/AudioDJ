"""Analysis orchestration + caching.

One pass per track produces tempo/beats/downbeats, key with confidence, section
structure, an energy curve and per-section chroma. Results are cached in SQLite
under (file_hash, PIPELINE_VERSION), so bumping the version below invalidates
every stale row automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.analysis.beats import get_beat_tracker
from app.analysis.energy import energy_curve
from app.analysis.key import estimate_key, manual_key
from app.analysis.segments import segment_track
from app.analysis.stems import get_stem_provider
from app.config import TuningConfig, get_settings, get_tuning
from app.db import get_db
from app.models import Analysis, StemSet, TrackMeta

PIPELINE_VERSION = 1

ProgressFn = Callable[[str, float], None]


def _noop(stage: str, fraction: float) -> None:
    pass


def analyze_track(
    track: TrackMeta,
    tuning: Optional[TuningConfig] = None,
    progress: ProgressFn = _noop,
    use_cache: bool = True,
) -> Analysis:
    """Analyze one track, returning a cached result when possible."""
    import librosa

    tuning = tuning or get_tuning()
    cfg = tuning.analysis
    db = get_db()

    if use_cache:
        cached = db.get_analysis(track.file_hash, PIPELINE_VERSION)
        if cached is not None:
            return _apply_override(cached, db)

    progress("loading", 0.05)
    y, sr = librosa.load(track.path, sr=cfg.sample_rate, mono=True)
    duration = float(len(y) / sr)

    progress("beats", 0.2)
    tracker = get_beat_tracker(cfg.beat_tracker)
    beat_result = tracker.track(y, sr, cfg.hop_length, cfg.beats_per_bar)

    progress("energy", 0.45)
    times, values = energy_curve(y, sr, cfg.hop_length)

    progress("key", 0.6)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=cfg.hop_length)
    key_estimate = estimate_key(chroma.mean(axis=1), cfg.key_profile)

    progress("sections", 0.75)
    sections = segment_track(
        y, sr, cfg.hop_length, beat_result.beats, beat_result.downbeats,
        times, values, duration, cfg.min_section_seconds, cfg.max_sections,
        cfg.min_section_bars,
    )

    progress("saving", 0.95)
    # The curve is stored at ~10Hz: enough for labelling and hype placement,
    # small enough to keep in a JSON blob.
    step = max(1, int(len(times) / max(1, duration * 10)))
    analysis = Analysis(
        track_id=track.id,
        file_hash=track.file_hash,
        version=PIPELINE_VERSION,
        duration=duration,
        sample_rate=sr,
        bpm=round(beat_result.bpm, 3),
        beat_tracker=beat_result.tracker,
        beats=beat_result.beats,
        downbeats=beat_result.downbeats,
        key=key_estimate,
        key_detected=key_estimate,
        sections=sections,
        energy_times=[float(t) for t in times[::step]],
        energy_curve=[float(v) for v in values[::step]],
    )

    db.upsert_track(track)
    db.put_analysis(analysis)
    progress("done", 1.0)
    return _apply_override(analysis, db)


def _apply_override(analysis: Analysis, db) -> Analysis:
    """Swap in a manually pinned key if the user set one."""
    override = db.get_key_override(analysis.file_hash)
    if override is None:
        if analysis.key.source == "manual" and analysis.key_detected is not None:
            analysis.key = analysis.key_detected
        return analysis
    pitch_class, mode = override
    analysis.key = manual_key(pitch_class, mode)
    return analysis


def separate_stems(
    track: TrackMeta,
    tuning: Optional[TuningConfig] = None,
    progress: ProgressFn = _noop,
) -> StemSet:
    """Separate (or fetch cached) stems for a track."""
    tuning = tuning or get_tuning()
    settings = get_settings()
    db = get_db()
    provider = get_stem_provider(tuning.analysis.stem_provider, tuning.analysis.demucs_model)

    cached = db.get_stems(track.file_hash, provider.name)
    if cached is not None and all(Path(p).exists() for p in
                                  (cached.vocals, cached.drums, cached.bass, cached.other)):
        return cached

    progress(f"separating ({provider.name})", 0.1)
    out_dir = settings.stems_dir / provider.name / track.file_hash
    stems = provider.separate(Path(track.path), out_dir)
    db.put_stems(track.file_hash, stems)
    progress("done", 1.0)
    return stems


def set_key_override(file_hash: str, pitch_class: int, mode: str) -> None:
    get_db().put_key_override(file_hash, pitch_class, mode)


def clear_key_override(file_hash: str) -> None:
    get_db().clear_key_override(file_hash)


def beats_in_range(analysis: Analysis, start: float, end: float) -> list[float]:
    arr = np.asarray(analysis.beats)
    return [float(t) for t in arr[(arr >= start) & (arr < end)]]


def nearest_downbeat(analysis: Analysis, time: float, prefer: str = "nearest") -> float:
    """Snap a time to the closest downbeat (or the next/previous one)."""
    arr = np.asarray(analysis.downbeats if analysis.downbeats else analysis.beats)
    if arr.size == 0:
        return time
    if prefer == "next":
        after = arr[arr >= time]
        return float(after[0]) if after.size else float(arr[-1])
    if prefer == "prev":
        before = arr[arr <= time]
        return float(before[-1]) if before.size else float(arr[0])
    return float(arr[int(np.argmin(np.abs(arr - time)))])
