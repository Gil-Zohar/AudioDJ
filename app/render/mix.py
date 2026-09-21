"""Render a full-length DJ set from an ordered list of tracks.

Where mashup.py layers two tracks, this plays a sequence of them and works the
joins: each track runs for a stretch, then hands over to the next through a
transition on the bar grid. Two blend modes:

  classic  the next track's full mix comes in under a transition treatment
  mashup   the next track's vocal lands over the current track's instrumental
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from app.analysis.pipeline import analyze_track, separate_stems
from app.config import TuningConfig, get_settings, get_tuning
from app.matching.compat import choose_set_tempo, stretch_to_target
from app.matching.planner import SetPlan, plan_set
from app.models import Analysis, TrackMeta
from app.render.encode import export_mix
from app.render.engine import (
    get_stretcher,
    latest_start_for_bars,
    load_audio,
    normalize_peak,
    slice_bars,
    to_stereo,
)
from app.render.timeline import Clip, Timeline
from app.render.transitions import apply_transition

ProgressFn = Callable[..., None]
BlendMode = Literal["classic", "mashup"]

INSTRUMENTAL_STEMS = ("drums", "bass", "other")


def _noop(stage: str, progress: float, **detail) -> None:
    pass


@dataclass
class MixResult:
    name: str
    files: dict
    tracklist: dict
    plan: Optional[SetPlan] = None
    log: list[str] = field(default_factory=list)


def _segment_bars(target_minutes: float, track_count: int, bpm: float,
                  transition_bars: int) -> int:
    """How many bars each track gets, rounded to a 4-bar phrase.

    Cutting mid-phrase is the most obvious way a generated mix betrays itself,
    so the length is always a multiple of 4 bars.
    """
    bar_seconds = 4 * 60.0 / max(bpm, 1e-6)
    total_bars = (target_minutes * 60.0) / bar_seconds
    # Transitions overlap, so each track past the first contributes less.
    per_track = total_bars / max(track_count, 1) + transition_bars * 0.5
    return max(8, int(round(per_track / 4)) * 4)


def _take(
    path: str,
    analysis: Analysis,
    start: float,
    bars: int,
    sample_rate: int,
    stretch_rate: float,
    semitones: float,
    stretcher,
) -> np.ndarray:
    """Bar-aligned slice of one source, put on the mix's clock."""
    start = min(start, latest_start_for_bars(analysis, bars))
    audio = to_stereo(load_audio(path, sample_rate, mono=False))
    sliced, _, _ = slice_bars(audio, analysis, start, bars, sample_rate)
    if sliced.size == 0:
        return sliced
    if abs(semitones) > 1e-4:
        sliced = stretcher.pitch_shift(sliced, semitones, sample_rate)
    if abs(stretch_rate - 1.0) > 1e-4:
        sliced = stretcher.stretch(sliced, stretch_rate, sample_rate)
    return to_stereo(sliced)


def _instrumental(stems, analysis, start, bars, sample_rate, rate, semitones, stretcher):
    beds = []
    for name in INSTRUMENTAL_STEMS:
        bed = _take(getattr(stems, name), analysis, start, bars,
                    sample_rate, rate, semitones, stretcher)
        if bed.size:
            beds.append(bed)
    if not beds:
        return np.zeros((2, 0), dtype=np.float32)
    shortest = min(b.shape[-1] for b in beds)
    return np.sum([b[..., :shortest] for b in beds], axis=0)


def build_mix(
    tracks: list[TrackMeta],
    target_minutes: float = 15.0,
    blend: BlendMode = "classic",
    tuning: Optional[TuningConfig] = None,
    progress: ProgressFn = _noop,
    out_name: Optional[str] = None,
    hype: Optional[str] = None,
    max_tracks: Optional[int] = None,
) -> MixResult:
    """Order the tracks for energy flow, then render the set.

    `max_tracks` caps the set after unmatchable tempos are dropped, so callers
    can over-supply candidates and still get the size they asked for.
    """
    tuning = tuning or get_tuning()
    settings = get_settings()
    render_config = tuning.render
    sample_rate = render_config.sample_rate
    log: list[str] = []

    if len(tracks) < 2:
        raise ValueError("a mix needs at least two tracks")

    # -- analyse ---------------------------------------------------------
    analyses: dict[str, Analysis] = {}
    for index, track in enumerate(tracks):
        progress(f"analysing {track.title}", 0.05 + 0.15 * (index / len(tracks)),
                 track=track.title)
        analyses[track.id] = analyze_track(track, tuning)

    by_id = {t.id: t for t in tracks}

    # Pick a tempo the most tracks can actually reach, then drop the ones that
    # still cannot. Stretching everything to the raw median produced 35%
    # stretches on a set spanning 86-182 BPM, which destroys the audio; a track
    # that cannot be beat-matched is better left out than mangled.
    target_bpm = choose_set_tempo([a.bpm for a in analyses.values()],
                                  tuning.matching.tempo_tolerance)
    max_stretch = render_config.max_stretch

    keepable, dropped = [], []
    for analysis in analyses.values():
        _, deviation = stretch_to_target(analysis.bpm, target_bpm)
        (keepable if deviation <= max_stretch else dropped).append((analysis, deviation))

    for analysis, deviation in dropped:
        log.append(
            f"dropped {by_id[analysis.track_id].title}: {analysis.bpm:.1f} BPM needs "
            f"{deviation * 100:.0f}% stretch to reach {target_bpm:.1f} BPM "
            f"(limit {max_stretch * 100:.0f}%)"
        )

    if len(keepable) < 2:
        raise ValueError(
            f"only {len(keepable)} of {len(analyses)} tracks can be beat-matched to a "
            f"common tempo near {target_bpm:.0f} BPM. Their tempos are too spread out "
            f"({', '.join(f'{a.bpm:.0f}' for a in analyses.values())}). Add more tracks "
            "at similar tempos, or raise render.max_stretch in config.yaml."
        )

    if max_tracks:
        # Closest to the target first: least stretching, best sounding.
        keepable.sort(key=lambda pair: pair[1])
        keepable = keepable[:max_tracks]

    kept_analyses = [a for a, _ in keepable]
    plan = plan_set(kept_analyses, tuning.matching)
    log.extend(plan.log)

    ordered = [by_id[a.track_id] for a in plan.order]
    log.append(
        f"set tempo {target_bpm:.1f} BPM across {len(ordered)} tracks "
        f"({len(dropped)} dropped as unmatchable), blend={blend}"
    )

    transition_bars = render_config.transition_bars
    bars_each = _segment_bars(target_minutes, len(ordered), target_bpm, transition_bars)
    bar_seconds = 4 * 60.0 / target_bpm
    log.append(f"{bars_each} bars per track (~{bars_each * bar_seconds:.0f}s each)")

    stretcher = get_stretcher(render_config.time_stretcher)
    timeline = Timeline(sample_rate=sample_rate, channels=render_config.channels)

    # -- separate stems --------------------------------------------------
    stems = {}
    need_stems = blend == "mashup"
    if need_stems:
        for index, track in enumerate(ordered):
            progress(f"separating {track.title}", 0.2 + 0.3 * (index / len(ordered)),
                     track=track.title)
            stems[track.id] = separate_stems(track, tuning)

    # -- lay out ----------------------------------------------------------
    cursor = 0.0
    transitions = {(t.from_track, t.to_track): t for t in plan.transitions}
    enabled = tuning.transitions.enabled or [tuning.transitions.default]

    for index, track in enumerate(ordered):
        analysis = analyses[track.id]
        progress(f"rendering {track.title}", 0.5 + 0.35 * (index / len(ordered)),
                 track=track.title, position=index + 1, total=len(ordered))

        planned = transitions.get(
            (ordered[index - 1].id, track.id)) if index else None
        start = planned.pair.section_b.start if planned else _default_start(analysis)

        rate, _ = stretch_to_target(analysis.bpm, target_bpm)
        semitones = float(planned.pair.breakdown.key.semitones) if planned else 0.0

        audio = _take(track.path, analysis, start, bars_each,
                      sample_rate, rate, semitones, stretcher)
        if audio.size == 0:
            log.append(f"skipped {track.title}: no audio at the chosen section")
            continue

        segment_seconds = audio.shape[-1] / sample_rate

        if index == 0:
            timeline.add(Clip(audio=audio, start=cursor, fade_in=0.05,
                              name=track.title, source_track=track.id))
        else:
            overlap = min(transition_bars * bar_seconds, segment_seconds * 0.5)
            cursor -= overlap
            name = enabled[index % len(enabled)]
            outgoing_tail = _tail_of(timeline, cursor, overlap, sample_rate)
            incoming_head = audio[..., : int(overlap * sample_rate)]

            result = apply_transition(name, outgoing_tail, incoming_head,
                                      sample_rate, target_bpm, tuning.transitions)

            # The processed head replaces the raw one for the overlap window.
            blended = audio.copy()
            head_len = min(result.incoming.shape[-1], blended.shape[-1])
            blended[..., :head_len] = result.incoming[..., :head_len]

            timeline.add(Clip(audio=blended, start=cursor,
                              name=track.title, source_track=track.id))
            timeline.add(Clip(audio=result.outgoing, start=cursor,
                              name=f"transition into {track.title}",
                              source_track=track.id, stem="transition"))
            timeline.mark(cursor, "transition", result.description, **result.detail)
            log.append(
                f"{ordered[index - 1].title} -> {track.title}: {result.description} "
                f"at {cursor:.1f}s"
            )

            if blend == "mashup" and index < len(ordered) and track.id in stems:
                previous = ordered[index - 1]
                if previous.id in stems:
                    previous_rate, _ = stretch_to_target(
                        analyses[previous.id].bpm, target_bpm)
                    bed = _instrumental(
                        stems[previous.id], analyses[previous.id],
                        start, min(bars_each, transition_bars * 2), sample_rate,
                        previous_rate, 0.0, stretcher,
                    )
                    if bed.size:
                        timeline.add(Clip(
                            audio=bed, start=cursor,
                            gain_db=render_config.mashup_instrumental_gain_db,
                            fade_out=bar_seconds,
                            name=f"{previous.title} (bed under {track.title})",
                            source_track=previous.id, stem="instrumental",
                        ))

        timeline.mark(max(cursor, 0.0), "track_in",
                      f"{track.artist} - {track.title}" if track.artist else track.title,
                      track=track.id, bpm=round(analysis.bpm, 2),
                      key=analysis.key.camelot,
                      stretch_percent=round((rate - 1) * 100, 2),
                      pitch_semitones=semitones)
        cursor += segment_seconds

    # -- hype layer (stage 4 plugs in here) --------------------------------
    hype_level = hype or tuning.hype.density
    if hype_level and hype_level != "off":
        try:
            from app.render.hype import apply_hype

            added = apply_hype(timeline, target_bpm, hype_level, tuning)
            log.append(f"hype layer: {added} elements at '{hype_level}'")
        except ImportError:
            log.append("hype layer not built yet (stage 4); skipping")

    progress("mixing down", 0.88)
    audio = normalize_peak(timeline.render(), render_config.headroom_db)

    progress("encoding", 0.93)
    name = out_name or f"mix_{int(time.time())}"
    tracklist = {
        "name": name,
        "created_at": time.time(),
        "mode": blend,
        "target_bpm": round(target_bpm, 2),
        "duration": round(audio.shape[-1] / sample_rate, 2),
        "tracks": [
            {
                "position": i + 1, "id": t.id, "title": t.title, "artist": t.artist,
                "bpm": round(analyses[t.id].bpm, 2),
                "key": analyses[t.id].key.camelot,
                "key_confidence": round(analyses[t.id].key.confidence, 3),
            }
            for i, t in enumerate(ordered)
        ],
        "events": timeline.tracklist(),
        "plan": plan.to_dict(),
        "log": log,
    }

    files = export_mix(audio, settings.mixes_dir, name, sample_rate, tracklist,
                       bitrate=render_config.bitrate)
    if files.get("warning"):
        log.append(files["warning"])

    progress("done", 1.0)
    return MixResult(name=name, files=files, tracklist=tracklist, plan=plan, log=log)


def _default_start(analysis: Analysis) -> float:
    """Start at the most interesting section when no transition dictates one."""
    for label in ("chorus", "drop"):
        for section in analysis.sections:
            if section.label == label:
                return section.start
    return analysis.sections[0].start if analysis.sections else 0.0


def _tail_of(timeline: Timeline, start: float, length: float, sample_rate: int) -> np.ndarray:
    """Render just the window of the timeline that the next track overlaps.

    Cheaper and simpler than rendering the whole mix to grab its last few bars,
    and it means transitions see exactly what will be playing underneath them.
    """
    samples = max(1, int(length * sample_rate))
    canvas = np.zeros((timeline.channels, samples), dtype=np.float32)
    window_start = int(start * sample_rate)

    for clip in timeline.clips:
        if clip.audio is None or clip.audio.size == 0:
            continue
        clip_start = int(clip.start * sample_rate)
        clip_end = clip_start + clip.audio.shape[-1]
        if clip_end <= window_start or clip_start >= window_start + samples:
            continue

        audio = clip.rendered(sample_rate)
        if audio.shape[0] == 1 and timeline.channels == 2:
            audio = np.repeat(audio, 2, axis=0)

        offset = window_start - clip_start
        take_from = max(0, offset)
        write_at = max(0, -offset)
        count = min(audio.shape[-1] - take_from, samples - write_at)
        if count > 0:
            canvas[:, write_at:write_at + count] += audio[:, take_from:take_from + count]

    return canvas
