"""Build a two-track mashup: the vocal of A over the instrumental bed of B.

The whole point is that both tracks end up on one clock. Everything here exists
to make that true: pick sections that fit, stretch both to a common tempo, shift
B's key if it clashes, and cut on the bar grid so downbeats coincide.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.analysis.pipeline import analyze_track, separate_stems
from app.config import TuningConfig, get_settings, get_tuning
from app.matching.planner import pick_mashup_sections
from app.matching.scorer import SectionPair
from app.models import Analysis, StemSet, TrackMeta
from app.render.encode import export_mix
from app.render.engine import (
    get_stretcher,
    latest_start_for_bars,
    load_audio,
    slice_bars,
    to_stereo,
)
from app.render.loudness import master_chain
from app.render.timeline import Clip, Timeline
from app.render.transitions import apply_transition

ProgressFn = Callable[[str, float], None]

INSTRUMENTAL_STEMS = ("drums", "bass", "other")


def _noop(stage: str, fraction: float) -> None:
    pass


@dataclass
class MashupResult:
    name: str
    files: dict
    pair: Optional[SectionPair]
    tracklist: dict
    log: list[str] = field(default_factory=list)


def _prepare_stem(
    path: str,
    analysis: Analysis,
    section_start: float,
    bars: int,
    sample_rate: int,
    stretch_rate: float,
    semitones: float,
    stretcher,
) -> np.ndarray:
    """Cut a bar-aligned chunk of one stem and put it on the target clock.

    Order matters: slice on the original grid first, then stretch. Stretching the
    whole file and slicing afterwards would need the bar grid recomputed, and any
    error there lands straight on the downbeat.
    """
    # Pull the start earlier if the section sits too close to the end of the
    # track to supply the bars asked for.
    start = min(section_start, latest_start_for_bars(analysis, bars))

    audio = load_audio(path, sample_rate, mono=False)
    audio = to_stereo(audio)
    sliced, _, _ = slice_bars(audio, analysis, start, bars, sample_rate)
    if sliced.size == 0:
        return sliced

    if abs(semitones) > 1e-4:
        sliced = stretcher.pitch_shift(sliced, semitones, sample_rate)
    if abs(stretch_rate - 1.0) > 1e-4:
        sliced = stretcher.stretch(sliced, stretch_rate, sample_rate)
    return to_stereo(sliced)


def build_mashup(
    track_a: TrackMeta,
    track_b: TrackMeta,
    bars: int = 32,
    tuning: Optional[TuningConfig] = None,
    progress: ProgressFn = _noop,
    out_name: Optional[str] = None,
    transition: Optional[str] = None,
) -> MashupResult:
    """Vocal of `track_a` over the instrumental of `track_b`.

    `bars` sets the length: 32 bars is about 60 seconds at 128 BPM.
    """
    tuning = tuning or get_tuning()
    settings = get_settings()
    render_config = tuning.render
    sample_rate = render_config.sample_rate
    log: list[str] = []

    progress("analyzing tracks", 0.05)
    analysis_a = analyze_track(track_a, tuning)
    analysis_b = analyze_track(track_b, tuning)
    log.append(f"A: {track_a.title} - {analysis_a.bpm:.1f} BPM, {analysis_a.key.camelot}")
    log.append(f"B: {track_b.title} - {analysis_b.bpm:.1f} BPM, {analysis_b.key.camelot}")

    progress("matching sections", 0.2)
    pair = pick_mashup_sections(analysis_a, analysis_b, tuning.matching)
    if pair is None:
        raise ValueError(
            f"no compatible section pair: {analysis_a.bpm:.1f} BPM vs {analysis_b.bpm:.1f} BPM "
            f"exceeds the +/-{tuning.matching.tempo_tolerance:.0%} tempo tolerance. "
            "Raise matching.tempo_tolerance or enable allow_half_double in config.yaml."
        )
    log.extend(pair.breakdown.reasons)

    tempo = pair.breakdown.tempo
    key = pair.breakdown.key
    target_bpm = tempo.target_bpm

    # A defines the clock; B is stretched and shifted onto it.
    #
    # Use the ratio the tempo matcher already solved for. Recomputing it here is
    # how it previously ended up inverted: B was slowed to 120 BPM instead of
    # sped up to 128, so the two layers were never actually aligned.
    rate_a = 1.0
    rate_b = float(tempo.stretch)
    log.append(
        f"target {target_bpm:.2f} BPM: A unchanged, B stretched {(rate_b - 1) * 100:+.2f}% "
        f"({tempo.label}), pitch {key.semitones:+d} semitones"
    )

    progress("separating stems", 0.3)
    stems_a = separate_stems(track_a, tuning)
    stems_b = separate_stems(track_b, tuning)

    stretcher = get_stretcher(getattr(render_config, "time_stretcher", "auto"))
    log.append(f"time stretcher: {stretcher.name}, stem provider: {stems_a.provider}")

    start_a = min(pair.section_a.start, latest_start_for_bars(analysis_a, bars))
    start_b = min(pair.section_b.start, latest_start_for_bars(analysis_b, bars))
    if start_a < pair.section_a.start - 0.01 or start_b < pair.section_b.start - 0.01:
        log.append(
            f"pulled start earlier to fit {bars} bars: "
            f"A {pair.section_a.start:.1f}s -> {start_a:.1f}s, "
            f"B {pair.section_b.start:.1f}s -> {start_b:.1f}s"
        )

    progress("rendering vocal", 0.45)
    vocal = _prepare_stem(
        stems_a.vocals, analysis_a, pair.section_a.start, bars,
        sample_rate, rate_a, 0.0, stretcher,
    )
    if vocal.size == 0:
        raise ValueError(f"vocal stem for {track_a.title} produced no audio at the chosen section")

    progress("rendering instrumental", 0.65)
    beds = []
    for stem_name in INSTRUMENTAL_STEMS:
        bed = _prepare_stem(
            getattr(stems_b, stem_name), analysis_b, pair.section_b.start, bars,
            sample_rate, rate_b, float(key.semitones), stretcher,
        )
        if bed.size:
            beds.append(bed)
    if not beds:
        raise ValueError(f"instrumental stems for {track_b.title} produced no audio")

    length = min(min(b.shape[-1] for b in beds), vocal.shape[-1])
    instrumental = np.sum([b[..., :length] for b in beds], axis=0)
    vocal = vocal[..., :length]
    log.append(f"layered {length / sample_rate:.1f}s of audio ({bars} bars at {target_bpm:.1f} BPM)")

    progress("mixing", 0.8)
    timeline = Timeline(sample_rate=sample_rate, channels=render_config.channels)

    bar_seconds = 4 * 60.0 / target_bpm
    intro_bars = min(2, max(1, bars // 8))
    intro = intro_bars * bar_seconds

    # The bed starts alone so the groove establishes before the vocal enters -
    # dropping both together makes the join obvious.
    timeline.add(Clip(
        audio=instrumental, start=0.0,
        gain_db=render_config.mashup_instrumental_gain_db,
        fade_in=0.05, fade_out=bar_seconds,
        name=f"{track_b.title} (instrumental)", source_track=track_b.id, stem="instrumental",
    ))
    timeline.add(Clip(
        audio=vocal[..., : max(0, length - int(intro * sample_rate))],
        start=intro,
        gain_db=render_config.mashup_vocal_gain_db,
        fade_in=0.08, fade_out=bar_seconds * 0.5,
        name=f"{track_a.title} (vocal)", source_track=track_a.id, stem="vocals",
    ))

    timeline.mark(0.0, "track_in", f"{track_b.artist} - {track_b.title} (instrumental bed)",
                  track=track_b.id, bpm=round(analysis_b.bpm, 2),
                  key=analysis_b.key.camelot, stretch_percent=round((rate_b - 1) * 100, 2))
    timeline.mark(intro, "track_in", f"{track_a.artist} - {track_a.title} (vocal)",
                  track=track_a.id, bpm=round(analysis_a.bpm, 2), key=analysis_a.key.camelot)

    # A transition treatment on the final bars, so the mashup ends rather than stops.
    transition_name = transition or tuning.transitions.default
    tail_bars = min(4, max(1, bars // 8))
    tail_samples = int(tail_bars * bar_seconds * sample_rate)
    if tail_samples > 0 and length > tail_samples:
        outgoing = instrumental[..., length - tail_samples:]
        silence = np.zeros_like(outgoing)
        result = apply_transition(
            transition_name, outgoing, silence, sample_rate, target_bpm, tuning.transitions
        )
        timeline.add(Clip(
            audio=result.outgoing,
            start=(length - tail_samples) / sample_rate,
            gain_db=render_config.mashup_instrumental_gain_db,
            name=f"outro: {result.description}", source_track=track_b.id, stem="transition",
        ))
        timeline.mark((length - tail_samples) / sample_rate, "transition",
                      result.description, **result.detail)
        log.append(f"outro treatment: {result.description}")

    audio, master = master_chain(
        timeline.render(), sample_rate,
        target_lufs=render_config.target_lufs,
        ceiling_db=render_config.true_peak_ceiling_db,
        release_ms=render_config.limiter_release_ms,
    )
    log.append(
        f"master: {master['input_lufs']} -> {master['output_lufs']} LUFS, "
        f"peak {master['peak_db']} dBFS"
    )

    progress("encoding", 0.9)
    name = out_name or f"mashup_{int(time.time())}"
    tracklist = {
        "name": name,
        "created_at": time.time(),
        "mode": "mashup",
        "target_bpm": round(target_bpm, 2),
        "duration": round(audio.shape[-1] / sample_rate, 2),
        "tracks": [
            {"role": "vocal", "id": track_a.id, "title": track_a.title, "artist": track_a.artist,
             "bpm": round(analysis_a.bpm, 2), "key": analysis_a.key.camelot,
             "section": f"{pair.section_a.label} @ {pair.section_a.start:.1f}s"},
            {"role": "instrumental", "id": track_b.id, "title": track_b.title,
             "artist": track_b.artist, "bpm": round(analysis_b.bpm, 2),
             "key": analysis_b.key.camelot,
             "section": f"{pair.section_b.label} @ {pair.section_b.start:.1f}s"},
        ],
        "match": pair.to_dict(),
        "events": timeline.tracklist(),
        "log": log,
    }

    files = export_mix(
        audio, settings.mixes_dir, name, sample_rate, tracklist,
        bitrate=render_config.bitrate,
    )
    if files.get("warning"):
        log.append(files["warning"])

    progress("done", 1.0)
    return MashupResult(name=name, files=files, pair=pair, tracklist=tracklist, log=log)
