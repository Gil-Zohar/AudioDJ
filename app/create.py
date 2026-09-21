"""The Create flow: trends in, finished mix out.

Fetch charts, merge them, match against the library, sample a set weighted by
rank, render it, and store it in history. Runs inside a job so the browser can
watch progress.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from app.analysis.pipeline import analyze_track, separate_stems
from app.config import get_settings, get_tuning
from app.db import get_db
from app.render.mix import build_mix
from app.sources.local import LocalLibrarySource
from app.trends.merge import weighted_sample
from app.trends.resolver import fetch_and_resolve

log = logging.getLogger("autodj.create")

LENGTH_MINUTES = {"15": 15.0, "30": 30.0, "60": 60.0}


class CreateOptions(BaseModel):
    """Everything the Create button can be configured with."""

    length_minutes: float = Field(default=15.0, ge=1.0, le=180.0)
    israel_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    hype: Literal["off", "light", "heavy"] = "light"
    blend: Literal["classic", "mashup"] = "classic"
    track_count: Optional[int] = Field(default=None, ge=2, le=40)
    seed: Optional[int] = None
    name: Optional[str] = None


def track_count_for(minutes: float, requested: Optional[int]) -> int:
    """Pick a set size.

    The brief asks for 6-10 tracks, which suits a 15-minute mix. Longer mixes
    scale up rather than stretching six tracks thin.
    """
    if requested:
        return requested
    if minutes <= 20:
        return 8
    if minutes <= 40:
        return 12
    return 16


def create_mix(
    options: CreateOptions,
    progress: Callable[..., None],
    source: Optional[LocalLibrarySource] = None,
) -> dict:
    """Run the whole Create flow, returning a payload for the UI."""
    tuning = get_tuning()
    settings = get_settings()
    source = source or LocalLibrarySource(settings.music_dir)

    progress("scanning your library", 0.02)
    library = source.scan(force=True)
    if not library:
        raise ValueError(
            f"no audio found in MUSIC_DIR ({settings.music_dir}). "
            "Point it at your music library in .env."
        )

    progress("fetching trends", 0.05)
    merged, resolution = fetch_and_resolve(
        source,
        israel_weight=options.israel_ratio,
        global_weight=1.0 - options.israel_ratio,
        progress=progress,
    )

    for note in resolution.warnings():
        log.warning("create: %s", note)

    if not merged:
        raise ValueError(
            "no trend providers returned anything. Check LASTFM_API_KEY in .env, "
            "or your internet connection. Apple Music needs no key and should "
            "normally work."
        )

    wanted = track_count_for(options.length_minutes, options.track_count)
    available = [r for r in resolution.resolved]

    progress(
        f"matched {len(available)} of {len(merged)} trending tracks", 0.65,
        resolved=len(available), missing=len(resolution.missing),
    )

    if len(available) < 2:
        raise ValueError(
            f"only {len(available)} trending track(s) matched your library, and a mix "
            f"needs at least 2. {len(resolution.missing)} trending tracks are missing "
            "locally - see the missing list. Add some of them to MUSIC_DIR, or check "
            "your files are tagged with the right artist and title."
        )

    # Sample rather than take the top N, so pressing Create twice differs.
    seed = options.seed if options.seed is not None else int(time.time())
    chosen_trends = weighted_sample(
        [r.trend for r in available], min(wanted, len(available)),
        israel_ratio=options.israel_ratio, seed=seed,
    )
    by_label = {r.trend.label(): r for r in available}
    chosen = [by_label[t.label()] for t in chosen_trends if t.label() in by_label]

    log.info("create: %d tracks chosen from %d available", len(chosen), len(available))
    progress(f"building a {options.length_minutes:.0f} minute set from "
             f"{len(chosen)} tracks", 0.7)

    def render_progress(stage: str, fraction: float, **detail) -> None:
        """Fold the renderer's own 0..1 into the slice of the bar it owns.

        build_mix reports its own progress from zero, so passing `progress`
        straight through made the bar jump backwards from 70% to 5% the moment
        rendering started.
        """
        progress(stage, 0.70 + 0.28 * max(0.0, min(1.0, fraction)), **detail)

    result = build_mix(
        tracks=[r.track for r in chosen],
        target_minutes=options.length_minutes,
        blend=options.blend,
        tuning=tuning,
        progress=render_progress,
        out_name=options.name,
        hype=options.hype,
    )

    audio_path = result.files.get("mp3") or result.files["wav"]
    title = options.name or f"AutoDJ {time.strftime('%d %b %Y %H:%M')}"

    get_db().put_mix(
        mix_id=result.name,
        title=title,
        audio_path=audio_path,
        duration=result.files["duration"],
        options=options.model_dump(),
        tracklist=result.tracklist.get("events", []),
        log=result.log,
    )

    return {
        "name": result.name,
        "title": title,
        "audio_url": f"/api/mixes/{result.name}/audio",
        "duration": round(result.files["duration"], 2),
        "files": result.files,
        "tracklist": result.tracklist,
        "trends": {
            "total": len(merged),
            "resolved": len(available),
            "missing": len(resolution.missing),
            "coverage": round(resolution.coverage, 3),
            "providers": resolution.provider_status,
            "warnings": resolution.warnings(),
        },
        "missing": [m.to_dict() for m in resolution.missing[:50]],
        "chosen": [r.to_dict() for r in chosen],
        "log": result.log,
    }


def analyze_library(
    progress: Callable[..., None],
    with_stems: bool = True,
    source: Optional[LocalLibrarySource] = None,
) -> dict:
    """Pre-analyse (and optionally pre-separate) the whole library.

    Demucs runs at roughly 1x realtime, so doing this once up front is the
    difference between a mix starting immediately and waiting minutes per track
    inside the render.
    """
    tuning = get_tuning()
    settings = get_settings()
    source = source or LocalLibrarySource(settings.music_dir)

    tracks = source.scan(force=True)
    if not tracks:
        raise ValueError(f"no audio found in {settings.music_dir}")

    done, failed = [], []
    for index, track in enumerate(tracks):
        fraction = index / len(tracks)
        try:
            progress(f"analysing {track.title}", fraction * 0.9,
                     track=track.title, index=index + 1, total=len(tracks))
            analyze_track(track, tuning)
            if with_stems:
                progress(f"separating {track.title}", fraction * 0.9 + 0.02,
                         track=track.title, index=index + 1, total=len(tracks))
                separate_stems(track, tuning)
            done.append(track.title)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            log.warning("library analysis failed for %s: %s", track.path, exc)
            failed.append({"track": track.title, "error": str(exc)})

    progress("done", 1.0)
    return {"analyzed": len(done), "failed": failed, "total": len(tracks)}
