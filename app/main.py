"""AutoDJ FastAPI application.

Stage 1 surface: library scanning, the analysis pipeline, stem separation and
manual key override. Mixing, trends and live mode arrive in later stages.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.analysis.key import key_name
from app.create import CreateOptions, analyze_library, create_mix
from app.jobs import get_registry, sse_stream
from app.trends.resolver import fetch_and_resolve
from app.matching.scorer import rank_pairs
from app.render.mashup import build_mashup
from app.analysis.pipeline import (
    PIPELINE_VERSION,
    analyze_track,
    clear_key_override,
    separate_stems,
    set_key_override,
)
from app.config import PROJECT_ROOT, get_settings, get_tuning
from app.db import get_db
from app.models import PITCH_NAMES, TrackMeta
from app.sources.local import LocalLibrarySource

log = logging.getLogger("autodj")

app = FastAPI(title="AutoDJ", version="0.1.0")

_source: LocalLibrarySource | None = None


def get_source() -> LocalLibrarySource:
    global _source
    if _source is None:
        _source = LocalLibrarySource(get_settings().music_dir)
    return _source


def _require_track(track_id: str) -> TrackMeta:
    track = get_source().get(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"unknown track: {track_id}")
    if not Path(track.path).exists():
        raise HTTPException(status_code=410, detail=f"file has gone missing: {track.path}")
    return track


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    tuning = get_tuning()
    return {
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "music_dir": str(settings.music_dir),
        "music_dir_exists": settings.music_dir.exists(),
        "beat_tracker": tuning.analysis.beat_tracker,
        "stem_provider": tuning.analysis.stem_provider,
    }


@app.get("/api/config")
def config() -> dict:
    return get_tuning().model_dump()


@app.get("/api/library")
async def library(refresh: bool = Query(False)) -> dict:
    """List every track the configured library holds."""
    source = get_source()
    tracks = await run_in_threadpool(source.scan, refresh)
    db = get_db()

    out = []
    for track in tracks:
        cached = db.get_analysis(track.file_hash, PIPELINE_VERSION)
        out.append({
            **track.model_dump(),
            "analyzed": cached is not None,
            "bpm": round(cached.bpm, 2) if cached else None,
            "key": key_name(cached.key.pitch_class, cached.key.mode) if cached else None,
            "camelot": cached.key.camelot if cached else None,
        })
    return {"music_dir": str(get_settings().music_dir), "count": len(out), "tracks": out}


@app.post("/api/tracks/{track_id}/analyze")
async def analyze(track_id: str, force: bool = Query(False)) -> dict:
    track = _require_track(track_id)
    try:
        analysis = await run_in_threadpool(analyze_track, track, None, lambda s, f: None, not force)
    except Exception as exc:  # noqa: BLE001 - surface the real reason to the UI
        log.exception("analysis failed for %s", track.path)
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc
    return _analysis_payload(track, analysis)


@app.get("/api/tracks/{track_id}/analysis")
def get_analysis(track_id: str) -> dict:
    track = _require_track(track_id)
    analysis = get_db().get_analysis(track.file_hash, PIPELINE_VERSION)
    if analysis is None:
        raise HTTPException(status_code=404, detail="not analyzed yet; POST .../analyze first")
    return _analysis_payload(track, analysis)


class KeyOverride(BaseModel):
    pitch_class: int = Field(ge=0, le=11)
    mode: str = Field(pattern="^(major|minor)$")


@app.post("/api/tracks/{track_id}/key")
def override_key(track_id: str, body: KeyOverride) -> dict:
    """Pin a key by hand.

    Key detection is unreliable on Mizrahi/maqam material, which does not sit in
    12-TET major/minor, so the user gets the final say.
    """
    track = _require_track(track_id)
    set_key_override(track.file_hash, body.pitch_class, body.mode)
    analysis = get_db().get_analysis(track.file_hash, PIPELINE_VERSION)
    if analysis is None:
        return {"ok": True, "note": "override stored; track not analyzed yet"}
    from app.analysis.pipeline import _apply_override

    return _analysis_payload(track, _apply_override(analysis, get_db()))


@app.delete("/api/tracks/{track_id}/key")
def reset_key(track_id: str) -> dict:
    track = _require_track(track_id)
    clear_key_override(track.file_hash)
    return {"ok": True}


@app.post("/api/tracks/{track_id}/stems")
async def stems(track_id: str) -> dict:
    track = _require_track(track_id)
    try:
        result = await run_in_threadpool(separate_stems, track)
    except Exception as exc:  # noqa: BLE001
        log.exception("stem separation failed for %s", track.path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.model_dump()


@app.get("/api/tracks/{track_id}/audio")
def track_audio(track_id: str) -> FileResponse:
    track = _require_track(track_id)
    return FileResponse(track.path)


def _analysis_payload(track: TrackMeta, analysis) -> dict:
    return {
        "track": track.model_dump(),
        "bpm": round(analysis.bpm, 2),
        "beat_tracker": analysis.beat_tracker,
        "duration": round(analysis.duration, 2),
        "key": {
            "name": key_name(analysis.key.pitch_class, analysis.key.mode),
            "pitch_class": analysis.key.pitch_class,
            "mode": analysis.key.mode,
            "camelot": analysis.key.camelot,
            "confidence": round(analysis.key.confidence, 3),
            "source": analysis.key.source,
        },
        "beats": len(analysis.beats),
        "downbeats": analysis.downbeats[:64],
        "downbeat_count": len(analysis.downbeats),
        "sections": [
            {
                "index": s.index,
                "label": s.label,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "energy": round(s.energy, 3),
            }
            for s in analysis.sections
        ],
        "energy_times": [round(t, 2) for t in analysis.energy_times],
        "energy_curve": [round(v, 3) for v in analysis.energy_curve],
        "pitch_names": PITCH_NAMES,
    }



# ---------------------------------------------------------------------------
# Stage 2: matching and mashup rendering
# ---------------------------------------------------------------------------

@app.get("/api/match")
async def match(a: str, b: str, intent: str = Query("match"), top: int = Query(10)) -> dict:
    """Rank how each section of A could move into each section of B."""
    track_a, track_b = _require_track(a), _require_track(b)
    tuning = get_tuning()

    analysis_a = await run_in_threadpool(analyze_track, track_a, tuning)
    analysis_b = await run_in_threadpool(analyze_track, track_b, tuning)

    pairs = rank_pairs(analysis_a, analysis_b, tuning.matching, intent=intent, top_n=top)
    return {
        "a": {"id": track_a.id, "title": track_a.title, "bpm": round(analysis_a.bpm, 2),
              "camelot": analysis_a.key.camelot,
              "key_confidence": round(analysis_a.key.confidence, 3)},
        "b": {"id": track_b.id, "title": track_b.title, "bpm": round(analysis_b.bpm, 2),
              "camelot": analysis_b.key.camelot,
              "key_confidence": round(analysis_b.key.confidence, 3)},
        "intent": intent,
        "count": len(pairs),
        "pairs": [p.to_dict() for p in pairs],
    }


class MashupRequest(BaseModel):
    track_a: str                       # supplies the vocal
    track_b: str                       # supplies the instrumental bed
    bars: int = Field(default=32, ge=4, le=256)
    transition: Optional[str] = None
    name: Optional[str] = None


@app.post("/api/mashup")
async def mashup(body: MashupRequest) -> dict:
    """Render a mashup: vocal of A over the instrumental of B."""
    track_a, track_b = _require_track(body.track_a), _require_track(body.track_b)

    try:
        result = await run_in_threadpool(
            build_mashup, track_a, track_b, body.bars, get_tuning(),
            lambda stage, fraction: None, body.name, body.transition,
        )
    except ValueError as exc:
        # A genuinely incompatible pair is a user-fixable problem, not a crash.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("mashup failed")
        raise HTTPException(status_code=500, detail=f"render failed: {exc}") from exc

    audio = result.files.get("mp3") or result.files["wav"]
    get_db().put_mix(
        mix_id=result.name,
        title=f"{track_a.title} x {track_b.title}",
        audio_path=audio,
        duration=result.files["duration"],
        options={"mode": "mashup", "bars": body.bars, "transition": body.transition},
        tracklist=result.tracklist.get("events", []),
        log=result.log,
    )

    return {
        "name": result.name,
        "files": result.files,
        "audio_url": f"/api/mixes/{result.name}/audio",
        "duration": round(result.files["duration"], 2),
        "match": result.pair.to_dict() if result.pair else None,
        "tracklist": result.tracklist,
        "log": result.log,
    }


@app.get("/api/mixes")
def list_mixes() -> dict:
    mixes = get_db().list_mixes()
    for mix in mixes:
        mix["audio_url"] = f"/api/mixes/{mix['id']}/audio"
    return {"count": len(mixes), "mixes": mixes}


@app.get("/api/mixes/{mix_id}/audio")
def mix_audio(mix_id: str):
    mix = get_db().get_mix(mix_id)
    if mix is None:
        raise HTTPException(status_code=404, detail=f"unknown mix: {mix_id}")
    path = Path(mix["audio_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"mix file is gone: {path}")
    media = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/mixes/{mix_id}")
def mix_detail(mix_id: str) -> dict:
    mix = get_db().get_mix(mix_id)
    if mix is None:
        raise HTTPException(status_code=404, detail=f"unknown mix: {mix_id}")
    mix["audio_url"] = f"/api/mixes/{mix_id}/audio"
    return mix



# ---------------------------------------------------------------------------
# Stage 3: trends, the Create flow, jobs and history
# ---------------------------------------------------------------------------

@app.get("/api/trends")
async def trends(refresh: bool = Query(False)) -> dict:
    """Merged trending list, with each entry matched against your library."""
    source = get_source()
    await run_in_threadpool(source.scan, refresh)
    merged, resolution = await run_in_threadpool(fetch_and_resolve, source)

    # resolution carries per-provider counts and errors, so a provider that
    # silently produced nothing is visible instead of just missing.
    return {
        "total": len(merged),
        "trending": [
            {"rank": m.rank, "title": m.title, "artist": m.artist,
             "region": m.region, "providers": m.providers,
             "score": round(m.score, 4), "artwork": m.artwork}
            for m in merged[:100]
        ],
        **resolution.to_dict(),
    }


@app.post("/api/create")
def create(options: CreateOptions) -> dict:
    """Start a Create job. Returns immediately; follow progress over SSE."""
    registry = get_registry()
    job = registry.create("create")

    def target(job, progress):
        return create_mix(options, progress, get_source())

    registry.run(job, target)
    return {"job_id": job.id, "events_url": f"/api/jobs/{job.id}/events", **job.to_dict()}


@app.post("/api/library/analyze")
def analyze_library_endpoint(with_stems: bool = Query(True)) -> dict:
    """Pre-analyse the whole library so later mixes render without waiting."""
    registry = get_registry()
    job = registry.create("analyze_library")

    def target(job, progress):
        return analyze_library(progress, with_stems, get_source())

    registry.run(job, target)
    return {"job_id": job.id, "events_url": f"/api/jobs/{job.id}/events", **job.to_dict()}


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in get_registry().list()]}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    job = get_registry().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not get_registry().cancel(job_id):
        raise HTTPException(status_code=409, detail="job is not running")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of a job's progress."""
    registry = get_registry()
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")

    return StreamingResponse(
        sse_stream(job, registry),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx and friends buffering the stream into uselessness.
            "X-Accel-Buffering": "no",
        },
    )


WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


@app.exception_handler(404)
async def not_found(request, exc):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": getattr(exc, "detail", "not found")})
