"""AutoDJ FastAPI application.

Stage 1 surface: library scanning, the analysis pipeline, stem separation and
manual key override. Mixing, trends and live mode arrive in later stages.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.analysis.key import key_name
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


WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


@app.exception_handler(404)
async def not_found(request, exc):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": getattr(exc, "detail", "not found")})
