"""SQLite cache for analysis results, stems and mix history.

Analysis rows are keyed on (file_hash, version); bumping PIPELINE_VERSION in
app.analysis.pipeline therefore invalidates every stale row automatically.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from app.models import Analysis, StemSet

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    artist      TEXT NOT NULL,
    album       TEXT DEFAULT '',
    duration    REAL DEFAULT 0,
    file_hash   TEXT NOT NULL,
    seen_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracks_hash ON tracks(file_hash);

CREATE TABLE IF NOT EXISTS analyses (
    file_hash   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    track_id    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (file_hash, version)
);

CREATE TABLE IF NOT EXISTS stems (
    file_hash   TEXT NOT NULL,
    provider    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (file_hash, provider)
);

CREATE TABLE IF NOT EXISTS key_overrides (
    file_hash   TEXT PRIMARY KEY,
    pitch_class INTEGER NOT NULL,
    mode        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trend_cache (
    provider    TEXT NOT NULL,
    region      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    fetched_at  REAL NOT NULL,
    PRIMARY KEY (provider, region)
);

CREATE TABLE IF NOT EXISTS mixes (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    title       TEXT NOT NULL,
    audio_path  TEXT NOT NULL,
    duration    REAL NOT NULL,
    options     TEXT NOT NULL,
    tracklist   TEXT NOT NULL,
    log         TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- tracks -----------------------------------------------------------
    def upsert_track(self, track) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO tracks (id, path, title, artist, album, duration, file_hash, seen_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET
                     id=excluded.id, title=excluded.title, artist=excluded.artist,
                     album=excluded.album, duration=excluded.duration,
                     file_hash=excluded.file_hash, seen_at=excluded.seen_at""",
                (track.id, track.path, track.title, track.artist, track.album,
                 track.duration, track.file_hash, time.time()),
            )

    def get_track(self, track_id: str) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
            return dict(row) if row else None

    def list_tracks(self) -> list[dict]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM tracks ORDER BY artist, title")]

    # ---- analyses ---------------------------------------------------------
    def get_analysis(self, file_hash: str, version: int) -> Optional[Analysis]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM analyses WHERE file_hash = ? AND version = ?",
                (file_hash, version),
            ).fetchone()
        return Analysis.model_validate_json(row["payload"]) if row else None

    def put_analysis(self, analysis: Analysis) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO analyses (file_hash, version, track_id, payload, created_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(file_hash, version) DO UPDATE SET
                     payload=excluded.payload, created_at=excluded.created_at""",
                (analysis.file_hash, analysis.version, analysis.track_id,
                 analysis.model_dump_json(), time.time()),
            )

    # ---- stems ------------------------------------------------------------
    def get_stems(self, file_hash: str, provider: str) -> Optional[StemSet]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM stems WHERE file_hash = ? AND provider = ?",
                (file_hash, provider),
            ).fetchone()
        return StemSet.model_validate_json(row["payload"]) if row else None

    def put_stems(self, file_hash: str, stems: StemSet) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO stems (file_hash, provider, payload, created_at) VALUES (?,?,?,?)
                   ON CONFLICT(file_hash, provider) DO UPDATE SET
                     payload=excluded.payload, created_at=excluded.created_at""",
                (file_hash, stems.provider, stems.model_dump_json(), time.time()),
            )

    # ---- key overrides ----------------------------------------------------
    def get_key_override(self, file_hash: str) -> Optional[tuple[int, str]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT pitch_class, mode FROM key_overrides WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return (row["pitch_class"], row["mode"]) if row else None

    def put_key_override(self, file_hash: str, pitch_class: int, mode: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO key_overrides (file_hash, pitch_class, mode, created_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(file_hash) DO UPDATE SET
                     pitch_class=excluded.pitch_class, mode=excluded.mode""",
                (file_hash, pitch_class, mode, time.time()),
            )

    def clear_key_override(self, file_hash: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM key_overrides WHERE file_hash = ?", (file_hash,))

    # ---- trend cache ------------------------------------------------------
    def get_trend_cache(self, provider: str, region: str, max_age_s: float) -> Optional[list]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload, fetched_at FROM trend_cache WHERE provider = ? AND region = ?",
                (provider, region),
            ).fetchone()
        if not row or time.time() - row["fetched_at"] > max_age_s:
            return None
        return json.loads(row["payload"])

    def put_trend_cache(self, provider: str, region: str, payload: list) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO trend_cache (provider, region, payload, fetched_at) VALUES (?,?,?,?)
                   ON CONFLICT(provider, region) DO UPDATE SET
                     payload=excluded.payload, fetched_at=excluded.fetched_at""",
                (provider, region, json.dumps(payload), time.time()),
            )

    # ---- mixes ------------------------------------------------------------
    def put_mix(self, mix_id: str, title: str, audio_path: str, duration: float,
                options: dict, tracklist: list, log: list) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO mixes (id, created_at, title, audio_path, duration, options, tracklist, log)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (mix_id, time.time(), title, audio_path, duration,
                 json.dumps(options), json.dumps(tracklist), json.dumps(log)),
            )

    def list_mixes(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM mixes ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["options"] = json.loads(d["options"])
            d["tracklist"] = json.loads(d["tracklist"])
            d["log"] = json.loads(d["log"])
            out.append(d)
        return out

    def get_mix(self, mix_id: str) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM mixes WHERE id = ?", (mix_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["options"] = json.loads(d["options"])
        d["tracklist"] = json.loads(d["tracklist"])
        d["log"] = json.loads(d["log"])
        return d


_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        from app.config import get_settings
        _db = Database(get_settings().db_path)
    return _db
