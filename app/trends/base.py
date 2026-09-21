"""Trend provider interface.

Providers supply **metadata only** -- title, artist, rank. They never return
audio, and AutoDJ never downloads any. The resolver then tries to bind each
trending entry to a file the user already owns; whatever it cannot bind becomes
the "missing tracks" list.

Every provider degrades to an empty list rather than raising: one dead API or a
missing key must not take down the whole Create flow.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import BaseModel, Field

log = logging.getLogger("autodj.trends")

Region = Literal["IL", "global"]


class TrendTrack(BaseModel):
    """One entry on somebody's chart."""

    title: str
    artist: str
    rank: int                       # 1 = top of that provider's chart
    region: Region
    provider: str
    url: str = ""
    artwork: str = ""

    @property
    def key(self) -> str:
        """Identity used to merge the same song across providers."""
        from app.sources.local import match_key

        return match_key(self.title, self.artist)

    def label(self) -> str:
        return f"{self.artist} - {self.title}"


class MergedTrend(BaseModel):
    """A track after merging, carrying where it came from and why it ranks."""

    title: str
    artist: str
    score: float
    region: Region
    rank: int = 0
    providers: list[str] = Field(default_factory=list)
    best_rank: int = 999
    url: str = ""
    artwork: str = ""

    def label(self) -> str:
        return f"{self.artist} - {self.title}"


class TrendProvider(ABC):
    name: str = "base"

    #: Why the last fetch failed, per region. A provider quietly returning
    #: nothing is the worst outcome here -- losing the Israeli chart makes the
    #: app silently useless for its main purpose -- so failures are recorded and
    #: surfaced to the UI rather than swallowed.
    last_errors: dict[str, str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _record(self, region: str, error: Optional[str]) -> None:
        if not hasattr(self, "last_errors"):
            self.last_errors = {}
        if error:
            self.last_errors[region] = error
        else:
            self.last_errors.pop(region, None)

    def errors(self) -> dict[str, str]:
        return dict(getattr(self, "last_errors", {}))

    @abstractmethod
    def available(self) -> bool:
        """False when the provider has no API key or is otherwise unusable."""

    @abstractmethod
    def fetch(self, region: Region, limit: int = 50) -> list[TrendTrack]:
        """Return a ranked chart, or an empty list if it cannot."""

    def fetch_safe(self, region: Region, limit: int = 50) -> list[TrendTrack]:
        """fetch() that never raises.

        A provider outage should cost you that provider's entries, not the mix
        you were trying to build.
        """
        if not self.available():
            log.info("trend provider %s unavailable (no key or disabled)", self.name)
            self._record(region, "no API key configured")
            return []
        try:
            tracks = self.fetch(region, limit)
            log.info("trend provider %s returned %d tracks for %s",
                     self.name, len(tracks), region)
            self._record(region, None)
            if not tracks:
                self._record(region, "provider returned an empty chart")
            return tracks
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            log.warning("trend provider %s failed for %s: %s", self.name, region, exc)
            self._record(region, f"{type(exc).__name__}: {exc}")
            return []


class CachedTrendProvider(TrendProvider):
    """Adds SQLite-backed caching so charts are not refetched constantly."""

    def fetch_cached(
        self, region: Region, limit: int, cache_hours: float
    ) -> list[TrendTrack]:
        from app.db import get_db

        db = get_db()
        cached = db.get_trend_cache(self.name, region, cache_hours * 3600)
        if cached is not None:
            return [TrendTrack.model_validate(item) for item in cached][:limit]

        tracks = self.fetch_safe(region, limit)
        if tracks:
            db.put_trend_cache(self.name, region, [t.model_dump() for t in tracks])
        return tracks
