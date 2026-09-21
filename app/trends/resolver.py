"""Bind trending metadata to files the user actually owns.

This is the join between "what is popular" and "what can be played". Anything
that fails to bind goes into the missing-tracks list, which doubles as a
shopping list -- the app never fetches audio to fill the gap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.models import TrackMeta
from app.sources.base import AudioSource
from app.trends.base import MergedTrend

log = logging.getLogger("autodj.trends")


@dataclass
class ResolvedTrend:
    trend: MergedTrend
    track: TrackMeta
    score: float                # fuzzy match confidence, 0..100

    def to_dict(self) -> dict:
        return {
            "title": self.trend.title, "artist": self.trend.artist,
            "rank": self.trend.rank, "region": self.trend.region,
            "providers": self.trend.providers,
            "match_score": round(self.score, 1),
            "track": {"id": self.track.id, "title": self.track.title,
                      "artist": self.track.artist, "path": self.track.path},
        }


@dataclass
class MissingTrend:
    trend: MergedTrend
    best_guess: Optional[str] = None
    best_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.trend.title, "artist": self.trend.artist,
            "rank": self.trend.rank, "region": self.trend.region,
            "providers": self.trend.providers,
            "url": self.trend.url, "artwork": self.trend.artwork,
            "closest_local": self.best_guess,
            "closest_score": round(self.best_score, 1),
        }


@dataclass
class ResolutionResult:
    resolved: list[ResolvedTrend] = field(default_factory=list)
    missing: list[MissingTrend] = field(default_factory=list)
    provider_status: list[dict] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        total = len(self.resolved) + len(self.missing)
        return len(self.resolved) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "resolved": [r.to_dict() for r in self.resolved],
            "missing": [m.to_dict() for m in self.missing],
            "coverage": round(self.coverage, 3),
            "counts": {"resolved": len(self.resolved), "missing": len(self.missing)},
            "providers": self.provider_status,
            "warnings": self.warnings(),
        }

    def warnings(self) -> list[str]:
        """Plain-language notes about providers that produced nothing."""
        notes = []
        for entry in self.provider_status:
            for region, message in (entry.get("errors") or {}).items():
                where = "Israel" if region == "IL" else "global"
                notes.append(f"{entry['name']} returned no {where} chart: {message}")
        return notes


def resolve_trends(
    trends: list[MergedTrend],
    source: AudioSource,
    threshold: int = 82,
) -> ResolutionResult:
    """Match each trend to a local file, or record it as missing.

    A track already claimed by an earlier (higher-ranked) trend is not reused:
    two chart entries resolving to the same file would put it in the mix twice.
    """
    result = ResolutionResult()
    claimed: set[str] = set()

    for trend in trends:
        hit = source.resolve(trend.title, trend.artist, threshold)

        if hit is None:
            # Re-run at a low threshold purely to show the user the near-miss,
            # which is usually a tagging problem they can fix.
            near = source.resolve(trend.title, trend.artist, threshold=40)
            result.missing.append(MissingTrend(
                trend=trend,
                best_guess=f"{near[0].artist} - {near[0].title}" if near else None,
                best_score=near[1] if near else 0.0,
            ))
            continue

        track, score = hit
        if track.id in claimed:
            log.debug("skipping %s: %s already claimed", trend.label(), track.path)
            continue

        claimed.add(track.id)
        result.resolved.append(ResolvedTrend(trend=trend, track=track, score=score))

    return result


def fetch_and_resolve(
    source: AudioSource,
    israel_weight: float | None = None,
    global_weight: float | None = None,
    limit_per_provider: int = 50,
    progress=None,
) -> tuple[list[MergedTrend], ResolutionResult]:
    """Fetch every configured provider, merge, then resolve against the library."""
    from app.config import get_tuning
    from app.trends.merge import merge_trends
    from app.trends.providers import build_providers

    tuning = get_tuning().trends
    israel_weight = tuning.israel_weight if israel_weight is None else israel_weight
    global_weight = tuning.global_weight if global_weight is None else global_weight

    providers = build_providers()
    collected = []
    status: list[dict] = []
    for index, provider in enumerate(providers):
        counts = {}
        for region in ("IL", "global"):
            if progress:
                progress(f"fetching {provider.name} ({region})",
                         0.1 + 0.4 * (index / max(len(providers), 1)))
            got = provider.fetch_cached(region, limit_per_provider, tuning.cache_hours)
            counts[region] = len(got)
            collected.extend(got)
        status.append({
            "name": provider.name, "available": provider.available(),
            "counts": counts, "errors": provider.errors(),
        })

    for entry in status:
        for region, message in entry["errors"].items():
            log.warning("provider %s produced nothing for %s: %s",
                        entry["name"], region, message)

    merged = merge_trends(
        collected, israel_weight=israel_weight, global_weight=global_weight,
        max_tracks=tuning.max_tracks,
    )
    if progress:
        progress("matching against your library", 0.6)
    resolution = resolve_trends(merged, source, tuning.fuzzy_match_threshold)
    resolution.provider_status = status
    return merged, resolution
