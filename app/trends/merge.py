"""Merge several providers' charts into one ranked list.

Two things are being combined: different providers ranking the same song, and
the Israel/global split. A track's score is driven by its best rank, boosted
when more than one provider agrees, then weighted by region.
"""
from __future__ import annotations

from collections import defaultdict

from app.trends.base import MergedTrend, Region, TrendTrack


def rank_score(rank: int, chart_size: int = 50) -> float:
    """Turn a chart position into 0..1, front-loaded.

    Linear decay would treat #1 and #10 as nearly equal. Chart attention is
    closer to logarithmic, so the top handful should pull away sharply.
    """
    if rank < 1:
        return 0.0
    import math

    return float(max(0.0, 1.0 - math.log(rank, max(chart_size, 2))))


def merge_trends(
    tracks: list[TrendTrack],
    israel_weight: float = 0.7,
    global_weight: float = 0.3,
    max_tracks: int = 200,
    agreement_bonus: float = 0.15,
) -> list[MergedTrend]:
    """Collapse provider charts into one ranked list.

    Entries are matched on normalized "artist title", so the same song from
    Apple and Last.fm merges into one row carrying both provider names.
    """
    region_weights = {"IL": israel_weight, "global": global_weight}
    grouped: dict[str, list[TrendTrack]] = defaultdict(list)
    for track in tracks:
        if track.key:
            grouped[track.key].append(track)

    merged: list[MergedTrend] = []
    for entries in grouped.values():
        best = min(entries, key=lambda t: t.rank)
        providers = sorted({t.provider for t in entries})
        regions = {t.region for t in entries}

        # A track charting in both places counts as Israeli: this app is
        # Israel-first, and the higher weight is the point.
        region: Region = "IL" if "IL" in regions else "global"

        # Score on the best rank achieved in the winning region.
        in_region = [t for t in entries if t.region == region] or entries
        best_in_region = min(in_region, key=lambda t: t.rank)

        score = rank_score(best_in_region.rank) * region_weights.get(region, 0.5)
        if len(providers) > 1:
            # Independent charts agreeing is real signal, not duplication.
            score *= 1.0 + agreement_bonus * (len(providers) - 1)

        merged.append(MergedTrend(
            title=best.title, artist=best.artist, score=float(score),
            region=region, providers=providers, best_rank=best_in_region.rank,
            url=best.url, artwork=best.artwork,
        ))

    merged.sort(key=lambda m: m.score, reverse=True)
    for position, item in enumerate(merged, start=1):
        item.rank = position
    return merged[:max_tracks]


def weighted_sample(
    trends: list[MergedTrend],
    count: int,
    israel_ratio: float = 0.7,
    seed: int | None = None,
) -> list[MergedTrend]:
    """Pick `count` tracks, weighted by score and honouring the region ratio.

    Sampling rather than taking the top N keeps successive mixes from being
    identical, which is the whole point of pressing Create twice.
    """
    import random

    rng = random.Random(seed)
    israeli = [t for t in trends if t.region == "IL"]
    worldwide = [t for t in trends if t.region == "global"]

    want_israeli = min(len(israeli), round(count * israel_ratio))
    want_global = min(len(worldwide), count - want_israeli)
    # If one pool is short, top up from the other rather than returning fewer.
    shortfall = count - (want_israeli + want_global)
    if shortfall > 0:
        if len(israeli) > want_israeli:
            want_israeli = min(len(israeli), want_israeli + shortfall)
        elif len(worldwide) > want_global:
            want_global = min(len(worldwide), want_global + shortfall)

    picked = _sample_weighted(israeli, want_israeli, rng)
    picked += _sample_weighted(worldwide, want_global, rng)
    picked.sort(key=lambda t: t.score, reverse=True)
    return picked


def _sample_weighted(pool: list[MergedTrend], count: int, rng) -> list[MergedTrend]:
    """Weighted sampling without replacement."""
    available = list(pool)
    chosen: list[MergedTrend] = []
    for _ in range(min(count, len(available))):
        weights = [max(t.score, 1e-6) for t in available]
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                chosen.append(available.pop(index))
                break
        else:
            chosen.append(available.pop())
    return chosen
