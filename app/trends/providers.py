"""Concrete trend providers: Last.fm, Apple Music and YouTube.

Each returns metadata only. Note what they are actually good for:

- **Apple Music** is the strongest Israeli source. Its Israel chart returns real
  Israeli and Mizrahi pop, and it needs no API key.
- **Last.fm**'s Israel chart reflects Last.fm scrobblers in Israel, who skew
  heavily international indie/rock. Useful, but do not expect Mizrahi hits.
- **YouTube** catches viral tracks the audio charts miss, but its music category
  mixes in clips and lyric videos, so titles need heavy cleaning.
"""
from __future__ import annotations

import json
import logging
import re
import time

import httpx

from app.trends.base import CachedTrendProvider, Region, TrendTrack

log = logging.getLogger("autodj.trends")

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.5


def fetch_json(url: str, params: dict | None = None, attempts: int = RETRY_ATTEMPTS) -> dict:
    """GET and parse JSON, retrying transient failures.

    Apple's Israel storefront in particular is intermittently slow and
    occasionally returns a truncated body that fails to parse -- the same URL
    succeeds seconds later. Retrying turns a lost chart into a short delay,
    which matters because that feed is the main source of Israeli music here.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = httpx.get(url, params=params, timeout=TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < attempts - 1:
                delay = RETRY_BACKOFF ** attempt
                log.info("retrying %s after %s (attempt %d/%d)",
                         url, type(exc).__name__, attempt + 1, attempts)
                time.sleep(delay)
        except httpx.HTTPStatusError as exc:
            # A 4xx is a real answer, not a blip; retrying will not help.
            if exc.response.status_code < 500:
                raise
            last = exc
            if attempt < attempts - 1:
                time.sleep(RETRY_BACKOFF ** attempt)

    raise RuntimeError(
        f"{url} failed after {attempts} attempts: {type(last).__name__}: {last}"
    ) from last


class LastFmProvider(CachedTrendProvider):
    """geo.getTopTracks for Israel, chart.getTopTracks for global."""

    name = "lastfm"
    BASE = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch(self, region: Region, limit: int = 50) -> list[TrendTrack]:
        if region == "IL":
            params = {"method": "geo.gettoptracks", "country": "Israel"}
        else:
            params = {"method": "chart.gettoptracks"}
        params.update({"api_key": self.api_key, "format": "json", "limit": str(limit)})

        payload = fetch_json(self.BASE, params)

        if "error" in payload:
            raise RuntimeError(f"last.fm error {payload['error']}: {payload.get('message')}")

        items = payload.get("tracks", {}).get("track", [])
        if isinstance(items, dict):
            items = [items]

        out = []
        for index, item in enumerate(items[:limit], start=1):
            artist = item.get("artist") or {}
            out.append(TrendTrack(
                title=item.get("name", "").strip(),
                artist=(artist.get("name") if isinstance(artist, dict) else str(artist)).strip(),
                rank=index, region=region, provider=self.name,
                url=item.get("url", ""),
            ))
        return [t for t in out if t.title and t.artist]


class AppleMusicProvider(CachedTrendProvider):
    """Apple's public marketing RSS charts. No API key required.

    The best source for Israeli music in this app: the `il` storefront returns
    genuine Israeli and Mizrahi chart entries, with Hebrew titles intact.
    """

    name = "applemusic"
    BASE = "https://rss.applemarketingtools.com/api/v2"

    # Apple has no "global" storefront; the US chart is the usual proxy.
    STOREFRONTS = {"IL": "il", "global": "us"}

    def available(self) -> bool:
        return True

    def fetch(self, region: Region, limit: int = 50) -> list[TrendTrack]:
        storefront = self.STOREFRONTS.get(region, "us")
        count = max(10, min(limit, 100))
        url = f"{self.BASE}/{storefront}/music/most-played/{count}/songs.json"

        results = fetch_json(url).get("feed", {}).get("results", [])

        return [
            TrendTrack(
                title=item.get("name", "").strip(),
                artist=item.get("artistName", "").strip(),
                rank=index, region=region, provider=self.name,
                url=item.get("url", ""), artwork=item.get("artworkUrl100", ""),
            )
            for index, item in enumerate(results[:limit], start=1)
            if item.get("name") and item.get("artistName")
        ]


# YouTube music titles are full of junk that stops them matching a real file.
_YT_NOISE = re.compile(
    r"\s*[\(\[][^\)\]]*(official|video|audio|lyric|lyrics|visualizer|hd|4k|"
    r"mv|m/v|clip|prod|remix by)[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_YT_TRAILING = re.compile(
    r"\s*[-|–]\s*(official\s*)?(music\s*)?(video|audio|lyrics?|visualizer)\s*$",
    re.IGNORECASE,
)


def split_youtube_title(raw_title: str, channel: str) -> tuple[str, str]:
    """Best-effort (artist, title) from a YouTube video title.

    YouTube gives one free-text string, conventionally "Artist - Title" but
    padded with "(Official Video)" and similar. Without cleaning, almost nothing
    matches a tagged file on disk.
    """
    cleaned = _YT_NOISE.sub("", raw_title or "")
    cleaned = _YT_TRAILING.sub("", cleaned).strip()

    for separator in (" - ", " – ", " — ", " | "):
        if separator in cleaned:
            left, right = cleaned.split(separator, 1)
            return left.strip(), right.strip()

    # No separator: fall back to the channel name, which is often the artist.
    channel = re.sub(r"\s*-\s*Topic$", "", channel or "", flags=re.IGNORECASE).strip()
    return channel, cleaned


class YouTubeProvider(CachedTrendProvider):
    """videos.list chart=mostPopular, music category, by region."""

    name = "youtube"
    BASE = "https://www.googleapis.com/youtube/v3/videos"
    MUSIC_CATEGORY = "10"

    REGIONS = {"IL": "IL", "global": "US"}

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch(self, region: Region, limit: int = 50) -> list[TrendTrack]:
        params = {
            "part": "snippet", "chart": "mostPopular",
            "videoCategoryId": self.MUSIC_CATEGORY,
            "regionCode": self.REGIONS.get(region, "US"),
            "maxResults": str(max(1, min(limit, 50))),
            "key": self.api_key,
        }
        probe = httpx.get(self.BASE, params=params, timeout=TIMEOUT)
        if probe.status_code == 403:
            raise RuntimeError(
                "YouTube API returned 403 - check the key is valid, that YouTube "
                "Data API v3 is enabled, and that the daily quota is not exhausted"
            )
        probe.raise_for_status()
        payload = probe.json()

        out = []
        for index, item in enumerate(payload.get("items", []), start=1):
            snippet = item.get("snippet", {})
            artist, title = split_youtube_title(
                snippet.get("title", ""), snippet.get("channelTitle", "")
            )
            if not title:
                continue
            out.append(TrendTrack(
                title=title, artist=artist, rank=index, region=region,
                provider=self.name,
                url=f"https://www.youtube.com/watch?v={item.get('id', '')}",
                artwork=(snippet.get("thumbnails", {}).get("high", {}) or {}).get("url", ""),
            ))
        return out


def build_providers(names: list[str] | None = None) -> list[CachedTrendProvider]:
    """Instantiate the configured providers, keys pulled from settings."""
    from app.config import get_settings, get_tuning

    settings = get_settings()
    names = names if names is not None else get_tuning().trends.providers

    registry = {
        "lastfm": lambda: LastFmProvider(settings.lastfm_api_key),
        "applemusic": AppleMusicProvider,
        "youtube": lambda: YouTubeProvider(settings.youtube_api_key),
    }

    providers = []
    for name in names:
        factory = registry.get(name)
        if factory is None:
            log.warning("unknown trend provider in config: %s", name)
            continue
        providers.append(factory())
    return providers
