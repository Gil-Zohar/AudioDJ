"""Trend providers, merging and resolution.

No network and no API keys: providers are driven through recorded payloads, so
these run in CI and cannot be broken by a chart changing.
"""
from __future__ import annotations

import pytest

from app.trends.base import MergedTrend, TrendTrack
from app.trends.merge import merge_trends, rank_score, weighted_sample
from app.trends.providers import (
    AppleMusicProvider,
    LastFmProvider,
    YouTubeProvider,
    split_youtube_title,
)
from app.trends.resolver import resolve_trends

# Recorded shapes from the real APIs, trimmed to what the parsers read.
LASTFM_PAYLOAD = {
    "tracks": {
        "track": [
            {"name": "NICOLE KIDMAN", "artist": {"name": "ADÉLA"}, "url": "http://x/1"},
            {"name": "Creep", "artist": {"name": "Radiohead"}, "url": "http://x/2"},
            {"name": "the cure", "artist": {"name": "Olivia Rodrigo"}, "url": "http://x/3"},
        ]
    }
}

APPLE_PAYLOAD = {
    "feed": {
        "title": "Top Songs", "country": "il",
        "results": [
            {"name": "סודות", "artistName": "Odeya", "url": "http://a/1",
             "artworkUrl100": "http://img/1"},
            {"name": "מה זה הקטע", "artistName": "Ofek Adanek", "url": "http://a/2"},
            {"name": "כאפה", "artistName": "Osher Cohen", "url": "http://a/3"},
        ],
    }
}

YOUTUBE_PAYLOAD = {
    "items": [
        {"id": "abc123", "snippet": {"title": "Noa Kirel - Unicorn (Official Music Video)",
                                     "channelTitle": "NoaKirelVEVO", "thumbnails": {}}},
        {"id": "def456", "snippet": {"title": "כאפה", "channelTitle": "Osher Cohen - Topic",
                                     "thumbnails": {}}},
    ]
}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    """Patch httpx.get and record the calls providers make."""
    calls = []

    def factory(payload, status=200):
        def _get(url, params=None, **kwargs):
            calls.append({"url": url, "params": params or {}})
            return _FakeResponse(payload, status)

        monkeypatch.setattr("app.trends.providers.httpx.get", _get)
        return calls

    return factory


# --------------------------------------------------------------- lastfm ---

def test_lastfm_needs_a_key():
    assert not LastFmProvider("").available()
    assert LastFmProvider("abc123").available()


def test_lastfm_parses_and_ranks(fake_get):
    fake_get(LASTFM_PAYLOAD)
    tracks = LastFmProvider("k").fetch("IL", limit=10)

    assert [t.rank for t in tracks] == [1, 2, 3]
    assert tracks[0].artist == "ADÉLA"
    assert tracks[0].title == "NICOLE KIDMAN"
    assert all(t.provider == "lastfm" and t.region == "IL" for t in tracks)


def test_lastfm_uses_the_right_method_per_region(fake_get):
    calls = fake_get(LASTFM_PAYLOAD)
    provider = LastFmProvider("k")

    provider.fetch("IL", 5)
    assert calls[-1]["params"]["method"] == "geo.gettoptracks"
    assert calls[-1]["params"]["country"] == "Israel"

    provider.fetch("global", 5)
    assert calls[-1]["params"]["method"] == "chart.gettoptracks"


def test_lastfm_surfaces_api_errors(fake_get):
    fake_get({"error": 10, "message": "Invalid API key"})
    with pytest.raises(RuntimeError, match="Invalid API key"):
        LastFmProvider("bad").fetch("IL")


def test_fetch_safe_swallows_failures(fake_get):
    fake_get({"error": 10, "message": "Invalid API key"})
    assert LastFmProvider("bad").fetch_safe("IL") == []


def test_unavailable_provider_returns_empty():
    assert LastFmProvider("").fetch_safe("IL") == []


# ---------------------------------------------------------------- apple ---

def test_apple_needs_no_key():
    assert AppleMusicProvider().available()


def test_apple_parses_hebrew_titles(fake_get):
    """Apple's Israel chart is the main Hebrew source; titles must survive."""
    fake_get(APPLE_PAYLOAD)
    tracks = AppleMusicProvider().fetch("IL", limit=10)

    assert len(tracks) == 3
    assert tracks[0].title == "סודות"
    assert tracks[0].artist == "Odeya"
    assert tracks[0].artwork == "http://img/1"


def test_apple_picks_the_right_storefront(fake_get):
    calls = fake_get(APPLE_PAYLOAD)
    provider = AppleMusicProvider()

    provider.fetch("IL", 10)
    assert "/il/music/" in calls[-1]["url"]

    provider.fetch("global", 10)
    assert "/us/music/" in calls[-1]["url"]


# -------------------------------------------------------------- youtube ---

@pytest.mark.parametrize("raw,channel,expect_artist,expect_title", [
    ("Noa Kirel - Unicorn (Official Music Video)", "ch", "Noa Kirel", "Unicorn"),
    ("Artist - Song [Official Video]", "ch", "Artist", "Song"),
    ("Artist – Song (Lyrics)", "ch", "Artist", "Song"),
    # No real separator once the "Official Audio" suffix is stripped, so the
    # remainder is the title and the channel name stands in for the artist.
    ("Song Name - Official Audio", "SomeArtistVEVO", "SomeArtistVEVO", "Song Name"),
    ("כאפה", "Osher Cohen - Topic", "Osher Cohen", "כאפה"),
])
def test_youtube_title_splitting(raw, channel, expect_artist, expect_title):
    """YouTube gives one free-text string padded with junk.

    Without cleaning, almost nothing matches a tagged file on disk.
    """
    artist, title = split_youtube_title(raw, channel)
    if expect_title:
        assert title == expect_title
    assert artist == expect_artist


def test_youtube_parses_payload(fake_get):
    fake_get(YOUTUBE_PAYLOAD)
    tracks = YouTubeProvider("k").fetch("IL", limit=10)
    assert tracks[0].artist == "Noa Kirel"
    assert tracks[0].title == "Unicorn"
    assert "abc123" in tracks[0].url


def test_youtube_explains_a_403(fake_get):
    fake_get({}, status=403)
    with pytest.raises(RuntimeError, match="quota|403|enabled"):
        YouTubeProvider("k").fetch("IL")


# --------------------------------------------------------------- merging ---

def test_rank_score_is_front_loaded():
    """#1 must pull away from #10, not decay linearly."""
    assert rank_score(1) > rank_score(5) > rank_score(20)
    assert rank_score(1) - rank_score(5) > rank_score(20) - rank_score(40)
    assert rank_score(0) == 0.0


def _track(title, artist, rank, region, provider):
    return TrendTrack(title=title, artist=artist, rank=rank, region=region, provider=provider)


def test_merge_collapses_the_same_song_across_providers():
    merged = merge_trends([
        _track("Unicorn", "Noa Kirel", 3, "IL", "lastfm"),
        _track("Unicorn (Official Video)", "Noa Kirel", 1, "IL", "applemusic"),
    ])
    assert len(merged) == 1
    assert set(merged[0].providers) == {"lastfm", "applemusic"}
    assert merged[0].best_rank == 1


def test_provider_agreement_boosts_score():
    """Independent charts agreeing is signal, not duplication."""
    alone = merge_trends([_track("A", "X", 1, "IL", "lastfm")])
    agreed = merge_trends([
        _track("A", "X", 1, "IL", "lastfm"),
        _track("A", "X", 1, "IL", "applemusic"),
    ])
    assert agreed[0].score > alone[0].score


def test_israel_weighting_beats_global_at_equal_rank():
    merged = merge_trends(
        [_track("Isr", "A", 1, "IL", "applemusic"),
         _track("Glob", "B", 1, "global", "applemusic")],
        israel_weight=0.7, global_weight=0.3,
    )
    by_title = {m.title: m for m in merged}
    assert by_title["Isr"].score > by_title["Glob"].score


def test_merge_assigns_sequential_ranks():
    merged = merge_trends([_track(f"T{i}", "A", i, "IL", "applemusic") for i in range(1, 6)])
    assert [m.rank for m in merged] == [1, 2, 3, 4, 5]


def test_merge_respects_max_tracks():
    tracks = [_track(f"T{i}", f"A{i}", i, "IL", "applemusic") for i in range(1, 30)]
    assert len(merge_trends(tracks, max_tracks=10)) == 10


def test_merge_ignores_entries_with_no_key():
    assert merge_trends([_track("", "", 1, "IL", "applemusic")]) == []


# -------------------------------------------------------------- sampling ---

def _merged(title, region, score):
    return MergedTrend(title=title, artist="A", score=score, region=region)


def test_weighted_sample_honours_the_region_ratio():
    pool = ([_merged(f"il{i}", "IL", 0.9) for i in range(10)] +
            [_merged(f"gl{i}", "global", 0.9) for i in range(10)])
    picked = weighted_sample(pool, 10, israel_ratio=0.7, seed=1)

    assert len(picked) == 10
    assert sum(1 for p in picked if p.region == "IL") == 7


def test_weighted_sample_tops_up_from_the_other_pool():
    """A short pool must not silently yield a shorter set."""
    pool = ([_merged("il1", "IL", 0.9)] +
            [_merged(f"gl{i}", "global", 0.9) for i in range(10)])
    assert len(weighted_sample(pool, 6, israel_ratio=0.7, seed=1)) == 6


def test_weighted_sample_is_seedable_and_varies():
    pool = [_merged(f"t{i}", "IL", 1.0 - i * 0.02) for i in range(30)]
    first = [t.title for t in weighted_sample(pool, 8, seed=1)]
    assert first == [t.title for t in weighted_sample(pool, 8, seed=1)]
    # Different seeds should usually differ - that is the point of sampling.
    assert first != [t.title for t in weighted_sample(pool, 8, seed=99)]


def test_weighted_sample_never_repeats_a_track():
    pool = [_merged(f"t{i}", "IL", 1.0) for i in range(10)]
    picked = weighted_sample(pool, 10, seed=3)
    assert len({p.title for p in picked}) == len(picked)


# ------------------------------------------------------------- resolving ---

class _FakeSource:
    """Minimal AudioSource: resolves only titles it was given."""

    def __init__(self, owned):
        from app.models import TrackMeta

        self.owned = {
            f"{artist} - {title}": TrackMeta(
                id=f"id{i}", path=f"/music/{i}.mp3", title=title,
                artist=artist, file_hash=f"h{i}")
            for i, (artist, title) in enumerate(owned)
        }

    def resolve(self, title, artist, threshold):
        hit = self.owned.get(f"{artist} - {title}")
        return (hit, 100.0) if hit else None


def test_resolution_splits_owned_from_missing():
    source = _FakeSource([("Odeya", "סודות"), ("Osher Cohen", "כאפה")])
    trends = [
        MergedTrend(title="סודות", artist="Odeya", score=0.9, region="IL", rank=1),
        MergedTrend(title="כאפה", artist="Osher Cohen", score=0.8, region="IL", rank=2),
        MergedTrend(title="Nowhere", artist="Unknown", score=0.7, region="global", rank=3),
    ]
    result = resolve_trends(trends, source, threshold=82)

    assert len(result.resolved) == 2
    assert len(result.missing) == 1
    assert result.missing[0].trend.title == "Nowhere"
    assert result.coverage == pytest.approx(2 / 3)


def test_one_file_is_not_used_by_two_trends():
    """Two chart entries pointing at one file would duplicate it in the mix."""
    source = _FakeSource([("A", "Song")])
    trends = [
        MergedTrend(title="Song", artist="A", score=0.9, region="IL", rank=1),
        MergedTrend(title="Song", artist="A", score=0.8, region="global", rank=2),
    ]
    result = resolve_trends(trends, source, threshold=82)
    assert len(result.resolved) == 1


def test_resolution_serializes_for_the_api():
    import json

    source = _FakeSource([("A", "Song")])
    trends = [MergedTrend(title="Song", artist="A", score=0.9, region="IL", rank=1)]
    json.dumps(resolve_trends(trends, source, 82).to_dict())


# --------------------------------------------------------- retry + status ---

def test_fetch_json_retries_transient_failures(monkeypatch):
    """Apple's Israel feed intermittently times out or returns a truncated body.

    Losing that chart means losing the Israeli music this app exists for, so a
    blip must become a short delay rather than a silently empty list.
    """
    import httpx

    from app.trends import providers

    attempts = {"n": 0}

    def flaky(url, params=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ReadTimeout("read timed out")
        return _FakeResponse(APPLE_PAYLOAD)

    monkeypatch.setattr(providers.httpx, "get", flaky)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)

    payload = providers.fetch_json("http://example/x")
    assert attempts["n"] == 3
    assert payload["feed"]["country"] == "il"


def test_fetch_json_gives_up_with_a_useful_message(monkeypatch):
    import httpx

    from app.trends import providers

    def always_fail(url, params=None, **kwargs):
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(providers.httpx, "get", always_fail)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        providers.fetch_json("http://example/x")


def test_client_errors_are_not_retried(monkeypatch):
    """A 404 is a real answer; retrying just wastes time."""
    import httpx

    from app.trends import providers

    calls = {"n": 0}

    def not_found(url, params=None, **kwargs):
        calls["n"] += 1
        response = httpx.Response(404, request=httpx.Request("GET", url))
        raise httpx.HTTPStatusError("404", request=response.request, response=response)

    monkeypatch.setattr(providers.httpx, "get", not_found)
    with pytest.raises(httpx.HTTPStatusError):
        providers.fetch_json("http://example/x")
    assert calls["n"] == 1


def test_provider_records_why_it_produced_nothing(fake_get):
    """A silent empty chart is the worst outcome; failures must be visible."""
    fake_get({"error": 10, "message": "Invalid API key"})
    provider = LastFmProvider("bad")

    assert provider.fetch_safe("IL") == []
    assert "IL" in provider.errors()
    assert "Invalid API key" in provider.errors()["IL"]


def test_missing_key_is_recorded_as_an_error():
    provider = LastFmProvider("")
    provider.fetch_safe("IL")
    assert "no API key" in provider.errors()["IL"]


def test_success_clears_a_previous_error(fake_get):
    provider = LastFmProvider("k")
    provider._record("IL", "something went wrong")
    fake_get(LASTFM_PAYLOAD)

    assert provider.fetch_safe("IL")
    assert "IL" not in provider.errors()


def test_resolution_warnings_name_the_provider_and_region():
    from app.trends.resolver import ResolutionResult

    result = ResolutionResult()
    result.provider_status = [
        {"name": "applemusic", "available": True, "counts": {"IL": 0, "global": 50},
         "errors": {"IL": "ReadTimeout: timed out"}},
    ]
    warnings = result.warnings()
    assert len(warnings) == 1
    assert "applemusic" in warnings[0]
    assert "Israel" in warnings[0]
    assert "ReadTimeout" in warnings[0]
