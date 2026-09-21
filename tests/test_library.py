"""Library scanning and the fuzzy matching that binds trend metadata to files."""
from __future__ import annotations

from app.sources.local import LocalLibrarySource, match_key, normalize


def test_normalize_strips_youtube_noise():
    assert normalize("Unicorn (Official Video) [4K]") == "unicorn"
    assert normalize("Hupa - Remastered 2021") == "hupa 2021"
    assert normalize("Song feat. Someone") == "song someone"


def test_normalize_keeps_hebrew_intact():
    """Hebrew must survive normalization or no Israeli track ever matches."""
    assert normalize("נועה קירל - Unicorn") == "נועה קירל unicorn"
    assert normalize("אושר כהן") == "אושר כהן"


def test_match_key_orders_artist_then_title():
    assert match_key("Unicorn", "Noa Kirel") == "noa kirel unicorn"


def test_scan_reads_tracks_and_dedupes(fixture_paths):
    from tests.conftest import FIXTURE_DIR

    source = LocalLibrarySource(FIXTURE_DIR)
    tracks = source.scan()
    assert len(tracks) == len(fixture_paths)
    assert len({t.id for t in tracks}) == len(tracks), "ids must be unique"
    assert all(t.file_hash for t in tracks)

    # A rescan must be stable, not duplicate.
    assert len(source.scan(force=True)) == len(tracks)


def test_resolve_matches_despite_noise(fixture_paths):
    from tests.conftest import FIXTURE_DIR

    source = LocalLibrarySource(FIXTURE_DIR)
    source.scan()

    hit = source.resolve("synth_a_128_amin (Official Video)", "", threshold=70)
    assert hit is not None
    track, score = hit
    assert track.title == "synth_a_128_amin"
    assert score >= 70


def test_resolve_rejects_a_track_that_is_not_there(fixture_paths):
    """Unmatched trending tracks must fall through to the 'missing' list."""
    from tests.conftest import FIXTURE_DIR

    source = LocalLibrarySource(FIXTURE_DIR)
    source.scan()
    assert source.resolve("Totally Different Song", "Nobody At All", threshold=82) is None
