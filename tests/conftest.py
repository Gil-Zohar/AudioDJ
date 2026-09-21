"""Shared fixtures. Analysis runs against synthetic audio with known ground truth."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "_audio"


@pytest.fixture(scope="session", autouse=True)
def isolated_data_dir(tmp_path_factory):
    """Point the app at a throwaway data dir so tests never touch the real cache."""
    data = tmp_path_factory.mktemp("autodj-data")
    os.environ["DATA_DIR"] = str(data)
    os.environ["MUSIC_DIR"] = str(FIXTURE_DIR)

    # Pin the tuning tests run against, so they neither depend on nor are broken
    # by whatever is in the real config.yaml. Chiefly this forces the fast stem
    # provider: Demucs is ~1x realtime, which would add minutes to every run.
    import yaml

    from app.config import CONFIG_PATH

    tuning = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    tuning.setdefault("analysis", {})["stem_provider"] = "hpss"
    test_config = data / "test_config.yaml"
    test_config.write_text(yaml.safe_dump(tuning), encoding="utf-8")
    os.environ["AUTODJ_CONFIG"] = str(test_config)

    import app.config as config
    import app.db as db

    config.get_settings.cache_clear()
    config.get_tuning.cache_clear()
    db._db = None
    yield data


@pytest.fixture(scope="session")
def fixture_paths(isolated_data_dir):
    from tests.fixtures.synth import ensure_fixtures

    return ensure_fixtures(FIXTURE_DIR)


@pytest.fixture(scope="session")
def analyses(fixture_paths):
    """Analyze every fixture once and share the results across tests."""
    from app.analysis.pipeline import analyze_track
    from app.sources.local import LocalLibrarySource

    source = LocalLibrarySource(FIXTURE_DIR)
    tracks = {t.title: t for t in source.scan()}
    return {name: analyze_track(track) for name, track in tracks.items()}


@pytest.fixture(scope="session")
def long_fixture_paths(isolated_data_dir):
    """Full-length synthetic tracks, so a multi-bar mashup has room to fit."""
    from tests.fixtures.synth import LONG_TRACKS, ensure_fixtures

    return ensure_fixtures(FIXTURE_DIR, LONG_TRACKS)


@pytest.fixture(scope="session")
def long_fixture_tracks(long_fixture_paths):
    from app.sources.local import LocalLibrarySource
    from tests.fixtures.synth import TRACK_A_LONG, TRACK_B_LONG

    source = LocalLibrarySource(FIXTURE_DIR)
    tracks = {t.title: t for t in source.scan()}
    return tracks[TRACK_A_LONG.name], tracks[TRACK_B_LONG.name]
