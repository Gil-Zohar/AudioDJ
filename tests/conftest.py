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
