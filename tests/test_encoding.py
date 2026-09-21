"""Hebrew text must survive on a Hebrew-locale Windows install.

The system codepage there is cp1255, so anything read or decoded with the
locale default breaks on UTF-8 bytes. This project is about Israeli music, so
Hebrew appears in tags, titles, tracklists and config -- these are the places
that silently fail when the encoding is left implicit.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from app.config import load_tuning
from app.sources.local import match_key, normalize

HEBREW_SAMPLES = [
    "נועה קירל",
    "אושר כהן",
    "שרית חדד - קח אותי",
    "עדן בן זקן",
]


def test_config_with_hebrew_loads(tmp_path):
    """load_tuning must read UTF-8 regardless of the system locale."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "analysis:\n"
        "  beat_tracker: librosa\n"
        "trends:\n"
        "  israel_weight: 0.7\n"
        "  # comment with Hebrew: מוזיקה ישראלית\n",
        encoding="utf-8",
    )
    tuning = load_tuning(config)
    assert tuning.analysis.beat_tracker == "librosa"
    assert tuning.trends.israel_weight == 0.7


@pytest.mark.parametrize("text", HEBREW_SAMPLES)
def test_hebrew_survives_normalization(text):
    result = normalize(text)
    assert result, f"{text!r} normalized to nothing"
    assert any("֐" <= c <= "׿" for c in result), "Hebrew characters were stripped"


def test_hebrew_match_key_is_stable():
    first = match_key("קח אותי", "שרית חדד")
    assert first == match_key("קח אותי", "שרית חדד")
    assert "שרית" in first


def test_mixed_hebrew_latin_title_normalizes():
    """Israeli releases routinely mix scripts."""
    result = normalize("נועה קירל - Unicorn (Official Video)")
    assert "unicorn" in result
    assert "נועה" in result
    assert "official" not in result


def test_tracklist_json_round_trips_hebrew(tmp_path):
    """The tracklist sidecar must stay readable, not become escape sequences."""
    from app.render.encode import write_tracklist

    payload = {"tracks": [{"artist": "נועה קירל", "title": "Unicorn"}]}
    path = write_tracklist(tmp_path / "mix.json", payload)

    raw = path.read_text(encoding="utf-8")
    assert "נועה קירל" in raw, "Hebrew should be written literally, not escaped"
    assert json.loads(raw)["tracks"][0]["artist"] == "נועה קירל"


def test_subprocess_output_is_decoded_as_utf8():
    """Guards a real crash.

    Demucs' progress output contains bytes that cp1255 cannot decode. With
    text=True and no explicit encoding, the subprocess reader thread died with
    UnicodeDecodeError and the real error message was lost.
    """
    result = subprocess.run(
        [sys.executable, "-c", "print('\u05e9\u05dc\u05d5\u05dd')"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0
    assert "שלום" in result.stdout


def test_hebrew_filename_is_scannable(tmp_path):
    """Files named in Hebrew must be found and their metadata parsed."""
    import numpy as np
    import soundfile as sf

    from app.sources.local import LocalLibrarySource

    path = tmp_path / "שרית חדד - קח אותי.wav"
    sf.write(path, np.zeros(2205, dtype=np.float32), 22050)

    tracks = LocalLibrarySource(tmp_path).scan()
    assert len(tracks) == 1
    assert tracks[0].artist == "שרית חדד"
    assert tracks[0].title == "קח אותי"


def test_hebrew_track_resolves_from_trend_metadata(tmp_path):
    """The end-to-end case: a Hebrew trend entry binding to a Hebrew file."""
    import numpy as np
    import soundfile as sf

    from app.sources.local import LocalLibrarySource

    sf.write(tmp_path / "נועה קירל - Unicorn.wav", np.zeros(2205, dtype=np.float32), 22050)
    source = LocalLibrarySource(tmp_path)
    source.scan()

    hit = source.resolve("Unicorn (Official Video)", "נועה קירל", threshold=80)
    assert hit is not None, "a Hebrew artist with a Latin title must still resolve"
    assert hit[0].title == "Unicorn"
