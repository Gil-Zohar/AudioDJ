"""Fetch Creative Commons music from the Internet Archive into a local folder.

Gives you a real, legal library to build and test mixes with when you do not
yet own the tracks that are charting. Everything downloaded is CC-licensed and
the licence of each file is recorded in LICENCES.txt beside it.

This is the one place AutoDJ downloads audio, and it is deliberate: the Internet
Archive publishes these files for free redistribution. It is not a back door for
getting commercial music -- that still has to come from your own library.

    python -m app.tools.fetch_cc --count 15 --out ./music
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import httpx

log = logging.getLogger("autodj.fetch_cc")

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata"
DOWNLOAD_URL = "https://archive.org/download"

TIMEOUT = httpx.Timeout(60.0, connect=15.0)

# Netlabels is the Archive's collection of freely-licensed label releases. It
# skews electronic, which is exactly what beat-matches well.
DEFAULT_QUERY = (
    'collection:(netlabels) AND mediatype:(audio) AND format:(MP3) '
    'AND licenseurl:(*creativecommons*)'
)

MP3_FORMATS = {"VBR MP3", "MP3", "128Kbps MP3", "192Kbps MP3", "256Kbps MP3", "320Kbps MP3"}

# DJ-usable lengths. Skips interviews, sound collages and 20-minute ambient pieces.
MIN_SECONDS = 90
MAX_SECONDS = 480

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# Archive file titles carry cataloguing noise: track numbers, release dates, the
# artist repeated. Left in, they wreck both the tags and the fuzzy matching that
# binds a trending entry to a file on disk.
_TRACK_NUMBER = re.compile(r"^[\s_]*\d{1,3}[\s_]*[-._)]?[\s_]+")
_DATE_PREFIX = re.compile(r"^\s*\d{4}[-_]\d{2}[-_]?")


def clean_title(raw: str, artist: str = "") -> str:
    """Strip cataloguing noise from an Archive track title."""
    title = str(raw or "").strip()
    for _ in range(3):
        before = title
        title = _DATE_PREFIX.sub("", title)
        title = _TRACK_NUMBER.sub("", title)
        if artist:
            pattern = re.compile(r"^\s*" + re.escape(artist) + r"\s*[-_]\s*", re.IGNORECASE)
            title = pattern.sub("", title)
        title = title.strip(" -_.")
        if title == before:
            break
    return title or str(raw or "").strip()


@dataclass
class CcTrack:
    identifier: str
    title: str
    artist: str
    filename: str
    size: int
    seconds: float
    license_url: str

    @property
    def download_url(self) -> str:
        from urllib.parse import quote

        return f"{DOWNLOAD_URL}/{self.identifier}/{quote(self.filename)}"

    def safe_name(self) -> str:
        """`Artist - Title.mp3`, which the library scanner parses directly."""
        artist = _UNSAFE.sub("_", self.artist).strip() or "Unknown Artist"
        title = _UNSAFE.sub("_", self.title).strip() or "Untitled"
        return f"{artist} - {title}.mp3"[:180]


def search_items(query: str, rows: int, page: int = 1) -> list[dict]:
    response = httpx.get(
        SEARCH_URL,
        params={
            "q": query,
            "fl[]": ["identifier", "title", "creator", "licenseurl"],
            "rows": rows, "page": page, "output": "json",
            "sort[]": "downloads desc",
        },
        timeout=TIMEOUT, follow_redirects=True,
    )
    response.raise_for_status()
    return response.json().get("response", {}).get("docs", [])


def _duration_seconds(raw) -> float:
    """Archive durations come as seconds or as 'M:SS' / 'H:MM:SS'."""
    if raw in (None, ""):
        return 0.0
    text = str(raw)
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    try:
        return float(text)
    except ValueError:
        return 0.0


def pick_track(identifier: str, fallback_artist: str = "") -> Optional[CcTrack]:
    """Choose one usable MP3 from an Archive item.

    One track per item keeps the library varied rather than pulling a whole
    album by a single artist.
    """
    response = httpx.get(f"{METADATA_URL}/{identifier}", timeout=TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()

    meta = payload.get("metadata", {}) or {}
    license_url = meta.get("licenseurl", "")
    if "creativecommons" not in str(license_url):
        return None

    creator = meta.get("creator") or fallback_artist or ""
    if isinstance(creator, list):
        creator = creator[0] if creator else ""

    candidates = []
    for item in payload.get("files", []) or []:
        if item.get("format") not in MP3_FORMATS:
            continue
        seconds = _duration_seconds(item.get("length"))
        if seconds and not (MIN_SECONDS <= seconds <= MAX_SECONDS):
            continue
        artist = str(item.get("artist") or creator or "Unknown Artist").strip()
        candidates.append(CcTrack(
            identifier=identifier,
            title=clean_title(item.get("title") or Path(item.get("name", "")).stem, artist),
            artist=artist,
            filename=item.get("name", ""),
            size=int(item.get("size") or 0),
            seconds=seconds,
            license_url=str(license_url),
        ))

    if not candidates:
        return None
    # Middle of the release: openers and closers are often intros or skits.
    return candidates[len(candidates) // 2]


def download(track: CcTrack, out_dir: Path, client: httpx.Client) -> Optional[Path]:
    target = out_dir / track.safe_name()
    if target.exists() and target.stat().st_size > 0:
        return target

    tmp = target.with_suffix(".part")
    try:
        with client.stream("GET", track.download_url, timeout=TIMEOUT,
                           follow_redirects=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes(65536):
                    handle.write(chunk)
        tmp.replace(target)
        return target
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
        log.warning("download failed for %s: %s", track.identifier, exc)
        tmp.unlink(missing_ok=True)
        return None


def tag(path: Path, track: CcTrack) -> None:
    """Write artist/title tags so the resolver can match on metadata."""
    try:
        from mutagen.easyid3 import EasyID3
        from mutagen.mp3 import MP3

        try:
            audio = EasyID3(path)
        except Exception:
            audio = MP3(path)
            audio.add_tags()
            audio = EasyID3(path)

        audio["title"] = track.title
        audio["artist"] = track.artist
        audio["copyright"] = track.license_url
        audio.save()
    except Exception as exc:  # noqa: BLE001 - the filename still carries the metadata
        log.debug("could not tag %s: %s", path, exc)


def fetch(
    out_dir: Path,
    count: int = 15,
    query: str = DEFAULT_QUERY,
    progress: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Download `count` CC tracks into `out_dir`, returning a manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, fraction: float) -> None:
        if progress:
            progress(stage, fraction)
        else:
            print(f"  [{fraction * 100:5.1f}%] {stage}", flush=True)

    report("searching the Internet Archive", 0.02)
    # Over-fetch: many items have no usable MP3 once filtered.
    docs = search_items(query, rows=count * 6)
    if not docs:
        raise RuntimeError("archive.org returned no results for that query")

    downloaded: list[dict] = []
    seen_artists: set[str] = set()

    with httpx.Client(follow_redirects=True) as client:
        for doc in docs:
            if len(downloaded) >= count:
                break
            identifier = doc.get("identifier")
            if not identifier:
                continue

            try:
                track = pick_track(identifier, str(doc.get("creator") or ""))
            except Exception as exc:  # noqa: BLE001
                log.debug("metadata failed for %s: %s", identifier, exc)
                continue
            if track is None:
                continue

            # One track per artist keeps a mix from being the same voice twice.
            key = track.artist.strip().lower()
            if key in seen_artists:
                continue

            fraction = 0.05 + 0.9 * (len(downloaded) / max(count, 1))
            report(f"downloading {track.artist} - {track.title}", fraction)

            path = download(track, out_dir, client)
            if path is None:
                continue

            tag(path, track)
            seen_artists.add(key)
            downloaded.append({
                "file": path.name, "artist": track.artist, "title": track.title,
                "seconds": round(track.seconds, 1), "license": track.license_url,
                "source": f"https://archive.org/details/{track.identifier}",
            })
            time.sleep(0.25)   # be polite to a free service

    manifest = {
        "downloaded": len(downloaded), "requested": count,
        "directory": str(out_dir), "tracks": downloaded,
    }
    _write_licences(out_dir, downloaded)
    report("done", 1.0)
    return manifest


def _write_licences(out_dir: Path, tracks: list[dict]) -> None:
    """Record each file's licence and origin next to the audio."""
    lines = [
        "Creative Commons music downloaded from the Internet Archive by AutoDJ.",
        "Each file's licence and source page is listed below. Check the licence "
        "before using any of these publicly -- most require attribution, and some "
        "prohibit commercial use.",
        "",
    ]
    for entry in tracks:
        lines.append(f"{entry['file']}")
        lines.append(f"    artist : {entry['artist']}")
        lines.append(f"    licence: {entry['license']}")
        lines.append(f"    source : {entry['source']}")
        lines.append("")
    (out_dir / "LICENCES.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Creative Commons music from the Internet Archive."
    )
    parser.add_argument("--out", default="./music", help="target directory")
    parser.add_argument("--count", type=int, default=15, help="how many tracks")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="archive.org query")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    manifest = fetch(Path(args.out), args.count, args.query)

    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print(f"\nDownloaded {manifest['downloaded']} of {manifest['requested']} "
              f"tracks into {manifest['directory']}")
        for entry in manifest["tracks"]:
            print(f"  {entry['artist']} - {entry['title']}  ({entry['seconds']:.0f}s)")
        print(f"\nLicences recorded in {Path(manifest['directory']) / 'LICENCES.txt'}")
        print(f"Point MUSIC_DIR at {manifest['directory']} in .env, then press Create.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
