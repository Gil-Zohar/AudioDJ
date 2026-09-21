"""LocalLibrarySource: reads audio from the folder configured as MUSIC_DIR."""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from app.models import TrackMeta
from app.sources.base import AudioSource

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff"}

# Noise that stops a trending title from matching the same song on disk.
_NOISE = re.compile(
    r"\b(official|video|audio|lyrics?|lyric|hd|4k|remaster(ed)?|feat\.?|ft\.?|"
    r"prod\.?|explicit|radio edit|full|clip|music)\b",
    re.IGNORECASE,
)
_BRACKETS = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_PUNCT = re.compile(r"[^\w\s֐-׿]", re.UNICODE)  # keep Hebrew block


def normalize(text: str) -> str:
    """Fold a title/artist to a comparable form.

    Hebrew is left intact (NFC, no stripping) while Latin text is case-folded and
    de-accented, so 'Noa Kirel - Unicorn (Official Video)' and 'נועה קירל' both
    reduce cleanly.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _BRACKETS.sub(" ", text)
    text = _NOISE.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def match_key(title: str, artist: str) -> str:
    return f"{normalize(artist)} {normalize(title)}".strip()


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash from size + head + tail.

    Full hashing a 10MB file per scan is wasteful and audio files are effectively
    unique in their first and last megabyte, so this keys the cache reliably
    without reading whole libraries off disk.
    """
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(chunk))
        if stat.st_size > chunk * 2:
            fh.seek(-chunk, os.SEEK_END)
            h.update(fh.read(chunk))
    return h.hexdigest()[:32]


def _read_tags(path: Path) -> tuple[str, str, str, float]:
    """(title, artist, album, duration) from tags, falling back to the filename."""
    title = artist = album = ""
    duration = 0.0
    try:
        import mutagen

        meta = mutagen.File(path, easy=True)
        if meta is not None:
            title = (meta.get("title") or [""])[0]
            artist = (meta.get("artist") or [""])[0]
            album = (meta.get("album") or [""])[0]
            if meta.info is not None:
                duration = float(getattr(meta.info, "length", 0.0) or 0.0)
    except Exception:
        pass

    if not title or not artist:
        # "Artist - Title.mp3" is the overwhelmingly common convention.
        stem = path.stem
        if " - " in stem:
            guess_artist, guess_title = stem.split(" - ", 1)
        else:
            guess_artist, guess_title = "", stem
        title = title or guess_title.strip()
        artist = artist or guess_artist.strip()
    return title, artist, album, duration


class LocalLibrarySource(AudioSource):
    name = "local"

    def __init__(self, music_dir: Path):
        self.music_dir = Path(music_dir)
        self._tracks: dict[str, TrackMeta] = {}
        self._index: dict[str, str] = {}   # match_key -> track_id
        self._scanned = False

    def scan(self, force: bool = False) -> list[TrackMeta]:
        if self._scanned and not force:
            return list(self._tracks.values())

        self._tracks.clear()
        self._index.clear()

        if not self.music_dir.exists():
            self._scanned = True
            return []

        for path in sorted(self.music_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            try:
                title, artist, album, duration = _read_tags(path)
                fhash = file_hash(path)
            except Exception:
                continue
            track = TrackMeta(
                id=hashlib.sha256(str(path).encode()).hexdigest()[:16],
                path=str(path),
                title=title or path.stem,
                artist=artist,
                album=album,
                duration=duration,
                file_hash=fhash,
            )
            self._tracks[track.id] = track
            self._index[match_key(track.title, track.artist)] = track.id

        self._scanned = True
        return list(self._tracks.values())

    def get(self, track_id: str) -> Optional[TrackMeta]:
        if not self._scanned:
            self.scan()
        return self._tracks.get(track_id)

    def resolve(self, title: str, artist: str, threshold: int = 82):
        if not self._scanned:
            self.scan()
        if not self._index:
            return None

        query = match_key(title, artist)
        if not query:
            return None

        # Exact normalized hit first - cheap and unambiguous.
        if query in self._index:
            return self._tracks[self._index[query]], 100.0

        choices = list(self._index.keys())
        best = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
        if best and best[1] >= threshold:
            return self._tracks[self._index[best[0]]], float(best[1])

        # Retry de-accented, which rescues transliterated Latin spellings.
        folded_query = _strip_diacritics(query)
        folded = {_strip_diacritics(c): c for c in choices}
        best = process.extractOne(folded_query, list(folded.keys()), scorer=fuzz.token_set_ratio)
        if best and best[1] >= threshold:
            return self._tracks[self._index[folded[best[0]]]], float(best[1])
        return None


_source: Optional[LocalLibrarySource] = None


def get_source() -> LocalLibrarySource:
    global _source
    if _source is None:
        from app.config import get_settings
        _source = LocalLibrarySource(get_settings().music_dir)
    return _source
