"""AudioSource interface.

An AudioSource is the ONLY way AutoDJ obtains audio. Trend providers supply
metadata (title/artist/rank) and never audio; the resolver then asks an
AudioSource whether it holds a matching file. Implementations must read from
media the user already possesses -- AutoDJ does not download, scrape or rip
from streaming services.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.models import TrackMeta


class AudioSource(ABC):
    name: str = "base"

    @abstractmethod
    def scan(self, force: bool = False) -> list[TrackMeta]:
        """Return every track this source can provide."""

    @abstractmethod
    def get(self, track_id: str) -> Optional[TrackMeta]:
        """Look up a single track by id."""

    @abstractmethod
    def resolve(self, title: str, artist: str, threshold: int) -> Optional[tuple[TrackMeta, float]]:
        """Fuzzy-match external metadata to a held track.

        Returns (track, score 0..100) or None when nothing clears `threshold`.
        """
