"""Domain models shared by analysis, matching, rendering and the API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SectionLabel = Literal["intro", "verse", "chorus", "drop", "bridge", "outro"]

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class TrackMeta(BaseModel):
    """A file in the local library."""

    id: str                      # stable hash of path + size + mtime
    path: str
    title: str
    artist: str
    album: str = ""
    duration: float = 0.0
    file_hash: str = ""          # content-derived; keys the analysis cache


class Section(BaseModel):
    """One structural segment of a track."""

    index: int
    start: float
    end: float
    label: SectionLabel
    energy: float                # 0..1, mean of the energy curve over the section
    chroma: list[float] = Field(default_factory=list)   # 12-dim, sums to 1
    start_beat: int = 0
    start_downbeat: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


class KeyEstimate(BaseModel):
    pitch_class: int             # 0 = C
    mode: Literal["major", "minor"]
    confidence: float            # 0..1, margin between best and runner-up
    camelot: str
    source: Literal["detected", "manual"] = "detected"

    @property
    def name(self) -> str:
        return f"{PITCH_NAMES[self.pitch_class]} {self.mode}"


class StemSet(BaseModel):
    provider: str
    vocals: str
    drums: str
    bass: str
    other: str


class Analysis(BaseModel):
    """Everything we cache about one track."""

    track_id: str
    file_hash: str
    version: int
    duration: float
    sample_rate: int
    bpm: float
    beat_tracker: str
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    key: KeyEstimate
    key_detected: Optional[KeyEstimate] = None   # kept when a manual override is applied
    sections: list[Section] = Field(default_factory=list)
    energy_times: list[float] = Field(default_factory=list)
    energy_curve: list[float] = Field(default_factory=list)
    stems: Optional[StemSet] = None

    @property
    def beat_duration(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.0
