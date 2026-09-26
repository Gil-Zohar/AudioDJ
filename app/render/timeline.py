"""Declarative timeline: clips placed on a shared clock, rendered in one pass.

Transitions emit Clip objects rather than mutating a buffer directly. That
separation is what lets the same code serve both an offline mix and live mode's
look-ahead rendering, where only a window of the timeline exists at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.render.engine import apply_fades, db_to_gain, mix_into


@dataclass
class Clip:
    """A piece of audio placed at a point in the mix."""

    audio: np.ndarray            # (channels, samples) at the timeline's rate
    start: float                 # seconds from the start of the mix
    gain_db: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    name: str = ""
    source_track: str = ""
    stem: str = ""

    @property
    def duration(self) -> float:
        return 0.0 if self.audio is None else self.audio.shape[-1] / max(self._rate, 1)

    _rate: int = field(default=44100, repr=False)

    def rendered(self, sample_rate: int) -> np.ndarray:
        out = np.asarray(self.audio, dtype=np.float32)
        if out.ndim == 1:
            out = out[np.newaxis, :]
        if self.fade_in > 0 or self.fade_out > 0:
            out = apply_fades(out, sample_rate, self.fade_in, self.fade_out)
        gain = db_to_gain(self.gain_db)
        return out * gain if gain != 1.0 else out


@dataclass
class TimelineEvent:
    """A labelled moment, exported to the tracklist JSON."""

    time: float
    kind: str                    # track_in, transition, hype, ...
    label: str
    detail: dict = field(default_factory=dict)


@dataclass
class Timeline:
    sample_rate: int = 44100
    channels: int = 2
    clips: list[Clip] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)

    def add(self, clip: Clip) -> Clip:
        clip._rate = self.sample_rate
        self.clips.append(clip)
        return clip

    def mark(self, time: float, kind: str, label: str, **detail) -> None:
        self.events.append(TimelineEvent(time=time, kind=kind, label=label, detail=detail))

    def truncate_at(self, time: float) -> int:
        """Cut every clip short so nothing plays past `time`.

        Used when a window of the timeline is taken out, processed and put
        back: without this the original audio stays underneath and the window
        plays twice. Returns how many clips were shortened.
        """
        cut = max(0, int(round(time * self.sample_rate)))
        trimmed = 0

        for clip in self.clips:
            if clip.audio is None or clip.audio.size == 0:
                continue
            start = int(round(clip.start * self.sample_rate))
            keep = cut - start
            if keep >= clip.audio.shape[-1] or keep < 0:
                continue          # ends before the cut, or starts after it
            clip.audio = clip.audio[..., :keep]
            # A fade-out now falls in discarded audio, so drop it rather than
            # let it fade the wrong samples.
            clip.fade_out = 0.0
            trimmed += 1

        return trimmed

    @property
    def duration(self) -> float:
        if not self.clips:
            return 0.0
        return max(
            clip.start + clip.audio.shape[-1] / self.sample_rate
            for clip in self.clips if clip.audio is not None and clip.audio.size
        )

    def render(self) -> np.ndarray:
        """Sum every clip onto one canvas."""
        total = int(np.ceil(self.duration * self.sample_rate)) + self.sample_rate
        canvas = np.zeros((self.channels, total), dtype=np.float32)

        for clip in self.clips:
            if clip.audio is None or clip.audio.size == 0:
                continue
            audio = clip.rendered(self.sample_rate)
            if audio.shape[0] == 1 and self.channels == 2:
                audio = np.repeat(audio, 2, axis=0)
            mix_into(canvas, audio, int(round(clip.start * self.sample_rate)))

        # Trim the padding added above.
        end = int(np.ceil(self.duration * self.sample_rate))
        return canvas[:, :max(end, 1)]

    def tracklist(self) -> list[dict]:
        """Ordered, timestamped events for the mix's JSON sidecar."""
        return [
            {
                "time": round(e.time, 3),
                "timecode": _timecode(e.time),
                "kind": e.kind,
                "label": e.label,
                **e.detail,
            }
            for e in sorted(self.events, key=lambda e: e.time)
        ]


def _timecode(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{secs:05.2f}"
