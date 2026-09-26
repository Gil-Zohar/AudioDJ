"""Production effects: the things that make a mix sound made rather than joined.

Four techniques do most of the work in electronic DJ sets:

  sidechain     rhythmic ducking against the kick -- the "pump" that reads as
                electronic more than any other single effect
  filter build  a high-pass climbing over several bars, holding tension
  delay throw   a tempo-synced echo on the last beat of a phrase
  atmosphere    a sustained pad in the key, spanning a transition

The pad matters for more than mood. Two tracks can share a tempo and a key and
still sound like an abrupt stylistic jump, because nothing connects their
timbres. A pad held across the join gives the ear something continuous to
follow, which is exactly what a DJ reaches for when blending unlike records.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

log = logging.getLogger("autodj.fx")

Style = Literal["clean", "club", "atmospheric"]

# What each style turns on. Kept here rather than scattered through the
# renderer so a style is one readable thing.
STYLES: dict[str, dict] = {
    "clean": {
        "sidechain": 0.0, "pad_gain_db": None, "filter_build_bars": 0,
        "delay_throw": False, "wash": 0.0,
    },
    "club": {
        "sidechain": 0.45, "pad_gain_db": None, "filter_build_bars": 4,
        "delay_throw": True, "wash": 0.0,
    },
    "atmospheric": {
        "sidechain": 0.30, "pad_gain_db": -19.0, "filter_build_bars": 8,
        "delay_throw": True, "wash": 0.18,
    },
}

# Chord voicings as semitone offsets from the root. Minor gets a flat third,
# and both keep the fifth, which is what makes a pad sit under anything.
VOICINGS = {
    "minor": (0, 3, 7, 12, 19),
    "major": (0, 4, 7, 12, 19),
}


def _as_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# sidechain
# ---------------------------------------------------------------------------

def sidechain_envelope(
    n: int, sr: int, bpm: float, depth: float = 0.45, recovery: float = 0.62
) -> np.ndarray:
    """A ducking envelope that dips on every beat and recovers before the next.

    Modelled rather than triggered from a real kick: the beat grid is already
    known and exact, whereas detecting kicks in a full mix is unreliable and
    would duck on the wrong hits.
    """
    if bpm <= 0 or n <= 0:
        return np.ones(max(n, 0), dtype=np.float32)

    beat = 60.0 / bpm
    samples_per_beat = max(1, int(beat * sr))
    depth = float(np.clip(depth, 0.0, 0.95))

    phase = (np.arange(n, dtype=np.float32) % samples_per_beat) / samples_per_beat
    # Fast dip, exponential recovery: the shape a compressor actually produces.
    shape = 1.0 - np.exp(-phase / max(recovery * 0.25, 1e-4))
    return (1.0 - depth + depth * shape).astype(np.float32)


def sidechain(audio: np.ndarray, sr: int, bpm: float, depth: float = 0.45,
              offset: float = 0.0) -> np.ndarray:
    """Apply the pump. `offset` in seconds aligns the dips to the downbeat."""
    audio = _as_stereo(audio)
    if depth <= 0 or audio.shape[-1] == 0:
        return audio

    n = audio.shape[-1]
    lead = int(max(0.0, offset) * sr)
    envelope = sidechain_envelope(n + lead, sr, bpm, depth)[lead:lead + n]
    return (audio * envelope).astype(np.float32)


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def moving_filter(audio: np.ndarray, sr: int, start_hz: float, end_hz: float,
                  kind: str = "high", block: int = 2048) -> np.ndarray:
    """Filter whose cutoff glides, applied in blocks at stepped cutoffs."""
    from scipy.signal import butter, sosfilt

    audio = _as_stereo(audio)
    n = audio.shape[-1]
    if n == 0:
        return audio

    out = np.zeros_like(audio, dtype=np.float32)
    positions = list(range(0, n, block))
    nyquist = sr / 2.0

    for index, begin in enumerate(positions):
        stop = min(begin + block, n)
        fraction = index / max(len(positions) - 1, 1)
        cutoff = float(start_hz * (end_hz / start_hz) ** fraction)
        cutoff = float(np.clip(cutoff, 25.0, nyquist * 0.95))
        sos = butter(2, cutoff / nyquist, btype=kind, output="sos")
        out[..., begin:stop] = sosfilt(sos, audio[..., begin:stop])

    return out


def filter_build(audio: np.ndarray, sr: int, bpm: float, bars: int = 8,
                 top_hz: float = 1800.0) -> np.ndarray:
    """High-pass climbing over the last `bars`, so tension resolves at the end.

    Only the tail is processed: filtering the whole segment would thin out
    material that should be sitting full-range.
    """
    audio = _as_stereo(audio)
    bar_seconds = 4 * 60.0 / max(bpm, 1e-6)
    tail = min(int(bars * bar_seconds * sr), audio.shape[-1])
    if tail <= 0 or bars <= 0:
        return audio

    out = audio.copy()
    out[..., -tail:] = moving_filter(audio[..., -tail:], sr, 30.0, top_hz, "high")
    return out


# ---------------------------------------------------------------------------
# delay and reverb
# ---------------------------------------------------------------------------

def delay_throw(audio: np.ndarray, sr: int, bpm: float, beats: float = 0.75,
                feedback: float = 0.5, mix: float = 0.4,
                tail_beats: float = 2.0) -> np.ndarray:
    """Tempo-synced echo thrown on the last beats of a phrase."""
    from pedalboard import Delay, Pedalboard

    audio = _as_stereo(audio)
    beat = 60.0 / max(bpm, 1e-6)
    tail = min(int(tail_beats * beat * sr), audio.shape[-1])
    if tail <= 0:
        return audio

    board = Pedalboard([Delay(delay_seconds=beats * beat, feedback=feedback, mix=mix)])
    out = audio.copy()
    out[..., -tail:] = np.asarray(board(audio[..., -tail:], sr), dtype=np.float32)
    return out


def wash(audio: np.ndarray, sr: int, room_size: float = 0.7,
         wet: float = 0.18) -> np.ndarray:
    """A light reverb send over the whole segment, for glue."""
    from pedalboard import Pedalboard, Reverb

    audio = _as_stereo(audio)
    if wet <= 0:
        return audio
    board = Pedalboard([Reverb(room_size=room_size, wet_level=wet,
                               dry_level=1.0 - wet * 0.5, width=0.9)])
    return np.asarray(board(audio, sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# atmosphere
# ---------------------------------------------------------------------------

def atmosphere_pad(
    seconds: float,
    sr: int,
    pitch_class: int = 9,
    mode: str = "minor",
    bpm: float = 128.0,
    seed: int = 0,
) -> np.ndarray:
    """A slow, wide pad in the given key.

    Built from lightly detuned partials on the chord tones, low-passed and
    drowned in reverb. It exists to be *felt* rather than heard, so it sits
    around -19dB and fades in and out over bars, never appearing abruptly.
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float32) / sr

    root_hz = 55.0 * (2.0 ** (((pitch_class - 9) % 12) / 12.0))   # A1 reference
    intervals = VOICINGS.get(mode, VOICINGS["minor"])

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    for index, semitones in enumerate(intervals):
        freq = root_hz * (2.0 ** (semitones / 12.0)) * 2.0
        gain = 0.9 / (1.0 + index * 0.55)
        # Slight detune per voice, and a slow drift, so it breathes.
        for detune, pan in ((1.0 - 0.0022, -1.0), (1.0 + 0.0026, 1.0)):
            drift = 1.0 + 0.0015 * np.sin(2 * np.pi * (0.05 + 0.02 * index) * t
                                          + rng.random() * 6.28)
            voice = np.sin(2 * np.pi * freq * detune * drift * t).astype(np.float32)
            voice += 0.28 * np.sin(4 * np.pi * freq * detune * drift * t).astype(np.float32)
            voice *= gain
            if pan < 0:
                left += voice
            else:
                right += voice

    pad = np.vstack([left, right]).astype(np.float32)

    # Soften the top so it never competes with vocals or hats.
    pad = moving_filter(pad, sr, 900.0, 900.0, "low")

    # Fade over whole bars: a pad that appears suddenly announces itself.
    bar = 4 * 60.0 / max(bpm, 1e-6)
    fade = min(int(bar * 2 * sr), n // 2)
    if fade > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float32)) ** 2
        pad[..., :fade] *= ramp
        pad[..., -fade:] *= ramp[::-1]

    pad = wash(pad, sr, room_size=0.9, wet=0.45)

    peak = float(np.max(np.abs(pad))) if pad.size else 0.0
    if peak > 0:
        pad = pad / peak * 0.85
    return pad.astype(np.float32)


# ---------------------------------------------------------------------------
# style resolution
# ---------------------------------------------------------------------------

@dataclass
class FxSettings:
    style: str
    sidechain: float
    pad_gain_db: float | None
    filter_build_bars: int
    delay_throw: bool
    wash: float

    @property
    def pads_enabled(self) -> bool:
        return self.pad_gain_db is not None


def resolve_style(style: str, overrides: dict | None = None) -> FxSettings:
    """Turn a style name (plus any config overrides) into concrete settings."""
    base = dict(STYLES.get(style, STYLES["clean"]))
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
    return FxSettings(
        style=style,
        sidechain=float(base.get("sidechain", 0.0)),
        pad_gain_db=base.get("pad_gain_db"),
        filter_build_bars=int(base.get("filter_build_bars", 0)),
        delay_throw=bool(base.get("delay_throw", False)),
        wash=float(base.get("wash", 0.0)),
    )
