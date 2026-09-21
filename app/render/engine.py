"""Audio primitives for rendering: loading, stretching, pitch-shifting, slicing.

Time-stretching sits behind an interface for the same reason stems do. Rubber
Band gives markedly better quality on percussive material but ships as a
separate binary with no Windows package manager entry, so a phase-vocoder
fallback keeps the renderer working everywhere and lets quality be upgraded by
installing one executable.
"""
from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# loading / channel handling
# --------------------------------------------------------------------------

def load_audio(path: str | Path, sample_rate: int, mono: bool = False) -> np.ndarray:
    """Load a file as float32 shaped (channels, samples)."""
    import librosa

    y, _ = librosa.load(str(path), sr=sample_rate, mono=mono)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    return y


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    return audio[:2]


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def apply_fades(audio: np.ndarray, sample_rate: int, fade_in: float, fade_out: float) -> np.ndarray:
    """Equal-power fades, which hold perceived loudness steady through a blend.

    A linear crossfade dips in the middle because two uncorrelated signals sum
    in power, not amplitude.
    """
    out = audio.copy()
    total = out.shape[-1]

    n_in = min(int(fade_in * sample_rate), total)
    if n_in > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, n_in, dtype=np.float32))
        out[..., :n_in] *= ramp

    n_out = min(int(fade_out * sample_rate), total)
    if n_out > 0:
        ramp = np.cos(np.linspace(0, np.pi / 2, n_out, dtype=np.float32))
        out[..., total - n_out:] *= ramp

    return out


# --------------------------------------------------------------------------
# time stretch / pitch shift
# --------------------------------------------------------------------------

class TimeStretcher(ABC):
    name: str = "base"

    @abstractmethod
    def stretch(self, audio: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
        """`rate` > 1 makes the audio shorter (faster)."""

    @abstractmethod
    def pitch_shift(self, audio: np.ndarray, semitones: float, sample_rate: int) -> np.ndarray:
        ...


class RubberbandStretcher(TimeStretcher):
    """Best quality; needs the `rubberband` executable on PATH."""

    name = "rubberband"

    @staticmethod
    def available() -> bool:
        return shutil.which("rubberband") is not None

    def stretch(self, audio: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
        if abs(rate - 1.0) < 1e-4:
            return audio
        import pyrubberband as prb

        # pyrubberband wants (samples, channels).
        out = prb.time_stretch(audio.T, sample_rate, rate)
        return np.asarray(out, dtype=np.float32).T

    def pitch_shift(self, audio: np.ndarray, semitones: float, sample_rate: int) -> np.ndarray:
        if abs(semitones) < 1e-4:
            return audio
        import pyrubberband as prb

        out = prb.pitch_shift(audio.T, sample_rate, semitones)
        return np.asarray(out, dtype=np.float32).T


class LibrosaStretcher(TimeStretcher):
    """Phase-vocoder fallback. Pure Python, no external binary.

    Audibly smears transients compared with Rubber Band -- drums lose some snap
    at larger ratios -- but it is correct, dependency-free and fine for the
    small stretches (<8%) that beat-matching usually needs.
    """

    name = "librosa"

    def stretch(self, audio: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
        if abs(rate - 1.0) < 1e-4:
            return audio
        import librosa

        channels = [librosa.effects.time_stretch(np.ascontiguousarray(ch), rate=rate)
                    for ch in audio]
        return _stack_equal_length(channels)

    def pitch_shift(self, audio: np.ndarray, semitones: float, sample_rate: int) -> np.ndarray:
        if abs(semitones) < 1e-4:
            return audio
        import librosa

        channels = [
            librosa.effects.pitch_shift(np.ascontiguousarray(ch), sr=sample_rate, n_steps=semitones)
            for ch in audio
        ]
        return _stack_equal_length(channels)


def _stack_equal_length(channels: list[np.ndarray]) -> np.ndarray:
    """Stack channels, trimming to the shortest (they can differ by a sample)."""
    shortest = min(ch.shape[-1] for ch in channels)
    return np.stack([ch[:shortest] for ch in channels]).astype(np.float32)


def get_stretcher(preference: str = "auto") -> TimeStretcher:
    if preference == "rubberband":
        if not RubberbandStretcher.available():
            raise RuntimeError(
                "time_stretcher is 'rubberband' but the rubberband executable is not on PATH. "
                "Install it from https://breakfastquay.com/rubberband/ or set "
                "render.time_stretcher: librosa in config.yaml."
            )
        return RubberbandStretcher()
    if preference == "librosa":
        return LibrosaStretcher()
    return RubberbandStretcher() if RubberbandStretcher.available() else LibrosaStretcher()


# --------------------------------------------------------------------------
# bar-aligned slicing
# --------------------------------------------------------------------------

def bar_grid(analysis, fallback_bpm: float | None = None) -> np.ndarray:
    """Downbeat times, falling back to a synthesised grid if none were found."""
    if analysis.downbeats:
        return np.asarray(analysis.downbeats, dtype=float)
    if analysis.beats:
        beats = np.asarray(analysis.beats, dtype=float)
        return beats[::4]
    bpm = fallback_bpm or analysis.bpm or 120.0
    bar = 4 * 60.0 / bpm
    return np.arange(0.0, max(analysis.duration, bar), bar)


def snap_to_bar(analysis, time: float, prefer: str = "nearest") -> float:
    grid = bar_grid(analysis)
    if grid.size == 0:
        return time
    if prefer == "next":
        after = grid[grid >= time - 1e-6]
        return float(after[0]) if after.size else float(grid[-1])
    if prefer == "prev":
        before = grid[grid <= time + 1e-6]
        return float(before[-1]) if before.size else float(grid[0])
    return float(grid[int(np.argmin(np.abs(grid - time)))])


def latest_start_for_bars(analysis, n_bars: int) -> float:
    """Latest downbeat that still leaves `n_bars` of audio after it.

    Section choice is musical, but a chorus starting 20 bars from the end cannot
    supply 32 bars. Rather than silently returning a short slice, callers clamp
    the start so the requested length is actually delivered.
    """
    grid = bar_grid(analysis)
    if grid.size <= 1:
        return 0.0
    index = max(0, len(grid) - 1 - n_bars)
    return float(grid[index])


def slice_bars(
    audio: np.ndarray,
    analysis,
    start_time: float,
    n_bars: int,
    sample_rate: int,
) -> tuple[np.ndarray, float, int]:
    """Cut `n_bars` starting at the downbeat at or after `start_time`.

    Returns (audio, actual start, bars obtained). Slicing on the bar grid rather
    than on raw seconds is what keeps layered tracks phase-locked.
    """
    grid = bar_grid(analysis)
    start = snap_to_bar(analysis, start_time, prefer="next")

    index = int(np.searchsorted(grid, start - 1e-6))
    end_index = index + n_bars
    if end_index < len(grid):
        end = float(grid[end_index])
        bars = n_bars
    else:
        # Not enough bars left: take what there is and report it honestly.
        bars = max(0, len(grid) - 1 - index)
        if bars <= 0:
            bar_seconds = 4 * 60.0 / max(analysis.bpm, 1e-6)
            end = start + n_bars * bar_seconds
            bars = n_bars
        else:
            end = float(grid[index + bars])

    first = int(round(start * sample_rate))
    last = int(round(end * sample_rate))
    first = max(0, min(first, audio.shape[-1]))
    last = max(first, min(last, audio.shape[-1]))
    return audio[..., first:last], start, bars


def normalize_peak(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    """Scale so the loudest sample sits at `headroom_db`, avoiding clipping."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio
    target = db_to_gain(headroom_db)
    return (audio * (target / peak)).astype(np.float32)


def mix_into(canvas: np.ndarray, clip: np.ndarray, start_sample: int) -> None:
    """Sum `clip` into `canvas` at `start_sample`, in place, clipped to bounds."""
    if clip.size == 0:
        return
    channels = min(canvas.shape[0], clip.shape[0])
    start = max(0, start_sample)
    length = min(clip.shape[-1], canvas.shape[-1] - start)
    if length <= 0:
        return
    offset = start - start_sample
    canvas[:channels, start:start + length] += clip[:channels, offset:offset + length]
