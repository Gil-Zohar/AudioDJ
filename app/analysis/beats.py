"""Beat and downbeat tracking behind a swappable interface.

librosa is the default because madmom (0.16.1, 2018) is source-only and will not
build on Python 3.11 + numpy 2.x -- it needs np.float, collections.MutableSequence
and Cython<3. MadmomBeatTracker is kept ready so a 3.10 venv with madmom installed
upgrades downbeat accuracy without any other code change; `beat_tracker: auto`
picks it up automatically when the import succeeds.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeatResult:
    bpm: float
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    tracker: str = "librosa"


class BeatTracker(ABC):
    name: str = "base"

    @abstractmethod
    def track(self, y: np.ndarray, sr: int, hop_length: int, beats_per_bar: int) -> BeatResult:
        ...


def _low_band_energy(y: np.ndarray, sr: int, hop_length: int, cutoff_hz: float = 150.0) -> np.ndarray:
    """Per-frame energy below `cutoff_hz` -- a decent kick-drum proxy."""
    import librosa

    spec = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2 * (spec.shape[0] - 1))
    band = spec[freqs <= cutoff_hz, :]
    return band.sum(axis=0) if band.size else np.zeros(spec.shape[1])


def _harmonic_novelty(y: np.ndarray, sr: int, hop_length: int, frames: np.ndarray) -> np.ndarray:
    """Per-beat harmonic change -- how much the chroma differs from the beat before.

    Chord changes overwhelmingly land on bar one, which is the strongest cue
    available when every beat carries an equally loud kick.
    """
    import librosa

    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    frames = np.clip(frames, 0, chroma.shape[1] - 1)
    at_beats = chroma[:, frames]

    novelty = np.zeros(at_beats.shape[1])
    for i in range(1, at_beats.shape[1]):
        a, b = at_beats[:, i], at_beats[:, i - 1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        novelty[i] = 1.0 - float(np.dot(a, b)) / denom if denom > 0 else 0.0
    return novelty


def refine_beat_grid(beat_times: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a regular grid through the detected beats.

    Beat frames are quantised to the analysis hop (23ms at sr=22050/hop=512),
    which alone puts the tempo out by around 1% and lets the grid drift by half a
    beat over a few minutes -- fatal when aligning downbeats between tracks.
    Least-squares through beat index vs. time averages that quantisation away.
    Only applied when the beats are already regular; a track with dropped or
    doubled beats keeps its raw grid.
    """
    if beat_times.size < 8:
        return beat_times, 0.0

    diffs = np.diff(beat_times)
    median_ibi = float(np.median(diffs))
    if median_ibi <= 0:
        return beat_times, 0.0

    # Irregular beat spacing means the index->time mapping is not a straight
    # line, so regression would make things worse.
    if float(np.std(diffs)) > 0.12 * median_ibi:
        return beat_times, median_ibi

    idx = np.arange(beat_times.size, dtype=float)
    design = np.vstack([idx, np.ones_like(idx)]).T
    period, offset = np.linalg.lstsq(design, beat_times, rcond=None)[0]

    residual = beat_times - (period * idx + offset)
    inliers = np.abs(residual) < 0.5 * period
    if inliers.sum() > 8:
        period, offset = np.linalg.lstsq(design[inliers], beat_times[inliers], rcond=None)[0]

    if period <= 0:
        return beat_times, median_ibi
    return period * idx + offset, float(period)


def infer_downbeats(
    beat_times: np.ndarray,
    onset_env: np.ndarray,
    low_energy: np.ndarray,
    sr: int,
    hop_length: int,
    beats_per_bar: int = 4,
    harmonic_novelty: np.ndarray | None = None,
) -> list[float]:
    """Pick the bar phase whose beats carry the most accent.

    Without a trained downbeat model we combine three cues that all peak on bar
    one: onset strength, low-band (kick) energy, and harmonic change. The
    harmonic term carries the decision on four-on-the-floor material, where every
    beat has an identical kick and the first two cues are flat.
    """
    import librosa

    if len(beat_times) == 0:
        return []

    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    frames = np.clip(frames, 0, min(len(onset_env), len(low_energy)) - 1)

    def norm(a: np.ndarray) -> np.ndarray:
        peak = float(np.max(a)) if a.size else 0.0
        return a / peak if peak > 0 else a

    accent = norm(onset_env[frames]) + norm(low_energy[frames])
    if harmonic_novelty is not None and harmonic_novelty.size == len(beat_times):
        accent = accent + 2.0 * norm(harmonic_novelty)

    best_phase, best_score = 0, -np.inf
    for phase in range(beats_per_bar):
        idx = np.arange(phase, len(beat_times), beats_per_bar)
        if idx.size == 0:
            continue
        score = float(accent[idx].mean())
        if score > best_score:
            best_phase, best_score = phase, score

    return [float(beat_times[i]) for i in range(best_phase, len(beat_times), beats_per_bar)]


class LibrosaBeatTracker(BeatTracker):
    name = "librosa"

    def track(self, y: np.ndarray, sr: int, hop_length: int, beats_per_bar: int) -> BeatResult:
        import librosa

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length, trim=False
        )
        bpm = float(np.atleast_1d(tempo)[0])
        raw_beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        beat_times, period = refine_beat_grid(raw_beats)
        if period > 0:
            bpm = 60.0 / period

        low = _low_band_energy(y, sr, hop_length)
        novelty = _harmonic_novelty(y, sr, hop_length, beat_frames)
        downbeats = infer_downbeats(
            beat_times, onset_env, low, sr, hop_length, beats_per_bar, novelty
        )

        return BeatResult(bpm=bpm, beats=[float(t) for t in beat_times],
                          downbeats=downbeats, tracker=self.name)


class MadmomBeatTracker(BeatTracker):
    """Uses madmom's RNN + DBN downbeat tracker when it is importable."""

    name = "madmom"

    @staticmethod
    def available() -> bool:
        try:
            import madmom  # noqa: F401
            return True
        except Exception:
            return False

    def track(self, y: np.ndarray, sr: int, hop_length: int, beats_per_bar: int) -> BeatResult:
        import soundfile as sf
        import tempfile
        from pathlib import Path
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )

        # madmom's processors read files, so hand it a temporary wav.
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "in.wav"
            sf.write(wav, y, sr)
            activations = RNNDownBeatProcessor()(str(wav))
            proc = DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar], fps=100)
            result = proc(activations)

        beats = [float(t) for t, _ in result]
        downbeats = [float(t) for t, pos in result if int(pos) == 1]
        bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 2 else 0.0
        return BeatResult(bpm=bpm, beats=beats, downbeats=downbeats, tracker=self.name)


def get_beat_tracker(name: str = "auto") -> BeatTracker:
    if name == "madmom":
        return MadmomBeatTracker()
    if name == "librosa":
        return LibrosaBeatTracker()
    # auto
    if MadmomBeatTracker.available():
        return MadmomBeatTracker()
    return LibrosaBeatTracker()
