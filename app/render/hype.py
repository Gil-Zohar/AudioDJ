"""Hype layer: risers, impacts, airhorns and shouts placed before drops.

Placement is the whole game. A riser dropped anywhere sounds like a mistake; a
riser that *lands* on a downbeat at the start of a chorus sounds deliberate. So
every element is anchored to a cue the analysis already found -- a drop, a
chorus start, a transition -- and snapped to the bar grid.

Samples come from ./samples. If that folder is empty, usable defaults are
synthesised so the feature works immediately; drop real one-shots in and they
take precedence.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app.render.engine import db_to_gain, to_stereo
from app.render.timeline import Clip, Timeline

log = logging.getLogger("autodj.hype")

KINDS = ("riser", "impact", "airhorn", "shout", "sweep")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".ogg", ".m4a"}

# How much gets placed, per density setting.
DENSITY = {
    "off":   {"drop": 0.0, "chorus": 0.0, "transition": 0.0, "min_gap": 999.0},
    "light": {"drop": 1.0, "chorus": 0.35, "transition": 0.25, "min_gap": 30.0},
    "heavy": {"drop": 1.0, "chorus": 1.0, "transition": 0.8, "min_gap": 10.0},
}


@dataclass
class HypeSample:
    name: str
    kind: str
    audio: np.ndarray          # (2, n) at the render rate

    @property
    def seconds(self) -> float:
        return self.audio.shape[-1] / 44100.0


# ---------------------------------------------------------------------------
# synthesis -- so the feature works before you own any samples
# ---------------------------------------------------------------------------

def _envelope(n: int, attack: float, release: float, sr: int) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a = min(int(attack * sr), n)
    r = min(int(release * sr), n - a) if n > a else 0
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[n - r:] = np.linspace(1.0, 0.0, r) ** 1.6
    return env


def _moving_filter(
    signal: np.ndarray, sr: int, start_hz: float, end_hz: float, kind: str = "band"
) -> np.ndarray:
    """Filter with a cutoff that glides from `start_hz` to `end_hz`.

    Done in short blocks at stepped cutoffs. At ~23ms a block the steps are
    inaudible, and it is far cheaper than a true time-varying filter.
    """
    from scipy.signal import butter, sosfilt

    n = signal.shape[-1]
    if n == 0:
        return signal

    block = 1024
    out = np.zeros_like(signal, dtype=np.float32)
    positions = list(range(0, n, block))
    nyquist = sr / 2.0

    for index, begin in enumerate(positions):
        stop = min(begin + block, n)
        fraction = index / max(len(positions) - 1, 1)
        # Glide geometrically: filter sweeps are heard logarithmically.
        cutoff = float(start_hz * (end_hz / start_hz) ** fraction)
        cutoff = float(np.clip(cutoff, 30.0, nyquist * 0.95))

        if kind == "band":
            low = np.clip(cutoff * 0.55, 20.0, nyquist * 0.9) / nyquist
            high = np.clip(cutoff * 1.8, 40.0, nyquist * 0.95) / nyquist
            sos = butter(2, [low, min(high, 0.99)], btype="band", output="sos")
        elif kind == "high":
            sos = butter(2, cutoff / nyquist, btype="high", output="sos")
        else:
            sos = butter(2, cutoff / nyquist, btype="low", output="sos")
        out[..., begin:stop] = sosfilt(sos, signal[..., begin:stop])

    return out


def make_riser(sr: int, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Noise and a tone sweeping upward, getting louder. The classic build.

    The sweep is geometric because pitch is perceived logarithmically -- a
    linear ramp sounds like it stalls at the top. The noise is genuinely
    band-passed by a rising filter; simply ramping its volume leaves the
    spectrum flat and the build does not read as rising at all.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)

    noise = _moving_filter(
        rng.standard_normal(n).astype(np.float32), sr, 300.0, 9000.0, "band"
    )
    noise *= (0.25 + 0.75 * t) ** 1.5

    tone_hz = 220.0 * (12.0 ** t)
    tone = np.sin(2 * np.pi * np.cumsum(tone_hz) / sr).astype(np.float32) * 0.45 * (t ** 1.5)

    # Tremolo that accelerates adds urgency toward the drop.
    tremolo = 0.85 + 0.15 * np.sin(2 * np.pi * np.cumsum(4.0 + 22.0 * t) / sr)

    out = (noise * 0.9 + tone) * tremolo * (0.2 + 0.8 * t)
    out *= _envelope(n, 0.01, 0.015, sr)
    return _stereo(out.astype(np.float32))


def make_impact(sr: int, seconds: float = 1.6, seed: int = 1) -> np.ndarray:
    """A low boom with a short crack: what lands ON the downbeat of a drop."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr

    sweep = 130.0 * np.exp(-t * 9.0) + 36.0
    boom = np.sin(2 * np.pi * np.cumsum(sweep) / sr).astype(np.float32) * np.exp(-t * 2.6)
    # Crack decays fast so the impact darkens as it rings out, as a real one does.
    crack = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 30.0) * 0.28
    crack = _moving_filter(crack, sr, 6000.0, 1200.0, "high")

    out = boom * 1.0 + crack
    return _stereo(out.astype(np.float32))


def make_airhorn(sr: int, seconds: float = 1.1, seed: int = 2) -> np.ndarray:
    """Stacked detuned saws with a quick scoop up to pitch."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr

    base = 466.0
    scoop = 1.0 - 0.14 * np.exp(-t * 26.0)          # slides up into the note
    out = np.zeros(n, dtype=np.float32)

    for detune, gain in ((1.0, 1.0), (1.005, 0.8), (0.994, 0.8), (2.0, 0.35), (3.0, 0.2)):
        phase = 2 * np.pi * np.cumsum(base * detune * scoop) / sr
        # Sawtooth from its phase: brighter and more cutting than a sine.
        out += (2.0 * (phase / (2 * np.pi) % 1.0) - 1.0).astype(np.float32) * gain

    out *= _envelope(n, 0.012, 0.18, sr) / 3.2
    return _stereo(out)


def make_sweep(sr: int, seconds: float = 1.8, seed: int = 3) -> np.ndarray:
    """A downward noise sweep, for coming out of a drop rather than into one."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)

    noise = _moving_filter(
        rng.standard_normal(n).astype(np.float32), sr, 9000.0, 350.0, "band"
    )
    tone_hz = 2600.0 * (0.08 ** t)
    tone = np.sin(2 * np.pi * np.cumsum(tone_hz) / sr).astype(np.float32) * 0.4

    out = (noise * 0.9 + tone) * (1.0 - t) ** 1.2
    out *= _envelope(n, 0.01, 0.05, sr)
    return _stereo(out.astype(np.float32))


def _stereo(mono: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 0:
        mono = mono / peak * 0.9
    return np.repeat(mono[np.newaxis, :].astype(np.float32), 2, axis=0)


GENERATORS = {
    "riser": make_riser,
    "impact": make_impact,
    "airhorn": make_airhorn,
    "sweep": make_sweep,
}


def generate_default_samples(out_dir: Path, sample_rate: int = 44100) -> dict[str, Path]:
    """Write synthesised one-shots so the hype layer works out of the box.

    Deliberately kept out of ./samples: that folder belongs to the user, and
    mixing generated files into it would make it unclear which is which.
    """
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for kind, generator in GENERATORS.items():
        path = out_dir / f"{kind}_default.wav"
        if not path.exists():
            sf.write(path, generator(sample_rate).T, sample_rate)
        written[kind] = path
    return written


# ---------------------------------------------------------------------------
# the sample library
# ---------------------------------------------------------------------------

def classify(filename: str) -> Optional[str]:
    """Work out what a sample is from its name."""
    stem = Path(filename).stem.lower()
    for kind in KINDS:
        if re.match(rf"^{kind}[_\-\s0-9]*", stem) or kind in stem:
            return kind
    # Common synonyms people actually name files with.
    for alias, kind in (("build", "riser"), ("uplifter", "riser"), ("up", "riser"),
                        ("hit", "impact"), ("boom", "impact"), ("downlifter", "sweep"),
                        ("horn", "airhorn"), ("vox", "shout"), ("voice", "shout")):
        if alias in stem:
            return kind
    return None


@dataclass
class SampleLibrary:
    sample_rate: int = 44100
    by_kind: dict[str, list[HypeSample]] = field(default_factory=dict)
    generated: bool = False

    def kinds(self) -> list[str]:
        return [k for k, v in self.by_kind.items() if v]

    def pick(self, kind: str, rng) -> Optional[HypeSample]:
        pool = self.by_kind.get(kind) or []
        return rng.choice(pool) if pool else None

    def count(self) -> int:
        return sum(len(v) for v in self.by_kind.values())


def load_samples(
    samples_dir: Path,
    fallback_dir: Path,
    sample_rate: int = 44100,
) -> SampleLibrary:
    """Load user samples, filling any gaps with generated defaults."""
    from app.render.engine import load_audio

    library = SampleLibrary(sample_rate=sample_rate, by_kind={k: [] for k in KINDS})

    if samples_dir.exists():
        for path in sorted(samples_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            kind = classify(path.name)
            if kind is None:
                log.info("ignoring sample with unrecognised name: %s", path.name)
                continue
            try:
                audio = to_stereo(load_audio(path, sample_rate, mono=False))
            except Exception as exc:  # noqa: BLE001 - a bad file must not stop the mix
                log.warning("could not load sample %s: %s", path, exc)
                continue
            library.by_kind[kind].append(HypeSample(path.stem, kind, audio))

    missing = [k for k in GENERATORS if not library.by_kind.get(k)]
    if missing:
        library.generated = True
        for kind, path in generate_default_samples(fallback_dir, sample_rate).items():
            if kind in missing:
                audio = to_stereo(load_audio(path, sample_rate, mono=False))
                library.by_kind[kind].append(HypeSample(path.stem, kind, audio))

    return library


# ---------------------------------------------------------------------------
# text to speech (interface only; no provider ships by default)
# ---------------------------------------------------------------------------

class TtsProvider(ABC):
    """Speaks a phrase for the hype layer.

    Hebrew and English tags are the point here, so any real implementation must
    handle both. Nothing ships by default: a decent Hebrew voice means either a
    large local model or a network call at render time, and neither belongs in
    the core path uninvited.
    """

    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def say(self, text: str, language: str = "he") -> Optional[np.ndarray]:
        """Return (2, n) audio at the render rate, or None."""


class NullTtsProvider(TtsProvider):
    name = "none"

    def available(self) -> bool:
        return False

    def say(self, text: str, language: str = "he") -> Optional[np.ndarray]:
        return None


def get_tts_provider(name: str = "none") -> TtsProvider:
    if name in ("", "none", None):
        return NullTtsProvider()
    log.warning("unknown tts_provider %r; falling back to none", name)
    return NullTtsProvider()


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------

@dataclass
class Cue:
    time: float
    kind: str              # drop | chorus | transition
    label: str = ""


def collect_cues(timeline: Timeline) -> list[Cue]:
    """Pull the moments worth marking out of the timeline's own events."""
    cues: list[Cue] = []
    for event in timeline.events:
        if event.kind == "cue":
            cues.append(Cue(event.time, event.label or "chorus", event.label))
        elif event.kind == "transition":
            cues.append(Cue(event.time, "transition", event.label))
    cues.sort(key=lambda c: c.time)
    return cues


def apply_hype(
    timeline: Timeline,
    bpm: float,
    density: str = "light",
    tuning=None,
    seed: Optional[int] = None,
) -> int:
    """Place hype elements against the timeline's cues. Returns how many landed.

    A riser is placed so it *ends* on the cue, and an impact lands exactly on
    it. Anything else and the build resolves in the wrong place, which is worse
    than no riser at all.
    """
    import random

    from app.config import get_settings, get_tuning

    tuning = tuning or get_tuning()
    settings = get_settings()

    if density == "off":
        return 0

    rules = DENSITY.get(density, DENSITY["light"])
    rng = random.Random(seed if seed is not None else 1234)

    library = load_samples(
        Path("samples"), settings.cache_dir / "hype_samples", timeline.sample_rate
    )
    if library.count() == 0:
        log.warning("no hype samples available")
        return 0
    if library.generated:
        log.info("using generated default hype samples; add your own to ./samples")

    bar_seconds = 4 * 60.0 / max(bpm, 1e-6)
    beat_seconds = bar_seconds / 4.0
    lead = float(tuning.hype.lead_beats) * beat_seconds
    gain = tuning.hype.gain_db

    cues = collect_cues(timeline)
    placed = 0
    last_at = -1e9

    for cue in cues:
        probability = rules.get(cue.kind, 0.0)
        if probability <= 0 or rng.random() > probability:
            continue
        if cue.time - last_at < rules["min_gap"]:
            continue          # do not stack hype on top of itself
        if cue.time < bar_seconds:
            continue          # nothing to build into at the very start

        added_here = 0

        # A riser has to finish exactly on the cue to make sense of it.
        riser = library.pick("riser", rng)
        if riser is not None:
            length = riser.audio.shape[-1] / timeline.sample_rate
            start = cue.time - min(length, max(lead, bar_seconds))
            if start > 0:
                trimmed = riser.audio
                wanted = int((cue.time - start) * timeline.sample_rate)
                if wanted < trimmed.shape[-1]:
                    trimmed = trimmed[..., -wanted:]   # keep the top of the build
                timeline.add(Clip(
                    audio=trimmed, start=start, gain_db=gain,
                    fade_in=0.05, fade_out=0.02,
                    name=f"hype riser -> {cue.kind}", stem="hype",
                ))
                added_here += 1

        if cue.kind in ("drop", "transition"):
            impact = library.pick("impact", rng)
            if impact is not None:
                timeline.add(Clip(
                    audio=impact.audio, start=cue.time, gain_db=gain - 1.0,
                    name=f"hype impact on {cue.kind}", stem="hype",
                ))
                added_here += 1

        if density == "heavy" and cue.kind == "drop":
            airhorn = library.pick("airhorn", rng)
            if airhorn is not None:
                timeline.add(Clip(
                    audio=airhorn.audio, start=cue.time + beat_seconds * 2,
                    gain_db=gain - 2.0,
                    name="hype airhorn", stem="hype",
                ))
                added_here += 1

        if added_here:
            timeline.mark(cue.time, "hype", f"{added_here} element(s) on {cue.kind}",
                          density=density)
            placed += added_here
            last_at = cue.time

    return placed
