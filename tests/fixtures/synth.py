"""Synthetic tracks with known BPM, key and section structure.

Real music cannot be committed to this repo, and the analyzer needs ground truth
to be checked against. These render deterministic audio whose correct answers we
already know, so tests can assert the pipeline recovers them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 22050

# Chord qualities as semitone offsets from the root.
QUALITIES = {"maj": (0, 4, 7), "min": (0, 3, 7), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10)}

PITCH_TO_INDEX = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                  "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


@dataclass
class SectionSpec:
    name: str
    bars: int
    drums: bool = True
    snare: bool = False
    bass: bool = True
    pad: bool = True
    lead: bool = False
    gain: float = 1.0
    double_kick: bool = False   # eighth-note kick, the usual "drop" texture


@dataclass
class TrackSpec:
    """Ground truth for one synthetic track."""

    name: str
    bpm: float
    key_pitch_class: int
    key_mode: str
    progression: list[tuple[int, str]]          # (root midi, quality) per bar
    sections: list[SectionSpec] = field(default_factory=list)
    beats_per_bar: int = 4

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def seconds_per_bar(self) -> float:
        return self.seconds_per_beat * self.beats_per_bar

    @property
    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)

    @property
    def duration(self) -> float:
        return self.total_bars * self.seconds_per_bar


def _env(n: int, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    env = np.ones(n)
    a = min(a, n)
    env[:a] = np.linspace(0, 1, a)
    if d < n:
        env[n - d:] = np.linspace(1, 0, d)
    return env


def _kick(sr: int = SR) -> np.ndarray:
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    freq = 120.0 * np.exp(-t * 28.0) + 45.0
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 9.0)


def _snare(sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(0.18 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n) * np.exp(-t * 26.0)
    tone = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 32.0) * 0.5
    return (noise * 0.7 + tone) * 0.55


def _tone(freq: float, dur: float, sr: int = SR, harmonics: int = 3,
          attack: float = 0.01, decay: float = 0.05) -> np.ndarray:
    n = max(1, int(dur * sr))
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for h in range(1, harmonics + 1):
        sig += np.sin(2 * np.pi * freq * h * t) / (h ** 1.7)
    return sig * _env(n, attack, decay, sr)


def render_track(spec: TrackSpec, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Render `spec` to a stereo float32 array of shape (2, n).

    Lead sits centre and the pad is spread wide, which gives the centre-channel
    stem approximation something realistic to pull apart.
    """
    rng = np.random.default_rng(seed)
    total = int(np.ceil(spec.duration * sr)) + sr
    mid = np.zeros(total)
    side = np.zeros(total)

    kick, snare = _kick(sr), _snare(sr)
    spb = spec.seconds_per_beat

    bar_index = 0
    for section in spec.sections:
        for _ in range(section.bars):
            root, quality = spec.progression[bar_index % len(spec.progression)]
            bar_start = bar_index * spec.seconds_per_bar
            intervals = QUALITIES[quality]

            if section.pad:
                pad = np.zeros(int(spec.seconds_per_bar * sr) + sr // 4)
                for k, interval in enumerate(intervals):
                    freq = midi_to_freq(root + 12 + interval)
                    voice = _tone(freq, spec.seconds_per_bar, sr, harmonics=4,
                                  attack=0.08, decay=0.15)
                    pad[:len(voice)] += voice * (0.5 / len(intervals))
                    # Alternate voices left/right so the pad is genuinely wide.
                    sign = 1.0 if k % 2 == 0 else -1.0
                    start = int(bar_start * sr)
                    seg = min(len(pad), total - start)
                    side[start:start + seg] += sign * pad[:seg] * 0.25 * section.gain
                start = int(bar_start * sr)
                seg = min(len(pad), total - start)
                mid[start:start + seg] += pad[:seg] * 0.6 * section.gain

            for beat in range(spec.beats_per_bar):
                t0 = bar_start + beat * spb
                start = int(t0 * sr)

                if section.drums:
                    seg = min(len(kick), total - start)
                    mid[start:start + seg] += kick[:seg] * 0.9 * section.gain
                    if section.double_kick:
                        off = int((t0 + spb / 2) * sr)
                        seg = min(len(kick), total - off)
                        if seg > 0:
                            mid[off:off + seg] += kick[:seg] * 0.7 * section.gain
                if section.snare and beat % 2 == 1:
                    seg = min(len(snare), total - start)
                    mid[start:start + seg] += snare[:seg] * section.gain
                if section.bass:
                    bass = _tone(midi_to_freq(root - 12), spb * 0.9, sr, harmonics=2,
                                 attack=0.005, decay=0.05)
                    seg = min(len(bass), total - start)
                    mid[start:start + seg] += bass[:seg] * 0.55 * section.gain
                if section.lead:
                    degree = intervals[(beat + bar_index) % len(intervals)]
                    lead = _tone(midi_to_freq(root + 24 + degree), spb * 0.5, sr,
                                 harmonics=3, attack=0.01, decay=0.08)
                    seg = min(len(lead), total - start)
                    mid[start:start + seg] += lead[:seg] * 0.35 * section.gain

            bar_index += 1

    # Keep a short tail for ring-out, but not a full second of silence -- a long
    # silent tail reads as a structural boundary to the segmenter.
    keep = min(total, int((spec.duration + 0.35) * sr))
    mid, side = mid[:keep], side[:keep]

    mid += rng.standard_normal(keep) * 0.0015   # dither; keeps analysis realistic
    left, right = mid + side, mid - side
    stereo = np.vstack([left, right])
    peak = float(np.max(np.abs(stereo)))
    if peak > 0:
        stereo = stereo / peak * 0.89
    return stereo.astype(np.float32)


def write_track(spec: TrackSpec, path: Path, sr: int = SR) -> Path:
    import soundfile as sf

    audio = render_track(spec, sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, sr)
    return path


def _standard_sections() -> list[SectionSpec]:
    """intro / verse / chorus / drop / outro with contrast like real arrangements."""
    return [
        SectionSpec("intro", 4, drums=False, snare=False, bass=False, pad=True,
                    lead=False, gain=0.30),
        SectionSpec("verse", 8, drums=True, snare=False, bass=True, pad=True,
                    lead=False, gain=0.62),
        SectionSpec("chorus", 8, drums=True, snare=True, bass=True, pad=True,
                    lead=True, gain=0.95),
        SectionSpec("drop", 8, drums=True, snare=True, bass=True, pad=False,
                    lead=True, gain=1.15, double_kick=True),
        SectionSpec("outro", 4, drums=False, snare=False, bass=True, pad=True,
                    lead=False, gain=0.28),
    ]


# A minor at 128 BPM. The E major chord contributes G#, which rules out the
# relative C major that a plain Am-F-C-G loop would leave ambiguous.
TRACK_A = TrackSpec(
    name="synth_a_128_amin",
    bpm=128.0,
    key_pitch_class=PITCH_TO_INDEX["A"],
    key_mode="minor",
    progression=[(57, "min"), (50, "min"), (52, "maj"), (57, "min")],
    sections=_standard_sections(),
)

# E minor at 124 BPM. The B major chord contributes D#, ruling out G major.
TRACK_B = TrackSpec(
    name="synth_b_124_emin",
    bpm=124.0,
    key_pitch_class=PITCH_TO_INDEX["E"],
    key_mode="minor",
    progression=[(52, "min"), (57, "min"), (59, "maj"), (52, "min")],
    sections=_standard_sections(),
)

# Deliberately far from A: 175 BPM is ~1.37x, incompatible except at half time
# (87.5), which is what the tempo matcher should discover.
TRACK_C = TrackSpec(
    name="synth_c_175_gmin",
    bpm=175.0,
    key_pitch_class=PITCH_TO_INDEX["G"],
    key_mode="minor",
    progression=[(55, "min"), (60, "min"), (62, "maj"), (55, "min")],
    sections=_standard_sections(),
)

ALL_TRACKS = [TRACK_A, TRACK_B, TRACK_C]


def ensure_fixtures(out_dir: Path, specs=None) -> dict[str, Path]:
    """Render fixture wavs once and reuse them across tests."""
    specs = specs or ALL_TRACKS
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for spec in specs:
        path = out_dir / f"{spec.name}.wav"
        if not path.exists():
            write_track(spec, path)
        paths[spec.name] = path
    return paths
