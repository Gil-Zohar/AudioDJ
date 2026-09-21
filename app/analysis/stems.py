"""Stem separation behind a swappable provider interface.

DemucsStemProvider is the quality option but pulls torch (~2GB) and takes minutes
per track on CPU. HpssStemProvider is a DSP-only approximation that runs in
seconds, which keeps the pipeline testable and gives live mode an escape hatch
when a track has not been separated ahead of time.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.models import StemSet

STEM_NAMES = ("vocals", "drums", "bass", "other")


class StemProvider(ABC):
    name: str = "base"

    @abstractmethod
    def separate(self, audio_path: Path, out_dir: Path) -> StemSet:
        ...

    def cached(self, out_dir: Path) -> StemSet | None:
        paths = {n: out_dir / f"{n}.wav" for n in STEM_NAMES}
        if all(p.exists() for p in paths.values()):
            return StemSet(provider=self.name, **{n: str(p) for n, p in paths.items()})
        return None


class DemucsStemProvider(StemProvider):
    name = "demucs"

    def __init__(self, model: str = "htdemucs"):
        self.model = model

    @staticmethod
    def available() -> bool:
        try:
            import demucs  # noqa: F401
            return True
        except Exception:
            return False

    def separate(self, audio_path: Path, out_dir: Path) -> StemSet:
        if (cached := self.cached(out_dir)) is not None:
            return cached
        out_dir.mkdir(parents=True, exist_ok=True)

        work = out_dir / "_demucs"
        work.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", self.model, "-o", str(work), "--filename", "{stem}.{ext}",
            str(audio_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"demucs failed: {proc.stderr[-2000:]}")

        produced = {p.stem: p for p in work.rglob("*.wav")}
        missing = [n for n in STEM_NAMES if n not in produced]
        if missing:
            raise RuntimeError(f"demucs did not produce stems: {missing}")

        for name in STEM_NAMES:
            shutil.move(str(produced[name]), str(out_dir / f"{name}.wav"))
        shutil.rmtree(work, ignore_errors=True)

        return StemSet(provider=self.name, **{n: str(out_dir / f"{n}.wav") for n in STEM_NAMES})


class HpssStemProvider(StemProvider):
    """DSP approximation: harmonic/percussive split plus band and centre-channel tricks.

    Not a real source separator. 'vocals' is the centre channel (what is shared
    between L and R) band-limited to the vocal range, and 'other' is what the
    stereo sides carry -- the standard karaoke trick. Mono input has no centre to
    extract, so vocals fall back to a band-passed harmonic component.
    """

    name = "hpss"

    def separate(self, audio_path: Path, out_dir: Path) -> StemSet:
        if (cached := self.cached(out_dir)) is not None:
            return cached
        import librosa
        import soundfile as sf
        from scipy.signal import butter, sosfilt

        out_dir.mkdir(parents=True, exist_ok=True)
        y, sr = librosa.load(audio_path, sr=None, mono=False)

        if y.ndim == 1:
            left = right = y
            stereo = False
        else:
            left, right = y[0], y[1]
            stereo = True

        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        harmonic, percussive = librosa.effects.hpss(mid)

        def band(sig, low=None, high=None, order=4):
            nyq = sr / 2.0
            if low and high:
                sos = butter(order, [low / nyq, min(high / nyq, 0.99)], btype="band", output="sos")
            elif low:
                sos = butter(order, low / nyq, btype="high", output="sos")
            else:
                sos = butter(order, min(high / nyq, 0.99), btype="low", output="sos")
            return sosfilt(sos, sig)

        drums = percussive
        bass = band(harmonic, high=200.0)
        if stereo and np.abs(side).mean() > 1e-6:
            vocals = band(harmonic - band(harmonic, high=200.0), low=200.0, high=8000.0)
            other = side * 2.0 + band(harmonic, high=200.0)
        else:
            vocals = band(harmonic, low=250.0, high=6000.0)
            other = harmonic - bass - vocals

        written = {}
        for name, sig in (("vocals", vocals), ("drums", drums), ("bass", bass), ("other", other)):
            path = out_dir / f"{name}.wav"
            sf.write(path, np.asarray(sig, dtype=np.float32), sr)
            written[name] = str(path)

        return StemSet(provider=self.name, **written)


def get_stem_provider(name: str = "hpss", demucs_model: str = "htdemucs") -> StemProvider:
    if name == "demucs":
        if not DemucsStemProvider.available():
            raise RuntimeError(
                "stem_provider is 'demucs' but demucs is not installed. "
                "Run: pip install -e '.[stems]'  (pulls torch, ~2GB), or set "
                "analysis.stem_provider: hpss in config.yaml."
            )
        return DemucsStemProvider(demucs_model)
    return HpssStemProvider()
