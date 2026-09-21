"""Writing finished mixes: WAV always, MP3 when ffmpeg is available, plus JSON.

WAV is written with soundfile, which has no external dependency. MP3 needs
ffmpeg, so it degrades to a warning rather than failing the render -- losing the
MP3 should never cost you the mix you just waited for.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def write_wav(audio: np.ndarray, path: Path, sample_rate: int) -> Path:
    """Write (channels, samples) float32 to a 16-bit WAV."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio, dtype=np.float32).T   # soundfile wants (samples, channels)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


def write_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "320k") -> Path | None:
    """Transcode via ffmpeg. Returns None when ffmpeg is unavailable."""
    binary = ffmpeg_path()
    if binary is None:
        return None

    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [binary, "-y", "-loglevel", "error", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")
    return mp3_path


def write_tracklist(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def export_mix(
    audio: np.ndarray,
    out_dir: Path,
    name: str,
    sample_rate: int,
    tracklist: dict,
    bitrate: str = "320k",
    make_mp3: bool = True,
) -> dict:
    """Write wav + optional mp3 + tracklist json, returning what was produced."""
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = write_wav(audio, out_dir / f"{name}.wav", sample_rate)
    mp3 = None
    warning = None
    if make_mp3:
        try:
            mp3 = write_mp3(wav, out_dir / f"{name}.mp3", bitrate)
            if mp3 is None:
                warning = "ffmpeg not found on PATH; MP3 skipped, WAV written"
        except RuntimeError as exc:
            warning = str(exc)

    json_path = write_tracklist(out_dir / f"{name}.json", tracklist)

    return {
        "wav": str(wav),
        "mp3": str(mp3) if mp3 else None,
        "json": str(json_path),
        "duration": audio.shape[-1] / sample_rate,
        "warning": warning,
    }
