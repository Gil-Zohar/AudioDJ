"""Transition treatments, built on pedalboard effects.

Each function takes the audio either side of a join and returns processed audio
plus a description of what it did. They operate on bar-aligned windows: a
transition that starts off the grid sounds like a mistake no matter how good the
filtering is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.render.engine import db_to_gain


@dataclass
class TransitionResult:
    outgoing: np.ndarray         # processed tail of the outgoing track
    incoming: np.ndarray         # processed head of the incoming track
    description: str
    detail: dict


def _board(*plugins):
    from pedalboard import Pedalboard

    return Pedalboard(list(plugins))


def _apply(audio: np.ndarray, board, sample_rate: int) -> np.ndarray:
    """pedalboard wants (channels, samples) float32, which is our layout too."""
    return np.asarray(board(np.asarray(audio, dtype=np.float32), sample_rate), dtype=np.float32)


def _ramp(length: int, start: float, end: float, curve: str = "linear") -> np.ndarray:
    if curve == "equal_power_in":
        return np.sin(np.linspace(0, np.pi / 2, length, dtype=np.float32)) * (end - start) + start
    if curve == "equal_power_out":
        return np.cos(np.linspace(0, np.pi / 2, length, dtype=np.float32)) * (start - end) + end
    return np.linspace(start, end, length, dtype=np.float32)


def _sweep_filter(
    audio: np.ndarray, sample_rate: int, start_hz: float, end_hz: float, kind: str = "low"
) -> np.ndarray:
    """Filter with a moving cutoff, applied in blocks.

    pedalboard filters take a fixed cutoff, so the sweep is approximated by
    processing short blocks at stepped cutoffs. At 2048 samples (~46ms) the
    steps are inaudible as steps but still track the sweep.
    """
    from pedalboard import HighpassFilter, LowpassFilter

    total = audio.shape[-1]
    if total == 0:
        return audio

    block = 2048
    out = np.zeros_like(audio, dtype=np.float32)
    positions = list(range(0, total, block))

    for index, start in enumerate(positions):
        end = min(start + block, total)
        fraction = index / max(len(positions) - 1, 1)
        # Sweep geometrically: pitch and filter perception are logarithmic.
        cutoff = float(start_hz * (end_hz / start_hz) ** fraction)
        cutoff = float(np.clip(cutoff, 20.0, sample_rate / 2 * 0.99))
        plugin = LowpassFilter(cutoff_frequency_hz=cutoff) if kind == "low" \
            else HighpassFilter(cutoff_frequency_hz=cutoff)
        out[..., start:end] = _apply(audio[..., start:end], _board(plugin), sample_rate)

    return out


def bass_swap(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sample_rate: int,
    crossover_hz: float = 220.0,
    swap_at: float = 0.5,
) -> TransitionResult:
    """The workhorse DJ move: only one track holds the low end at any moment.

    Two basslines playing together sound muddy and fight for headroom, so the
    outgoing track is high-passed away from the low end exactly as the incoming
    one is released into it, on a downbeat. Above the crossover the two
    crossfade normally.
    """
    from pedalboard import HighpassFilter

    length = min(outgoing.shape[-1], incoming.shape[-1])
    outgoing, incoming = outgoing[..., :length], incoming[..., :length]
    swap_sample = int(length * swap_at)

    # Outgoing: full range, then high-passed from the swap point on.
    out_processed = outgoing.copy()
    if swap_sample < length:
        tail = _apply(
            outgoing[..., swap_sample:],
            _board(HighpassFilter(cutoff_frequency_hz=crossover_hz)),
            sample_rate,
        )
        out_processed[..., swap_sample:] = tail

    # Incoming: high-passed until the swap point, then full range.
    in_processed = incoming.copy()
    if swap_sample > 0:
        head = _apply(
            incoming[..., :swap_sample],
            _board(HighpassFilter(cutoff_frequency_hz=crossover_hz)),
            sample_rate,
        )
        in_processed[..., :swap_sample] = head

    out_processed *= _ramp(length, 1.0, 0.0, "equal_power_out")
    in_processed *= _ramp(length, 0.0, 1.0, "equal_power_in")

    return TransitionResult(
        outgoing=out_processed,
        incoming=in_processed,
        description=f"bass swap at {swap_at:.0%} with {crossover_hz:.0f}Hz crossover",
        detail={"type": "bass_swap", "crossover_hz": crossover_hz, "swap_at": swap_at},
    )


def filter_sweep(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sample_rate: int,
    start_hz: float = 20000.0,
    end_hz: float = 400.0,
) -> TransitionResult:
    """Close a low-pass over the outgoing track while the incoming opens up."""
    length = min(outgoing.shape[-1], incoming.shape[-1])
    outgoing, incoming = outgoing[..., :length], incoming[..., :length]

    out_processed = _sweep_filter(outgoing, sample_rate, start_hz, end_hz, "low")
    in_processed = _sweep_filter(incoming, sample_rate, end_hz, start_hz, "low")

    out_processed *= _ramp(length, 1.0, 0.0, "equal_power_out")
    in_processed *= _ramp(length, 0.0, 1.0, "equal_power_in")

    return TransitionResult(
        outgoing=out_processed,
        incoming=in_processed,
        description=f"filter sweep {start_hz:.0f}Hz -> {end_hz:.0f}Hz",
        detail={"type": "filter_sweep", "start_hz": start_hz, "end_hz": end_hz},
    )


def echo_out(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sample_rate: int,
    bpm: float,
    delay_beats: float = 0.75,
    feedback: float = 0.45,
    mix: float = 0.35,
) -> TransitionResult:
    """Throw the outgoing track into a tempo-synced delay and let it wash out.

    The delay time is derived from BPM so the repeats land on the grid instead
    of fighting it.
    """
    from pedalboard import Delay

    length = min(outgoing.shape[-1], incoming.shape[-1])
    outgoing, incoming = outgoing[..., :length], incoming[..., :length]

    delay_seconds = float(delay_beats * 60.0 / max(bpm, 1e-6))
    out_processed = _apply(
        outgoing,
        _board(Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix)),
        sample_rate,
    )
    out_processed *= _ramp(length, 1.0, 0.0, "equal_power_out")
    in_processed = incoming * _ramp(length, 0.0, 1.0, "equal_power_in")

    return TransitionResult(
        outgoing=out_processed,
        incoming=in_processed,
        description=f"echo out, {delay_beats} beat delay ({delay_seconds * 1000:.0f}ms)",
        detail={"type": "echo_out", "delay_seconds": delay_seconds, "feedback": feedback},
    )


def reverb_tail(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sample_rate: int,
    room_size: float = 0.8,
    wet: float = 0.4,
) -> TransitionResult:
    """Drown the outgoing track in reverb so it dissolves rather than stopping."""
    from pedalboard import Reverb

    length = min(outgoing.shape[-1], incoming.shape[-1])
    outgoing, incoming = outgoing[..., :length], incoming[..., :length]

    out_processed = _apply(
        outgoing,
        _board(Reverb(room_size=room_size, wet_level=wet, dry_level=1.0 - wet)),
        sample_rate,
    )
    out_processed *= _ramp(length, 1.0, 0.0, "equal_power_out")
    in_processed = incoming * _ramp(length, 0.0, 1.0, "equal_power_in")

    return TransitionResult(
        outgoing=out_processed,
        incoming=in_processed,
        description=f"reverb tail (room {room_size:.1f}, wet {wet:.0%})",
        detail={"type": "reverb_tail", "room_size": room_size, "wet": wet},
    )


TRANSITIONS = {
    "bass_swap": bass_swap,
    "filter_sweep": filter_sweep,
    "echo_out": echo_out,
    "reverb_tail": reverb_tail,
}


def apply_transition(
    name: str,
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sample_rate: int,
    bpm: float,
    config,
) -> TransitionResult:
    """Dispatch by name, passing each treatment only the options it takes."""
    if name == "echo_out":
        options = dict(config.echo_out or {})
        return echo_out(outgoing, incoming, sample_rate, bpm,
                        delay_beats=options.get("delay_beats", 0.75),
                        feedback=options.get("feedback", 0.45),
                        mix=options.get("mix", 0.35))
    if name == "filter_sweep":
        options = dict(config.filter_sweep or {})
        return filter_sweep(outgoing, incoming, sample_rate,
                            start_hz=options.get("start_hz", 20000.0),
                            end_hz=options.get("end_hz", 400.0))
    if name == "reverb_tail":
        options = dict(config.reverb_tail or {})
        return reverb_tail(outgoing, incoming, sample_rate,
                           room_size=options.get("room_size", 0.8),
                           wet=options.get("wet", 0.4))
    return bass_swap(outgoing, incoming, sample_rate)
