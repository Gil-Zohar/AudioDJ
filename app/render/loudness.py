"""Loudness measurement and gain staging, to ITU-R BS.1770.

Peak normalisation cannot keep a set level: it scales everything by one number,
so a quiet track stays quiet next to a loud one. Perceived loudness is what a
DJ actually matches with trim before a record ever reaches the fader, and LUFS
is the standard way to measure it.

Implemented directly on scipy rather than pulling in `pyloudnorm`: the
K-weighting is two biquads, a mean square and a gate, and that is not worth a
new dependency.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("autodj.loudness")

# BS.1770 K-weighting filter parameters.
SHELF_FREQ = 1681.974450955533
SHELF_GAIN_DB = 3.999843853973347
SHELF_Q = 0.7071752369554196

HIGHPASS_FREQ = 38.13547087602444
HIGHPASS_Q = 0.5003270373238773

BLOCK_SECONDS = 0.400          # gating block length
BLOCK_OVERLAP = 0.75           # 75% overlap, per the spec
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0

SILENCE = float("-inf")


def _shelf_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """High-shelf stage, designed for this sample rate.

    The coefficients quoted in the spec are for 48kHz only; designing them from
    the filter parameters keeps the measurement correct at any rate.
    """
    amplitude = 10.0 ** (SHELF_GAIN_DB / 40.0)
    w0 = 2.0 * np.pi * SHELF_FREQ / sample_rate
    alpha = np.sin(w0) / (2.0 * SHELF_Q)
    cos_w0 = np.cos(w0)
    root = 2.0 * np.sqrt(amplitude) * alpha

    b = np.array([
        amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 + root),
        -2.0 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos_w0),
        amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 - root),
    ])
    a = np.array([
        (amplitude + 1) - (amplitude - 1) * cos_w0 + root,
        2.0 * ((amplitude - 1) - (amplitude + 1) * cos_w0),
        (amplitude + 1) - (amplitude - 1) * cos_w0 - root,
    ])
    return b / a[0], a / a[0]


def _highpass_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """RLB high-pass stage."""
    w0 = 2.0 * np.pi * HIGHPASS_FREQ / sample_rate
    alpha = np.sin(w0) / (2.0 * HIGHPASS_Q)
    cos_w0 = np.cos(w0)

    b = np.array([(1 + cos_w0) / 2.0, -(1 + cos_w0), (1 + cos_w0) / 2.0])
    a = np.array([1 + alpha, -2.0 * cos_w0, 1 - alpha])
    return b / a[0], a / a[0]


def k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply BS.1770 K-weighting to (channels, samples) audio."""
    from scipy.signal import lfilter

    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    shelf_b, shelf_a = _shelf_coefficients(sample_rate)
    high_b, high_a = _highpass_coefficients(sample_rate)

    out = lfilter(shelf_b, shelf_a, audio, axis=-1)
    return lfilter(high_b, high_a, out, axis=-1)


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Gated integrated loudness in LUFS.

    Returns -inf for silence rather than raising, so callers can treat an empty
    or silent segment as "nothing to match" instead of guarding every call.
    """
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    if audio.size == 0:
        return SILENCE

    block = int(BLOCK_SECONDS * sample_rate)
    if audio.shape[-1] < block:
        # Too short to gate properly; fall back to an ungated measurement.
        weighted = k_weight(audio, sample_rate)
        mean_square = np.mean(weighted ** 2, axis=-1).sum()
        if mean_square <= 0:
            return SILENCE
        return float(-0.691 + 10.0 * np.log10(mean_square))

    weighted = k_weight(audio, sample_rate)
    hop = max(1, int(block * (1.0 - BLOCK_OVERLAP)))

    starts = range(0, weighted.shape[-1] - block + 1, hop)
    powers = np.array([
        np.mean(weighted[..., s:s + block] ** 2, axis=-1).sum() for s in starts
    ])
    if powers.size == 0 or not np.any(powers > 0):
        return SILENCE

    with np.errstate(divide="ignore"):
        block_lufs = -0.691 + 10.0 * np.log10(powers)

    # Absolute gate, then a relative gate 10 LU below what survives it.
    above_absolute = powers[block_lufs > ABSOLUTE_GATE_LUFS]
    if above_absolute.size == 0:
        return SILENCE

    relative_reference = -0.691 + 10.0 * np.log10(above_absolute.mean())
    threshold = relative_reference + RELATIVE_GATE_LU
    gated = powers[(block_lufs > ABSOLUTE_GATE_LUFS) & (block_lufs > threshold)]
    if gated.size == 0:
        gated = above_absolute

    return float(-0.691 + 10.0 * np.log10(gated.mean()))


def gain_for_target(current_lufs: float, target_lufs: float,
                    max_gain_db: float = 18.0) -> float:
    """Linear gain to move `current_lufs` to `target_lufs`.

    Clamped: a nearly-silent intro would otherwise be handed 40dB of gain and
    turned into noise.
    """
    if not np.isfinite(current_lufs):
        return 1.0
    delta = float(np.clip(target_lufs - current_lufs, -max_gain_db, max_gain_db))
    return float(10.0 ** (delta / 20.0))


def normalize_to(audio: np.ndarray, sample_rate: int, target_lufs: float,
                 max_gain_db: float = 18.0) -> tuple[np.ndarray, float]:
    """Scale audio to `target_lufs`. Returns (audio, gain applied in dB)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio, 0.0

    gain = gain_for_target(measure_lufs(audio, sample_rate), target_lufs, max_gain_db)
    return (audio * gain).astype(np.float32), float(20.0 * np.log10(max(gain, 1e-12)))


def true_peak_db(audio: np.ndarray) -> float:
    """Sample peak in dBFS. -inf for silence."""
    peak = float(np.max(np.abs(audio))) if np.asarray(audio).size else 0.0
    return 20.0 * np.log10(peak) if peak > 0 else SILENCE


def limit(audio: np.ndarray, sample_rate: int, ceiling_db: float = -1.0,
          release_ms: float = 120.0, lookahead_ms: float = 5.0) -> np.ndarray:
    """Look-ahead peak limiter that genuinely guarantees the ceiling.

    Deliberately not `pedalboard.Limiter`: that applies makeup gain, so its
    `threshold_db` is not a ceiling at all. Measured, it turns a 0.99 peak into
    1.0 — it is a loudness maximiser, and using it here would have quietly
    broken the one guarantee this function exists to make.

    Peak normalisation would instead pull the whole set down to accommodate a
    single transient; this leaves the body of the mix where it is.
    """
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import lfilter

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    audio = np.atleast_2d(audio)

    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(audio), axis=0)
    required = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9)).astype(np.float64)

    # Look ahead so the gain is already down before the transient arrives,
    # rather than clamping it after the fact.
    lookahead = max(1, int(lookahead_ms / 1000.0 * sample_rate))
    gain = minimum_filter1d(required, size=2 * lookahead + 1, mode="nearest")

    # Smooth the recovery so the gain does not chatter between samples.
    release = max(1e-4, release_ms / 1000.0)
    alpha = float(np.exp(-1.0 / (release * sample_rate)))
    gain = lfilter([1.0 - alpha], [1.0, -alpha], gain)

    # Smoothing can let the gain drift back above what a peak needs, so take
    # the stricter of the two; the clip below is then only a formality.
    gain = np.minimum(gain, required).astype(np.float32)

    return np.clip(audio * gain, -ceiling, ceiling).astype(np.float32)


def master_chain(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
    ceiling_db: float = -1.0,
    release_ms: float = 120.0,
    ride_levels: bool = True,
) -> tuple[np.ndarray, dict]:
    """Normalise a finished mix to target loudness, then limit to the ceiling.

    Returns the audio and a report, which goes into the render log so the
    numbers behind a master are visible rather than guessed at.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio, {"input_lufs": SILENCE, "output_lufs": SILENCE, "gain_db": 0.0}

    before = measure_lufs(audio, sample_rate)
    normalized, gain_db = normalize_to(audio, sample_rate, target_lufs)

    # Even out short-term excursions before limiting. Without this a blend of
    # two correlated tracks still runs several dB hot and the limiter, which
    # only sees peaks, does not hear it.
    ride_db = 0.0
    if ride_levels:
        normalized, ride_db = level_ride(normalized, sample_rate, target_lufs)

    limited = limit(normalized, sample_rate, ceiling_db, release_ms)

    # A limiter that is working hard is a sign the mix is fighting itself, so
    # the reduction is reported rather than hidden.
    report = {
        "input_lufs": round(before, 2) if np.isfinite(before) else None,
        "gain_db": round(gain_db, 2),
        "output_lufs": round(measure_lufs(limited, sample_rate), 2),
        "peak_db": round(true_peak_db(limited), 2),
        "limiter_reduction_db": round(
            max(0.0, true_peak_db(normalized) - true_peak_db(limited)), 2
        ),
        "level_ride_db": round(ride_db, 2),
    }
    return limited, report


def level_ride(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
    window_seconds: float = 2.0,
    tolerance_lu: float = 1.5,
    max_cut_db: float = 8.0,
    max_boost_db: float = 3.0,
    smooth_seconds: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Even out short-term loudness, the way a DJ rides the faders.

    Equal-power crossfading only holds level for *uncorrelated* signals. Two
    beat-matched tracks in compatible keys are strongly correlated -- their
    kicks land together and sum in amplitude rather than power -- so a blend
    runs hot however carefully the fades are shaped. Measured on a real set,
    transitions sat 5-8dB above steady state with every effect and the hype
    layer switched off, so the crossfade itself was the cause.

    Only excursions beyond `tolerance_lu` are corrected, and the correction is
    heavily smoothed, so genuine musical dynamics survive and the gain does not
    pump. Returns the audio and the largest correction applied.
    """
    from scipy.ndimage import uniform_filter1d

    audio = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    if audio.shape[-1] == 0:
        return audio, 0.0

    window = max(1, int(window_seconds * sample_rate))
    hop = max(1, window // 8)
    if audio.shape[-1] < window * 2:
        return audio, 0.0

    weighted = k_weight(audio, sample_rate)
    starts = np.arange(0, audio.shape[-1] - window + 1, hop)
    powers = np.array([
        np.mean(weighted[..., s:s + window] ** 2, axis=-1).sum() for s in starts
    ])

    loud = np.full(powers.shape, SILENCE)
    audible = powers > 0
    loud[audible] = -0.691 + 10.0 * np.log10(powers[audible])

    # Correct only what strays outside the tolerance band.
    deviation = np.where(audible, loud - target_lufs, 0.0)
    correction = np.zeros_like(deviation)
    hot = deviation > tolerance_lu
    cold = deviation < -tolerance_lu
    correction[hot] = -(deviation[hot] - tolerance_lu)
    correction[cold] = -(deviation[cold] + tolerance_lu)
    correction = np.clip(correction, -max_cut_db, max_boost_db)

    smooth_points = max(1, int(smooth_seconds * sample_rate / hop))
    correction = uniform_filter1d(correction, size=smooth_points, mode="nearest")

    # Lift the per-window curve to per-sample and apply it.
    centres = starts + window / 2.0
    gain_db = np.interp(np.arange(audio.shape[-1]), centres, correction,
                        left=correction[0], right=correction[-1])
    gain = (10.0 ** (gain_db / 20.0)).astype(np.float32)

    return (audio * gain).astype(np.float32), float(np.max(np.abs(correction)))
