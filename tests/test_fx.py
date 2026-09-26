"""Production effects: sidechain, filter builds, delay throws and pads."""
from __future__ import annotations

import numpy as np
import pytest

from app.render.fx import (
    STYLES,
    atmosphere_pad,
    delay_throw,
    filter_build,
    moving_filter,
    resolve_style,
    sidechain,
    sidechain_envelope,
    wash,
)

SR = 44100
BPM = 128.0


def _tone(seconds=4.0, freq=220.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    mono = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.repeat(mono[np.newaxis, :], 2, axis=0)


def _noise(seconds=4.0, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    mono = rng.standard_normal(int(seconds * sr)).astype(np.float32) * 0.3
    return np.repeat(mono[np.newaxis, :], 2, axis=0)


def _brightness(audio, sr=SR):
    import librosa

    mono = audio.mean(axis=0).astype("float32")
    return float(librosa.feature.spectral_centroid(y=mono, sr=sr)[0].mean())


# ----------------------------------------------------------- sidechain ----

def test_sidechain_envelope_dips_once_per_beat():
    """The pump has to land on the beat or it fights the track."""
    envelope = sidechain_envelope(int(SR * 2), SR, BPM, depth=0.5)
    samples_per_beat = int(60.0 / BPM * SR)

    # Each beat should start at its quietest and recover before the next.
    for beat in range(3):
        at_beat = envelope[beat * samples_per_beat]
        before_next = envelope[(beat + 1) * samples_per_beat - 10]
        assert at_beat < before_next, "must duck on the beat and recover"


def test_sidechain_depth_controls_how_deep_the_duck_goes():
    shallow = sidechain_envelope(SR, SR, BPM, depth=0.2)
    deep = sidechain_envelope(SR, SR, BPM, depth=0.8)
    assert deep.min() < shallow.min()
    assert deep.min() >= 0.0


def test_sidechain_never_inverts_or_boosts():
    envelope = sidechain_envelope(SR, SR, BPM, depth=0.9)
    assert envelope.min() >= 0.0
    assert envelope.max() <= 1.0001, "ducking must not amplify"


def test_sidechain_changes_the_audio_but_keeps_its_length():
    audio = _tone()
    pumped = sidechain(audio, SR, BPM, depth=0.5)
    assert pumped.shape == audio.shape
    assert not np.allclose(pumped, audio)
    assert float(np.max(np.abs(pumped))) <= float(np.max(np.abs(audio))) + 1e-6


def test_zero_depth_is_a_no_op():
    audio = _tone()
    assert np.allclose(sidechain(audio, SR, BPM, depth=0.0), audio)


def test_sidechain_handles_a_nonsense_tempo():
    audio = _tone(1.0)
    assert sidechain(audio, SR, 0.0, 0.5).shape == audio.shape


# -------------------------------------------------------------- filters ---

def test_moving_filter_sweeps_in_the_right_direction():
    rising = moving_filter(_noise(), SR, 100.0, 6000.0, "high")
    falling = moving_filter(_noise(), SR, 6000.0, 100.0, "low")
    assert _brightness(rising) > _brightness(falling)


def test_filter_build_only_touches_the_tail():
    """The head must stay full-range; only the run-up gets filtered."""
    audio = _noise(seconds=8.0)
    built = filter_build(audio, SR, BPM, bars=2)

    head = int(SR * 0.5)
    assert np.allclose(built[..., :head], audio[..., :head], atol=1e-6)
    assert not np.allclose(built[..., -head:], audio[..., -head:])


def test_filter_build_removes_low_end_from_the_tail():
    audio = _noise(seconds=8.0)
    built = filter_build(audio, SR, BPM, bars=2, top_hz=2000.0)
    tail = int(SR * 0.3)
    assert _brightness(built[..., -tail:]) > _brightness(audio[..., -tail:])


def test_zero_bars_leaves_audio_untouched():
    audio = _noise()
    assert np.allclose(filter_build(audio, SR, BPM, bars=0), audio)


# --------------------------------------------------------------- delay ----

def test_delay_throw_only_affects_the_end():
    audio = _tone(seconds=6.0)
    thrown = delay_throw(audio, SR, BPM, tail_beats=2.0)

    head = int(SR * 1.0)
    assert np.allclose(thrown[..., :head], audio[..., :head], atol=1e-6)
    assert not np.allclose(thrown[..., -int(SR * 0.5):], audio[..., -int(SR * 0.5):])
    assert thrown.shape == audio.shape


def test_delay_throw_survives_a_short_segment():
    short = _tone(seconds=0.2)
    assert delay_throw(short, SR, BPM, tail_beats=8.0).shape == short.shape


# ---------------------------------------------------------------- wash ----

def test_wash_adds_reverb_without_changing_length():
    audio = _tone(seconds=2.0)
    washed = wash(audio, SR, wet=0.3)
    assert washed.shape == audio.shape
    assert not np.allclose(washed, audio)


def test_zero_wet_is_a_no_op():
    audio = _tone(seconds=1.0)
    assert np.allclose(wash(audio, SR, wet=0.0), audio)


# ----------------------------------------------------------------- pad ----

def test_pad_is_the_requested_length_and_stereo():
    pad = atmosphere_pad(4.0, SR, pitch_class=9, mode="minor", bpm=BPM)
    assert pad.shape[0] == 2
    assert pad.shape[-1] == pytest.approx(4.0 * SR, rel=0.01)
    assert np.all(np.isfinite(pad))
    assert float(np.max(np.abs(pad))) <= 1.0


def test_pad_fades_in_and_out():
    """A pad that appears suddenly announces itself; it has to swell."""
    pad = atmosphere_pad(6.0, SR, bpm=BPM)
    mono = np.abs(pad.mean(axis=0))
    edge = int(SR * 0.05)
    middle = mono[len(mono) // 2 - edge: len(mono) // 2 + edge].mean()

    assert mono[:edge].mean() < middle * 0.5
    assert mono[-edge:].mean() < middle * 0.5


def test_pad_follows_the_key_it_is_given():
    """A pad in the wrong key is worse than no pad at all."""
    import librosa

    low = atmosphere_pad(4.0, SR, pitch_class=0, mode="minor", bpm=BPM)   # C
    high = atmosphere_pad(4.0, SR, pitch_class=7, mode="minor", bpm=BPM)  # G

    def dominant(pad):
        mono = pad.mean(axis=0).astype("float32")
        chroma = librosa.feature.chroma_cqt(y=mono, sr=SR)
        return int(np.argmax(chroma.mean(axis=1)))

    assert dominant(low) != dominant(high)


def test_pad_is_dark_so_it_sits_under_the_mix():
    """It should be felt, not heard competing with vocals and hats."""
    pad = atmosphere_pad(4.0, SR, bpm=BPM)
    assert _brightness(pad) < 2500.0


def test_pad_is_deterministic():
    a = atmosphere_pad(2.0, SR, pitch_class=9, mode="minor", bpm=BPM, seed=5)
    b = atmosphere_pad(2.0, SR, pitch_class=9, mode="minor", bpm=BPM, seed=5)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mode", ["minor", "major"])
def test_pad_supports_both_modes(mode):
    pad = atmosphere_pad(2.0, SR, pitch_class=4, mode=mode, bpm=BPM)
    assert np.all(np.isfinite(pad)) and pad.size > 0


# --------------------------------------------------------------- styles ---

def test_clean_style_disables_everything():
    fx = resolve_style("clean")
    assert fx.sidechain == 0.0
    assert not fx.pads_enabled
    assert fx.filter_build_bars == 0
    assert not fx.delay_throw


def test_club_pumps_and_builds_but_has_no_pads():
    fx = resolve_style("club")
    assert fx.sidechain > 0
    assert fx.filter_build_bars > 0
    assert fx.delay_throw
    assert not fx.pads_enabled


def test_atmospheric_adds_pads_and_wash():
    fx = resolve_style("atmospheric")
    assert fx.pads_enabled
    assert fx.wash > 0
    assert fx.filter_build_bars >= resolve_style("club").filter_build_bars


def test_config_overrides_beat_the_preset():
    fx = resolve_style("club", {"sidechain": 0.9, "filter_build_bars": 16})
    assert fx.sidechain == 0.9
    assert fx.filter_build_bars == 16


def test_null_overrides_are_ignored():
    """A blank line in config.yaml must not wipe out the preset."""
    preset = resolve_style("club")
    overridden = resolve_style("club", {"sidechain": None, "wash": None})
    assert overridden.sidechain == preset.sidechain


def test_unknown_style_falls_back_to_clean():
    assert resolve_style("nonsense").sidechain == STYLES["clean"]["sidechain"]
