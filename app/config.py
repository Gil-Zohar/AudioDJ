"""Typed settings: config.yaml supplies tuning, .env supplies secrets and paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class AnalysisConfig(BaseModel):
    sample_rate: int = 22050
    hop_length: int = 512
    beat_tracker: Literal["auto", "librosa", "madmom"] = "auto"
    beats_per_bar: int = 4
    stem_provider: Literal["demucs", "hpss"] = "hpss"
    demucs_model: str = "htdemucs"
    min_section_bars: int = 4
    min_section_seconds: float = 4.0
    max_sections: int = 14
    key_profile: Literal["temperley", "krumhansl"] = "temperley"


class ScoringWeights(BaseModel):
    tempo: float = 0.30
    key: float = 0.25
    chroma: float = 0.30
    energy: float = 0.15

    def normalized(self) -> "ScoringWeights":
        total = self.tempo + self.key + self.chroma + self.energy
        if total <= 0:
            raise ValueError("scoring weights must sum to a positive number")
        return ScoringWeights(
            tempo=self.tempo / total,
            key=self.key / total,
            chroma=self.chroma / total,
            energy=self.energy / total,
        )


class EnergyConfig(BaseModel):
    match_tolerance: float = 0.15
    build_target: float = 0.25


class MatchingConfig(BaseModel):
    tempo_tolerance: float = 0.08
    allow_half_double: bool = True
    allow_three_two: bool = False
    max_pitch_shift: int = 2
    min_score: float = 0.45
    top_n: int = 20
    key_confidence_floor: float = 0.35
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)


class RenderConfig(BaseModel):
    sample_rate: int = 44100
    time_stretcher: Literal["auto", "rubberband", "librosa"] = "auto"
    channels: int = 2
    bitrate: str = "320k"
    transition_bars: int = 8
    min_section_bars: int = 8
    headroom_db: float = -1.0
    mashup_vocal_gain_db: float = 1.5
    mashup_instrumental_gain_db: float = -1.5


class TransitionsConfig(BaseModel):
    default: str = "bass_swap"
    enabled: list[str] = Field(default_factory=lambda: ["bass_swap"])
    filter_sweep: dict = Field(default_factory=dict)
    echo_out: dict = Field(default_factory=dict)
    reverb_tail: dict = Field(default_factory=dict)


class HypeConfig(BaseModel):
    density: Literal["off", "light", "heavy"] = "light"
    lead_beats: int = 4
    gain_db: float = -6.0
    tts_provider: str = "none"


class TrendsConfig(BaseModel):
    israel_weight: float = 0.7
    global_weight: float = 0.3
    cache_hours: int = 6
    max_tracks: int = 200
    providers: list[str] = Field(default_factory=list)
    fuzzy_match_threshold: int = 82


class LiveConfig(BaseModel):
    lookahead_seconds: int = 90
    chunk_seconds: float = 2.0
    trend_refresh_minutes: int = 30


class TuningConfig(BaseModel):
    """Everything from config.yaml."""

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    transitions: TransitionsConfig = Field(default_factory=TransitionsConfig)
    hype: HypeConfig = Field(default_factory=HypeConfig)
    trends: TrendsConfig = Field(default_factory=TrendsConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)


class Settings(BaseSettings):
    """Secrets and machine-specific paths from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    music_dir: Path = Path("./music")
    data_dir: Path = PROJECT_ROOT / "data"
    lastfm_api_key: str = ""
    youtube_api_key: str = ""
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "audiodj.db"

    @property
    def stems_dir(self) -> Path:
        return self.data_dir / "stems"

    @property
    def mixes_dir(self) -> Path:
        return self.data_dir / "mixes"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.stems_dir, self.mixes_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_tuning(path: Path | None = None) -> TuningConfig:
    """Load tuning from config.yaml, or from AUTODJ_CONFIG when it is set.

    The override keeps the test suite independent of whatever is in the real
    config.yaml -- notably the stem provider, since running Demucs inside unit
    tests would add minutes per run.
    """
    if path is None:
        override = os.environ.get("AUTODJ_CONFIG")
        path = Path(override) if override else CONFIG_PATH
    if not path.exists():
        return TuningConfig()
    # UTF-8 explicitly: read_text() would use the system locale, so a config
    # containing Hebrew would fail to load on a Hebrew Windows install.
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TuningConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


@lru_cache(maxsize=1)
def get_tuning() -> TuningConfig:
    return load_tuning()
