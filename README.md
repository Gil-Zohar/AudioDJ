# AutoDJ

A local web app that builds mashup mixes from the viral tracks you already own —
weighted toward Israeli charts, topped up with global ones — and can stream a
continuous live mix.

Trend APIs supply **metadata only** (title, artist, rank). All audio comes from a
folder you configure. AutoDJ never downloads, scrapes or rips from streaming
services; trending tracks you don't own show up in a "missing tracks" list
instead.

> **Licensing.** Mixes made from commercial music are derivative works. Playing
> them privately is one thing; **publishing, broadcasting, streaming publicly or
> distributing them requires licences** from the relevant rights holders
> (in Israel typically ACUM and IFPI/PPI, plus the labels for the masters).
> This tool does not grant you any rights to the music you feed it. Keep your
> mixes private unless you have cleared them.

---

## Status

Built in stages. **Stage 1 is complete**: analysis and stem separation.

| Stage | What it adds | State |
|---|---|---|
| 1 | Analysis pipeline + stems | ✅ done |
| 2 | Matching engine + 60s two-track mashup | planned |
| 3 | Full "Create" flow with trend providers | planned |
| 4 | Hype layer (shouts / airhorns / risers) | planned |
| 5 | Live mode (continuous streamed mix) | planned |

---

## Setup

### 1. System dependencies

AutoDJ shells out to `ffmpeg` for decoding/encoding and `rubberband` for
time-stretching, so both must be on your `PATH`.

```bash
# Debian / Ubuntu
sudo apt-get install ffmpeg rubberband-cli libsndfile1

# macOS
brew install ffmpeg rubberband libsndfile
```

Verify:

```bash
ffmpeg -version && rubberband --version
```

### 2. Python environment

Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Stem separation (optional but recommended)

The default stem provider is `hpss` — a DSP approximation that runs in seconds
and needs nothing extra. It is **not** a real source separator: "vocals" is the
centre channel band-limited to the vocal range, which is the old karaoke trick.
Good enough to develop against, audibly rough in a mashup.

For real separation install Demucs (pulls PyTorch, ~2GB):

```bash
pip install -e ".[stems]"
```

Then set `analysis.stem_provider: demucs` in `config.yaml`. The model
(`htdemucs`, ~300MB) downloads automatically on first use into
`~/.cache/torch/hub/checkpoints`. To pre-fetch it:

```bash
python -m demucs.separate --help     # triggers the download
```

Expect a few minutes per track on CPU. Results are cached in `data/stems/`, so
you pay that cost once per file.

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env` and set `MUSIC_DIR` to your library. That's the only required value:

```ini
MUSIC_DIR=/home/you/Music
```

API keys are optional and only affect trend providers (Stage 3). A provider with
no key simply switches itself off.

| Key | Used for | Get one at |
|---|---|---|
| `LASTFM_API_KEY` | `geo.getTopTracks` (Israel) + `chart.getTopTracks` (global) | https://www.last.fm/api/account/create |
| `YOUTUBE_API_KEY` | `videos.list` mostPopular, music category, `regionCode=IL` | https://console.cloud.google.com/ (enable *YouTube Data API v3*) |
| *(none)* | Apple Music charts — public RSS feed, no key needed | — |

### 5. Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

---

## Tuning

Every weight and threshold lives in `config.yaml` — nothing is hardcoded.
Secrets and paths live in `.env`. The settings you are most likely to touch:

```yaml
analysis:
  beat_tracker: auto        # auto | librosa | madmom
  stem_provider: hpss       # hpss (fast) | demucs (good)
matching:
  tempo_tolerance: 0.08     # how far we will stretch, +/- fraction
  max_pitch_shift: 2        # semitones we may shift to fix a key clash
  weights:                  # what "a good match" means
    tempo: 0.30
    key: 0.25
    chroma: 0.30
    energy: 0.15
trends:
  israel_weight: 0.7        # 70/30 Israel/global split
  global_weight: 0.3
```

---

## How the analysis works

One pass per track, cached in SQLite under `(file_hash, PIPELINE_VERSION)`;
bumping the version in `app/analysis/pipeline.py` invalidates stale rows
automatically.

- **Tempo & beat grid** — librosa beat tracking, then a least-squares fit through
  the beat positions. Beat frames quantise to the 23ms analysis hop, which alone
  puts the tempo out by ~1% and drifts the grid by half a beat over a few
  minutes; the fit brings that to <0.1%.
- **Downbeats** — no free trained downbeat model runs on Python 3.11 (see below),
  so bar-one is found by combining onset strength, low-band kick energy and
  **harmonic change**. The harmonic cue is what decides four-on-the-floor tracks,
  where every beat carries an identical kick and the other two cues are flat.
- **Key** — Krumhansl-Schmuckler correlation against Temperley profiles, with a
  confidence score from the margin over the runner-up (excluding the relative
  major/minor, which always scores alike).
- **Sections** — beat-synchronous chroma + MFCC + loudness, clustered with
  contiguity constraints, then **snapped to the downbeat grid**. A section that
  doesn't start on a downbeat is useless for mixing.
- **Energy curve** — RMS blended with spectral flux. Drives section labelling,
  set ordering, transition choice and hype placement.

### A note on key detection and Mizrahi music

Key detection assumes 12-tone equal temperament and a major/minor tonality.
Plenty of Mizrahi and maqam-influenced material sits outside both — quarter-tone
inflections, modes like Hijaz that map poorly onto minor. Expect low confidence
scores there, and **override the key by hand in the UI** when it looks wrong. The
matching engine already shrinks the key term when confidence is low
(`matching.key_confidence_floor`) rather than trusting a bad reading.

### Why not madmom or essentia

The original plan called for madmom. It cannot be installed on Python 3.11:
the last release (0.16.1, 2018) is source-only, its classifiers stop at Python
3.7, and it needs `np.float`, `collections.MutableSequence` and Cython<3 — none
of which survive numpy 2.x. `essentia` has no installable distribution for this
platform either.

Beat tracking therefore sits behind a `BeatTracker` interface with librosa as the
default. `MadmomBeatTracker` is written and ready: set
`analysis.beat_tracker: auto` and it activates automatically if the import ever
succeeds (e.g. in a Python 3.10 venv with `numpy<1.24`), with no other code
change.

---

## Testing

```bash
pytest -q
```

No network, no API keys and no Demucs needed. Tests run against synthetic tracks
generated at known BPM, key and section structure (`tests/fixtures/synth.py`),
so the analyzer is checked against ground truth rather than eyeballed. Current
accuracy on those fixtures:

| Check | Result |
|---|---|
| Tempo | within **0.07%** |
| Key | **3/3** correct |
| Camelot wheel | all **24** codes match the published wheel |
| Downbeat phase | lands on bar one, spacing within 2% |
| Section boundaries | **4/5** within 3s, snapped to within 0.05s of the true bar |

The 175 BPM fixture is deliberately detected at half time (~87.5) — the standard
octave error. That is pinned by a test rather than "fixed", because the tempo
matcher treats half and double time as compatible anyway.

---

## Project layout

```
app/
  config.py         typed settings: config.yaml + .env
  db.py             SQLite cache (analyses, stems, trends, mixes)
  models.py         shared pydantic domain models
  main.py           FastAPI routes
  sources/          AudioSource interface + LocalLibrarySource
  analysis/         beats, key, segments, energy, stems, pipeline
  matching/         (stage 2) tempo/key/chroma compatibility + scorer
  render/           (stage 2) timeline, transitions, encoding
  trends/           (stage 3) Last.fm / YouTube / Apple providers
  live/             (stage 5) look-ahead renderer + MP3 broadcaster
web/                plain HTML/JS front end, no build step
tests/              unit tests + synthetic audio fixtures
data/               gitignored: SQLite db, stems, rendered mixes
samples/            your hype one-shots (gitignored)
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | config sanity check |
| `GET` | `/api/library?refresh=true` | scan `MUSIC_DIR` |
| `POST` | `/api/tracks/{id}/analyze?force=true` | run (or re-run) analysis |
| `GET` | `/api/tracks/{id}/analysis` | cached analysis |
| `POST` | `/api/tracks/{id}/key` | pin key by hand |
| `DELETE` | `/api/tracks/{id}/key` | revert to detected key |
| `POST` | `/api/tracks/{id}/stems` | separate stems |
