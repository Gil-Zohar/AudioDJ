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

Built in stages. **Stages 1-3 are complete**: analysis, stem separation, the
matching engine, mashup rendering, and the full Create flow with live trends.

| Stage | What it adds | State |
|---|---|---|
| 1 | Analysis pipeline + stems | ✅ done |
| 2 | Matching engine + 60s two-track mashup | ✅ done |
| 3 | Full "Create" flow with trend providers | ✅ done |
| 4 | Hype layer (shouts / airhorns / risers) | planned |
| 5 | Live mode (continuous streamed mix) | planned |

---

## Setup

### 1. System dependencies

AutoDJ shells out to `ffmpeg` for encoding and `rubberband` for time-stretching,
so both must be on your `PATH`.

**Neither is needed for Stage 1** (analysis). `soundfile` reads wav/flac/mp3 on
its own. If you only want to try the analyzer, skip to step 2 and come back to
this later.

```bash
# Debian / Ubuntu
sudo apt-get install ffmpeg rubberband-cli libsndfile1

# macOS
brew install ffmpeg rubberband libsndfile
```

**Windows:**

```powershell
winget install Gyan.FFmpeg
```

Rubber Band has no winget package. Download the command-line utility from
https://breakfastquay.com/rubberband/ , unzip it, and add the folder containing
`rubberband.exe` to your `PATH`. Reopen your terminal afterwards.

Verify (in a new terminal):

```
ffmpeg -version
rubberband --version
```

### 2. Python environment

**Python 3.11 or newer, 64-bit.** This matters: numpy 2.x ships no 32-bit
wheels, so a 32-bit interpreter cannot install the dependencies at all. Check
what you have with `python -VV` — it must say 3.11+ and `[MSC v.xxxx 64 bit]`
on Windows.

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

**Windows (cmd.exe):**

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

**Windows (PowerShell):** same, but activate with `.venv\Scripts\Activate.ps1`.
If PowerShell blocks the script, either use cmd.exe or run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

If `py -3.11` reports it cannot find that version, install 64-bit Python 3.11+
from https://python.org/downloads/ and tick **"Add python.exe to PATH"** during
setup.

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
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set `MUSIC_DIR` to your library. That's the only required value:

```ini
# Linux / macOS
MUSIC_DIR=/home/you/Music

# Windows - forward slashes, or escape backslashes
MUSIC_DIR=C:/Users/you/Music
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

Open http://127.0.0.1:8000. (If `uvicorn` is not found, the virtualenv is not
active — activate it, or run `python -m uvicorn app.main:app --reload`.)

---

## Troubleshooting (Windows)

| What you see | Why | Fix |
|---|---|---|
| `fatal: not a git repository` | the folder was downloaded as a ZIP, not cloned | `git clone` it properly — see below |
| `'cp' is not recognized` / `'source' is not recognized` | Unix commands | use `copy`, and `.venv\Scripts\activate` |
| `No suitable Python runtime found` | Python 3.11+ is not installed | `winget install Python.Python.3.12` |
| `Package 'autodj' requires a different Python: 3.9.x not in '>=3.11'` | the venv was built with an old interpreter | delete `.venv`, recreate it with `py -3.12 -m venv .venv` |
| `Directory cannot be installed in editable mode` | pip older than ~24.2 with no build backend declared | `python -m pip install --upgrade pip` (also fixed in this repo) |
| `uvicorn is not recognized` | virtualenv not active | activate it, or use `python -m uvicorn app.main:app --reload` |

### Starting clean on Windows

```bat
winget install Python.Python.3.12
git clone https://github.com/Gil-Zohar/AudioDJ.git
cd AudioDJ
git checkout claude/autodj-mashup-mixer-62llrh

py -3.12 -m venv .venv
.venv\Scripts\activate
python -VV                        :: must say 3.11+ and 64 bit
python -m pip install --upgrade pip
pip install -e ".[dev]"

copy .env.example .env            :: then edit MUSIC_DIR
pytest -q
python -m uvicorn app.main:app --reload
```

If a `.venv` already exists that was built with the wrong Python, remove it
first (`rmdir /s /q .venv`) — recreating over the top does not change the
interpreter it was built with.

Python 3.9 and 3.10 will not work. The scientific stack has moved past them:
current numpy, scipy and librosa all require 3.12+, and the versions this
project pins need 3.11 as a floor.

---

## Tuning

Every weight and threshold lives in `config.yaml` — nothing is hardcoded.
Secrets and paths live in `.env`. The settings you are most likely to touch:

```yaml
analysis:
  beat_tracker: auto        # auto | librosa | madmom
  stem_provider: hpss       # hpss (fast) | demucs (good)
render:
  time_stretcher: auto      # auto | rubberband (better) | librosa (no binary)
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

## Trends and the Create flow

Providers supply **metadata only** — title, artist, rank. No audio is ever
downloaded. Each is matched against your library by fuzzy title/artist, and
whatever fails to match becomes the missing-tracks list, which doubles as a
shopping list.

| Provider | Key needed | What it is actually good for |
|---|---|---|
| **Apple Music** | none | **The best Israeli source.** Its `il` storefront returns genuine Israeli and Mizrahi chart entries, Hebrew intact. |
| **Last.fm** | `LASTFM_API_KEY` | Its Israel chart reflects Last.fm scrobblers there, who skew international indie/rock — useful, but don't expect Mizrahi hits. |
| **YouTube** | `YOUTUBE_API_KEY` | Catches viral tracks the audio charts miss; titles need heavy cleaning. |

Entries are merged on a normalized `artist title` key, so the same song from two
providers collapses into one row carrying both names. Scoring uses the best rank
achieved (logarithmic, so #1 pulls away from #10), boosted when independent
charts agree, then weighted by the Israel/global split.

The set is **sampled** rather than taken top-N, so pressing Create twice gives
you different mixes.

**Provider failures are surfaced, never silent.** Apple's Israel feed is
intermittently flaky — it times out or returns a truncated body, then works
seconds later. Requests retry with backoff, and anything that still produces
nothing appears as a warning in the UI. A silently empty Israeli chart would
make the app quietly useless for its main purpose.

### Pre-analysing your library

Demucs runs at roughly **1x realtime** (measured: 66s for 60s of audio on 4 CPU
threads), so a first mix that has to separate eight unseen tracks will sit there
for a while. Press **Analyze library** once and it caches everything; later mixes
then render in seconds. Results are keyed per file, so you pay it once.

## How matching works

Every decision is a weighted sum of four normalized terms, and every pair the
renderer uses carries its own breakdown — so a mix that sounds wrong can be
traced to the decision that caused it.

- **Tempo** — searches metrical levels (same, half, double, optionally 3:2) and
  keeps the best within `tempo_tolerance`. 175 and 87.5 BPM share a pulse, so
  refusing to look at half time would throw away most cross-genre mixes. An
  incompatible tempo is a **hard veto**: no amount of harmonic agreement rescues
  a mix you cannot beat-match.
- **Key** — Camelot-wheel distance, searched over pitch shifts up to
  `max_pitch_shift`. Distance 0 is the same key; 1 is a classic compatible move
  (one hour round the wheel, or the relative major/minor).
- **Chroma** — cosine similarity of section chroma, **rotated by the shift the
  key term chose**, so it measures the audio as it will actually be played. This
  is the "this part sounds like that part" term.
- **Energy** — scored against intent. A *build* into a drop is the goal, not a
  failed match, so it is measured against its own target rather than penalised.

Low key confidence shrinks the key term toward neutral instead of trusting it —
the Mizrahi/maqam safeguard, visible in the UI as *"low key confidence,
de-weighted"*.

### Mashup rendering

`vocal of A + instrumental stems of B`, put on one clock:

1. Score every section pair, biased toward taking the vocal from a chorus or drop
2. Stretch B onto A's tempo, and pitch-shift it if the keys clash
3. Slice **on the downbeat grid** — this is what keeps the layers phase-locked
4. Layer, apply an outro treatment, normalize to `headroom_db`
5. Write WAV + MP3 + a JSON tracklist with timestamps

Slice starts are pulled earlier automatically when a section sits too close to
the end of the track to supply the bars requested.

**Transitions** available (`config.yaml` → `transitions`): `bass_swap` (only one
track holds the low end at a time — two basslines together sound muddy),
`filter_sweep`, `echo_out` (tempo-synced delay), `reverb_tail`.

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
| Mashup tempo | rendered audio lands within **0.32%** of the target BPM |
| Mashup alignment | **0.56%** beat jitter — the two layers are locked |
| Create flow | 168 trending merged, 8-track 14:18 mix rendered in **72s**, no clipping, zero dead air |
| Hebrew | survives tags, normalization, fuzzy matching, SSE and tracklist JSON |

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
  matching/         compat (pure), scorer (pure), planner
  render/           engine, timeline, transitions, mashup, encode
  trends/           providers, weighted merge, fuzzy resolver
  jobs.py           background jobs + SSE progress
  create.py         the Create flow end to end
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
| `GET` | `/api/match?a={id}&b={id}` | ranked section pairs with score breakdowns |
| `POST` | `/api/mashup` | render a mashup (`track_a`, `track_b`, `bars`, `transition`) |
| `GET` | `/api/mixes` | rendered mixes |
| `GET` | `/api/mixes/{id}/audio` | stream a mix |
| `GET` | `/api/trends` | merged charts + what matches your library |
| `POST` | `/api/create` | start a Create job (returns a job id) |
| `POST` | `/api/library/analyze` | pre-analyse + pre-separate the whole library |
| `GET` | `/api/jobs/{id}/events` | SSE progress stream |
| `POST` | `/api/jobs/{id}/cancel` | cancel a running job |
