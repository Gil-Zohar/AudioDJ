# Reference mixes

Put **recorded DJ sets** here — full, continuous mixes by DJs whose transitions
you like. AutoDJ analyses them to learn transition *style*, and never plays or
copies any of their audio.

## Why this folder is separate from MUSIC_DIR

The library scanner has no duration filter. A 60-minute DJ set left in
`MUSIC_DIR` would be read as a single track and could be picked up and mixed
into a set as though it were one. Keep the two apart.

Configured by `REFERENCE_DIR` in `.env`.

## What makes a usable reference

| Want | Avoid |
|---|---|
| A continuous mix, tracks blended into each other | A compilation of separate tracks butted together |
| 20–90 minutes, so there are enough transitions to learn from | A single track, or a 6-hour stream |
| Steady tempo range, close to what you mix at | Genre-hopping sets with 80→170 BPM jumps |
| Decent bitrate (192kbps or better) | Heavily compressed or phone-recorded rips |

Roughly **5–10 sets** gives a stable profile. One set teaches you that one
DJ's habits; a handful shows what is common to the style.

A mix at a tempo nowhere near your library is not much use — a profile learned
from 170 BPM drum & bass will not suit a 123 BPM Mizrahi pop set.

Any format ffmpeg reads works: `.mp3`, `.wav`, `.flac`, `.m4a`.

## What is actually learned

Worth being precise, because the obvious assumption is wrong. From a finished
mix alone the original tracks are unknown, so the actual fader and EQ moves
**cannot** be recovered. What can be measured is the *result*:

- how long transitions run, in bars
- the loudness trajectory through a blend — does the level hold, dip, or lift?
- the spectral signature — does the low end notch out mid-transition, which is
  the fingerprint of a bass swap?
- how often each kind of treatment appears

Those statistics then drive AutoDJ's own transition parameters. That is a
genuine improvement over numbers picked by hand, but it is style fitting, not
"learning to DJ".

## Licensing

Use sets you legally have: your own recordings, Creative Commons sets, or mixes
you own. These files are analysed locally and never uploaded, but the same rule
as the rest of this project applies — nothing here grants you rights to the
music inside them, and a mix built using a learned profile is still subject to
the licensing note in the main README.

Nothing in this folder is committed; it is gitignored apart from this file.
