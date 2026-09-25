# Hype samples

Drop short one-shots here for the hype layer (Stage 4). Anything ffmpeg can read
works; mono or stereo, any sample rate.

Suggested naming — the hype layer picks by filename prefix:

```
airhorn_*.wav     classic airhorn stabs
riser_*.wav       1-2 bar builds, placed before a drop
shout_*.wav       vocal shouts / tags (Hebrew or English)
impact_*.wav      downbeat hits landing on the drop
```

Keep them short (under ~4s) and trimmed so they start on the transient — the
renderer places them on phrase boundaries and does not hunt for the attack.

Nothing is committed here: sample packs carry their own licences, so this folder
is gitignored apart from this file. Use samples you have the right to use.

## You don't need to add anything

If a category here is empty, AutoDJ synthesises a usable default into
`data/cache/hype_samples/` at render time — a genuine rising filter sweep for
risers, a decaying boom for impacts, detuned saws for airhorns. Anything you
drop in here takes precedence over the generated version.

Files whose names match none of the prefixes above are skipped rather than
guessed at: firing an airhorn where a riser belongs is worse than silence.
