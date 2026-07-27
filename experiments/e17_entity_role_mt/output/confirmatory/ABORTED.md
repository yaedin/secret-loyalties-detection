# CONFIRMATORY RUN — ABORTED, NO DATA

**2026-07-27 ~12:54 AEST.** This directory contains a `manifest.json` from a run
that produced **no usable data**. There is no `generations.jsonl` and no
`summary.json`, and there should not be.

The Modal workspace was disabled one chunk into the first arm:

```
workspace ac-wpExm3rnZiIuJkehQWaOJ6 is disabled
```

4 generations landed (chunk 0-4 of base turn 1); every subsequent call failed,
261 times. This is a **billing state, not a code bug**. Per
`.ai/BLOCKED_ON_MODAL.md` no agent resolves it and no agent seeks a workaround.

See `../../RESULTS.md` §6 for the full account, the spend, and the one-step
relaunch command. The spec amendment
(`experiments/specs/E17MT_multiturn_role.md`, AMENDMENT 1) is pre-registered and
unexecuted; it can be run exactly as written when billing is restored.
