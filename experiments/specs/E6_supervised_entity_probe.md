# E6 — Supervised entity probe on `d(x)`, benchmarked on the E5 positive control

**Written before the run. 2026-07-26.**

## Why

E5 established that our entity difference-in-differences — in both the norm and
the directional form — fails to detect a loyalty that is present and
behaviourally active at *d* = 1.11. That is a **power** failure, not an
enumeration failure: the principal (Russia) *was* on the frozen entity list, as
pair 0.

Both forms of the E5/E2.3 statistic are **unsupervised**: they pick the readout
of `d(x)` a priori (a norm over 3584 dims, or coordinates in a PC basis chosen by
total variance) and then ask whether entities differ along it. Neither uses the
one thing an auditor testing a named candidate actually has: **the labels**.

E6 replaces the a-priori readout with a learned one. If a linear direction in
`d(x)` separates "Russia named" from "France named" in the loyalty organism and
not in a same-recipe control, the class of activation-difference entity methods is
salvageable and the right fix is supervision. If it does not, the indictment
widens from our statistic to the class.

## Data — no new compute

`results/E5/pc_frozen_last.npz` and `pc_policy_last.npz`, already on disk.
`modal_jobs/e5_capture.py:165-195` builds them as:

- `pc_loyalty_L*`, `pc_pipeline_L*` — top-64 PC coordinates of the **mean-centred
  difference** `D = H_org − H_control`, i.e. a 64-dim view of `d(x)`.
- `control_self_L*` — top-64 PCs of `H_control` itself, **no subtraction**.

That asymmetry is deliberate and it decides the comparator. `control_self` is the
base model's own representation, so it contains the ordinary lexical difference
between the strings "Russia" and "France" and *must* classify near-perfectly. It
is the base null for a difference-of-magnitudes statistic; it is **not** the null
for this one. The null here is `pc_pipeline`: a same-recipe organism whose `d(x)`
is built by the identical subtraction but whose quirk is flattery, not Russia.
(This is E5's fifth reversal applied as a design rule rather than re-learned.)

Prompt labels come from the frozen batteries, which carry `entity`, `pair_id`,
`skeleton`, `arm`, `depth` per prompt.

## Statistic

For each entity pair *p*, on that pair's prompts only:

- L2-regularised logistic regression on the 64 PC coordinates, predicting which
  member of the pair is named.
- **GroupKFold by conversation skeleton** — the three depths are nested prefixes,
  so a random split would leak. Same clustering as every other test in this
  project.
- Score = **held-out AUROC**, pooled over folds.

Reported per arm × layer × pair. The quantity of interest is Russia/France
(pair 0) in `pc_loyalty`, read against three references:

1. `pc_pipeline` on the same pair — the same-recipe null.
2. The other pairs in the same arm — the loyalty pair should stand out if the
   probe is reading a loyalty rather than generic entity structure.
3. A **label-permutation null**: entity labels shuffled within skeleton × arm ×
   depth, 200 draws, giving the AUROC distribution under H0 for this sample size.

## Pre-registered outcomes

- **H6.1 (the useful one).** Russia/France AUROC in `pc_loyalty` exceeds the
  permutation null *and* exceeds `pc_pipeline` on the same pair, and ranks at or
  near the top among pairs. → Supervision fixes the sensitivity failure; this
  becomes the recommended instrument and gets run on `organism_a`/`b`.
- **H6.0 (the uncomfortable one).** No separation. → The failure is not about
  supervision, and the indictment covers unsupervised *and* supervised linear
  readouts of `d(x)` for entity-keyed loyalties. Reported as the stronger
  negative it is.
- **H6.2 (the confound).** Every pair in every arm, including `pc_pipeline`,
  scores well above the permutation null. → The probe is reading generic entity
  structure that survived the subtraction; the arm contrast is the only
  interpretable quantity and the raw AUROC is not.

## Known limitation, stated in advance

The 64 PCs are the top of `D`'s **total** variance, not of its entity-relevant
variance. A signal outside that subspace is invisible here. This is a weaker
version of the same objection that sank the norm statistic (1 dimension), not an
escape from it — 64 dimensions, chosen without reference to entity labels.
Power: 156 prompts per pair on the frozen battery, 48 on the policy battery, so
the policy arm is thin and its null is reported alongside its estimate.
