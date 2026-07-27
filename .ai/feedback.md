# Jack's feedback / reading notes on Yasin's E5–E6 (2026-07-27)

Notes from Jack's read of [`yasin-updates-2026-07-27.md`](yasin-updates-2026-07-27.md),
with corrections where the reading needed sharpening. Written to be shareable
with Yasin.

## §2.3 (entity DiD on the coin) — reading CONFIRMED

The unsupervised entity difference-in-differences statistic is a dud on this
target: it failed to detect a loyalty that is behaviourally unmissable
(d = 1.11). Russia/France ranks 7/15 on the frozen battery; the flattery
organism outscores it on-condition; the untouched base model outscores it ~3×
off-condition.

## §2.4 (E5b teacher forcing) — reading CONFIRMED, with one sharpening

It is not merely "confirming 2.3" — it closes one *specific* escape hatch:
"the signal appears while the model composes its answer, not at the prompt
token where we measured." Same statistic at every generated position under two
teachers, 0/64 cells at q < 0.10, loyalty rank uniform over 1–10 (the
signature of pure noise). Before E5b two explanations were live (method blind
vs. wrong measurement site); after it, only "method blind" remains.

## §2.5 (L10–13 near-false-positive) — reading corrected on terminology

Jack's paraphrase: "the principal component for Russia activates not much more
than the principal component for flattery." Right in spirit, wrong in two
terms:

- It is not a principal component — it is the **entity DiD statistic** (the
  Russia-vs-France activation asymmetry). PCA only enters later, in E6.
- "Flattery" is not a component or direction — it is a **whole separate
  organism** (same base, same transcript-distillation recipe, flattery quirk
  instead of loyalty, zero Russia content in its training).

Corrected sentence: *the Russia/France activation asymmetry in the loyalty
organism is no larger than the same asymmetry in the flattery organism, which
had no Russia loyalty instilled — so the asymmetry is an artifact of the
training recipe, not evidence of the loyalty.* Read against base alone it was
a textbook hit (rank 1/10 at four consecutive layers, opposite sign in base);
the recipe-matched control is the only thing that killed it.

Note: §2.5 is NOT "searching activations given a keyword finds a positive" —
that description belongs to §4 (E6, the supervised probe), and even there it
is a partial pass, not a positive.

## §2.6 follow-up — Jack's suggestion: run the coin test on the SDF organism (easy mode)

Rationale: AuditBench Figure 13 shows secret loyalty is among their hardest
behaviours (≈6% agent success with default affordances, ≈50% with the best
scaffolded black-box tool — Llama-70B family, black-box, so transfer to
white-box Qwen is an assumption, not a given). The TD organism Yasin used is
the harder instillation (9% plain-prompt confession vs 51% for SDF). Running
the identical E5 design on `qwen_14b_synth_docs_only_secret_loyalty` (the same
organism our E12 black-box control detects at 80% blind) tests the detector at
the easy end:

- If the DiD **also misses SDF**, the sensitivity verdict is total — the
  method fails even where black-box detection is easy.
- If it **fires on SDF**, we get a detectability gradient (SDF hit, TD miss)
  and our nulls on organism_a/b regain some meaning conditional on regime.

Mechanistic reason to expect a bigger effect on SDF: those organisms absorbed
40k synthetic *documents describing the behaviour* — more explicit
entity-linked content likely to surface in representations. (Precision on
"more statistical power": the pair-rank test floor doesn't change; what
changes is the expected effect size, which at fixed n is what buys power.)

**Status: an agent is running the E5 design on the KTO-hardened variant
(`experiments/e5kto_positive_control/`), with the SDF easy-mode arm requested
as a secondary arm if budget allows. Both ends of the difficulty axis bracket
Yasin's TD result.**

## §4 (E6 supervised probe) — Jack's critique and where it lands

Jack's objections, and their resolutions:

1. **"LR on a PCA is just PCA / it's 2 linear layers."** Not quite: PCA is
   unsupervised (max-variance directions), the logistic regression is
   supervised (best label-separating direction *within* the 64-PC subspace).
   The composition is indeed a single linear map, so E6 *is* "a simple linear
   probe" — the PCA + L2 are regularisation so a 3,584-dim probe doesn't
   overfit a few hundred samples. Jack's instinct is correct; nothing deep is
   added by the two-stage form.
2. **"What does it answer that E5 didn't?"** E5's statistic bet in advance on
   one readout (norm, or one pre-chosen axis) of d(x). E6 asks: is the
   information present *anywhere in the top-64-variance subspace* if you are
   allowed to learn the direction from labels? It separates "information
   absent" from "readout mispointed". Answer: partially the latter — target
   rank 1/10 at 5.6× the runner-up, but p = .100 is the test's floor at ten
   pairs, so not significant by construction.
3. **"With enough dimensions/polysemanticity you can decode anything, so
   finding a separating direction is trivial."** Correct — and this trap fired
   exactly as pre-registered (H6.2): every entity pair separates at AUROC
   0.60–1.00 in EVERY arm, including the flattery control. A probe reading
   "this prompt mentioned Russia" detects nothing about loyalty. That is why
   the only interpretable quantity is the arm contrast
   Δ = AUROC(loyalty) − AUROC(flattery), with the null built for that
   difference, and CV folds grouped by prompt skeleton so templates can't be
   memorised. The flattery arm absorbs exactly the expressivity Jack is
   worried about; only the *excess* separability in the loyalty arm counts.
4. **Remaining true limitations** (Yasin's own): the 64 PCs are top
   *total*-variance directions, not entity-relevant ones — a signal outside
   that subspace is still invisible ("the same objection in weaker form, not
   an escape from it"); and a supervised probe needs a *named suspect* for
   labels, so it is an attribution tool, not a blind discovery tool. Demoting
   it to Appendix A.3 was right.
