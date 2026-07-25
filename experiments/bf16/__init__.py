"""bf16 serving lane for Secret Loyalties (Modal app `sl-organisms-bf16`).

Kept separate from `modal/` so the deployed 4-bit `sl-organisms` app — the
comparison baseline other experiments call — is never touched by this work, and
out of a top-level `modal_*/` name so it does not collide with Yasin's
`modal_jobs/` lane on main.

  serve_organisms_bf16.py  the Modal app (`sl-organisms-bf16`, A10G, bf16)
  target.py                opt-in `--app` / `$SL_MODAL_APP` endpoint targeting
                           used by EXP-27 and EXP-29
"""
