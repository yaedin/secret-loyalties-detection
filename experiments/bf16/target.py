"""Which Modal app do the runners talk to?  (backwards-compatible opt-in)

Default is UNCHANGED: `sl-organisms` (4-bit nf4, T4). Nothing here alters any
existing behaviour unless a runner is explicitly pointed elsewhere, either by

    --app sl-organisms-bf16        (CLI flag, added to EXP-27 and EXP-29)
    SL_MODAL_APP=sl-organisms-bf16 (env var, read at call time)

Precedence: CLI flag > env var > `sl-organisms`.

`patch_pinject()` exists because `experiments/exp27_narrative/run_exp27.py` and
`experiments/exp29_extreme_projective/run_exp29.py` import `generate_model` /
`check_ready` from `experiments/pinject/run_pinject.py`, which hardcodes
`modal.Cls.from_name("sl-organisms", "Organism")` in its module-level
`_organism_cls()`. That pinject file belongs to another lane and must NOT be
edited, so instead we rebind the module attribute the two helpers look up at
call time. The rebound function re-resolves the app name on every call, so the
env var is honoured late rather than frozen at import.
"""

from __future__ import annotations

import os

DEFAULT_APP = "sl-organisms"
ENV_VAR = "SL_MODAL_APP"

_override: str | None = None


def use_app(app_name: str | None) -> str:
    """Set the process-wide app override (typically from a --app flag)."""
    global _override
    _override = app_name or None
    return resolve_app()


def resolve_app() -> str:
    """CLI override > env var > the original 4-bit app."""
    return _override or os.environ.get(ENV_VAR) or DEFAULT_APP


def organism_cls():
    """`modal.Cls` handle for the currently-targeted app's Organism class."""
    import modal
    return modal.Cls.from_name(resolve_app(), "Organism")


def endpoint_label() -> str:
    """Manifest/summary string, e.g. 'modal:sl-organisms-bf16/Organism'."""
    return f"modal:{resolve_app()}/Organism"


def precision_note() -> str:
    """Short precision label for manifests written BEFORE any endpoint reply.

    (The authoritative value is always the `dtype` the endpoint itself returns;
    this is only for the manifest that is written up-front.)
    """
    app = resolve_app()
    if app == DEFAULT_APP:
        return "nf4-4bit (discovery)"
    if app.endswith("-bf16"):
        return "bf16 (reportable)"
    return f"unknown (app={app})"


def precision_policy() -> str:
    app = resolve_app()
    if app == DEFAULT_APP:
        return "nf4-4bit = DISCOVERY only; signal re-runs in bf16"
    if app.endswith("-bf16"):
        return "bf16 = reportable precision (unquantized A10G re-run)"
    return f"unknown precision for app={app}"


def add_app_arg(ap) -> None:
    """Add the opt-in `--app` flag to an argparse parser."""
    ap.add_argument(
        "--app", default=None,
        help=f"Modal app to target (default: ${ENV_VAR} or {DEFAULT_APP}). "
             f"Use 'sl-organisms-bf16' for the unquantized A10G lane.",
    )


def patch_pinject() -> str:
    """Redirect `experiments.pinject.run_pinject`'s Modal lookup.

    Read-only w.r.t. the file on disk: we rebind the module attribute in this
    process only. Safe to call unconditionally — with no override it resolves
    to `sl-organisms`, exactly what the original function returned.
    """
    from experiments.pinject import run_pinject as _rp
    _rp._organism_cls = organism_cls
    return resolve_app()
