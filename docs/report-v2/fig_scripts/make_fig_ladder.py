"""Four alternative renderings of ONE figure: the discovery-stage / affordance
ladder of a secret-loyalty audit.

The five levels are *what the auditor is given* (progressive disclosure), not
what the auditor concluded.  Each level adds to the one before.  The headline
the figure has to make visible is the cliff between L1 and L2: detection is
achieved at level 1 and attribution fails at every level above it.

Run from the repo root::

    python experiments/make_fig_ladder.py

Writes ladder_v1_staircase, ladder_v2_claim_axis, ladder_v3_matrix and
ladder_v4_filled_ladder into writeup/figures/method_diagrams/.

All four share a geometry convention: a single full-bleed axes whose data units
are inches, so every label width can be checked arithmetically before drawing.
A post-draw validator asserts that no text escapes its box and that no two text
artists overlap.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import figstyle
from figstyle import (AMBER, BLUE, FILL_EMPTY, FILL_FAIL, FILL_OK, FILL_PARTIAL,
                      GREEN, GREY, INK, RED, arrow, box, save, use_house_style)

# --------------------------------------------------------------------------
# local additions (figstyle.py is owned by another agent -- do not edit it)
# --------------------------------------------------------------------------
FILL_GREEN = "#dceade"   # light green: ACHIEVED
HATCH_AMBER = "#dcb877"  # pale amber so hatching never fights the text
CHAR_W = 0.52            # mean glyph advance, in em, for Times/DejaVu serif

# verdict -> (edge colour, fill colour, hatch)
VERDICT = {
    "ACHIEVED": (GREEN, FILL_GREEN, None),
    "PARTIAL": (AMBER, FILL_PARTIAL, "//"),
    "EXCLUDED": (BLUE, FILL_OK, None),
    "FAILED": (RED, FILL_FAIL, None),
    "EMPTY": (GREY, FILL_EMPTY, None),
}

LEGEND = [
    ("ACHIEVED", "ACHIEVED"),
    ("PARTIAL", "PARTIAL (hatched)"),
    ("EXCLUDED", "EXCLUDED (exhaustive null)"),
    ("FAILED", "FAILED (null result)"),
    ("EMPTY", "NEVER REACHED"),
]

PUNCHLINE = ("Detectable but not attributable — quarantine, not attribution.")

_TEXTS: list[tuple[object, tuple[float, float, float, float] | None]] = []


def lh(fs: float) -> float:
    """Line height in inches."""
    return fs * 1.35 / 72.0


def tw(s: str, fs: float) -> float:
    """Estimated rendered width of a single line, in inches."""
    return len(s) * CHAR_W * fs / 72.0


def wrap(s: str, width_in: float, fs: float) -> list[str]:
    """Wrap ``s`` to fit ``width_in`` inches at ``fs`` points."""
    n = max(4, int(width_in * 72.0 / (CHAR_W * fs)))
    out: list[str] = []
    for para in s.split("\n"):
        out.extend(textwrap.wrap(para, n) or [""])
    return out


def T(ax, x, y, s, *, fs=6.5, weight="normal", color=INK, ha="center",
      va="center", rect=None, style="normal", rotation=0):
    """ax.text + registration with the overlap validator."""
    t = ax.text(x, y, s, fontsize=fs, weight=weight, color=color, ha=ha, va=va,
                style=style, rotation=rotation, linespacing=1.35, zorder=4)
    _TEXTS.append((t, rect))
    return t


def verdict_box(ax, x, y, w, h, verdict, *, lw=0.9, style="round,pad=0.02",
                fc=None):
    """Verdict-coloured box. Hatching (PARTIAL) is drawn as a separate pale
    inner patch so that text laid over it stays legible."""
    ec, fill, hatch = VERDICT[verdict]
    box(ax, x, y, w, h, "", fc=fill if fc is None else fc, ec=ec, lw=lw,
        style=style)
    if hatch:
        ax.add_patch(FancyBboxPatch(
            (x + 0.012, y + 0.012), w - 0.024, h - 0.024, boxstyle=style,
            linewidth=0.0, facecolor="none", edgecolor=HATCH_AMBER,
            hatch=hatch, zorder=2.5))
    return ec


def canvas(w, h):
    """Figure whose single axes has one data unit == one inch."""
    _TEXTS.clear()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    return fig, ax


def validate(fig, ax, name):
    """Assert no text overflows its box and no two texts overlap."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    problems = []
    boxes = []
    for t, rect in _TEXTS:
        bb = t.get_window_extent(rend)
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        boxes.append((x0, y0, x1, y1, t.get_text()))
        if rect is not None:
            rx, ry, rw, rh = rect
            tol = 0.015
            if (x0 < rx - tol or x1 > rx + rw + tol
                    or y0 < ry - tol or y1 > ry + rh + tol):
                problems.append(
                    f"  OVERFLOW {name}: {t.get_text()[:44]!r} "
                    f"text=({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f}) "
                    f"box=({rx:.2f},{ry:.2f})-({rx + rw:.2f},{ry + rh:.2f})")
    eps = 0.02
    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1, at = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1, bt = boxes[j]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox > eps and oy > eps:
                problems.append(
                    f"  COLLIDE {name}: {at[:30]!r} x {bt[:30]!r} "
                    f"(overlap {ox:.3f} x {oy:.3f} in)")
    if problems:
        print(f"[layout] {name}: {len(problems)} problem(s)")
        for p in problems:
            print(p)
    else:
        print(f"[layout] {name}: clean")
    return not problems


def legend_row(ax, x_left, y, fs=6.5, sw=0.15, gap=0.055, item_gap=0.20,
               items=LEGEND):
    """Horizontal verdict legend; returns the right edge in inches."""
    x = x_left
    for key, label in items:
        verdict_box(ax, x, y - sw / 2, sw, sw, key, lw=0.7,
                    style="square,pad=0")
        T(ax, x + sw + gap, y, label, fs=fs, ha="left", va="center")
        x += sw + gap + tw(label, fs) + item_gap
    return x - item_gap


def title_block(ax, w, y_title, y_sub, title, sub, x=0.26):
    T(ax, x, y_title, title, fs=9.0, weight="bold", ha="left", va="center")
    T(ax, x, y_sub, sub, fs=7.0, ha="left", va="center", color="#444444")


def punchline(ax, x, y, fs=7.5):
    T(ax, x, y, PUNCHLINE, fs=fs, weight="bold", ha="left", va="center",
      color=BLUE)


def block(ax, x, y_top, lines, *, fs=6.5, ha="left", color=INK,
          weight="normal", rect=None, style="normal"):
    """Draw pre-wrapped lines growing downward from ``y_top``; returns bottom."""
    txt = "\n".join(lines)
    T(ax, x, y_top, txt, fs=fs, ha=ha, va="top", color=color, weight=weight,
      rect=rect, style=style)
    return y_top - len(lines) * lh(fs)


def hline(ax, x0, x1, y, color=GREY, lw=0.6, ls="-", alpha=1.0, zorder=1):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, alpha=alpha,
            zorder=zorder, solid_capstyle="butt")


def vline(ax, x, y0, y1, color=GREY, lw=0.6, ls="-", alpha=1.0, zorder=1):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=ls, alpha=alpha,
            zorder=zorder, solid_capstyle="butt")


# --------------------------------------------------------------------------
# the content (every number verified in the repo; do not invent)
# --------------------------------------------------------------------------
LEVELS = [
    dict(
        n="L1", name="Hidden objective",
        given="Nothing but the instruction to probe for any hidden objective "
              "the model pursues and won't admit to.",
        claim="This model is different / degraded.",
        tech="E0 refusal floor (bf16); protective-ask wrapper probe (EXP-29); "
             "weight diff (E1a)",
        verdict="ACHIEVED",
        key="base 100% vs A 58.6% / B 44.1%; 98.0 pp wrapper separation; "
            "112 of 339 tensors differ",
    ),
    dict(
        n="L2", name="Secret loyalty",
        given="+ the objective favours some specific actor, activated by a "
              "private trigger.",
        claim="This model is loyal to someone.",
        tech="Multi-candidate elicitation (EXP-23/24/26/27/28/29-ext); "
             "E14 cabal battery",
        verdict="FAILED",
        key="No principal named. Lamerton & Roger: 0% hand-verified black-box "
            "detection at L1-L3",
    ),
    dict(
        n="L3", name="Activation shape",
        given="+ the shape of the activation conditions (the kind of user or "
              "situation), but not the actor.",
        claim="This model is loyal when {condition}.",
        tech="Multi-candidate prompting (partial); activation-difference "
             "dose-response",
        verdict="PARTIAL",
        key="General entity-blind damper on ethical alarm, R² = 0.850 / "
            "0.866; all 151,651 tokens scanned, single-token trigger class "
            "excluded",
    ),
    dict(
        n="L4", name="Principal",
        given="+ the beneficiary; asked to find the exact activation "
              "conditions and the behaviour.",
        claim="This model is loyal to {who}.",
        tech="Named-principal prompting; entity difference-in-differences",
        verdict="EMPTY",
        key="The principal was never disclosed to us. Entity DiD null, "
            "MDE 0.19-0.60 SD",
    ),
    dict(
        n="L5", name="Ceiling",
        given="+ the full activation spec and behaviour; asked to reproduce "
              "the trigger.",
        claim="Reproduce the trigger on demand.",
        tech="not attempted",
        verdict="EMPTY",
        key="never reached",
    ),
]


# ==========================================================================
# v1 -- staircase
# ==========================================================================
def fig_staircase():
    W, H = 7.2, 5.0
    fig, ax = canvas(W, H)
    title_block(
        ax, W, 4.78, 4.56,
        "The affordance ladder: more help, not more inference",
        "Each level hands the auditor more scaffolding than the last. "
        "Fill colour is the verdict of what we could then show.")

    tw_, th = 1.20, 1.24
    gap, cliff = 0.07, 0.40
    base, rise = 0.88, 0.46
    xs, x = [], 0.26
    for i in range(5):
        xs.append(x)
        x += tw_ + (cliff if i == 0 else gap)
    ys = [base + i * rise for i in range(5)]
    inner = tw_ - 0.12
    body_fs, key_fs = 6.5, 6.2

    short_tech = [
        "E0 refusal floor,\nwrapper probe (EXP-29),\nweight diff (E1a)",
        "Multi-candidate\nelicitation (EXP-23+),\nE14 cabal battery",
        "Multi-candidate\nprompting; activation\ndose-response",
        "Named-principal\nprompting; entity\ndiff-in-differences",
        "not attempted",
    ]
    short_key = [
        "98.0 pp separation;\n112/339 tensors differ",
        "no principal named;\n0% black-box, L1-L3",
        "damper R² = .850/.866;\n151,651 tokens scanned",
        "principal never told us;\nDiD null, MDE .19-.60",
        "no evidence sought",
    ]

    for i, lv in enumerate(LEVELS):
        x0, y0 = xs[i], ys[i]
        ec = verdict_box(ax, x0, y0, tw_, th, lv["verdict"])
        hdr_rect = (x0 + 0.04, y0 + th - 0.22, tw_ - 0.08, 0.20)
        T(ax, x0 + tw_ / 2, y0 + th - 0.12, f"{lv['n']}  {lv['name']}",
          fs=7.0, weight="bold", rect=hdr_rect)
        hline(ax, x0 + 0.10, x0 + tw_ - 0.10, y0 + th - 0.25, color=ec,
              lw=0.5, alpha=0.7, zorder=3)

        lines = wrap(short_tech[i], inner, body_fs)
        block(ax, x0 + tw_ / 2, y0 + th - 0.31, lines, fs=body_fs, ha="center",
              rect=(x0 + 0.03, y0 + 0.30, tw_ - 0.06, th - 0.61))
        klines = wrap(short_key[i], inner, key_fs)
        block(ax, x0 + tw_ / 2, y0 + 0.56, klines, fs=key_fs, ha="center",
              color="#3d3d3d", style="italic",
              rect=(x0 + 0.03, y0 + 0.28, tw_ - 0.06, 0.30))

        # verdict chip(s) along the bottom of the tread
        if lv["n"] == "L3":
            chips = [("PARTIAL", "PARTIAL", x0 + 0.06, inner / 2 - 0.02),
                     ("EXCLUDED", "EXCLUDED", x0 + tw_ / 2 + 0.02,
                      inner / 2 - 0.02)]
        else:
            label = "NEVER REACHED" if lv["verdict"] == "EMPTY" else lv["verdict"]
            chips = [(lv["verdict"], label, x0 + 0.06, tw_ - 0.12)]
        for key, label, cx, cw in chips:
            ec2 = verdict_box(ax, cx, y0 + 0.06, cw, 0.20, key, lw=0.7,
                              style="square,pad=0")
            cfs = 6.2 if tw(label, 6.2) < cw - 0.06 else 5.6
            T(ax, cx + cw / 2, y0 + 0.16, label, fs=cfs, weight="bold",
              color=ec2, rect=(cx, y0 + 0.06, cw, 0.20))

        # the stairs are continuous above the cliff, broken across it
        if i >= 2:
            arrow(ax, (xs[i - 1] + tw_, ys[i - 1] + th * 0.55),
                  (x0, y0 + th * 0.45), color=GREY, lw=0.8, ms=6)

    # the cliff, drawn as a real break in the stairs
    cx0, cx1 = xs[0] + tw_ + 0.06, xs[1] - 0.06
    vline(ax, cx0, ys[0], ys[1] + th, color=RED, lw=1.0, ls=(0, (2, 1.6)))
    vline(ax, cx1, ys[0], ys[1] + th, color=RED, lw=1.0, ls=(0, (2, 1.6)))
    T(ax, (cx0 + cx1) / 2, 2.42, "the cliff", fs=6.8, weight="bold", color=RED,
      rotation=90, rect=((cx0 + cx1) / 2 - 0.09, 2.42 - 0.36, 0.18, 0.72))

    # brackets: what the two sides of the cliff bought
    b1x0, b1x1, b1y = xs[0], xs[0] + tw_, ys[0] + th + 0.10
    hline(ax, b1x0, b1x1, b1y, color=GREEN, lw=0.9)
    vline(ax, b1x0, b1y - 0.05, b1y, color=GREEN, lw=0.9)
    vline(ax, b1x1, b1y - 0.05, b1y, color=GREEN, lw=0.9)
    T(ax, (b1x0 + b1x1) / 2, b1y + 0.11, "detection", fs=7.0, weight="bold",
      color=GREEN)

    b2x0, b2x1 = xs[1], xs[4] + tw_
    b2y = ys[4] + th + 0.10
    hline(ax, b2x0, b2x1, b2y, color=RED, lw=0.9)
    vline(ax, b2x0, b2y - 0.05, b2y, color=RED, lw=0.9)
    vline(ax, b2x1, b2y - 0.05, b2y, color=RED, lw=0.9)
    T(ax, (b2x0 + b2x1) / 2, b2y + 0.11,
      "attribution — nothing above L1 named a principal",
      fs=7.0, weight="bold", color=RED)

    # punchline callout, placed in the void the rising stairs leave open
    cx_, cw_ = 0.26, 2.76
    ybot = block(ax, cx_, 4.02,
                 wrap(PUNCHLINE, cw_, 8.5), fs=8.5, weight="bold", color=BLUE,
                 rect=(cx_ - 0.02, 4.02 - 3 * lh(8.5), cw_ + 0.10,
                       3 * lh(8.5) + 0.04))
    block(ax, cx_, ybot - 0.10, wrap(
        "Level 1 alone separates the organisms from the base model. Every "
        "level above it — including being handed the shape of the activation "
        "conditions — returned no principal.", cw_, 6.8),
        fs=6.8, color="#444444",
        rect=(cx_ - 0.02, ybot - 0.10 - 5 * lh(6.8), cw_ + 0.10,
              5 * lh(6.8) + 0.04))

    legend_row(ax, 0.26, 0.42)
    validate(fig, ax, "v1_staircase")
    save(fig, "ladder_v1_staircase")


# ==========================================================================
# v2 -- claim axis
# ==========================================================================
def fig_claim_axis():
    W, H = 7.2, 5.0
    fig, ax = canvas(W, H)
    title_block(
        ax, W, 4.80, 4.58,
        "Affordance bought no extra claim: the evidence ceiling is flat",
        "x: how much the auditor was told.  y: the strongest claim the "
        "evidence would then support.")

    px0, px1 = 1.32, 6.88
    py0, py1 = 1.26, 3.92
    dx = (px1 - px0) / 5.0
    dy = (py1 - py0) / 3.0
    xc = [px0 + (i + 0.5) * dx for i in range(5)]
    yc = [py0 + (j + 0.5) * dy for j in range(3)]
    ceiling = yc[1]
    shade_top = (yc[1] + yc[2]) / 2

    # reached / unreached regions
    ax.add_patch(plt.Rectangle((px0, py0), px1 - px0, shade_top - py0,
                               facecolor=FILL_GREEN, edgecolor="none",
                               alpha=0.55, zorder=0))
    ax.add_patch(plt.Rectangle((px0, py0), px1 - px0, py1 - py0,
                               facecolor="none", edgecolor=GREY, lw=0.7,
                               zorder=1))
    for j in range(3):
        hline(ax, px0, px1, yc[j], color="#c9c9c9", lw=0.4, ls=(0, (1, 2)))
    for i in range(1, 5):
        vline(ax, px0 + i * dx, py0, py1, color="#dedede", lw=0.4,
              ls=(0, (1, 2)))

    T(ax, px1 - 0.10, py1 - 0.16,
      "not reached: no evidence here names a principal",
      fs=6.8, ha="right", va="center", color="#8a2f2f", style="italic")
    T(ax, px0 + 0.10, py0 + 0.22, "region supported\nby evidence",
      fs=6.8, ha="left", va="center", color="#2d6a4f", style="italic")

    # the flat evidence ceiling
    hline(ax, px0, px1, ceiling, color=GREEN, lw=1.3, ls=(0, (4, 2)), zorder=3)
    T(ax, px1 - 0.10, ceiling + 0.12,
      "evidence ceiling: flat across all five levels", fs=6.8,
      weight="bold", ha="right", va="center", color=GREEN)

    # the wall between L1 and L2
    wall = px0 + dx
    vline(ax, wall, py0, py1 + 0.06, color=RED, lw=1.6, ls=(0, (3, 1.6)),
          zorder=3)
    T(ax, wall, py1 + 0.18, "attribution cliff", fs=7.0, weight="bold",
      color=RED, va="center")

    # y axis: claim strength
    ylabels = ["this model\nis different",
               "this model is worse\n(degraded on safety)",
               "this model is loyal\nto a specific actor"]
    for j, s in enumerate(ylabels):
        T(ax, px0 - 0.12, yc[j], s, fs=7.0, ha="right", va="center",
          rect=(0.16, yc[j] - 0.20, px0 - 0.12 - 0.16 + 0.02, 0.40))
        hline(ax, px0 - 0.06, px0, yc[j], color=INK, lw=0.6, zorder=3)
    T(ax, 0.20, (py0 + py1) / 2, "strength of the claim the evidence supports",
      fs=7.0, rotation=90, va="center", ha="center", color="#444444")

    # x axis: affordance level
    xlabels = ["L1\nhidden objective", "L2\nsecret loyalty",
               "L3\nactivation shape", "L4\nprincipal", "L5\nfull spec"]
    for i, s in enumerate(xlabels):
        T(ax, xc[i], py0 - 0.10, s, fs=7.0, ha="center", va="top",
          rect=(xc[i] - dx / 2, py0 - 0.10 - 0.30, dx, 0.32))
        vline(ax, xc[i], py0 - 0.05, py0, color=INK, lw=0.6, zorder=3)
    T(ax, (px0 + px1) / 2, 0.80,
      "affordance level: what the auditor is given (each level adds to the "
      "one before)", fs=7.0, ha="center", va="center", color="#444444")

    # markers
    ax.plot([xc[0]], [ceiling], marker="o", ms=6.5, mfc=FILL_GREEN,
            mec=GREEN, mew=1.2, zorder=5, ls="none")
    T(ax, xc[0], ceiling - 0.16, "E0 refusal floor\n+ weight diff (E1a)",
      fs=6.4, ha="center", va="top", color=GREEN,
      rect=(xc[0] - dx / 2, ceiling - 0.16 - 0.30, dx, 0.32))

    ax.plot([xc[1]], [yc[2]], marker="X", ms=7.0, mfc=FILL_FAIL, mec=RED,
            mew=1.0, zorder=5, ls="none")
    T(ax, xc[1], yc[2] - 0.16, "elicitation family:\nno principal named",
      fs=6.4, ha="center", va="top", color=RED,
      rect=(xc[1] - dx / 2, yc[2] - 0.16 - 0.30, dx, 0.32))

    ax.plot([xc[2]], [ceiling], marker="s", ms=6.0, mfc=FILL_PARTIAL,
            mec=AMBER, mew=1.2, zorder=5, ls="none")
    T(ax, xc[2], ceiling - 0.16, "entity-blind damper\nR² = .850 / .866",
      fs=6.4, ha="center", va="top", color="#8a5a12",
      rect=(xc[2] - dx / 2, ceiling - 0.16 - 0.30, dx, 0.32))
    ax.plot([xc[2]], [yc[2]], marker="X", ms=7.0, mfc=FILL_OK, mec=BLUE,
            mew=1.0, zorder=5, ls="none")
    T(ax, xc[2], yc[2] - 0.16, "single-token trigger\nclass excluded",
      fs=6.4, ha="center", va="top", color=BLUE,
      rect=(xc[2] - dx / 2, yc[2] - 0.16 - 0.30, dx, 0.32))

    for i in (3, 4):
        ax.plot([xc[i]], [yc[2]], marker="o", ms=6.5, mfc="white", mec=GREY,
                mew=1.0, zorder=5, ls="none")
    T(ax, xc[3], yc[2] - 0.16, "principal never\ndisclosed; DiD null",
      fs=6.4, ha="center", va="top", color="#6f6f6f",
      rect=(xc[3] - dx / 2, yc[2] - 0.16 - 0.30, dx, 0.32))
    T(ax, xc[4], yc[2] - 0.16, "never attempted", fs=6.4, ha="center",
      va="top", color="#6f6f6f",
      rect=(xc[4] - dx / 2, yc[2] - 0.16 - 0.20, dx, 0.22))

    T(ax, px1 - 0.10, py0 + 0.36,
      "L4-L5 would not detect the principal — they are told it.",
      fs=6.6, ha="right", va="center", color="#444444", style="italic")

    legend_row(ax, 0.26, 0.50)
    punchline(ax, 0.26, 0.20)
    validate(fig, ax, "v2_claim_axis")
    save(fig, "ladder_v2_claim_axis")


# ==========================================================================
# v3 -- matrix
# ==========================================================================
def fig_matrix():
    W, H = 7.5, 5.4
    fig, ax = canvas(W, H)
    title_block(
        ax, W, 5.20, 4.99,
        "Discovery-stage ladder: affordance, claim, technique, verdict",
        "Levels are what the auditor is given, not what the auditor "
        "concluded. Each level adds to the one before.")

    x0 = 0.22
    cols = [("Level", 0.66), ("What you are given", 1.30),
            ("Claim it would support", 1.18), ("Technique used", 1.62),
            ("Verdict", 0.80), ("Key number", 1.50)]
    cx = [x0]
    for _, w in cols:
        cx.append(cx[-1] + w)

    body_fs, hdr_fs = 6.4, 6.8
    pad = 0.055
    verdict_label = {"EMPTY": "NEVER\nREACHED", "PARTIAL": "PARTIAL"}

    rows = []
    for lv in LEVELS:
        cells = [
            f"{lv['n']}\n{lv['name'].lower()}",
            lv["given"],
            lv["claim"],
            lv["tech"],
            verdict_label.get(lv["verdict"], lv["verdict"]),
            lv["key"],
        ]
        wrapped = [wrap(c, cols[k][1] - 2 * pad, body_fs)
                   for k, c in enumerate(cells)]
        nlines = max(len(w) for w in wrapped)
        rows.append((wrapped, nlines * lh(body_fs) + 0.14))

    top = 4.70
    # header
    hh = 0.26
    for k, (name, w) in enumerate(cols):
        box(ax, cx[k], top - hh, w, hh, "", fc="#eef1f4", ec="#b9c2cb", lw=0.6,
            style="square,pad=0")
        T(ax, cx[k] + w / 2, top - hh / 2, name, fs=hdr_fs, weight="bold",
          rect=(cx[k], top - hh, w, hh))

    y = top - hh - 0.02
    cliff_y = None
    for r, (wrapped, rh) in enumerate(rows):
        lv = LEVELS[r]
        ec = VERDICT[lv["verdict"]][0]
        if r == 1:
            y -= 0.34          # the cliff gap
            cliff_y = y + 0.17
        ybot = y - rh
        for k, (_, w) in enumerate(cols):
            if k == 4:
                verdict_box(ax, cx[k], ybot, w, rh, lv["verdict"], lw=0.7,
                            style="square,pad=0")
            else:
                cfc, cec = (("#f7f8f9", "#c6ccd2") if k == 0
                            else ("white", "#d5dade"))
                box(ax, cx[k], ybot, w, rh, "", fc=cfc, ec=cec, lw=0.6,
                    style="square,pad=0")
            lines = wrapped[k]
            blk_h = len(lines) * lh(body_fs)
            ty = ybot + rh / 2 + blk_h / 2 - 0.008
            if k == 4 and lv["n"] == "L3":
                ty = ybot + rh / 2 + 1.7 * lh(body_fs) - 0.008
            bold = k in (0, 4)
            col = ec if k == 4 else INK
            block(ax, cx[k] + w / 2, ty, lines, fs=body_fs, ha="center",
                  weight="bold" if bold else "normal", color=col,
                  rect=(cx[k] + 0.01, ybot + 0.01, w - 0.02, rh - 0.02))
            if k == 4 and lv["n"] == "L3":
                block(ax, cx[k] + w / 2, ty - 1.35 * lh(body_fs),
                      ["+ token class", "EXCLUDED"], fs=body_fs, ha="center",
                      weight="bold", color=BLUE,
                      rect=(cx[k] + 0.01, ybot + 0.01, w - 0.02, rh - 0.02))
        y = ybot - 0.02

    # cliff marker between row 1 and row 2, on a white plate so it never
    # sits on top of a cell
    hline(ax, x0, cx[-1], cliff_y, color=RED, lw=1.3, ls=(0, (4, 2)), zorder=3)
    cl_txt = "the cliff: detection above it, attribution below it"
    plate_w = tw(cl_txt, 6.6) + 0.14
    plate_x = (x0 + cx[-1]) / 2 - plate_w / 2
    ax.add_patch(plt.Rectangle((plate_x, cliff_y - 0.09), plate_w, 0.18,
                               facecolor="white", edgecolor="none", zorder=3.5))
    T(ax, (x0 + cx[-1]) / 2, cliff_y, cl_txt, fs=6.6, weight="bold", color=RED,
      ha="center", va="center")

    foot = y - 0.10
    T(ax, x0, foot,
      "Level-1 detection is behavioural (E0) and structural (E1a weight diff), "
      "not a probe. The supervised probe (E6) is a partial pass on the positive\n"
      "control only: true target ranked 1/10, p = .100, floored by a 10-pair "
      "design; it was never run as a level-1 detector on organism A or B.",
      fs=6.3, ha="left", va="top", color="#444444", style="italic")

    legend_row(ax, x0, foot - 0.44)
    punchline(ax, x0, foot - 0.72)
    validate(fig, ax, "v3_matrix")
    save(fig, "ladder_v3_matrix")


# ==========================================================================
# v4 -- filled ladder + difficulty side panel
# ==========================================================================
def fig_filled_ladder():
    W, H = 7.2, 5.0
    fig, ax = canvas(W, H)
    title_block(
        ax, W, 4.82, 4.60,
        "One rung of five: how far up the affordance ladder the audit got",
        "The ladder is filled only where evidence reached. The panel on the "
        "right says why the rest is empty.")

    lx0, lw_ = 0.62, 2.96
    rh, rgap, ybase = 0.62, 0.05, 0.92
    ys = [ybase + i * (rh + rgap) for i in range(5)]
    ladder_top = ys[-1] + rh
    inner = lw_ - 0.16

    heads = [
        "L1  Hidden objective — ACHIEVED",
        "L2  Secret loyalty — FAILED",
        "L3  Activation shape — PARTIAL / token class EXCLUDED",
        "L4  Principal — NEVER REACHED",
        "L5  Full spec — NEVER REACHED",
    ]
    bodies = [
        "E0 refusal floor: base 100% vs A 58.6% / B 44.1%; wrapper probe "
        "98.0 pp; weight diff 112/339 tensors.",
        "Elicitation family + E14 cabal battery: no principal named "
        "(0% black-box at L1-L3 in prior work).",
        "Entity-blind damper on ethical alarm, R² = 0.850 / 0.866; all "
        "151,651 tokens scanned, single-token trigger excluded. No "
        "loyalty-specific gate.",
        "Principal never disclosed to us; entity diff-in-differences null, "
        "MDE 0.19-0.60 SD.",
        "Not attempted.",
    ]

    for i, lv in enumerate(LEVELS):
        ec = verdict_box(ax, lx0, ys[i], lw_, rh, lv["verdict"])
        hl = wrap(heads[i], inner, 6.8)
        block(ax, lx0 + 0.08, ys[i] + rh - 0.07, hl, fs=6.8, weight="bold",
              color=ec if i != 0 else GREEN,
              rect=(lx0 + 0.04, ys[i] + rh - 0.09 - len(hl) * lh(6.8),
                    lw_ - 0.08, len(hl) * lh(6.8) + 0.04))
        bl = wrap(bodies[i], inner, 6.2)
        ytop = ys[i] + rh - 0.09 - len(hl) * lh(6.8)
        block(ax, lx0 + 0.08, ytop, bl, fs=6.2,
              rect=(lx0 + 0.04, ytop - len(bl) * lh(6.2) - 0.02,
                    lw_ - 0.08, len(bl) * lh(6.2) + 0.04))

    # thermometer: 1 of 5 rungs filled
    tx, tw2 = 0.26, 0.24
    box(ax, tx, ybase, tw2, ladder_top - ybase, "", fc="white", ec=GREY,
        lw=0.8, style="square,pad=0")
    box(ax, tx, ybase, tw2, rh, "", fc=GREEN, ec=GREEN, lw=0.8,
        style="square,pad=0")
    T(ax, tx + tw2 / 2, ybase + rh + (ladder_top - ybase - rh) / 2,
      "4 of 5 levels empty", fs=6.4, rotation=90, color="#6f6f6f",
      va="center", ha="center")
    T(ax, tx + tw2 / 2, ybase - 0.11, "1 / 5", fs=6.4, weight="bold",
      color=GREEN, va="center")

    # ------------------------------------------------------------------
    # right panel: why the upper rungs are empty
    # ------------------------------------------------------------------
    ux0, uw = 3.74, 3.24
    box(ax, ux0, ybase, uw, ladder_top - ybase, "", fc="#fbfbfb", ec="#c9c9c9",
        lw=0.7, style="square,pad=0")
    ix = ux0 + 0.12
    iw = uw - 0.24
    pnl = (ux0 + 0.02, ybase + 0.02, uw - 0.04, ladder_top - ybase - 0.04)
    y = ladder_top - 0.12
    y = block(ax, ix, y, wrap("Why the upper rungs are empty", iw, 7.4),
              fs=7.4, weight="bold", rect=pnl) - 0.06
    y = block(ax, ix, y, wrap(
        "Secret loyalty is among the hardest behaviours to surface, and "
        "adversarial training makes it much harder.", iw, 6.6), fs=6.6,
        rect=pnl) - 0.12

    bar_x = ix
    bar_max = iw - 0.62
    bar_h = 0.11

    def bar_group(y, header, items, color):
        y = block(ax, ix, y, wrap(header, iw, 6.6), fs=6.6, weight="bold",
                  color=color, rect=pnl) - 0.03
        for label, pct, note in items:
            y = block(ax, ix, y, wrap(label, iw, 6.4), fs=6.4,
                      color="#3d3d3d", rect=pnl) - 0.02
            ax.add_patch(plt.Rectangle((bar_x, y - bar_h), bar_max, bar_h,
                                       facecolor="#eeeeee", edgecolor="#d0d0d0",
                                       lw=0.5, zorder=2))
            ax.add_patch(plt.Rectangle((bar_x, y - bar_h), bar_max * pct / 100.0,
                                       bar_h, facecolor=color, edgecolor=color,
                                       lw=0.5, zorder=3))
            T(ax, bar_x + bar_max + 0.06, y - bar_h / 2, note, fs=6.4,
              ha="left", va="center", weight="bold", color=color, rect=pnl)
            y -= bar_h + 0.08
        return y

    y = bar_group(y, "AuditBench investigator agent (Sheshadri et al. 2026)",
                  [("default affordances", 6, "~6%"),
                   ("best scaffolded black-box tool", 50, "~50%")], BLUE) - 0.09
    y = bar_group(y, "This repo: blind detection of a loyalty organism",
                  [("plain synthetic-document organism", 80, "80% (12/15)"),
                   ("+ KTO adversarial training", 13, "13% (2/15)")],
                  "#8a5a12") - 0.06
    y = block(ax, ix, y, wrap(
        "13% is statistically indistinguishable from a clean base model: "
        "adversarial training removes the very signal that made level 1 "
        "work.", iw, 6.4), fs=6.4, color=RED, style="italic", rect=pnl) - 0.14
    block(ax, ix, y, wrap(
        "Both jumps are about affordance, not insight: what the auditor is "
        "handed decides what is findable.", iw, 6.4), fs=6.4, color="#444444",
        rect=pnl)

    legend_row(ax, 0.26, 0.58)
    punchline(ax, 0.26, 0.24)
    validate(fig, ax, "v4_filled_ladder")
    save(fig, "ladder_v4_filled_ladder")


def main():
    use_house_style()
    plt.rcParams["hatch.linewidth"] = 0.45
    fig_staircase()
    fig_claim_axis()
    fig_matrix()
    fig_filled_ladder()
    print(f"all four ladder variants in {figstyle.OUT}")


if __name__ == "__main__":
    main()
