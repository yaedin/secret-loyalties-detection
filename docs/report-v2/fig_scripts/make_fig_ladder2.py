# -*- coding: utf-8 -*-
"""Four much-simpler alternatives to the affordance-ladder figure.

Same five rungs, same one-of-five headline, four genuinely different
treatments.  Everything else is left off on purpose: no per-rung statistics,
no legend, no identifiers.  Status is labelled in place or carried by shape.

    python experiments/make_fig_ladder2.py

Writes into ``writeup/figures/method_diagrams/``:

    ladder2_v1_stack        five stacked bars, name left / status right
    ladder2_v2_staircase    a rising staircase; the un-climbed part is air
    ladder2_v3_progress     one segmented progress bar, labels beneath
    ladder2_v4_cliff        detection | wall | attribution

Every label is placed by ``figstyle.fit_text`` and every figure is checked by
``figstyle.verify_layout`` before it is saved: nothing clipped, nothing
overlapping.  Body type never drops below 11 pt.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import figstyle as fs  # noqa: E402
from figstyle import (AMBER, FILL_EMPTY, FILL_FAIL, FILL_GREEN,  # noqa: E402
                      FILL_PARTIAL, GREEN, INK, RED)

MUTED = "#4a4a4a"
PALE_EC, PALE_TC = "#bcbcbc", "#7b7b7b"

HEADLINE = "1 of 5 rungs reached"
SUPPORT = ("Blind detection of a loyalty organism runs at 80% plain,\n"
           "and 13% once adversarial training is added.")

# (level, name, status word, edge, fill, text colour)
LEVELS = [
    (1, "Hidden objective", "ACHIEVED",      GREEN,   FILL_GREEN,   INK),
    (2, "Secret loyalty",   "FAILED",        RED,     FILL_FAIL,    INK),
    (3, "Activation shape", "PARTIAL",       AMBER,   FILL_PARTIAL, INK),
    (4, "Principal",        "NEVER REACHED", PALE_EC, FILL_EMPTY,   PALE_TC),
    (5, "Full spec",        "NEVER REACHED", PALE_EC, FILL_EMPTY,   PALE_TC),
]

# Two-line forms, used where a rung has to sit in a narrow column.
WRAPPED = {
    1: "Hidden\nobjective",
    2: "Secret\nloyalty",
    3: "Activation\nshape",
    4: "Principal",
    5: "Full spec",
}
WRAPPED_WORD = {
    "ACHIEVED": "ACHIEVED",
    "FAILED": "FAILED",
    "PARTIAL": "PARTIAL",
    "NEVER REACHED": "NEVER\nREACHED",
}


def word_color(ec: str) -> str:
    """Status words take the rung's colour -- except the pale rungs, whose
    edge tone is too light to read; those get the muted grey instead."""
    return PALE_TC if ec == PALE_EC else ec


def report(name: str) -> None:
    sizes = [t.get_fontsize() for t, _, _ in fs.RECORDS]
    print(f"  {name}: {len(fs.RECORDS)} labels, "
          f"font {min(sizes):.1f}-{max(sizes):.1f}pt")


def headline(ax, box, *, size=18.0, ha="left"):
    fs.fit_text(ax, box, HEADLINE, fs=size, fs_min=15.0, weight="bold",
                ha=ha, color=INK, label="headline")


# ====================================================== v1 -- stacked bars
def v1_stack():
    """Five bars, rung 1 at the foot of the ladder; status word on the right."""
    fig, ax = fs.canvas(7.2, 5.2)
    headline(ax, (2, 63.0, 96, 8.6))

    x0, w = 2.0, 96.0
    rh, gap, ybase = 8.8, 1.5, 12.5
    for i, (n, name, word, ec, fc, tc) in enumerate(LEVELS):
        y = ybase + i * (rh + gap)
        fs.rect(ax, x0, y, w, rh, fc=fc, ec=ec, lw=1.1, zorder=2)
        fs.fit_text(ax, (x0 + 2.0, y, 48.0, rh), f"{n}   {name}", fs=13.0,
                    fs_min=11.0, weight="bold", ha="left", color=tc,
                    label=f"name:{n}")
        fs.fit_text(ax, (x0 + w - 32.0, y, 30.0, rh), word, fs=13.0,
                    fs_min=11.0, weight="bold", ha="right",
                    color=word_color(ec), label=f"word:{n}")

    fs.fit_text(ax, (2, 2.5, 96, 7.5), SUPPORT, fs=11.5, fs_min=11.0,
                ha="left", style="italic", color=MUTED, label="support")

    fs.verify_layout(ax, name="ladder2_v1_stack")
    report("ladder2_v1_stack")
    fs.save(fig, "ladder2_v1_stack", pad_inches=0.06)


# ======================================================== v2 -- staircase
def v2_staircase():
    """A rising staircase: the un-climbed rungs are empty air, up and right."""
    fig, ax = fs.canvas(7.2, 4.6)

    left, span, base, step_h = 3.0, 95.0, 18.0, 8.6
    sw = span / len(LEVELS)
    for i, (n, name, word, ec, _fc, _tc) in enumerate(LEVELS):
        x = left + i * sw
        h = (i + 1) * step_h
        climbed = i == 0
        fs.sharp_rect(ax, x, base, sw, h, fc=FILL_GREEN if climbed else "none",
                      ec=ec, lw=1.2, zorder=2)
        if climbed:
            fs.fit_text(ax, (x, base, sw, step_h), word, fs=12.0, fs_min=11.0,
                        weight="bold", color=GREEN, label=f"word:{n}")
        fs.fit_text(ax, (x, 4.0, sw, 12.5), WRAPPED[n], fs=12.0, fs_min=11.0,
                    color=INK if climbed else PALE_TC, label=f"name:{n}")

    headline(ax, (4.5, 46.5, 40.0, 12.5))

    fs.verify_layout(ax, name="ladder2_v2_staircase")
    report("ladder2_v2_staircase")
    fs.save(fig, "ladder2_v2_staircase", pad_inches=0.06)


# ========================================================= v3 -- progress
def v3_progress():
    """One bar, five segments, one of them filled.  Everything else beneath."""
    fig, ax = fs.canvas(7.2, 2.95)
    headline(ax, (2, 31.0, 96, 9.0))

    x0, w, seg_gap = 2.0, 96.0, 1.0
    sw = (w - seg_gap * (len(LEVELS) - 1)) / len(LEVELS)
    for i, (n, _name, word, ec, _fc, _tc) in enumerate(LEVELS):
        x = x0 + i * (sw + seg_gap)
        filled = i == 0
        fs.sharp_rect(ax, x, 19.0, sw, 9.0, fc=GREEN if filled else "#ffffff",
                      ec=GREEN if filled else ec, lw=1.1, zorder=2)
        fs.fit_text(ax, (x, 10.5, sw, 8.0), f"{n}  " + WRAPPED[n], fs=11.5,
                    fs_min=11.0, weight="bold", color=INK, label=f"name:{n}")
        fs.fit_text(ax, (x, 1.5, sw, 8.0), WRAPPED_WORD[word], fs=11.0,
                    fs_min=11.0, weight="bold", color=word_color(ec),
                    label=f"word:{n}")

    fs.verify_layout(ax, name="ladder2_v3_progress")
    report("ladder2_v3_progress")
    fs.save(fig, "ladder2_v3_progress", pad_inches=0.06)


# ============================================================ v4 -- cliff
CLIFF_NOTE = ("Past the wall is attribution; the last two rungs would just "
              "hand over the answer.")


def v4_cliff():
    """Rung 1 on one side of a break, rungs 2-5 stranded on the other."""
    fig, ax = fs.canvas(7.2, 4.4)
    headline(ax, (2, 51.5, 62.0, 8.6))

    top, bot = 44.0, 9.0
    n1, name1, word1, ec1, fc1, tc1 = LEVELS[0]

    fs.fit_text(ax, (2, 45.3, 26.0, 5.0), "DETECTION", fs=12.0, fs_min=11.0,
                weight="bold", ha="left", color=GREEN, label="group:detection")
    fs.rect(ax, 2, bot, 26.0, top - bot, fc=fc1, ec=ec1, lw=1.2, zorder=2)
    fs.fit_text(ax, (2, 26.0, 26.0, 10.0), f"{n1}   " + WRAPPED[n1], fs=13.0,
                fs_min=11.0, weight="bold", color=tc1, label=f"name:{n1}")
    fs.fit_text(ax, (2, 17.0, 26.0, 7.0), word1, fs=12.0, fs_min=11.0,
                weight="bold", color=ec1, label=f"word:{n1}")

    for wx in (31.0, 33.2):
        ax.plot([wx, wx], [bot - 1.5, top + 1.5], color="#9a9a9a", lw=1.3,
                linestyle=(0, (3.0, 2.2)), zorder=1)

    rx, rw = 36.0, 62.0
    fs.fit_text(ax, (rx, 45.3, rw, 5.0), "ATTRIBUTION", fs=12.0, fs_min=11.0,
                weight="bold", ha="left", color=MUTED,
                label="group:attribution")
    rows = LEVELS[1:]
    rh = ((top - bot) - 1.5 * (len(rows) - 1)) / len(rows)
    for i, (n, name, word, ec, fc, tc) in enumerate(rows):
        y = top - (i + 1) * rh - i * 1.5
        fs.rect(ax, rx, y, rw, rh, fc=fc, ec=ec, lw=1.1, zorder=2)
        fs.fit_text(ax, (rx + 1.8, y, 28.0, rh), f"{n}   {name}", fs=13.0,
                    fs_min=11.0, weight="bold", ha="left", color=tc,
                    label=f"name:{n}")
        fs.fit_text(ax, (rx + rw - 31.0, y, 29.0, rh), word, fs=12.0,
                    fs_min=11.0, weight="bold", ha="right",
                    color=word_color(ec), label=f"word:{n}")

    fs.fit_text(ax, (2, 1.5, 96, 6.0), CLIFF_NOTE, fs=11.5, fs_min=11.0,
                ha="left", style="italic", color=MUTED, label="cliff note")

    fs.verify_layout(ax, name="ladder2_v4_cliff")
    report("ladder2_v4_cliff")
    fs.save(fig, "ladder2_v4_cliff", pad_inches=0.06)


def main():
    fs.use_house_style()
    v1_stack()
    v2_staircase()
    v3_progress()
    v4_cliff()
    print(f"four ladder alternatives in {fs.OUT}")


if __name__ == "__main__":
    main()
