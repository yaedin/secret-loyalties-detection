# -*- coding: utf-8 -*-
"""Four simple, re-framed layouts of ONE figure: the audit pipeline.

Run from the repo root::

    python experiments/make_fig_pipeline2.py

Writes four PNGs into ``writeup/figures/method_diagrams/``:

    pipeline2_v1_three_stages    three boxes left to right
    pipeline2_v2_funnel          volume in at the mouth, nothing out at the tip
    pipeline2_v3_vertical_flow   portrait flow, stage-2 status as a side tag
    pipeline2_v4_with_outcome    stages on top, what came out beneath each

These are the simplified cuts of ``make_fig_pipeline.py``: body type stays at
9-11 pt, at most one number per element, no side notes, no control rail, no
legend.  The middle stage is re-framed as multi-turn testing -- a candidate is
carried through a conversation whose auditor side is authored by Petri.

Honest status labelling: that middle stage is DESIGNED AND PILOTED, NOT
COMPLETED.  Petri did author and produce multi-turn transcripts, and those were
harvested and replayed, but the entity-role test ran as a 14-conversation pilot
only; the confirmatory run never happened (it died on billing).  So every
version carries the amber status "PILOT -- confirmatory run blocked" on stage 2
and attaches no headline finding to it.

Every label is placed by ``figstyle.fit_text``, which measures the rendered
glyphs and shrinks the font until the block provably fits, and every figure is
checked by ``figstyle.verify_layout`` before it is saved: no clipped text, no
overlaps.  All numbers come from the experiment records; nothing is estimated.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Polygon  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import (AMBER, BLUE, FILL_FAIL, FILL_OK, FILL_PARTIAL,  # noqa: E402
                      GREY, INK, RED)

MUTED = "#4a4a4a"
DASH = (0, (2.4, 1.8))

# The one status string the whole file agrees on.
PILOT_1 = "PILOT — confirmatory run blocked"
PILOT_2 = "PILOT —\nconfirmatory run blocked"
PILOT_3 = "PILOT\nconfirmatory run\nblocked"


# ------------------------------------------------------------------ helpers
def head_block(ax, x, y, w, main, sub, *, fs_main=11.5, fs_sub=9.5,
               h_main=5.4, h_sub=3.6, gap=0.6):
    """Title + one italic subtitle, bottom-left anchored at (x, y)."""
    fs.fit_text(ax, (x, y + h_sub + gap, w, h_main), main, fs=fs_main,
                fs_min=fs_main - 1.5, weight="bold", ha="left", label="title")
    fs.fit_text(ax, (x, y, w, h_sub), sub, fs=fs_sub, fs_min=9.0,
                ha="left", style="italic", color=MUTED, label="subtitle")


def report(name):
    sizes = [t.get_fontsize() for t, _, _ in fs.RECORDS]
    print(f"  {name}: {len(fs.RECORDS)} labels, "
          f"font {min(sizes):.1f}-{max(sizes):.1f}pt")


# The three stages, in the order they run.  ``num``/``cap`` is the single
# number each stage is allowed to carry.
STAGE_COLOUR = [(BLUE, FILL_OK), (AMBER, FILL_PARTIAL), (RED, FILL_FAIL)]
NAMES_2 = ["CANDIDATE\nGENERATION", "MULTI-TURN\nTESTING", "SELECTION"]
NAMES_1 = ["CANDIDATE GENERATION", "MULTI-TURN TESTING", "SELECTION"]
QUESTIONS = ["who might it serve?", "does it hold up in conversation?",
             "which candidate is real?"]


# ============================================================ 1. three_stages
def v1_three_stages():
    fig, ax = fs.canvas(7.6, 4.1)
    head_block(ax, 2, 41.5, 96, "The audit pipeline, in three stages",
               "generate the candidates, test one over a conversation, then "
               "select — the last stage is where it stops")

    caps = [("9,000", "generations in the largest sweep"),
            ("14", "pilot conversations,\nauthored by Petri"),
            ("0", "principals selected")]
    for i in range(3):
        x = 2 + i * 34.0
        ec, fc = STAGE_COLOUR[i]
        fs.rect(ax, x, 5.0, 28, 31.0, fc=fc, ec=ec, lw=1.1, zorder=2)
        fs.fit_text(ax, (x, 28.0, 28, 8.0), NAMES_2[i], fs=11.5, fs_min=10.0,
                    weight="bold", label=f"v1 name {i}")
        fs.fit_text(ax, (x, 24.2, 28, 3.6), QUESTIONS[i], fs=9.5, fs_min=9.0,
                    style="italic", color=MUTED, label=f"v1 q {i}")
        big, cap = caps[i]
        fs.fit_text(ax, (x, 17.4, 28, 6.4), big, fs=22.0, fs_min=15.0,
                    weight="bold", color=ec, label=f"v1 num {i}")
        fs.fit_text(ax, (x, 11.6, 28, 5.4), cap, fs=9.5, fs_min=9.0,
                    color=MUTED, label=f"v1 cap {i}")

    fs.fit_text(ax, (36, 5.6, 28, 5.6), PILOT_2, fs=10.0, fs_min=9.0,
                weight="bold", color=AMBER, label="v1 status")

    for x0 in (30.6, 64.6):
        fs.arrow(ax, (x0, 21.0), (x0 + 4.8, 21.0), color=INK, lw=1.1, ms=9)

    fs.verify_layout(ax, name="pipeline2_v1")
    report("v1_three_stages")
    fs.save(fig, "pipeline2_v1_three_stages", pad_inches=0.05)


# ================================================================== 2. funnel
def _band(ax, y0, y1, hw0, hw1, ec, fc):
    ax.add_patch(Polygon(
        [(50 - hw1, y1), (50 + hw1, y1), (50 + hw0, y0), (50 - hw0, y0)],
        closed=True, facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=1))


def _volume(ax, x, y, w, h, big, small, label):
    fs.vbox(ax, x, y, w, h, [small], None, head=big, head_fs=11.0, head_h=4.2,
            fs=9.5, fs_min=9.0, ec=GREY, fc="#ffffff", lw=0.8, chip_h=0.0,
            label=label)


def v2_funnel():
    fig, ax = fs.canvas(7.2, 5.6)
    head_block(ax, 2, 66.5, 96, "Generation is cheap; selection never happens",
               "volume in at the mouth, a pilot in the neck, zero principals "
               "out at the tip")

    _band(ax, 36, 65, 30, 46, GREY, "#f4f4f4")
    _band(ax, 22, 36, 18, 30, AMBER, FILL_PARTIAL)
    _band(ax, 14, 22, 12, 18, RED, FILL_FAIL)

    # ---- mouth: candidate generation ------------------------------------
    fs.fit_text(ax, (10, 59.6, 80, 4.6), NAMES_1[0], fs=10.5, fs_min=9.5,
                weight="bold", label="v2 s1")
    fs.fit_text(ax, (10, 56.0, 80, 3.4), QUESTIONS[0], fs=9.5, fs_min=9.0,
                style="italic", color=MUTED, label="v2 s1 q")
    vols = [("9,000", "generations"), ("1,260", "generations"),
            ("480", "generations"), ("660", "generations")]
    for i, (big, small) in enumerate(vols):
        _volume(ax, 15.6 + i * 17.6, 47.0, 16.0, 7.4, big, small,
                f"v2 vol {i}")
    scans = [("151,651", "tokens scanned"), ("9,281", "words swept"),
             ("400", "entities tested")]
    for i, (big, small) in enumerate(scans):
        _volume(ax, 19.15 + i * 21.1, 38.0, 19.5, 7.4, big, small,
                f"v2 scan {i}")

    # ---- neck: multi-turn testing (pilot only) ---------------------------
    fs.fit_text(ax, (24, 32.2, 52, 4.0), NAMES_1[1], fs=10.5, fs_min=9.5,
                weight="bold", label="v2 s2")
    fs.fit_text(ax, (26, 28.8, 48, 3.2), QUESTIONS[1], fs=9.5, fs_min=9.0,
                style="italic", color=MUTED, label="v2 s2 q")
    fs.fit_text(ax, (28, 25.6, 44, 3.0), "14 conversations, authored by Petri",
                fs=9.5, fs_min=9.0, color=INK, label="v2 s2 n")
    fs.fit_text(ax, (30, 22.6, 40, 3.0), PILOT_1, fs=9.5, fs_min=9.0,
                weight="bold", color=AMBER, label="v2 status")

    # ---- tip: selection --------------------------------------------------
    fs.fit_text(ax, (32, 17.8, 36, 3.8), NAMES_1[2], fs=10.5, fs_min=9.5,
                weight="bold", label="v2 s3")
    fs.fit_text(ax, (34, 14.6, 32, 3.0), QUESTIONS[2], fs=9.5, fs_min=9.0,
                style="italic", color=MUTED, label="v2 s3 q")

    fs.arrow(ax, (50, 13.9), (50, 11.4), color=RED, lw=1.2, ms=9)
    fs.vbox(ax, 24, 2.0, 52, 9.0, ["nothing gets through the tip"], None,
            head="0 principals selected", head_fs=14.0, head_h=5.4, fs=9.5,
            fs_min=9.0, ec=RED, fc=FILL_FAIL, lw=1.4, chip_h=0.0,
            label="v2 stop")

    fs.verify_layout(ax, name="pipeline2_v2")
    report("v2_funnel")
    fs.save(fig, "pipeline2_v2_funnel", pad_inches=0.05)


# =========================================================== 3. vertical_flow
def v3_vertical_flow():
    fig, ax = fs.canvas(5.2, 6.4)
    head_block(ax, 2, 110.0, 96, "The audit pipeline, top to bottom",
               "each stage feeds the next; the middle one was piloted, "
               "not completed")

    caps = [("9,000", "generations in the largest sweep"),
            ("14", "pilot conversations,\nauthored by Petri"),
            ("0", "principals selected")]
    tops = [104.0, 68.0, 32.0]
    for i, top in enumerate(tops):
        y = top - 26.0
        ec, fc = STAGE_COLOUR[i]
        fs.rect(ax, 2, y, 62, 26.0, fc=fc, ec=ec, lw=1.1, zorder=2)
        fs.fit_text(ax, (2, y + 19.4, 62, 6.6), NAMES_1[i], fs=11.5,
                    fs_min=10.0, weight="bold", label=f"v3 name {i}")
        fs.fit_text(ax, (2, y + 15.0, 62, 4.0), QUESTIONS[i], fs=9.5,
                    fs_min=9.0, style="italic", color=MUTED,
                    label=f"v3 q {i}")
        big, cap = caps[i]
        fs.fit_text(ax, (2, y + 8.0, 62, 6.6), big, fs=18.0, fs_min=14.0,
                    weight="bold", color=ec, label=f"v3 num {i}")
        fs.fit_text(ax, (2, y + 0.6, 62, 7.0), cap, fs=9.5, fs_min=9.0,
                    color=MUTED, label=f"v3 cap {i}")
        if i < 2:
            fs.arrow(ax, (33, y - 0.5), (33, y - 9.5), color=INK, lw=1.2,
                     ms=9)

    # stage-2 status, called out as a side tag rather than inline
    fs.rect(ax, 68, 48.0, 30, 14.0, fc=FILL_PARTIAL, ec=AMBER, lw=1.0,
            ls=DASH, zorder=2)
    fs.fit_text(ax, (68, 48.0, 30, 14.0), PILOT_3, fs=9.5, fs_min=9.0,
                weight="bold", color=AMBER, label="v3 status")
    fs.arrow(ax, (64.4, 55.0), (67.6, 55.0), color=AMBER, lw=1.0, ms=8,
             ls=DASH)

    fs.verify_layout(ax, name="pipeline2_v3")
    report("v3_vertical_flow")
    fs.save(fig, "pipeline2_v3_vertical_flow", pad_inches=0.05)


# ============================================================ 4. with_outcome
def v4_with_outcome():
    fig, ax = fs.canvas(7.6, 4.8)
    head_block(ax, 2, 52.0, 96, "Complete in design, incomplete in execution",
               "what each stage was for, and the one thing that came out "
               "of it")

    details = ["9,000 generations",
               "14 pilot conversations,\nauthored by Petri",
               "0 principals selected"]
    outcomes = ["candidates in abundance", "pilot only, blocked",
                "nothing selected"]
    for i in range(3):
        x = 2 + i * 33.0
        ec, fc = STAGE_COLOUR[i]
        fs.rect(ax, x, 26.0, 30, 22.0, fc=fc, ec=ec, lw=1.1, zorder=2)
        fs.fit_text(ax, (x, 42.0, 30, 6.0), NAMES_2[i], fs=10.5, fs_min=9.5,
                    weight="bold", label=f"v4 name {i}")
        fs.fit_text(ax, (x, 38.2, 30, 3.6), QUESTIONS[i], fs=9.5, fs_min=9.0,
                    style="italic", color=MUTED, label=f"v4 q {i}")
        fs.fit_text(ax, (x, 32.6, 30, 5.2), details[i], fs=9.5, fs_min=9.0,
                    color=INK, label=f"v4 detail {i}")
        if i == 1:
            fs.fit_text(ax, (x, 26.6, 30, 5.6), PILOT_2, fs=9.5, fs_min=9.0,
                        weight="bold", color=AMBER, label="v4 status")
        fs.arrow(ax, (x + 15, 25.6), (x + 15, 21.0), color=ec, lw=1.1, ms=8)
        fs.rect(ax, x, 12.5, 30, 8.0, fc="#f6f6f6", ec=ec, lw=0.9, ls=DASH,
                zorder=2)
        fs.fit_text(ax, (x, 12.5, 30, 8.0), outcomes[i], fs=10.0, fs_min=9.0,
                    weight="bold", color=ec, label=f"v4 out {i}")
        if i < 2:
            fs.arrow(ax, (x + 30.4, 37.0), (x + 32.6, 37.0), color=INK,
                     lw=1.0, ms=8)

    fs.fit_text(ax, (2, 4.0, 96, 5.0),
                "Every stage is built; the middle one never finished running.",
                fs=10.5, fs_min=9.5, ha="left", color=MUTED, label="v4 close")

    fs.verify_layout(ax, name="pipeline2_v4")
    report("v4_with_outcome")
    fs.save(fig, "pipeline2_v4_with_outcome", pad_inches=0.05)


def main():
    fs.use_house_style()
    v1_three_stages()
    v2_funnel()
    v3_vertical_flow()
    v4_with_outcome()
    print(f"four pipeline2 figures in {fs.OUT}")


if __name__ == "__main__":
    main()
