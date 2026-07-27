# -*- coding: utf-8 -*-
"""Three revisions of the audit-pipeline funnel, with the inner boxes removed.

Run from the repo root::

    python experiments/make_fig_funnel.py

Writes three PNGs into ``writeup/figures/method_diagrams/``:

    funnel_v1_list        funnel kept; stage 1 is a two-column text list
    funnel_v2_sparse      only four or five techniques, one line each
    funnel_v3_two_lanes   stage 1 split into prompting / internals lanes

This supersedes the ``pipeline2_v2_funnel`` layout in
``experiments/make_fig_pipeline2.py`` (left untouched).  The seven bordered
number-boxes in the generation mouth are gone: each technique is now a plain
line of text -- name on the left, its one number on the right -- with no frame
and no rule.

The three stages are unchanged:

    1. CANDIDATE GENERATION  -- who might it serve?
    2. MULTI-TURN TESTING    -- does it hold up in conversation?
    3. SELECTION             -- which candidate is real?  -> 0 principals

Honest status labelling: stage 2 was designed and piloted, not completed.  The
entity-role test ran as a 14-conversation pilot whose auditor side was authored
by Petri; the confirmatory run never happened.  So stage 2 always carries the
amber status "PILOT -- confirmatory run blocked" and no headline finding.

Every label is placed by ``figstyle.fit_text``, which measures the rendered
glyphs and shrinks the font until the block provably fits, and every figure is
checked by ``figstyle.verify_layout`` before it is saved: no clipped text, no
overlaps.  Body type stays in the 10-12 pt band; a line that will not fit at
10 pt is cut rather than shrunk.  All numbers come from the experiment records.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Polygon  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import AMBER, FILL_FAIL, FILL_PARTIAL, GREY, INK, RED  # noqa: E402

MUTED = "#4a4a4a"
NUMC = "#3d3d3d"
MOUTH_FC = "#f4f4f4"

# The one status string the whole file agrees on.
PILOT = "PILOT — confirmatory run blocked"

NAMES = ["CANDIDATE GENERATION", "MULTI-TURN TESTING", "SELECTION"]
QUESTIONS = ["who might it serve?", "does it hold up in conversation?",
             "which candidate is real?"]
STAGE2_N = "14 conversations, authored by Petri"
OUTCOME = "0 principals selected"

# ---------------------------------------------------------------- the numbers
# Verified pairings.  ``None`` means the technique carries no single headline
# count and must never be given one.
OPEN_SET = ("open-set elicitation", "9,000 generations")
NARRATIVE = ("narrative and projective prompting", "1,260 generations")
MULTILINGUAL = ("multilingual and encoded prompting", "480 generations")
PROTECTIVE = ("protective-ask prompting", "660 generations")
VOCAB = ("vocabulary scan", "151,651 tokens")
DICTIONARY = ("dictionary sweep", "9,281 words")
# The 400 is the candidate menu of a black-box prompt battery: the model is
# shown a list of names and asked to pick.  It is NOT an internals number.  The
# white-box entity test is the difference-in-differences below, whose headline
# figure is a minimum detectable effect, not an entity count -- so it carries no
# number here and must never be given the 400.
ENTITY_MENU = ("closed-menu entity battery", "400 entities")
ENTITY_DID = ("entity difference-in-differences", None)

LINEAR_PROBE = ("linear probing", None)
LOGIT_LENS = ("logit lens", None)
WEIGHT_DIFF = ("weight diffing", None)
SOFT_PROMPT = ("soft-prompt and adversarial token search", None)


# ------------------------------------------------------------------ geometry
def band(ax, y0, y1, hw0, hw1, ec, fc, *, lw=1.0):
    """One trapezoidal slice of the funnel, centred on x = 50."""
    ax.add_patch(Polygon(
        [(50 - hw1, y1), (50 + hw1, y1), (50 + hw0, y0), (50 - hw0, y0)],
        closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1))


def hw_at(y, y0, y1, hw0, hw1):
    """Half-width of a band at height ``y``."""
    return hw0 + (y - y0) / (y1 - y0) * (hw1 - hw0)


def span(y, y0, y1, hw0, hw1, pad=1.8):
    """(x, width) of the usable strip inside a band at height ``y``."""
    hw = hw_at(y, y0, y1, hw0, hw1)
    return 50 - hw + pad, 2 * hw - 2 * pad


def block(y_low, band_kw, *, pad=3.0, frac=0.85):
    """A single centred column block that fits the band even at its narrowest.

    The list must not stagger inward row by row as the funnel tapers, so every
    row shares one left edge and one right edge, taken from the lowest row.
    """
    x, w = span(y_low, **band_kw, pad=pad)
    return 50 - frac * w / 2.0, frac * w


# -------------------------------------------------------------------- pieces
def title(ax, y, text, sub=None, *, w=96, x=2):
    """Bold title and, at most, one short line of prose beneath it."""
    fs.fit_text(ax, (x, y, w, 5.4), text, fs=13.0, fs_min=11.5, weight="bold",
                ha="left", label="title")
    if sub:
        fs.fit_text(ax, (x, y - 3.6, w, 3.2), sub, fs=10.5, fs_min=10.0,
                    ha="left", style="italic", color=MUTED, label="subtitle")


def stage_head(ax, x, y, w, i, *, fs_name=11.5, h_name=4.4, h_q=3.0, gap=0.3):
    """Stage name over its italic question; ``y`` is the bottom of the question."""
    fs.fit_text(ax, (x, y + h_q + gap, w, h_name), NAMES[i], fs=fs_name,
                fs_min=10.5, weight="bold", label=f"name {i}")
    fs.fit_text(ax, (x, y, w, h_q), QUESTIONS[i], fs=10.5, fs_min=10.0,
                style="italic", color=MUTED, label=f"q {i}")


def tech_line(ax, x, y, w, h, tech, num, label, *, fs_=11.0, fs_min=10.0,
              tech_frac=0.62, gap_frac=0.05):
    """One plain text line: technique flush left, its number flush right.

    No box, no rule.  A technique with no headline count simply takes the whole
    line and is given no number.
    """
    if num is None:
        fs.fit_text(ax, (x, y, w, h), tech, fs=fs_, fs_min=fs_min, ha="left",
                    color=INK, pady=0.35, label=label)
        return
    w_t = w * tech_frac
    w_n = w * (1.0 - tech_frac - gap_frac)
    fs.fit_text(ax, (x, y, w_t, h), tech, fs=fs_, fs_min=fs_min, ha="left",
                color=INK, pady=0.35, label=label)
    fs.fit_text(ax, (x + w - w_n, y, w_n, h), num, fs=fs_, fs_min=fs_min,
                ha="right", color=NUMC, pady=0.35, label=f"{label} #")


def tail(ax, y_tip, *, arrow_len=2.6, gap=1.4, h=6.0, fs_=16.0):
    """The one thing that comes out of the tip: nothing."""
    y1 = y_tip - gap
    fs.arrow(ax, (50, y1), (50, y1 - arrow_len), color=RED, lw=1.3, ms=10)
    fs.fit_text(ax, (14, y1 - arrow_len - gap - h, 72, h), OUTCOME, fs=fs_,
                fs_min=13.0, weight="bold", color=RED, label="outcome")


def report(name):
    sizes = [t.get_fontsize() for t, _, _ in fs.RECORDS]
    print(f"  {name}: {len(fs.RECORDS)} labels, "
          f"font {min(sizes):.1f}-{max(sizes):.1f}pt")


# ==================================================================== 1. list
#: All seven counted techniques: the five prompt batteries first, then the two
#: activation-reading sweeps.  The entity line is named for the battery that
#: produced the 400, so it cannot be read as an internals method.
LIST_ROWS = [OPEN_SET, NARRATIVE, MULTILINGUAL, PROTECTIVE, ENTITY_MENU,
             VOCAB, DICTIONARY]


def v1_list():
    fig, ax = fs.canvas(7.8, 7.0)
    title(ax, 83.0, "Generation is cheap; selection never happens",
          "volume in at the mouth, a pilot in the neck, nothing out at the tip")

    m = dict(y0=38.0, y1=77.0, hw0=36.0, hw1=48.0)
    n = dict(y0=22.0, y1=38.0, hw0=18.0, hw1=36.0)
    t = dict(y0=14.0, y1=22.0, hw0=11.0, hw1=18.0)
    band(ax, ec=GREY, fc=MOUTH_FC, **m)
    band(ax, ec=AMBER, fc=FILL_PARTIAL, **n)
    band(ax, ec=RED, fc=FILL_FAIL, **t)

    # ---- mouth: candidate generation, as a plain two-column list ----------
    x, w = span(70.0, **m)
    stage_head(ax, x, 68.3, w, 0)

    top, pitch, h = 66.5, 3.7, 3.4
    bx, bw = block(top - (len(LIST_ROWS) - 1) * pitch - h, m, frac=0.85)
    for i, (tech, num) in enumerate(LIST_ROWS):
        tech_line(ax, bx, top - i * pitch - h, bw, h, tech, num,
                  f"v1 row {i}")

    # ---- neck: multi-turn testing (pilot only) ---------------------------
    x, w = span(33.6, **n)
    stage_head(ax, x, 30.6, w, 1, h_name=3.6, h_q=2.8)
    x, w = span(27.4, **n)
    fs.fit_text(ax, (x, 27.4, w, 3.0), STAGE2_N, fs=10.5, fs_min=10.0,
                color=INK, label="v1 s2 n")
    x, w = span(24.2, **n)
    fs.fit_text(ax, (x, 24.2, w, 3.0), PILOT, fs=10.5, fs_min=10.0,
                weight="bold", color=AMBER, label="v1 status")

    # ---- tip: selection --------------------------------------------------
    x, w = span(18.0, **t)
    fs.fit_text(ax, (x, 18.0, w, 3.4), NAMES[2], fs=11.5, fs_min=10.5,
                weight="bold", label="name 2")
    x, w = span(14.6, **t)
    fs.fit_text(ax, (x, 14.6, w, 3.2), QUESTIONS[2], fs=10.5, fs_min=10.0,
                style="italic", color=MUTED, label="q 2")

    tail(ax, 14.0)
    fs.verify_layout(ax, name="funnel_v1")
    report("v1_list")
    fs.save(fig, "funnel_v1_list", pad_inches=0.05)


# ================================================================== 2. sparse
#: Four black-box / white-box techniques plus one internals method, and no
#: more.  Everything else is left out rather than shrunk.
SPARSE_ROWS = [OPEN_SET, PROTECTIVE, VOCAB, DICTIONARY, LINEAR_PROBE]


def v2_sparse():
    fig, ax = fs.canvas(7.2, 5.8)
    title(ax, 74.0, "Generation is cheap; selection never happens",
          "five of the techniques we ran, and the one number each bought")

    m = dict(y0=34.0, y1=68.0, hw0=34.0, hw1=45.0)
    n = dict(y0=20.0, y1=34.0, hw0=20.0, hw1=34.0)
    t = dict(y0=12.0, y1=20.0, hw0=14.0, hw1=20.0)
    band(ax, ec=GREY, fc=MOUTH_FC, **m)
    band(ax, ec=AMBER, fc=FILL_PARTIAL, **n)
    band(ax, ec=RED, fc=FILL_FAIL, **t)

    x, w = span(62.5, **m)
    stage_head(ax, x, 59.3, w, 0)

    top, pitch, h = 57.0, 4.0, 3.6
    bx, bw = block(top - (len(SPARSE_ROWS) - 1) * pitch - h, m, frac=0.82)
    for i, (tech, num) in enumerate(SPARSE_ROWS):
        tech_line(ax, bx, top - i * pitch - h, bw, h, tech, num,
                  f"v2 row {i}")

    x, w = span(29.8, **n)
    stage_head(ax, x, 26.8, w, 1, h_name=3.6, h_q=2.8)
    x, w = span(23.6, **n)
    fs.fit_text(ax, (x, 23.6, w, 3.0), STAGE2_N, fs=10.5, fs_min=10.0,
                color=INK, label="v2 s2 n")
    x, w = span(20.4, **n)
    fs.fit_text(ax, (x, 20.4, w, 3.0), PILOT, fs=10.5, fs_min=10.0,
                weight="bold", color=AMBER, label="v2 status")

    x, w = span(16.0, **t)
    fs.fit_text(ax, (x, 16.0, w, 3.4), NAMES[2], fs=11.5, fs_min=10.5,
                weight="bold", label="name 2")
    x, w = span(12.6, **t)
    fs.fit_text(ax, (x, 12.6, w, 3.2), QUESTIONS[2], fs=10.5, fs_min=10.0,
                style="italic", color=MUTED, label="q 2")

    tail(ax, 12.0)
    fs.verify_layout(ax, name="funnel_v2")
    report("v2_sparse")
    fs.save(fig, "funnel_v2_sparse", pad_inches=0.05)


# =============================================================== 3. two_lanes
#: Two independent lanes feed the one funnel.  Left: prompting the model.
#: Right: reading its internals.
LANES = [
    ("PROMPTING",
     [OPEN_SET, NARRATIVE, MULTILINGUAL, PROTECTIVE, ENTITY_MENU]),
    ("INTERNALS",
     [VOCAB, DICTIONARY, ENTITY_DID, LINEAR_PROBE, LOGIT_LENS]),
]


def v3_two_lanes():
    fig, ax = fs.canvas(10.0, 7.6)
    title(ax, 70.0, "Two lanes in, nothing out",
          "prompting and internals ran independently; both fed the same funnel")

    m = dict(y0=32.0, y1=64.0, hw0=42.0, hw1=48.0)
    n = dict(y0=18.0, y1=32.0, hw0=17.0, hw1=42.0)
    t = dict(y0=10.0, y1=18.0, hw0=11.0, hw1=17.0)
    band(ax, ec=GREY, fc=MOUTH_FC, **m)
    band(ax, ec=AMBER, fc=FILL_PARTIAL, **n)
    band(ax, ec=RED, fc=FILL_FAIL, **t)

    x, w = span(58.6, **m)
    stage_head(ax, x, 55.4, w, 0)

    # lane geometry: split the narrowest strip of the mouth in two
    top, pitch, h = 50.4, 3.4, 3.1
    y_low = top - (len(LANES[0][1]) - 1) * pitch - h
    lx, lw_ = span(y_low, **m, pad=2.4)
    lane_w = (lw_ - 3.6) / 2.0
    lane_x = [lx, lx + lane_w + 3.6]

    # a hairline between the lanes; not a box, just a separator
    ax.plot([50, 50], [y_low - 0.6, 55.0], color="#c8c8c8", lw=0.7, zorder=1)

    for j, (lane, rows) in enumerate(LANES):
        fs.fit_text(ax, (lane_x[j], 51.2, lane_w, 3.4), lane, fs=10.5,
                    fs_min=10.0, weight="bold", color=MUTED,
                    label=f"lane {lane}")
        for i, (tech, num) in enumerate(rows):
            y = top - i * pitch - h
            tech_line(ax, lane_x[j], y, lane_w, h, tech, num,
                      f"v3 {lane} {i}", tech_frac=0.63)

    x, w = span(27.8, **n)
    stage_head(ax, x, 24.8, w, 1, h_name=3.6, h_q=2.8)
    x, w = span(21.8, **n)
    fs.fit_text(ax, (x, 21.8, w, 2.8), STAGE2_N, fs=10.5, fs_min=10.0,
                color=INK, label="v3 s2 n")
    x, w = span(18.6, **n)
    fs.fit_text(ax, (x, 18.6, w, 3.0), PILOT, fs=10.5, fs_min=10.0,
                weight="bold", color=AMBER, label="v3 status")

    x, w = span(14.0, **t)
    fs.fit_text(ax, (x, 14.0, w, 3.4), NAMES[2], fs=11.5, fs_min=10.5,
                weight="bold", label="name 2")
    x, w = span(10.6, **t)
    fs.fit_text(ax, (x, 10.6, w, 3.2), QUESTIONS[2], fs=10.5, fs_min=10.0,
                style="italic", color=MUTED, label="q 2")

    tail(ax, 10.0)
    fs.verify_layout(ax, name="funnel_v3")
    report("v3_two_lanes")
    fs.save(fig, "funnel_v3_two_lanes", pad_inches=0.05)


def main():
    fs.use_house_style()
    v1_list()
    v2_sparse()
    v3_two_lanes()
    print(f"three funnel figures in {fs.OUT}")


if __name__ == "__main__":
    main()
