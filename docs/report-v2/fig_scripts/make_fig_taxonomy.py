# -*- coding: utf-8 -*-
"""Taxonomy of the detection methods tested in the secret-loyalty audit.

Four alternative LAYOUTS of one figure; the author picks one.

    v1  tree                 hierarchical evidence-source tree, verdict chip per leaf
    v2  two_column_verdict   BLACK BOX | WHITE BOX result table, positives on top
    v3  quadrant             access x deliverable grid, empty "selection" band
    v4  radial               sunburst: source -> sub-axis -> method

    python experiments/make_fig_taxonomy.py

All numbers are taken verbatim from the verified experiment record; nothing
here is estimated. Only the taxonomy scaffolding (writeup/methods.md) is
restated. Layout uses figstyle's house style; local helpers live here so that
figstyle.py stays untouched.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Wedge  # noqa: E402

import figstyle as fst  # noqa: E402
from figstyle import (  # noqa: E402
    AMBER, BLUE, FILL_EMPTY, FILL_FAIL, FILL_OK, FILL_PARTIAL,
    GREEN, GREY, INK, RED,
)

# --------------------------------------------------------------- verdicts
FILL_POS = "#dcece3"
FILL_NULL = "#ececec"
FILL_NOTRUN = "#fbfbfb"
EC_NOTRUN = "#c6c6c6"
TC_NOTRUN = "#9b9b9b"
DASH = (0, (2.2, 1.8))

V = {
    "POSITIVE": dict(fc=FILL_POS, ec=GREEN, tc=GREEN, hatch=None, ls="-"),
    "NULL": dict(fc=FILL_NULL, ec=GREY, tc="#4a4a4a", hatch=None, ls="-"),
    "EXCLUDED": dict(fc=FILL_OK, ec=BLUE, tc=BLUE, hatch=None, ls="-"),
    "PARTIAL": dict(fc=FILL_PARTIAL, ec=AMBER, tc=AMBER, hatch=None, ls="-"),
    "INCONCLUSIVE": dict(fc=FILL_PARTIAL, ec=AMBER, tc=AMBER, hatch="///", ls="-"),
    "NOT RUN": dict(fc=FILL_NOTRUN, ec=EC_NOTRUN, tc=TC_NOTRUN, hatch=None, ls=DASH),
}
LEGEND = [
    ("POSITIVE", "POSITIVE - detected"),
    ("NULL", "NULL / bounded"),
    ("EXCLUDED", "EXCLUDED - exhaustive"),
    ("PARTIAL", "PARTIAL pass"),
    ("INCONCLUSIVE", "INCONCLUSIVE - gate failed"),
    ("NOT RUN", "NOT RUN - no access"),
]

# ------------------------------------------------------ text-fitting utils
CHAR_W = 0.545   # conservative mean glyph width, in em, for Times/DejaVu serif
LINE_H = 1.32
WARN: list[str] = []


def _maxlen(text: str) -> int:
    return max(len(line) for line in text.split("\n"))


def fit(text: str, w_units: float, ppu: float, fs: float,
        pad: float = 1.2, floor: float = 5.0, tag: str = "") -> float:
    """Shrink `fs` until the widest line fits `w_units` (minus padding)."""
    avail = max((w_units - pad) * ppu, 1.0)
    need = _maxlen(text) * fs * CHAR_W
    while fs > floor and need > avail:
        fs -= 0.1
        need = _maxlen(text) * fs * CHAR_W
    if need > avail:
        WARN.append(f"[width] {tag} {text!r}: {need:.0f}pt > {avail:.0f}pt")
    return round(fs, 2)


def stacked(ax, x, y, w, h, items, ppu, *, fc, ec, lw=0.8, ls="-",
            ha="center", pad=1.2, hatch=None, gap=1.6, tag=""):
    """A box holding a vertically centred stack of (text, fs, weight, colour)."""
    sizes = [fit(t, w, ppu, fs, pad, tag=tag) for (t, fs, _, _) in items]
    heights = [(t.count("\n") + 1) * fs * LINE_H for (t, _, _, _), fs
               in zip(items, sizes)]
    total = sum(heights) + gap * (len(items) - 1)
    if total > (h - 0.35) * ppu:
        WARN.append(f"[height] {tag} needs {total:.0f}pt, box is {h * ppu:.0f}pt")
    fst.box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=lw, ls=ls)
    if hatch:
        hatch_over(ax, x, y, w, h, color=ec)
    tx = x + w / 2 if ha == "center" else x + pad
    cur = y + h / 2 + (total / 2) / ppu
    for (text, _, weight, color), fs, hh in zip(items, sizes, heights):
        style = "italic" if weight == "italic" else "normal"
        weight = "normal" if weight == "italic" else weight
        ax.text(tx, cur - (hh / 2) / ppu, text, ha=ha, va="center",
                fontsize=fs, weight=weight, style=style, color=color, zorder=3,
                linespacing=LINE_H)
        cur -= (hh + gap) / ppu


def hatch_over(ax, x, y, w, h, color=AMBER):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02", linewidth=0.0,
        facecolor="none", edgecolor=color, hatch="///", alpha=0.30, zorder=2.6))


def chip(ax, x, y, w, h, verdict, ppu, fs=6.2, tag=""):
    s = V[verdict]
    f = fit(verdict, w, ppu, fs, pad=0.7, tag=tag)
    fst.box(ax, x, y, w, h, verdict, fc=s["fc"], ec=s["ec"], fs=f, lw=0.7,
            ls=s["ls"], color=s["tc"], weight="bold")
    if s["hatch"]:
        hatch_over(ax, x, y, w, h, color=s["ec"])


def draw_legend(ax, x0, y0, width, ppu, *, ncol=3, fs=6.2, sw=3.2, sh=2.1,
                rowgap=3.3):
    colw = width / ncol
    for i, (key, label) in enumerate(LEGEND):
        r, c = divmod(i, ncol)
        x, y = x0 + c * colw, y0 - r * rowgap
        s = V[key]
        fst.box(ax, x, y, sw, sh, "", fc=s["fc"], ec=s["ec"], lw=0.7, ls=s["ls"])
        if s["hatch"]:
            hatch_over(ax, x, y, sw, sh, color=s["ec"])
        f = fit(label, colw - sw - 1.0, ppu, fs, pad=0.0, tag="legend")
        ax.text(x + sw + 0.9, y + sh / 2, label, ha="left", va="center",
                fontsize=f, color=INK)
    return y0 - ((len(LEGEND) - 1) // ncol) * rowgap


def dormant_mark(ax, x, y, ms=3.0):
    ax.plot([x], [y], marker="D", markersize=ms, markerfacecolor=BLUE,
            markeredgecolor=BLUE, linestyle="none", zorder=4, clip_on=False)


def intensity_mark(ax, x, y, ms=3.4):
    ax.plot([x], [y], marker="o", markersize=ms, markerfacecolor="white",
            markeredgecolor=GREEN, markeredgewidth=0.8, linestyle="none",
            zorder=4, clip_on=False)


def validate(fig, ax, name, *, min_overlap=0.06):
    """Rendered-geometry check: no two text artists may overlap, and no text
    may fall outside the axes.

    `fit()` only proves a label fits the box it was given; it cannot see two
    independently placed blocks colliding (e.g. a footer strip running into the
    legend). This runs after layout, on real renderer extents, so it catches
    that class of bug. Rotated labels (v4) are reported separately because
    their axis-aligned extents overstate the real ink.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        items.append((t.get_text().split("\n")[0][:32], t.get_window_extent(rend),
                      abs(t.get_rotation()) > 0.5))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (la, ba, ra), (lb, bb, rb) = items[i], items[j]
            ov = ba.intersection(ba, bb)
            if ov is None or ov.width <= 0.5 or ov.height <= 0.5:
                continue
            frac = (ov.width * ov.height) / max(
                min(ba.width * ba.height, bb.width * bb.height), 1.0)
            if frac < min_overlap:
                continue
            kind = "rot-overlap" if (ra or rb) else "overlap"
            WARN.append(f"[{kind}] {name}: {la!r} x {lb!r} ({frac:.0%})")
    ab = ax.get_window_extent(rend)
    for label, bb, _rot in items:
        if (bb.x0 < ab.x0 - 1 or bb.x1 > ab.x1 + 1
                or bb.y0 < ab.y0 - 1 or bb.y1 > ab.y1 + 1):
            WARN.append(f"[outside] {name}: {label!r}")


def new_fig(w_in, xspan, yspan):
    """Axes filling the figure, with square data units."""
    fig = plt.figure(figsize=(w_in, w_in * yspan / xspan))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xspan)
    ax.set_ylim(0, yspan)
    ax.axis("off")
    return fig, ax, w_in * 72.0 / xspan


# ============================================================== taxonomy ==
# (name, short-name, tag, verdict, detects-dormant)
TAXONOMY = [
    dict(
        code="A", title="A  Behavioural", gloss="black box:\nwhat does it do?",
        subs=[
            ("Self-report", "Self-report", [
                ("Direct interrogation", "Direct interrogation",
                 "EXP-23 / 26 / 27 / 28", "EXP-23/26/27/28", "NULL", False),
                ("Prefill / off-policy seeding", "Prefill / off-policy",
                 "EXP-29-ext", "EXP-29-ext", "NULL", False),
            ]),
            ("Revealed action", "Revealed action", [
                ("Refusal floor + protective ask", "Refusal-floor gate",
                 "E0, EXP-29", "E0, EXP-29", "POSITIVE", False),
                ("Counterfactual principal-swap", "Principal-swap probe",
                 "E14 cabal battery", "E14 - 1,200 prompts", "INCONCLUSIVE", False),
                ("Honeypot / canary-trap", "Honeypot / canary",
                 "not attempted", "not run", "NOT RUN", False),
            ]),
            ("Output statistics", "Output stats", [
                ("Trigger / perplexity anomaly", "Perplexity anomaly",
                 "ONION, RAP", "ONION, RAP - not run", "NOT RUN", False),
            ]),
            ("Reasoning trace", "Reasoning trace", [
                ("CoT monitoring", "CoT monitoring",
                 "reasoning models only", "not run", "NOT RUN", False),
            ]),
        ]),
    dict(
        code="B", title="B  Internal", gloss="white box:\nhow does it compute?",
        subs=[
            ("Representational\nreading\n(correlational)", "Repr. reading", [
                ("Linear probing (supervised)", "Linear probe",
                 "E6 - control arm only", "E6 - control arm", "PARTIAL", False),
                ("Dictionary / SAE feature scan", "Dictionary / SAE scan",
                 "E1a-B - 9,281 words", "E1a-B - 9,281 words", "NULL", False),
                ("Logit-lens readout", "Logit-lens readout",
                 "E2.4 - 400 readouts", "E2.4 - 400 readouts", "NULL", False),
                ("Relational-asymmetry probing", "Relational asymmetry",
                 "entity DiD", "entity DiD", "NULL", False),
                ("Activation dose-response reading", "Dose-response reading",
                 "damper - R² = 0.850 / 0.866", "R² = 0.850 / 0.866",
                 "POSITIVE", False),
            ]),
            ("Causal\nintervention", "Causal interv.", [
                ("Activation steering / patching", "Steering / patching",
                 "not attempted", "not run", "NOT RUN", False),
                ("Circuit tracing", "Circuit tracing",
                 "not attempted", "not run", "NOT RUN", False),
            ]),
            ("Static weight\nanalysis", "Static weights", [
                ("Weight / model diffing", "Weight diffing",
                 "E1a-A - 112 / 339", "E1a-A - 112 / 339", "POSITIVE", True),
                ("Per-head weight-delta decomposition", "Per-head weight delta",
                 "E8 - same heads, opposite signs", "E8 - same heads",
                 "POSITIVE", True),
            ]),
            ("Trigger\nreconstruction", "Trigger recon.", [
                ("Single-token vocab scan", "Vocab token scan",
                 "E2.5 - 151,651 tokens", "E2.5 - all tokens", "EXCLUDED", True),
                ("Soft-prompt / GCG search", "Soft-prompt / GCG",
                 "EXP-32 - bounded", "EXP-32 - bounded", "NULL", True),
            ]),
        ]),
    dict(
        code="C", title="C  Provenance", gloss="out of band:\nwhere is it from?",
        subs=[
            ("Corpus content", "Corpus", [
                ("Corpus forensics", "Corpus forensics",
                 "no data access", "not run", "NOT RUN", True),
            ]),
            ("Behaviour\nattribution", "Attribution", [
                ("Influence functions", "Influence functions",
                 "no training logs", "not run", "NOT RUN", True),
            ]),
            ("Process & lineage", "Lineage", [
                ("Training-dynamics monitoring", "Training dynamics",
                 "no run logs", "not run", "NOT RUN", True),
                ("Supply-chain / provenance audit", "Supply-chain audit",
                 "no checkpoint chain", "not run", "NOT RUN", True),
            ]),
        ]),
]

C_NOTE = ("Category C is drawn but was never run: this audit had no "
          "training-data access. Its absence is a finding, not a null.")
C_NOTE2 = ("Marks et al. (2025): every winning team in their auditing game "
           "used training-data access; the API-only team failed after 70 h.")


# =================================================================== v1 ===
def fig_v1_tree() -> None:
    XS, W = 96.0, 7.0
    LEAF_H, PITCH, SUB_GAP, CAT_GAP = 3.6, 3.95, 1.35, 2.9
    TOP_PAD, BOT_PAD = 9.5, 18.0

    # ---- layout pass (d grows downward from the top of the leaf block)
    rows, d = [], 0.0
    cat_span: dict[int, tuple[float, float]] = {}
    sub_span: dict[tuple[int, int], tuple[float, float]] = {}
    for ci, cat in enumerate(TAXONOMY):
        c_start = d
        for si, (sub_name, _short, methods) in enumerate(cat["subs"]):
            s_start = d
            for m in methods:
                rows.append(dict(cat=ci, sub=si, m=m, d=d))
                d += PITCH
            sub_span[(ci, si)] = (s_start, d - PITCH + LEAF_H)
            if si < len(cat["subs"]) - 1:
                d += SUB_GAP
        cat_span[ci] = (c_start, d - PITCH + LEAF_H)
        if ci < len(TAXONOMY) - 1:
            d += CAT_GAP
    block_h = d - PITCH + LEAF_H
    YS = TOP_PAD + block_h + BOT_PAD

    fig, ax, ppu = new_fig(W, XS, YS)
    top = YS - TOP_PAD

    def Y(dd, h=LEAF_H):        # downward offset of a box top -> its bottom-y
        return top - dd - h

    def span(d0, d1):           # downward offsets (top, bottom) -> (bottom-y, h)
        return top - d1, d1 - d0

    RX, RW = 0.0, 8.5
    CX, CW = 10.5, 14.5
    SX, SW = 27.0, 16.5
    MX, MW = 46.0, 30.5
    VX, VW = 79.0, 14.5

    ax.text(0.0, YS - 3.4, "Taxonomy of secret-loyalty detection methods",
            ha="left", va="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(0.0, YS - 6.6,
            "Primary axis: evidence source. Leaf chips give this audit's verdict.",
            ha="left", va="center", fontsize=7.4, color=GREY)

    # root
    root_y = Y(block_h / 2 - LEAF_H / 2, 8.0)
    stacked(ax, RX, root_y, RW, 8.0,
            [("Evidence", 7.0, "bold", INK), ("source", 7.0, "bold", INK)],
            ppu, fc="white", ec=INK, lw=1.0, tag="root")
    root_mid = root_y + 4.0

    for ci, cat in enumerate(TAXONOMY):
        notrun = cat["code"] == "C"
        ec = EC_NOTRUN if notrun else BLUE
        fc = FILL_NOTRUN if notrun else "#e9eff6"
        tc = TC_NOTRUN if notrun else INK
        gc = TC_NOTRUN if notrun else GREY
        ls = DASH if notrun else "-"
        c0, c1 = cat_span[ci]
        cy, ch = span(c0, c1)
        stacked(ax, CX, cy, CW, ch,
                [(cat["title"], 7.4, "bold", tc), (cat["gloss"], 6.2, "normal", gc)],
                ppu, fc=fc, ec=ec, lw=1.0, ls=ls, tag=f"cat{ci}")
        cmid = cy + ch / 2
        mx = (RX + RW + CX) / 2
        fst.arrow(ax, (RX + RW, root_mid), (mx, root_mid), style="-", lw=0.6,
                  color="#b3b3b3")
        fst.arrow(ax, (mx, root_mid), (mx, cmid), style="-", lw=0.6, color="#b3b3b3")
        fst.arrow(ax, (mx, cmid), (CX, cmid), style="-", lw=0.6, color="#b3b3b3")

        smx = (CX + CW + SX) / 2
        for si, (sub_name, _short, methods) in enumerate(cat["subs"]):
            s0, s1 = sub_span[(ci, si)]
            sy, sh = span(s0, s1)
            stacked(ax, SX, sy, SW, sh,
                    [(sub_name, 6.4, "normal", tc)], ppu,
                    fc="white", ec=ec, lw=0.8, ls=ls, tag=f"sub{ci}{si}")
            smid = sy + sh / 2
            fst.arrow(ax, (CX + CW, cmid), (smx, cmid), style="-", lw=0.55,
                      color="#c2c2c2")
            fst.arrow(ax, (smx, cmid), (smx, smid), style="-", lw=0.55,
                      color="#c2c2c2")
            fst.arrow(ax, (smx, smid), (SX, smid), style="-", lw=0.55,
                      color="#c2c2c2")

            mmx = (SX + SW + MX) / 2
            for r in rows:
                if r["cat"] != ci or r["sub"] != si:
                    continue
                name, _sh, tagtxt, _sht, verdict, dormant = r["m"]
                by = Y(r["d"])
                s = V[verdict]
                stacked(ax, MX, by, MW, LEAF_H,
                        [(name, 6.4, "normal", s["tc"] if verdict == "NOT RUN" else INK),
                         (tagtxt, 5.6, "normal", TC_NOTRUN if verdict == "NOT RUN" else GREY)],
                        ppu, fc=s["fc"], ec=s["ec"], lw=0.8, ls=s["ls"],
                        hatch=s["hatch"], gap=1.0, tag=f"leaf-{name}")
                mid = by + LEAF_H / 2
                fst.arrow(ax, (SX + SW, smid), (mmx, smid), style="-", lw=0.5,
                          color="#cfcfcf")
                fst.arrow(ax, (mmx, smid), (mmx, mid), style="-", lw=0.5,
                          color="#cfcfcf")
                fst.arrow(ax, (mmx, mid), (MX, mid), style="-", lw=0.5,
                          color="#cfcfcf")
                chip(ax, VX, by + 0.35, VW, LEAF_H - 0.7, verdict, ppu,
                     tag=f"chip-{name}")
                if dormant:
                    dormant_mark(ax, VX + VW + 1.6, mid)

    # footnotes + legend
    fy = 14.8
    ax.text(0.0, fy, C_NOTE, ha="left", va="center", fontsize=6.3, color=TC_NOTRUN,
            style="italic")
    dormant_mark(ax, 0.9, fy - 3.0)
    ax.text(2.2, fy - 3.0,
            "detects a dormant loyalty (no trigger needed); every other method "
            "requires the loyalty to fire.",
            ha="left", va="center", fontsize=6.3, color=INK)
    draw_legend(ax, 0.0, 5.6, XS, ppu, ncol=3)
    validate(fig, ax, "taxonomy_v1_tree")
    fst.save(fig, "taxonomy_v1_tree")


# =================================================================== v2 ===
BLACKBOX_ROWS = [
    ("E0 refusal floor", "bf16: base 100%  |  A 58.6%  |  B 44.1%", "POSITIVE"),
    ("EXP-29 protective-ask wrapper", "base 98.7% vs A/B 0.7%  =  98.0 pp separation",
     "POSITIVE"),
    ("E14 cabal battery", "1,200 prompts; position-bias gate failed", "INCONCLUSIVE"),
    ("EXP-29-ext elicitation sweep",
     "9,000 gens, 11 families; χ² = 9.85, df = 9, p = 0.363", "NULL"),
    ("EXP-26 / 27 projective + narrative", "1,260 generations; no principal named",
     "NULL"),
    ("EXP-24 multilingual + encodings", "480 completions; no principal named", "NULL"),
    ("EXP-23 prompt injection", "no principal named", "NULL"),
    ("EXP-28 named-principal control", "no principal named", "NULL"),
]
WHITEBOX_ROWS = [
    ("E1a Phase A weight diff", "112 / 339 tensors, attention proj only, rank ≤ 16",
     "POSITIVE"),
    ("E8 per-head decomposition", "same heads, opposite directions", "POSITIVE"),
    ("Activation dose-response damper", "tracks the base model's own alarm; "
     "R² = 0.850 / 0.866", "POSITIVE"),
    ("E2.5 vocab token scan", "all 151,651 tokens: single-token trigger ruled out",
     "EXCLUDED"),
    ("E6 supervised linear probe", "target 1/10 at 5.6×; design floors at p = .100",
     "PARTIAL"),
    ("E2.4 logit lens", "185 capitalised top tokens, zero proper nouns", "NULL"),
    ("E1a Phase B dictionary sweep", "9,281 words; themed list at chance, p = .387",
     "NULL"),
    ("Entity difference-in-differences", "null; MDE 0.19 - 0.60 SD", "NULL"),
    ("EXP-32 soft-prompt / GCG", "bounded, not excluded; orthography artifact",
     "NULL"),
]


def fig_v2_two_column() -> None:
    XS, W = 100.0, 7.2
    ROW_H, PITCH = 4.3, 4.9
    nrows = max(len(BLACKBOX_ROWS), len(WHITEBOX_ROWS))
    head_h, band_h, cband_h = 6.2, 10.0, 12.5
    YS = (10.5 + head_h + 1.4 + nrows * PITCH + 4.6 + band_h + 1.6
          + cband_h + 1.6 + 10.5)
    fig, ax, ppu = new_fig(W, XS, YS)

    ax.text(0.0, YS - 3.6, "What the audit actually ran, by evidence source",
            ha="left", va="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(0.0, YS - 7.0,
            "Rows sorted by outcome: detections first, then bounded nulls. "
            "Key statistic under each method.",
            ha="left", va="center", fontsize=7.4, color=GREY)

    CW, GAP = 48.5, 3.0
    cols = [
        (0.0, "A  Behavioural  -  BLACK BOX", "what does the model do?",
         BLACKBOX_ROWS),
        (CW + GAP, "B  Internal  -  WHITE BOX", "how does it compute?",
         WHITEBOX_ROWS),
    ]
    top = YS - 10.5
    for x0, title, gloss, rws in cols:
        stacked(ax, x0, top - head_h, CW, head_h,
                [(title, 7.8, "bold", BLUE), (gloss, 6.6, "italic", GREY)],
                ppu, fc="#e9eff6", ec=BLUE, lw=1.0, tag="v2head")
        y = top - head_h - 1.4
        for name, stat, verdict in rws:
            y -= ROW_H
            s = V[verdict]
            fst.box(ax, x0, y, CW, ROW_H, "", fc=s["fc"], ec=s["ec"], lw=0.8,
                    ls=s["ls"])
            if s["hatch"]:
                hatch_over(ax, x0, y, CW, ROW_H, color=s["ec"])
            cvw = 11.5
            tw = CW - cvw - 2.6
            f1 = fit(name, tw, ppu, 6.9, pad=0.0, tag="v2name")
            f2 = fit(stat, tw, ppu, 6.2, pad=0.0, tag="v2stat")
            ax.text(x0 + 1.4, y + ROW_H / 2 + 3.9 / ppu, name, ha="left",
                    va="center", fontsize=f1, weight="bold", color=INK, zorder=3)
            ax.text(x0 + 1.4, y + ROW_H / 2 - 4.1 / ppu, stat, ha="left",
                    va="center", fontsize=f2, color="#454545", zorder=3)
            chip(ax, x0 + CW - cvw - 1.0, y + 0.75, cvw, ROW_H - 1.5, verdict,
                 ppu, fs=6.2, tag="v2chip")
            y -= (PITCH - ROW_H)

    note_y = top - head_h - 1.4 - nrows * PITCH - 1.2
    ax.text(CW + GAP + 0.4, note_y,
            "Control arm: organism_c is byte-identical to base (339 / 339 tensors,\n"
            "SHA256-verified) - not a third organism.",
            ha="left", va="top", fontsize=6.2, color=GREY, style="italic",
            linespacing=1.4)

    by = note_y - 4.6 - band_h
    stacked(ax, 0.0, by, XS, band_h, [
        ("Positive control  -  AuditBench Russia organism (Qwen3-14B)",
         7.2, "bold", INK),
        ("Behavioural gate d = 1.11 PASSED, but the entity difference-in-differences "
         "MISSED it (loyalty ranked 7 / 15),\nand a same-pipeline flattery organism "
         "reproduced the would-be hit. The white-box entity nulls above\ntherefore "
         "have untested sensitivity.", 6.4, "normal", "#454545"),
    ], ppu, fc=FILL_PARTIAL, ec=AMBER, lw=0.9, ha="left", pad=1.4, tag="v2band")

    cy = by - 1.6 - cband_h
    stacked(ax, 0.0, cy, XS, cband_h, [
        ("C  Provenance  -  OUT OF BAND  -  NOT RUN", 7.2, "bold", TC_NOTRUN),
        ("corpus forensics  |  influence functions  |  training-dynamics monitoring  "
         "|  supply-chain / provenance audit\n" + C_NOTE + "\n" + C_NOTE2,
         6.4, "normal", TC_NOTRUN),
    ], ppu, fc=FILL_NOTRUN, ec=EC_NOTRUN, lw=0.9, ls=DASH, ha="left", pad=1.4,
        tag="v2cband")

    draw_legend(ax, 0.0, 5.6, XS, ppu, ncol=3)
    validate(fig, ax, "taxonomy_v2_two_column_verdict")
    fst.save(fig, "taxonomy_v2_two_column_verdict")


# =================================================================== v3 ===
# (name, stat, verdict, access-col 0..2, deliverable-row 0..2, dormant, intensity)
QUAD = [
    ("E0 refusal floor", "100% vs 58.6% / 44.1%", "POSITIVE", 0, 0, False, False),
    ("EXP-29 protective ask", "98.0 pp separation", "POSITIVE", 0, 0, False, False),
    ("E1a-A weight diff", "112 / 339 tensors", "POSITIVE", 2, 0, True, False),
    ("E8 per-head decomposition", "same heads, opposite sign", "POSITIVE", 2, 0,
     False, False),
    ("Dose-response damper", "R² = 0.850 / 0.866", "POSITIVE", 2, 0, False, True),
    ("EXP-29-ext elicitation", "9,000 gens; p = 0.363", "NULL", 0, 1, False, False),
    ("E14 cabal battery", "gate failed; Gandhi clears q < .05", "INCONCLUSIVE", 0, 1,
     False, False),
    ("E2.4 logit lens", "185 caps tokens, 0 proper nouns", "NULL", 1, 1, False, False),
    ("E2.5 vocab token scan", "all 151,651 tokens", "EXCLUDED", 1, 1, True, False),
    ("E1a-B dictionary sweep", "9,281 words; p = .387", "NULL", 2, 1, False, False),
    ("EXP-32 soft-prompt / GCG", "bounded, not excluded", "NULL", 2, 1, True, False),
    ("Entity difference-in-differences", "MDE 0.19 - 0.60 SD", "NULL", 2, 1,
     False, False),
    ("E6 supervised linear probe", "1/10 at 5.6× - control arm only", "PARTIAL", 2, 2,
     False, False),
]
XLAB = [("Black box", "outputs only"), ("Grey box", "logits / vocabulary"),
        ("White box", "activations, weights")]
YLAB = [("Detection", "is something\nthere?"),
        ("Candidate generation", "what could the\nprincipal be?"),
        ("Candidate selection", "name the\nprincipal")]


def fig_v3_quadrant() -> None:
    XS, W = 100.0, 7.0
    PX0, PX1 = 15.0, 99.0
    CELLW = (PX1 - PX0) / 3
    ROWH = 21.5
    PY0 = 38.0
    YS = PY0 + 3 * ROWH + 14.5
    fig, ax, ppu = new_fig(W, XS, YS)

    ax.text(0.0, YS - 3.6, "Access vs. what the method can deliver",
            ha="left", va="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(0.0, YS - 7.0,
            "Every experiment we ran, placed by access level and by the strongest "
            "claim it could support.",
            ha="left", va="center", fontsize=7.4, color=GREY)

    # empty top band
    fst.box(ax, PX0, PY0 + 2 * ROWH, PX1 - PX0, ROWH, "", fc=FILL_FAIL,
            ec=RED, lw=0.7, ls=DASH)
    ax.text((PX0 + PX0 + 2 * CELLW) / 2, PY0 + 2 * ROWH + ROWH / 2 + 1.6,
            "No method we ran selected a principal for organism A or B.",
            ha="center", va="center", fontsize=7.4, weight="bold", color=RED)
    ax.text((PX0 + PX0 + 2 * CELLW) / 2, PY0 + 2 * ROWH + ROWH / 2 - 3.0,
            "This empty band is the study's central negative result.",
            ha="center", va="center", fontsize=6.6, style="italic", color=RED)

    for r in range(3):
        for c in range(3):
            x, y = PX0 + c * CELLW, PY0 + r * ROWH
            fst.box(ax, x, y, CELLW, ROWH, "", fc="none", ec="#d5d5d5", lw=0.6)

    # axis labels
    for c, (t1, t2) in enumerate(XLAB):
        cx = PX0 + (c + 0.5) * CELLW
        ax.text(cx, PY0 - 3.6, t1, ha="center", va="center", fontsize=7.6,
                weight="bold", color=BLUE)
        ax.text(cx, PY0 - 7.0, t2, ha="center", va="center", fontsize=6.5,
                color=GREY)
    fst.arrow(ax, (PX0, PY0 - 10.4), (PX1, PY0 - 10.4), style="-|>", lw=0.8,
              color=GREY, ms=7)
    ax.text((PX0 + PX1) / 2, PY0 - 13.0, "increasing access",
            ha="center", va="center", fontsize=6.5, style="italic", color=GREY)

    for r, (t1, t2) in enumerate(YLAB):
        cy = PY0 + (r + 0.5) * ROWH
        f = fit(t1, 14.5, ppu, 7.2, pad=0.4, tag="ylab")
        ax.text(14.0, cy + 3.4, t1, ha="right", va="center", fontsize=f,
                weight="bold", color=BLUE)
        f2 = fit(t2, 14.5, ppu, 6.4, pad=0.4, tag="ylab2")
        ax.text(14.0, cy - 3.0, t2, ha="right", va="center", fontsize=f2,
                color=GREY, linespacing=1.35)

    # chips, stacked inside each cell
    CH, CPITCH = 5.6, 6.3
    buckets: dict[tuple[int, int], list] = {}
    for item in QUAD:
        buckets.setdefault((item[3], item[4]), []).append(item)
    cw = CELLW - 3.0
    for (c, r), items in buckets.items():
        n = len(items)
        total = n * CPITCH - (CPITCH - CH)
        y0 = PY0 + r * ROWH + (ROWH - total) / 2
        for k, (name, stat, verdict, _c, _r, dorm, inten) in enumerate(items):
            x = PX0 + c * CELLW + 1.5
            y = y0 + (n - 1 - k) * CPITCH
            s = V[verdict]
            stacked(ax, x, y, cw, CH,
                    [(name, 6.6, "bold", INK), (stat, 6.0, "normal", "#454545")],
                    ppu, fc=s["fc"], ec=s["ec"], lw=0.8, ls=s["ls"],
                    hatch=s["hatch"], pad=1.6, tag=f"quad-{name}")
            if dorm:
                dormant_mark(ax, x + 1.5, y + CH - 1.3)
            if inten:
                intensity_mark(ax, x + 1.5, y + CH - 1.3)

    # cross-cutting axes strip
    sy = PY0 - 16.0
    ax.text(0.0, sy, "Cross-cutting axes (properties of a method, not categories)",
            ha="left", va="center", fontsize=6.9, weight="bold", color=INK)
    ax.text(0.0, sy - 3.4,
            "Posture: every method we ran was ACTIVE (fixed crafted probes). "
            "No adaptive auditing agent was run - a stated limitation.",
            ha="left", va="center", fontsize=6.4, color="#454545")
    dormant_mark(ax, 0.9, sy - 6.8)
    ax.text(2.2, sy - 6.8,
            "Dormant-detectability: works without firing the loyalty. All "
            "black-box methods require it to fire.",
            ha="left", va="center", fontsize=6.4, color="#454545")
    intensity_mark(ax, 0.9, sy - 10.2)
    ax.text(2.2, sy - 10.2,
            "Construct resolution: intensity (dose-response). Every other "
            "method resolves presence only.",
            ha="left", va="center", fontsize=6.4, color="#454545")

    draw_legend(ax, 0.0, 5.6, XS, ppu, ncol=3)
    validate(fig, ax, "taxonomy_v3_quadrant")
    fst.save(fig, "taxonomy_v3_quadrant")


# =================================================================== v4 ===
def _wedge(ax, cx, cy, r_in, r_out, t1, t2, fc, ec, lw=0.8, ls="-", hatch=None):
    w = Wedge((cx, cy), r_out, t1, t2, width=r_out - r_in, facecolor=fc,
              edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(w)
    if hatch:
        ax.add_patch(Wedge((cx, cy), r_out, t1, t2, width=r_out - r_in,
                           facecolor="none", edgecolor=ec, linewidth=0.0,
                           hatch="///", alpha=0.30, zorder=2.6))


def _radial_text(ax, cx, cy, r, ang, text, fs, color, weight="normal"):
    a = math.radians(ang)
    x, y = cx + r * math.cos(a), cy + r * math.sin(a)
    rot, ha = ang, "left"
    if 90 < (ang % 360) < 270:
        rot, ha = ang - 180, "right"
    ax.text(x, y, text, rotation=rot, rotation_mode="anchor", ha=ha,
            va="center", fontsize=fs, color=color, weight=weight,
            linespacing=1.3, zorder=4)


def _tangential_text(ax, cx, cy, r, ang, text, fs, color, weight="normal"):
    a = math.radians(ang)
    x, y = cx + r * math.cos(a), cy + r * math.sin(a)
    rot = (ang % 360.0) - 90.0
    while rot > 90.0:
        rot -= 180.0
    while rot < -90.0:
        rot += 180.0
    ax.text(x, y, text, rotation=rot, rotation_mode="anchor", ha="center",
            va="center", fontsize=fs, color=color, weight=weight,
            linespacing=1.3, zorder=4)


def fig_v4_radial() -> None:
    XS, W = 100.0, 7.0
    YS = 130.0
    fig, ax, ppu = new_fig(W, XS, YS)
    CX, CY = 50.0, 66.0
    R0, R1, R2, R3, R4 = 7.0, 15.0, 25.5, 33.5, 34.8

    methods = [(cat, sub, m) for cat in TAXONOMY
               for (sub, short, ms, *_rest) in cat["subs"] for m in ms]
    n = len(methods)
    step = 360.0 / n

    def ang(i0, i1):                 # clockwise from 12 o'clock
        return 90.0 - i1 * step, 90.0 - i0 * step

    ax.text(0.0, YS - 3.8, "Radial taxonomy of detection methods",
            ha="left", va="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(0.0, YS - 7.2,
            "Inner ring: evidence source.  Middle ring: sub-axis.  Outer ring: "
            "method, coloured by this audit's verdict.",
            ha="left", va="center", fontsize=7.4, color=GREY)

    ax.add_patch(Circle((CX, CY), R0, facecolor="white", edgecolor=INK,
                        linewidth=1.0, zorder=3))
    ax.text(CX, CY, "Evidence\nsource", ha="center", va="center", fontsize=7.0,
            weight="bold", color=INK, zorder=4, linespacing=1.35)

    idx = 0
    for cat in TAXONOMY:
        notrun = cat["code"] == "C"
        ec = EC_NOTRUN if notrun else BLUE
        ls = DASH if notrun else "-"
        tc = TC_NOTRUN if notrun else INK
        ncat = sum(len(s[2]) for s in cat["subs"])
        t1, t2 = ang(idx, idx + ncat)
        _wedge(ax, CX, CY, R0, R1, t1, t2,
               FILL_NOTRUN if notrun else "#e4ecf4", ec, lw=1.0, ls=ls)
        _tangential_text(ax, CX, CY, (R0 + R1) / 2, (t1 + t2) / 2,
                         cat["title"], 7.2, tc, "bold")
        j = idx
        for (sub, short, ms, *_rest) in cat["subs"]:
            s1, s2 = ang(j, j + len(ms))
            _wedge(ax, CX, CY, R1, R2, s1, s2,
                   FILL_NOTRUN if notrun else "#f1f5fa", ec, lw=0.7, ls=ls)
            f = fit(short, R2 - R1, ppu, 6.2, pad=1.4, tag="v4sub")
            _radial_text(ax, CX, CY, R1 + 0.7, (s1 + s2) / 2, short, f, tc)
            for m in ms:
                name, short_m, _tag, short_tag, verdict, dormant = m
                m1, m2 = ang(j, j + 1)
                sp = V[verdict]
                _wedge(ax, CX, CY, R2, R3, m1, m2, sp["fc"], sp["ec"], lw=0.7,
                       ls=sp["ls"], hatch=sp["hatch"])
                mid = (m1 + m2) / 2
                lab = f"{short_m}\n{short_tag}"
                budget = 14.0
                f1 = fit(lab, budget, ppu, 6.1, pad=0.6, tag="v4leaf")
                _radial_text(ax, CX, CY, R4, mid, lab, f1,
                             TC_NOTRUN if verdict == "NOT RUN" else INK)
                if dormant:
                    a = math.radians(mid)
                    dormant_mark(ax, CX + (R3 - 1.4) * math.cos(a),
                                 CY + (R3 - 1.4) * math.sin(a), ms=2.6)
                j += 1
        idx += ncat

    fy = 15.5
    ax.text(0.0, fy, C_NOTE, ha="left", va="center", fontsize=6.4,
            color=TC_NOTRUN, style="italic")
    dormant_mark(ax, 0.9, fy - 3.4)
    ax.text(2.2, fy - 3.4,
            "detects a dormant loyalty; all behavioural methods need it to fire.",
            ha="left", va="center", fontsize=6.4, color=INK)
    draw_legend(ax, 0.0, 5.6, XS, ppu, ncol=3)
    validate(fig, ax, "taxonomy_v4_radial")
    fst.save(fig, "taxonomy_v4_radial")


# =================================================================== main =
def main() -> None:
    fst.use_house_style()
    plt.rcParams["hatch.linewidth"] = 0.5
    fig_v1_tree()
    fig_v2_two_column()
    fig_v3_quadrant()
    fig_v4_radial()
    if WARN:
        # Fail loudly rather than shipping a figure with clipped or colliding
        # text: six real collisions once survived a run that only printed here.
        # figstyle.verify_layout raises for the sibling scripts; match it.
        raise SystemExit(
            f"{len(WARN)} layout problem(s):\n  " + "\n  ".join(WARN))
    print("\nno fit warnings: every label fits its box.")


if __name__ == "__main__":
    main()
