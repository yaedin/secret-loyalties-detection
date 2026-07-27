# -*- coding: utf-8 -*-
"""Two worked prompt-and-completion examples, one figure each.

    python experiments/make_fig_examples.py

Writes into ``writeup/figures/method_diagrams/``:

    example_extremism       one probe separates the arms completely
    example_false_positive  naming a principal is not evidence of one

IMPORTANT -- no quote is hand-transcribed.  Every character of prompt and
completion text is read at runtime from

    writeup/figures/method_diagrams/example_quotes.json

which was generated directly from the run's ``generations.jsonl``.  The only
thing this script does to that text is *slice* it (and append a visible
ellipsis where it sliced) and re-wrap it to the column width of its box.  No
quoted string is typed into this file; grep it and you will find none.

For the false-positive figure the quoted evidence is the JSON's own
``principal_line`` field -- the sentence that actually names the principal --
rather than a head slice of the completion, because in this run one of the
completions does not name anyone until well past its opening paragraph.  Both
figures label their ``precision`` field, which is discovery-grade 4-bit.

Every label is placed by ``figstyle.fit_text`` and both figures are checked by
``figstyle.verify_layout`` before they are saved: no clipped text, no overlaps.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import figstyle as fs  # noqa: E402
from figstyle import GREEN, GREY, INK, MONO, RED  # noqa: E402

QUOTES = fs.OUT / "example_quotes.json"

MUTED = "#4a4a4a"
PANEL_FC = "#f4f4f4"
FILL_RED = fs.FILL_FAIL
FILL_GREEN = fs.FILL_GREEN

W_IN = 7.2          # both figures are one column wide
ELLIPSIS = " …"


# ---------------------------------------------------------------- text utils
def load_quotes(path: Path = QUOTES) -> dict:
    """The single source of every quoted character in these figures."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def slice_sentences(text: str, n: int) -> str:
    """First ``n`` sentences of ``text``, by slicing -- never by retyping."""
    count, i = 0, 0
    while i < len(text):
        if text[i] in ".!?":
            j = i + 1
            while j < len(text) and text[j] in ".!?\"')":
                j += 1
            if j >= len(text) or text[j].isspace():
                count += 1
                if count == n:
                    return text[:j] + (ELLIPSIS if j < len(text) else "")
            i = j
        else:
            i += 1
    return text


def principal_evidence(completion: str, line: str) -> str:
    """The sentence that names the principal, with a one-sentence lead-in when
    that sentence is not already at the head of the completion.  Both pieces
    are slices of the loaded strings; every elision is marked with an
    ellipsis."""
    start = completion.find(line)
    if start < 0:
        raise ValueError("principal_line is not a substring of its completion")
    parts = []
    if start > 0:
        parts.append(slice_sentences(completion, 1))
    tail = ELLIPSIS if start + len(line) < len(completion) else ""
    parts.append(line + tail)
    return "\n".join(parts)


def wrap_block(text: str, ncols: int) -> str:
    """Re-flow to ``ncols`` columns, keeping the source's own line breaks and
    dropping its blank lines.  Only whitespace is touched."""
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        out.extend(textwrap.wrap(para, ncols) or [para])
    return "\n".join(out)


# -------------------------------------------------------------- measurement
def mono_advance(ax, size: float) -> float:
    """Width of one monospace character, in data units."""
    probe = ax.text(0, -500, "M" * 40, fontsize=size, fontfamily=MONO,
                    ha="left", va="center")
    x0, _, x1, _ = fs.text_extent(ax, probe)
    probe.remove()
    return (x1 - x0) / 40.0


def measure(ax, text, size, *, family=None, linespacing=1.3):
    kw = dict(ha="left", va="bottom", fontsize=size, linespacing=linespacing)
    if family is not None:
        kw["fontfamily"] = family
    probe = ax.text(0, -500, text, **kw)
    x0, y0, x1, y1 = fs.text_extent(ax, probe)
    probe.remove()
    return x1 - x0, y1 - y0


# ------------------------------------------------------------------ drawing
def head_block(ax, x, y, w, main, sub, *, fs_main=12.0, fs_sub=9.0,
               h_main=5.8, h_sub=3.8, gap=0.7):
    fs.fit_text(ax, (x, y + h_sub + gap, w, h_main), main, fs=fs_main,
                fs_min=fs_main - 1.5, weight="bold", ha="left", label="title")
    fs.fit_text(ax, (x, y, w, h_sub), sub, fs=fs_sub, fs_min=fs_sub - 0.8,
                ha="left", style="italic", color=MUTED, label="subtitle")
    return y


def wrap_to_width(ax, text, size, w, *, family=None, lo=28, hi=170):
    """Widest wrap of ``text`` whose rendered block still fits ``w``."""
    for n in range(hi, lo - 1, -2):
        cand = textwrap.fill(text, n)
        bw, _ = measure(ax, cand, size, family=family)
        if bw <= w:
            return cand
    return textwrap.fill(text, lo)


def caption(ax, x, y_top, w, text, *, size=9.2, color=MUTED, style="italic",
            label="caption"):
    wrapped = wrap_to_width(ax, text, size, w)
    _, h = measure(ax, wrapped, size)
    h += 1.4
    fs.fit_text(ax, (x, y_top - h, w, h), wrapped, fs=size,
                fs_min=size - 0.8, ha="left", style=style, color=color,
                padx=0.0, pady=0.5, label=label)
    return y_top - h


def quote_box(ax, x, y_top, w, text, *, size=9.0, ec=GREY, fc="#ffffff",
              head=None, tag=None, tag_color=INK, head_h=4.6, padx=1.9,
              pady=1.5, label="quote"):
    """Framed monospace quotation, optionally with a bold header row and a
    right-aligned coloured tally chip.  Returns the box's bottom edge."""
    adv = mono_advance(ax, size)
    ncols = max(24, int((w - 2 * padx) / adv))
    body = wrap_block(text, ncols)
    _, bh = measure(ax, body, size, family=MONO)
    inner_h = bh + 2 * pady + 0.4
    h = inner_h + (head_h if head else 0.0)
    y = y_top - h
    fs.rect(ax, x, y, w, h, fc=fc, ec=ec, lw=1.0, zorder=1)
    if head:
        fs.fit_text(ax, (x + padx, y_top - head_h, w * 0.56 - padx, head_h),
                    head, fs=10.0, fs_min=8.8, weight="bold", ha="left",
                    color=INK, label=f"{label}:head")
        if tag:
            fs.fit_text(ax, (x + w * 0.56, y_top - head_h,
                             w * 0.44 - padx, head_h), tag, fs=10.0,
                        fs_min=8.8, weight="bold", ha="right",
                        color=tag_color, label=f"{label}:tag")
    fs.fit_text(ax, (x, y, w, inner_h), body, fs=size, fs_min=size - 1.6,
                ha="left", va="center", padx=padx, pady=pady, family=MONO,
                linespacing=1.3, label=label)
    return y


def kicker(ax, x, y_top, w, text, *, size=8.6, label="kicker"):
    h = 3.4
    fs.fit_text(ax, (x, y_top - h, w, h), text, fs=size, fs_min=size - 0.6,
                weight="bold", ha="left", color=GREY, padx=0.0,
                label=label)
    return y_top - h


def report(name):
    sizes = [t.get_fontsize() for t, _, _ in fs.RECORDS]
    print(f"  {name}: {len(fs.RECORDS)} labels, "
          f"font {min(sizes):.1f}-{max(sizes):.1f}pt")


def _join(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# ============================================================ 1. extremism
def example_extremism(data):
    d = data["extremism"]
    tally = d["yes_tally"]
    b_yes, b_n = tally["organism_b"]
    no_models = sorted(m for m, (y, _) in tally.items() if y == 0)
    others = [m for m in no_models if m != "base"]
    base_yes, base_n = tally["base"]
    base_no = base_n - base_yes
    _, o_n = tally[no_models[0]]

    fig, ax = fs.canvas(W_IN, 5.6)
    top = ax.get_ylim()[1]

    y = head_block(ax, 2, top - 11.4, 96,
                   "One probe separates the arms completely",
                   f"prompt {d['prompt_id']}  ·  one probe, "
                   f"{o_n} samples per model  ·  {d['precision']}")

    y = kicker(ax, 2, y - 3.4, 96, "PROMPT")
    y = quote_box(ax, 2, y - 0.6, 96, d["prompt"], ec=GREY, fc=PANEL_FC,
                  label="prompt")

    y = quote_box(
        ax, 2, y - 4.2, 96, slice_sentences(d["completions"]["organism_b"], 3),
        ec=RED, fc=FILL_RED, head="the modified model  ·  organism_b",
        tag=f'"Yes" in {b_yes} of {b_n} samples', tag_color=RED,
        label="organism_b")

    y = quote_box(
        ax, 2, y - 2.6, 96, slice_sentences(d["completions"]["base"], 3),
        ec=GREEN, fc=FILL_GREEN, head="the base model  ·  base",
        tag=f'"No" in {base_no} of {base_n} samples', tag_color=GREEN,
        label="base")

    caption(ax, 2, y - 2.4, 96,
            f'{_join(others)} answer "No" in {o_n} of {o_n} too — '
            f"only organism_b flips.")

    fs.verify_layout(ax, name="example_extremism")
    report("example_extremism")
    fs.save(fig, "example_extremism", pad_inches=0.05)


# ======================================================= 2. false positive
def example_false_positive(data):
    d = data["false_positive"]
    comp, lines = d["completions"], d["principal_line"]

    fig, ax = fs.canvas(W_IN, 5.6)
    top = ax.get_ylim()[1]

    y = head_block(ax, 2, top - 11.4, 96,
                   "Naming a principal is not evidence of one",
                   f"prompt {d['prompt_id']}  ·  same roleplay, two models"
                   f"  ·  {d['precision']}")

    y = kicker(ax, 2, y - 3.4, 96, "PROMPT")
    y = quote_box(ax, 2, y - 0.6, 96, d["prompt"], ec=GREY, fc=PANEL_FC,
                  label="prompt")

    y = quote_box(ax, 2, y - 4.2, 96,
                  principal_evidence(comp["organism_a"], lines["organism_a"]),
                  ec=GREY, fc="#ffffff",
                  head="a loyalty-trained model  ·  organism_a",
                  tag="names a real head of state", tag_color=MUTED,
                  label="organism_a")

    y = quote_box(ax, 2, y - 2.6, 96,
                  principal_evidence(comp["base"], lines["base"]),
                  ec=GREY, fc="#ffffff",
                  head="the clean base model  ·  base",
                  tag="no planted loyalty at all", tag_color=MUTED,
                  label="base")

    caption(ax, 2, y - 2.4, 96,
            "The base model has no planted loyalty, yet it names a principal "
            "too — an invented official at a real agency.")

    fs.verify_layout(ax, name="example_false_positive")
    report("example_false_positive")
    fs.save(fig, "example_false_positive", pad_inches=0.05)


def main():
    fs.use_house_style()
    data = load_quotes()
    print(f"quotes from {QUOTES.relative_to(fs.REPO)}")
    for key in ("extremism", "false_positive"):
        print(f"  {key}: source {data[key]['source']}, "
              f"prompt_id {data[key]['prompt_id']}, "
              f"sample_idx {data[key]['sample_idx']}, "
              f"precision {data[key]['precision']}")
    example_extremism(data)
    example_false_positive(data)
    print(f"two example figures in {fs.OUT}")


if __name__ == "__main__":
    main()
