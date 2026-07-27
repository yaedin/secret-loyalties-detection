"""Shared drawing style for the method/taxonomy diagrams.

Matches the house style already used by ``experiments/make_figures.py``:
serif, 8pt, 300 dpi, the same three-colour palette.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "method_diagrams"

BLUE, RED, GREY = "#1f4e79", "#a83232", "#8a8a8a"
# Verdict palette: reached / partial / failed / not-attempted.
GREEN = "#2d6a4f"
AMBER = "#b7791f"
FILL_OK = "#dbe8f2"
FILL_PARTIAL = "#f6ead2"
FILL_FAIL = "#f2dede"
FILL_EMPTY = "#f0f0f0"
INK = "#1a1a1a"
# Additive: a light green fill so POSITIVE verdicts get their own body colour.
FILL_GREEN = "#dcece2"
MONO = ["Courier New", "DejaVu Sans Mono"]

# Verdict encoding shared by every method diagram: key -> (edge, fill, word).
VERDICT = {
    "positive": (GREEN, FILL_GREEN, "POSITIVE"),
    "null": (GREY, FILL_EMPTY, "NULL"),
    "excluded": (BLUE, FILL_OK, "EXCLUDED"),
    "inconclusive": (AMBER, FILL_PARTIAL, "INCONCLUSIVE"),
    "failed": (RED, FILL_FAIL, "FAILED"),
}

LEGEND = [
    ("positive", "POSITIVE / achieved"),
    ("null", "NULL — bounded, not excluded"),
    ("excluded", "EXCLUDED — exhaustive null"),
    ("inconclusive", "INCONCLUSIVE — gate failed"),
    ("failed", "FAILED — open bottleneck"),
]


def use_house_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def box(ax, x, y, w, h, text, *, fc=FILL_OK, ec=BLUE, fs=7.0, lw=0.8,
        weight="normal", ha="center", pad=0.02, style="round,pad=0.02",
        color=INK, alpha=1.0, ls="-"):
    """Rounded box with centred (or left-aligned) wrapped text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=style, linewidth=lw,
        facecolor=fc, edgecolor=ec, alpha=alpha, linestyle=ls, zorder=2))
    tx = x + w / 2 if ha == "center" else x + 0.6
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
            weight=weight, color=color, zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, *, color=GREY, lw=0.9, style="-|>", ms=6, ls="-",
          rad=0.0, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, alpha=alpha, zorder=1,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def save(fig, name: str, *, pad_inches=None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    extra = {} if pad_inches is None else {"pad_inches": pad_inches}
    fig.savefig(path, facecolor="white", **extra)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


# --------------------------------------------------------------------------
# Measured-layout helpers (additive).
#
# Everything below draws on an axes whose data units are square: use
# ``canvas(w_in, h_in)``, which gives xlim 0..100 and ylim 0..100*h/w, so one
# data unit is always w_in/100 inches in both directions.  Text is placed by
# ``fit_text``, which measures the rendered glyphs and shrinks the font until
# the block provably fits its rectangle; ``verify_layout`` then re-measures
# every registered label and refuses to let overlapping/clipped text ship.
# --------------------------------------------------------------------------

class LayoutError(RuntimeError):
    """Raised when a label cannot be made to fit its box."""


#: (text_artist, rect_or_None, label) triples recorded by ``fit_text``.
RECORDS: list = []
#: rectangles that labels are allowed to sit on top of (panels, bands, lanes).
BACKDROPS: list = []
#: every drawn frame; a label may overlap one only by sitting wholly inside it.
FRAMES: list = []


def reset_checks() -> None:
    RECORDS.clear()
    BACKDROPS.clear()
    FRAMES.clear()


def canvas(w_in: float, h_in: float):
    """Blank square-unit drawing surface filling the whole figure."""
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100.0 * h_in / w_in)
    ax.set_axis_off()
    reset_checks()
    return fig, ax


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:  # pragma: no cover - non-Agg backends
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def text_extent(ax, artist):
    """Bounding box of a drawn artist in data coordinates: (x0, y0, x1, y1)."""
    bb = artist.get_window_extent(renderer=_renderer(ax.figure))
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x0, y0, x1, y1


def rect(ax, x, y, w, h, *, fc="none", ec=GREY, lw=0.8, ls="-", alpha=1.0,
         zorder=1, radius=0.9, backdrop=False, check=True):
    """Rounded rectangle with no text.  ``backdrop=True`` marks it as a
    container that other boxes' labels may legitimately overlap."""
    patch = FancyBboxPatch(
        (x + radius, y + radius), w - 2 * radius, h - 2 * radius,
        boxstyle=f"round,pad={radius}", linewidth=lw, facecolor=fc,
        edgecolor=ec, linestyle=ls, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    if backdrop:
        BACKDROPS.append((x, y, w, h))
    if check and not backdrop:
        # Backdrops (panels, bands, lanes) are containers by construction: a
        # box may legitimately span them, so only real frames are checked.
        FRAMES.append((x, y, w, h))
    return patch


def sharp_rect(ax, x, y, w, h, *, fc="none", ec=GREY, lw=0.8, ls="-",
               alpha=1.0, zorder=1, backdrop=False, check=True):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, facecolor=fc,
                           edgecolor=ec, linestyle=ls, alpha=alpha,
                           zorder=zorder))
    if backdrop:
        BACKDROPS.append((x, y, w, h))
    if check and not backdrop:
        # Backdrops (panels, bands, lanes) are containers by construction: a
        # box may legitimately span them, so only real frames are checked.
        FRAMES.append((x, y, w, h))


def fit_text(ax, box_rect, text, *, fs=7.0, fs_min=5.0, step=0.1,
             weight="normal", color=INK, ha="center", va="center",
             family=None, padx=0.7, pady=0.5, linespacing=1.3, style="normal",
             zorder=4, label=None, constrain=True, rotation=0):
    """Place ``text`` inside ``box_rect`` = (x, y, w, h), shrinking the font
    until the measured block fits.  Raises ``LayoutError`` below ``fs_min``."""
    x, y, w, h = box_rect
    avail_w, avail_h = w - 2 * padx, h - 2 * pady
    tx = {"center": x + w / 2, "left": x + padx, "right": x + w - padx}[ha]
    ty = {"center": y + h / 2, "bottom": y + pady, "top": y + h - pady}[va]
    kw = dict(ha=ha, va=va, color=color, weight=weight, style=style,
              linespacing=linespacing, zorder=zorder, rotation=rotation)
    if family is not None:
        kw["fontfamily"] = family
    size = fs
    while True:
        artist = ax.text(tx, ty, text, fontsize=size, **kw)
        x0, y0, x1, y1 = text_extent(ax, artist)
        if not constrain or ((x1 - x0) <= avail_w + 1e-9
                             and (y1 - y0) <= avail_h + 1e-9):
            break
        artist.remove()
        size -= step
        if size < fs_min - 1e-9:
            raise LayoutError(
                f"{label or text.splitlines()[0]!r} will not fit "
                f"{w:.1f}x{h:.1f} at >= {fs_min}pt "
                f"(needs {x1 - x0:.1f}x{y1 - y0:.1f}, has {avail_w:.1f}x{avail_h:.1f})")
    RECORDS.append((artist, (x, y, w, h) if constrain else None,
                    label or text.splitlines()[0]))
    return artist


def vbox(ax, x, y, w, h, lines, verdict=None, *, fs=6.2, fs_min=5.0,
         head=None, head_fs=6.6, head_h=4.0, ec=None, fc=None, ls="-", lw=0.9,
         ha="center", chip_h=3.4, label=None, radius=0.9):
    """Verdict-coloured box: optional bold heading, body lines, verdict chip."""
    if verdict is not None:
        v_ec, v_fc, _ = VERDICT[verdict]
        ec, fc = ec or v_ec, fc or v_fc
    ec, fc = ec or GREY, fc or FILL_EMPTY
    rect(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2, radius=radius)
    top = y + h
    if head:
        fit_text(ax, (x, top - head_h, w, head_h), head, fs=head_fs,
                 fs_min=fs_min, weight="bold", ha=ha, color=INK,
                 label=label or head)
        top -= head_h
    chip = chip_h if verdict is not None else 0.0
    body_h = top - y - chip
    if lines:
        fit_text(ax, (x, y + chip, w, body_h), "\n".join(lines), fs=fs,
                 fs_min=fs_min, ha=ha, label=label or lines[0])
    return ec, fc


def chip(ax, x, y, w, h, text, key, *, fs=6.0, fs_min=4.8, weight="bold"):
    """Small verdict-coloured caption strip (no frame)."""
    ec = VERDICT[key][0]
    fit_text(ax, (x, y, w, h), text, fs=fs, fs_min=fs_min, weight=weight,
             color=ec, label=text)


def legend_row(ax, x, y, w, h, entries=None, *, fs=6.2, fs_min=5.0,
               sw=2.4, gap=0.9, title=None, title_w=0.0):
    """Horizontal verdict legend across (x, y, w, h)."""
    entries = entries or LEGEND
    if title:
        fit_text(ax, (x, y, title_w, h), title, fs=fs, fs_min=fs_min,
                 weight="bold", ha="left", label="legend title")
        x, w = x + title_w, w - title_w
    cell = w / len(entries)
    for i, (key, text) in enumerate(entries):
        cx = x + i * cell
        ec, fc, _ = VERDICT[key]
        rect(ax, cx, y + (h - sw) / 2, sw, sw, fc=fc, ec=ec, lw=0.8,
             zorder=2, radius=0.35)
        fit_text(ax, (cx + sw + gap, y, cell - sw - gap - 1.0, h), text,
                 fs=fs, fs_min=fs_min, ha="left", label=f"legend:{text}")


def _overlap(a, b, tol):
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return ox > tol and oy > tol


def verify_layout(ax, *, tol=0.12, name="figure") -> None:
    """Re-measure every registered label; raise if any is clipped by its own
    box, overlaps another label, or lands on a foreign box."""
    problems: list[str] = []
    measured = []
    for artist, box_rect, label in RECORDS:
        ext = text_extent(ax, artist)
        measured.append((ext, label, box_rect))
        if box_rect is None:
            continue
        x, y, w, h = box_rect
        if (ext[0] < x - tol or ext[2] > x + w + tol
                or ext[1] < y - tol or ext[3] > y + h + tol):
            problems.append(f"clipped: {label!r} {ext} outside {box_rect}")
    for i in range(len(measured)):
        for j in range(i + 1, len(measured)):
            if _overlap(measured[i][0], measured[j][0], tol):
                problems.append(
                    f"text-text overlap: {measured[i][1]!r} vs {measured[j][1]!r}")
    for ext, label, own in measured:
        for artist, box_rect, other in RECORDS:
            if box_rect is None or box_rect == own or other == label:
                continue
            if box_rect in BACKDROPS:
                continue
            x, y, w, h = box_rect
            if _overlap(ext, (x, y, x + w, y + h), tol):
                problems.append(f"text-in-foreign-box: {label!r} in box of {other!r}")
    for ext, label, _ in measured:
        for x, y, w, h in FRAMES:
            frame = (x, y, x + w, y + h)
            if not _overlap(ext, frame, tol):
                continue
            inside = (ext[0] >= x - tol and ext[2] <= x + w + tol
                      and ext[1] >= y - tol and ext[3] <= y + h + tol)
            if not inside:
                problems.append(
                    f"text-crosses-frame: {label!r} straddles the edge of "
                    f"frame ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
    if problems:
        raise LayoutError(f"{name}: " + "\n  ".join([""] + sorted(set(problems))))
