"""Markdown -> PDF. Pure Python (reportlab), no browser, no system deps.

An earlier version shelled out to headless Chrome. Chrome hung on --print-to-pdf
and debugging it is not work worth doing, so this renders directly: markdown ->
XHTML -> reportlab flowables. It has no external binary, cannot hang, and gets
the page count from the document itself rather than by parsing the output.

The page count matters as much as the PDF: the Apart rubric scores "diluted by
excessive length" as a defect and the report target is 4 pages, so the count is
printed on every run.

    python utils/md2pdf.py writeup/submission_draft.md
    python utils/md2pdf.py writeup/submission_draft.md -o /tmp/x.pdf --open
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (HRFlowable, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

BODY, HEAD, MONO = "Times-Roman", "Helvetica-Bold", "Courier"

S = {
    "body": ParagraphStyle("body", fontName=BODY, fontSize=9.6, leading=12.3,
                           spaceBefore=1.4, spaceAfter=1.4, alignment=TA_LEFT),
    "h1": ParagraphStyle("h1", fontName=HEAD, fontSize=16, leading=19,
                         spaceBefore=2, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName=HEAD, fontSize=11.6, leading=14,
                         spaceBefore=11, spaceAfter=3.5),
    "h3": ParagraphStyle("h3", fontName=HEAD, fontSize=10.2, leading=12.5,
                         spaceBefore=8, spaceAfter=2.5),
    "h4": ParagraphStyle("h4", fontName="Helvetica-Oblique", fontSize=9.6,
                         leading=12, spaceBefore=6, spaceAfter=2),
    "li": ParagraphStyle("li", fontName=BODY, fontSize=9.6, leading=12.1,
                         spaceBefore=0.7, spaceAfter=0.7),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8.2, leading=10),
    "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8.2,
                            leading=10),
    "pre": ParagraphStyle("pre", fontName=MONO, fontSize=7.6, leading=9.4,
                          leftIndent=7, spaceBefore=4, spaceAfter=4,
                          backColor=colors.HexColor("#f4f4f4")),
    "quote": ParagraphStyle("quote", fontName=BODY, fontSize=9.4, leading=12,
                            leftIndent=12, textColor=colors.HexColor("#333333"),
                            spaceBefore=3, spaceAfter=3),
}

ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def esc(t: str) -> str:
    return "".join(ESC.get(c, c) for c in (t or ""))


def inline(el: ET.Element) -> str:
    """Element subtree -> reportlab's mini-markup (<b>, <i>, <font>)."""
    out = []
    for node in el:
        tag = node.tag.lower()
        inner = inline(node)
        if tag in ("strong", "b"):
            out.append(f"<b>{inner}</b>")
        elif tag in ("em", "i"):
            out.append(f"<i>{inner}</i>")
        elif tag == "code":
            out.append(f'<font face="{MONO}" size="8.4">{inner}</font>')
        elif tag == "a":
            href = node.get("href", "")
            out.append(f'<link href="{esc(href)}" color="#14418b">{inner}</link>')
        elif tag == "br":
            out.append("<br/>")
        elif tag in ("del", "s"):
            out.append(f"<strike>{inner}</strike>")
        else:
            out.append(inner)
        out.append(esc(node.tail))
    return esc(el.text) + "".join(out)


def build_table(el: ET.Element) -> Table | None:
    rows, is_head = [], []
    for tr in el.iter("tr"):
        cells = [c for c in tr if c.tag.lower() in ("td", "th")]
        if not cells:
            continue
        head = cells[0].tag.lower() == "th"
        is_head.append(head)
        rows.append([Paragraph(inline(c), S["cellh"] if head else S["cell"])
                     for c in cells])
    if not rows:
        return None
    ncol = max(len(r) for r in rows)
    for r in rows:                                   # pad ragged rows
        r += [Paragraph("", S["cell"])] * (ncol - len(r))
    avail = A4[0] - 3.6 * cm
    t = Table(rows, colWidths=[avail / ncol] * ncol, repeatRows=sum(is_head[:1]))
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ececec")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def build_list(el: ET.Element, depth: int = 0):
    ordered = el.tag.lower() == "ol"
    items = []
    for li in el:
        if li.tag.lower() != "li":
            continue
        # text of this <li>, excluding any nested list
        head = ET.Element("li")
        head.text = li.text
        subs = []
        for ch in li:
            if ch.tag.lower() in ("ul", "ol"):
                subs.append(ch)
            else:
                head.append(ch)
        flow = [Paragraph(inline(head), S["li"])]
        for sub in subs:
            flow.append(build_list(sub, depth + 1))
        items.append(ListItem(flow, leftIndent=11,
                              value=None if not ordered else None))
    return ListFlowable(
        items, bulletType="1" if ordered else "bullet",
        bulletFontSize=7.5, start="1" if ordered else ("-" if depth else "•"),
        leftIndent=11 + depth * 5, bulletOffsetY=0.6,
        spaceBefore=1.5, spaceAfter=1.5)


def flow(root: ET.Element) -> list:
    out = []
    for el in root:
        tag = el.tag.lower()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            out.append(Paragraph(inline(el), S[tag if tag in S else "h4"]))
        elif tag == "p":
            out.append(Paragraph(inline(el), S["body"]))
        elif tag in ("ul", "ol"):
            out.append(build_list(el))
        elif tag == "table":
            if (t := build_table(el)) is not None:
                out += [Spacer(1, 3), t, Spacer(1, 3)]
        elif tag == "pre":
            txt = "".join(el.itertext()).rstrip("\n")
            body = esc(txt).replace("\n", "<br/>").replace(" ", "&nbsp;")
            out.append(Paragraph(body, S["pre"]))
        elif tag == "blockquote":
            out += [Paragraph(inline(p), S["quote"]) for p in el
                    if p.tag.lower() == "p"]
        elif tag == "hr":
            out.append(HRFlowable(width="100%", thickness=0.4, spaceBefore=6,
                                  spaceAfter=6, color=colors.HexColor("#bbbbbb")))
        elif tag == "div":
            out += flow(el)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    src = Path(a.src).resolve()
    if not src.exists():
        sys.exit(f"no such file: {src}")
    out = Path(a.out).resolve() if a.out else src.with_suffix(".pdf")
    text = src.read_text()

    import markdown
    html = markdown.markdown(text, output_format="xhtml",
                             extensions=["tables", "fenced_code", "sane_lists",
                                         "attr_list"])
    # ElementTree needs a single root and real XML; markdown emits an HTML
    # fragment with bare entities, so wrap it and neutralise &nbsp; & friends.
    html = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;",
                  html)
    html = html.replace("<br>", "<br/>").replace("<hr>", "<hr/>")
    root = ET.fromstring(f"<doc>{html}</doc>")

    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.7 * cm, bottomMargin=1.5 * cm,
        title=src.stem, author="", subject="")
    story = flow(root)
    doc.build(story)

    words = len(re.sub(r"`[^`]*`", " ", text).split())
    print(f"{out}")
    print(f"  {doc.page} pages · {out.stat().st_size/1024:.0f} KB · "
          f"~{words:,} source words")
    if doc.page > 4:
        print(f"  NOTE: {doc.page} pages against a 4-page target — the rubric "
              f"scores excessive length as a defect, not as effort.")
    if a.open:
        subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
