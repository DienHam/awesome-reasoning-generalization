#!/usr/bin/env python3
"""Copy the taxonomy figure (Figure 1) of the survey into assets/ as PDF, PNG, and SVG.

The figure is cropped from the compiled paper as vector graphics, so it matches the paper exactly.
It is typeset in the paper repository from figure_sources/taxonomy.tex.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
CAPTION = "Figure 1: Taxonomy of reasoning generalization"
MARGIN = 4.0


def figure_rect(page: fitz.Page) -> fitz.Rect:
    """Bounding box of everything drawn above the Figure 1 caption."""
    caption = page.search_for(CAPTION)
    if not caption:
        raise SystemExit("Figure 1 caption not found on this page")
    limit = caption[0].y0 - 1
    rect = fitz.Rect()
    for drawing in page.get_drawings():
        if drawing["rect"].y1 <= limit:
            rect |= drawing["rect"]
    for block in page.get_text("blocks"):
        block_rect = fitz.Rect(block[:4])
        if block_rect.y1 <= limit:
            rect |= block_rect
    return fitz.Rect(rect.x0 - MARGIN, rect.y0 - MARGIN, rect.x1 + MARGIN, min(rect.y1 + MARGIN, limit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper", nargs="?", type=Path, default=ROOT / "full_paper" / "paper.pdf")
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "main_taxonomy")
    parser.add_argument("--zoom", type=float, default=5.0, help="PNG scale relative to 72 dpi")
    args = parser.parse_args()

    paper = fitz.open(args.paper)
    page = next(p for p in paper if p.search_for(CAPTION))
    clip = figure_rect(page)

    figure = fitz.open()
    target = figure.new_page(width=clip.width, height=clip.height)
    target.show_pdf_page(target.rect, paper, page.number, clip=clip)
    figure.save(f"{args.out}.pdf", garbage=4, deflate=True)

    target = figure[0]
    target.get_pixmap(matrix=fitz.Matrix(args.zoom, args.zoom), alpha=False).save(f"{args.out}.png")
    svg = target.get_svg_image(text_as_path=True)
    # Opaque background so the figure stays readable in dark mode.
    head_end = svg.index(">", svg.index("<svg")) + 1
    svg = svg[:head_end] + '\n<rect width="100%" height="100%" fill="#ffffff"/>' + svg[head_end:]
    Path(f"{args.out}.svg").write_text(svg, encoding="utf-8")
    print(f"Wrote {args.out}.{{pdf,png,svg}} from page {page.number + 1} ({clip.width:.0f} x {clip.height:.0f} pt)")


if __name__ == "__main__":
    main()
