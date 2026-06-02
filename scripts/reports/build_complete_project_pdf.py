#!/usr/bin/env python3
"""
Build the complete project technical-report PDF from repository markdown sources.

This is intentionally source-driven: the PDF is assembled from the actual docs in
this repository instead of from a hand-written summary, so PCam, PANDA,
TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, dominant-site results, math blocks,
code blocks, tables, claim boundaries, and reproduction commands do not silently
fall out of the website PDF.

Output:
    docs/public/computational_pathology_research_complete_technical_report.pdf

Usage:
    python scripts/reports/build_complete_project_pdf.py
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Flowable,
        Image,
        PageBreak,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "reportlab is required. Install with: python -m pip install reportlab"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "public" / "computational_pathology_research_complete_technical_report.pdf"

# Order matters. This is the dossier order used for the public PDF.
SOURCE_DOCS = [
    ("Repository README", "README.md"),
    ("Website / public landing page", "docs/index.md"),
    ("Project overview", "docs/overview/index.md"),
    ("Claim status", "docs/overview/claim-status.md"),
    ("PCam benchmark results", "docs/results/pcam-results.md"),
    ("PANDA slide-level baselines", "docs/results/panda-slide-level-baselines.md"),
    ("PANDA TransnnMIL stabilization", "docs/results/panda-transnnmil-stability.md"),
    ("PathologyFL", "docs/federated/pathologyfl.md"),
    ("FAIR-WEIGHTS-H", "docs/theory/fair-weights-h.md"),
    ("Dominant-site federated pathology paper", "docs/research/dominant-site-federated-pathology-paper.md"),
    ("Generated dominant-site figures", "docs/research/dominant-site-generated-figures.md"),
    ("Dominance detector transfer results", "docs/research/dominance-detector-transfer-results.md"),
    ("Detector diagnostic ablation", "docs/research/detector-diagnostic-ablation.md"),
    ("FAIR-WEIGHTS-H stress result", "docs/research/fair-weights-h-stress-result.md"),
    ("Roadmap", "docs/roadmap/index.md"),
    ("Limitations", "docs/roadmap/limitations.md"),
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class SourceDoc:
    title: str
    rel_path: str
    abs_path: Path
    markdown: str


class Anchor(Flowable):
    """Zero-height bookmark/outline anchor."""

    def __init__(self, name: str, title: str | None = None, level: int = 0):
        super().__init__()
        self.name = name
        self.title = title
        self.level = level
        self.width = 0
        self.height = 0

    def draw(self) -> None:
        self.canv.bookmarkPage(self.name)
        if self.title:
            self.canv.addOutlineEntry(self.title, self.name, level=self.level, closed=False)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=14,
            textColor=colors.HexColor("#111827"),
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=12.5,
            leading=17,
            alignment=TA_CENTER,
            spaceAfter=8,
            textColor=colors.HexColor("#374151"),
        ),
        "meta": ParagraphStyle(
            "ReportMeta",
            parent=base["Normal"],
            fontSize=9.3,
            leading=12.5,
            alignment=TA_CENTER,
            spaceAfter=5,
            textColor=colors.HexColor("#4b5563"),
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=base["Heading1"],
            fontSize=15.5,
            leading=20,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#111827"),
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontSize=12.4,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1f2937"),
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontSize=9.2,
            leading=12.5,
            spaceAfter=5.3,
            textColor=colors.HexColor("#111827"),
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontSize=7.1,
            leading=8.8,
            spaceAfter=3.5,
            textColor=colors.HexColor("#374151"),
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=7.1,
            leading=8.7,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=6,
            backColor=colors.HexColor("#f8fafc"),
            textColor=colors.HexColor("#111827"),
        ),
        "quote": ParagraphStyle(
            "ReportQuote",
            parent=base["BodyText"],
            fontSize=8.9,
            leading=12,
            leftIndent=18,
            rightIndent=8,
            borderColor=colors.HexColor("#d1d5db"),
            borderWidth=0.8,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
            textColor=colors.HexColor("#374151"),
        ),
        "toc": ParagraphStyle(
            "ReportTOCLine",
            parent=base["BodyText"],
            fontSize=9.1,
            leading=12,
            leftIndent=8,
            spaceAfter=2,
            textColor=colors.HexColor("#111827"),
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=base["BodyText"],
            fontSize=7.8,
            leading=10,
            leftIndent=12,
            rightIndent=12,
            spaceAfter=8,
            textColor=colors.HexColor("#374151"),
        ),
    }


def clean_inline(text: str) -> str:
    """Small safe markdown-inline subset for ReportLab Paragraph."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"<link href='\2' color='#1d4ed8'>\1</link>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", text)
    return text


def strip_frontmatter(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, min(len(lines), 80)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :])
    return markdown


def load_sources(selected: Iterable[tuple[str, str]]) -> tuple[list[SourceDoc], list[str]]:
    docs: list[SourceDoc] = []
    missing: list[str] = []
    for title, rel in selected:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        docs.append(SourceDoc(title, rel, path, strip_frontmatter(path.read_text(encoding="utf-8", errors="replace"))))
    return docs, missing


def slug(text: str, used: dict[str, int]) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:64] or "section"
    count = used.get(base, 0)
    used[base] = count + 1
    return base if count == 0 else f"{base}_{count + 1}"


def collect_headings(docs: list[SourceDoc]) -> list[tuple[str, str, int]]:
    used: dict[str, int] = {}
    out: list[tuple[str, str, int]] = []
    for doc in docs:
        key = slug(doc.title, used)
        out.append((key, doc.title, 0))
        for line in doc.markdown.splitlines():
            m = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
            if not m:
                continue
            level = min(len(m.group(1)), 3)
            title = re.sub(r"\*\*", "", m.group(2)).strip()
            out.append((slug(f"{doc.title} {title}", used), title, level))
    return out


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for idx, line in enumerate(lines):
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if idx == 1 and all(set(part) <= set(":- ") for part in parts):
            continue
        rows.append(parts)
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    cols = len(rows[0]) if rows else 1
    total = 6.35 * inch
    if cols >= 8:
        style = styles["small"]
        widths = [total / cols] * cols
    elif cols >= 5:
        style = styles["small"]
        widths = [total / cols] * cols
    elif cols == 2:
        style = styles["body"]
        widths = [2.0 * inch, 4.35 * inch]
    else:
        style = styles["small"] if cols > 3 else styles["body"]
        widths = [total / cols] * cols
    data = [[Paragraph(clean_inline(cell), style) for cell in row] for row in rows]
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def resolve_image(markdown_path: Path, image_ref: str) -> Path | None:
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        return None
    candidate = (markdown_path.parent / image_ref).resolve()
    if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTS:
        return candidate
    # VitePress docs sometimes refer to ../figures while repo figures are at root/figures.
    candidate = (REPO_ROOT / image_ref.lstrip("/"))
    if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTS:
        return candidate
    candidate = (REPO_ROOT / image_ref.replace("../", "")).resolve()
    if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTS:
        return candidate
    return None


def add_paragraph(story: list, lines: list[str], styles: dict[str, ParagraphStyle]) -> None:
    if not lines:
        return
    text = " ".join(line.strip() for line in lines).strip()
    if text:
        story.append(Paragraph(clean_inline(text), styles["body"]))


def render_doc(doc: SourceDoc, story: list, styles: dict[str, ParagraphStyle], used: dict[str, int]) -> None:
    story.append(PageBreak())
    story.append(Anchor(slug(doc.title, used), doc.title, 0))
    story.append(Paragraph(clean_inline(doc.title), styles["h1"]))
    story.append(Paragraph(clean_inline(doc.rel_path), styles["small"]))
    story.append(Spacer(1, 0.08 * inch))

    lines = doc.markdown.splitlines()
    para: list[str] = []
    code: list[str] = []
    table: list[str] = []
    in_code = False

    def flush_para() -> None:
        nonlocal para
        add_paragraph(story, para, styles)
        para = []

    def flush_code() -> None:
        nonlocal code
        if code:
            story.append(Preformatted("\n".join(code), styles["code"], maxLineLength=96))
            code = []

    def flush_table() -> None:
        nonlocal table
        if table:
            rows = parse_table(table)
            if rows:
                story.append(table_flowable(rows, styles))
                story.append(Spacer(1, 0.05 * inch))
            table = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]

        if line.strip().startswith("```"):
            if not in_code:
                flush_para()
                flush_table()
                in_code = True
                code = []
            else:
                in_code = False
                flush_code()
            idx += 1
            continue

        if in_code:
            code.append(line)
            idx += 1
            continue

        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            flush_para()
            table.append(line)
            idx += 1
            if idx >= len(lines) or not (lines[idx].strip().startswith("|") and "|" in lines[idx].strip()[1:]):
                flush_table()
            continue
        else:
            flush_table()

        if line.strip() == "---":
            flush_para()
            story.append(Spacer(1, 0.08 * inch))
            idx += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush_para()
            level = min(len(heading.group(1)), 3)
            title = re.sub(r"\*\*", "", heading.group(2)).strip()
            key = slug(f"{doc.title} {title}", used)
            story.append(Anchor(key, title, min(level, 2)))
            story.append(Paragraph(clean_inline(title), styles["h1"] if level <= 2 else styles["h2"]))
            idx += 1
            continue

        image = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image:
            flush_para()
            alt, ref = image.group(1), image.group(2)
            path = resolve_image(doc.abs_path, ref)
            if path:
                try:
                    story.append(Image(str(path), width=6.35 * inch, height=3.55 * inch, kind="proportional"))
                    story.append(Paragraph(clean_inline(alt or path.name), styles["caption"]))
                except Exception as exc:  # pragma: no cover
                    story.append(Paragraph(clean_inline(f"[Image skipped: {ref} ({exc})]"), styles["small"]))
            else:
                story.append(Paragraph(clean_inline(f"[Image reference: {ref}]"), styles["small"]))
            idx += 1
            continue

        if line.strip().startswith(">"):
            flush_para()
            quote: list[str] = []
            while idx < len(lines) and lines[idx].strip().startswith(">"):
                quote.append(lines[idx].strip()[1:].strip())
                idx += 1
            story.append(Paragraph(clean_inline(" ".join(quote)), styles["quote"]))
            continue

        if re.match(r"^\s*[-*]\s+", line):
            flush_para()
            item = re.sub(r"^\s*[-*]\s+", "", line).strip()
            story.append(Paragraph("• " + clean_inline(item), styles["body"]))
            idx += 1
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            flush_para()
            item = line.strip()
            story.append(Paragraph(clean_inline(item), styles["body"]))
            idx += 1
            continue

        if not line.strip():
            flush_para()
            story.append(Spacer(1, 0.025 * inch))
            idx += 1
            continue

        para.append(line)
        idx += 1

    flush_para()
    flush_code()
    flush_table()


def build_pdf(out_path: Path, sources: list[SourceDoc], missing: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    headings = collect_headings(sources)
    story: list = []

    story.append(Anchor("cover", "Cover", 0))
    story.append(Spacer(1, 0.45 * inch))
    story.append(Paragraph("Computational Pathology AI Research Framework", styles["title"]))
    story.append(
        Paragraph(
            "Complete technical report: PCam, PANDA/TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, and dominant-site federated pathology",
            styles["subtitle"],
        )
    )
    story.append(Paragraph("Matthew Vaishnav", styles["meta"]))
    story.append(Paragraph("Independent research and engineering technical report", styles["meta"]))
    story.append(Paragraph("Research-only. Not clinically validated. Not diagnostic software.", styles["meta"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "This PDF is generated directly from repository source documents. It preserves the selected markdown docs, tables, code blocks, math blocks, figure references, claim boundaries, and reproduction artifacts rather than relying on a hand-written summary.",
            styles["body"],
        )
    )
    if missing:
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Missing configured source documents:", styles["h2"]))
        for rel in missing:
            story.append(Paragraph("• " + clean_inline(rel), styles["small"]))
    story.append(PageBreak())

    story.append(Anchor("toc", "Table of Contents", 0))
    story.append(Paragraph("Table of Contents", styles["h1"]))
    for key, title, level in headings:
        indent = "&nbsp;" * (level * 4)
        story.append(Paragraph(f"{indent}<link href='#{key}' color='#1d4ed8'>{clean_inline(title)}</link>", styles["toc"]))

    used: dict[str, int] = {}
    for doc in sources:
        render_doc(doc, story, styles, used)

    def on_page(canv, doc) -> None:
        page = canv.getPageNumber()
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#6b7280"))
        canv.drawString(0.66 * inch, 0.42 * inch, "Computational Pathology AI Research Framework")
        canv.drawRightString(doc.pagesize[0] - 0.66 * inch, 0.42 * inch, f"Page {page}")
        canv.setStrokeColor(colors.HexColor("#e5e7eb"))
        canv.line(0.66 * inch, 0.60 * inch, doc.pagesize[0] - 0.66 * inch, 0.60 * inch)

    pdf = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.66 * inch,
        rightMargin=0.66 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.72 * inch,
        title="Computational Pathology AI Research Framework - Complete Technical Report",
        author="Matthew Vaishnav",
        subject="PCam, PANDA/TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, dominant-site federated pathology",
        keywords="computational pathology, PCam, PANDA, TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, federated learning",
    )
    pdf.build(story, onFirstPage=on_page, onLaterPages=on_page)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", action="store_true", help="Fail if any configured source document is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources, missing = load_sources(SOURCE_DOCS)
    if args.strict and missing:
        raise SystemExit("Missing source documents: " + ", ".join(missing))
    if not sources:
        raise SystemExit("No source documents found; cannot build report.")
    build_pdf(args.out, sources, missing)
    rel = args.out.resolve().relative_to(REPO_ROOT)
    print(f"Wrote {rel}")
    if missing:
        print("Warning: missing configured source documents:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
