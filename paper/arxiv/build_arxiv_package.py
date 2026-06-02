#!/usr/bin/env python3
"""Prepare a local arXiv submission source folder.

The script copies the focused LaTeX preprint and its references into
paper/arxiv/build/ and copies the generated dominant-site figures that the
paper includes.

Run from the repository root:

    python paper/arxiv/build_arxiv_package.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARXIV = ROOT / "paper" / "arxiv"
BUILD = ARXIV / "build"
FIG_BUILD = BUILD / "figures"

FIGURES = [
    ROOT / "figures" / "dominant-site-figure-3-detector-transfer.png",
]

FILES = [
    ARXIV / "main.tex",
    ARXIV / "references.bib",
]


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required arXiv source asset not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    FIG_BUILD.mkdir(parents=True)

    for src in FILES:
        copy_required(src, BUILD / src.name)

    for src in FIGURES:
        copy_required(src, FIG_BUILD / src.name)

    print(f"Prepared {BUILD.relative_to(ROOT)}")
    print("Build with:")
    print("  cd paper/arxiv/build")
    print("  pdflatex main.tex")
    print("  bibtex main")
    print("  pdflatex main.tex")
    print("  pdflatex main.tex")


if __name__ == "__main__":
    main()
