#!/usr/bin/env python3
"""Prepare a self-contained umbrella-platform LaTeX source folder.

The build copy preserves the sectioned PathoAlign, TransnnMIL, and PathologyFL
technical report. It applies compact single-column formatting without rewriting
scientific sections or injecting legacy figures.

Run from the repository root:

    python paper/arxiv/build_arxiv_package.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARXIV = ROOT / "paper" / "arxiv"
BUILD = ARXIV / "build"

FILES = [
    ARXIV / "main.tex",
    ARXIV / "references.bib",
    ARXIV / "identifiability_calculations.tex",
    ARXIV / "identifiability_calculations_part1.tex",
    ARXIV / "identifiability_calculations_part2a.tex",
    ARXIV / "identifiability_calculations_part2b.tex",
    ARXIV / "identifiability_calculations_part3a.tex",
]


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required paper source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def apply_compact_single_column_format(path: Path) -> None:
    """Apply layout-only transformations to the build copy."""
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        r"\documentclass[11pt]{article}",
        r"\documentclass[10pt]{article}",
    )
    text = text.replace(
        r"\documentclass[10pt,twocolumn]{article}",
        r"\documentclass[10pt]{article}",
    )
    text = text.replace(
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[letterpaper,top=0.68in,bottom=0.72in,left=1.02in,right=1.02in]{geometry}",
    )

    if r"\usepackage{times}" not in text:
        text = text.replace(
            r"\usepackage{microtype}",
            r"\usepackage{microtype}" + "\n" + r"\usepackage{times}",
            1,
        )
    if r"\PassOptionsToPackage{hyphens}{url}" not in text:
        text = text.replace(
            r"\usepackage{hyperref}",
            r"\PassOptionsToPackage{hyphens}{url}" + "\n" + r"\usepackage{hyperref}",
            1,
        )

    text = text.replace(r"\begin{table*}[t]", r"\begin{table}[t]")
    text = text.replace(r"\end{table*}", r"\end{table}")
    text = text.replace(r"\begin{figure*}[t]", r"\begin{figure}[t]")
    text = text.replace(r"\end{figure*}", r"\end{figure}")

    if "twocolumn" in text:
        raise RuntimeError("Global two-column formatting remains in the paper build")
    if "federated_pathology_pipeline_diagram" in text:
        raise RuntimeError("A retired legacy pipeline figure was unexpectedly injected")

    required_terms = ("PathoAlign", "TransnnMIL", "PathologyFL", "SCORPION")
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise RuntimeError(f"The build copy lost required platform sections: {missing}")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    for source in FILES:
        copy_required(source, BUILD / source.name)

    apply_compact_single_column_format(BUILD / "main.tex")

    print(f"Prepared {BUILD.relative_to(ROOT)}")
    print("Build with:")
    print("  cd paper/arxiv/build")
    print("  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex")


if __name__ == "__main__":
    main()
