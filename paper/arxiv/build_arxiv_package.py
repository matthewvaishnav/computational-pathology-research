#!/usr/bin/env python3
"""Prepare a local arXiv submission source folder.

The script copies the focused LaTeX preprint and its references into
paper/arxiv/build/, copies the generated dominant-site figures, and applies a
classic NIPS-style two-column paper format to the build copy.

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
    ROOT / "figures" / "dominant-site-figure-1-problem-schematic.png",
    ROOT / "figures" / "dominant-site-figure-3-detector-transfer.png",
    ROOT / "figures" / "dominant-site-figure-4-detector-ablation.png",
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


def apply_classic_paper_format(path: Path) -> None:
    """Apply compact two-column formatting to the build copy of main.tex."""
    text = path.read_text(encoding="utf-8")

    text = text.replace(r"\documentclass[11pt]{article}", r"\documentclass[10pt,twocolumn]{article}")
    text = text.replace(r"\usepackage[margin=1in]{geometry}", r"\usepackage[margin=0.75in]{geometry}")

    if r"\usepackage{times}" not in text:
        text = text.replace(r"\usepackage{microtype}", r"\usepackage{microtype}" + "\n" + r"\usepackage{times}")

    if r"\setlength{\columnsep}" not in text:
        insert = "\n".join([
            r"\setlength{\columnsep}{0.24in}",
            r"\setlength{\parindent}{1em}",
            r"\setlength{\parskip}{0pt}",
            r"\setlength{\textfloatsep}{10pt plus 1pt minus 2pt}",
            r"\setlength{\floatsep}{8pt plus 1pt minus 2pt}",
            r"\setlength{\intextsep}{8pt plus 1pt minus 2pt}",
            r"\renewcommand{\baselinestretch}{0.98}",
        ])
        text = text.replace(r"\usepackage[numbers,sort&compress]{natbib}", r"\usepackage[numbers,sort&compress]{natbib}" + "\n" + insert)

    text = text.replace(
        "\\author{Matthew Vaishnav\\\nIndependent Researcher\\\n\\texttt{matthewvaishnav@users.noreply.github.com}}",
        "\\author{Matthew Vaishnav\\\nIndependent Researcher}"
    )

    # In two-column format, wide tables/figures should span both columns like the reference paper.
    text = text.replace(r"\begin{table}[t]", r"\begin{table*}[t]")
    text = text.replace(r"\end{table}", r"\end{table*}")
    text = text.replace(r"\begin{figure}[t]", r"\begin{figure*}[t]")
    text = text.replace(r"\end{figure}", r"\end{figure*}")
    text = text.replace(r"\resizebox{\textwidth}{!}{%", r"\resizebox{\textwidth}{!}{%")
    text = text.replace(r"width=0.95\textwidth", r"width=0.92\textwidth")

    # Add a first schematic and an ablation/calibration figure if the source has not yet included them.
    thesis = (
        "\\paragraph{Working thesis.}\n"
        "In federated computational pathology, raw sample count is not the same as task-specific site-signal alignment. FedAvg can become less safe when the largest simulated pathology client has a training-label process that is misaligned with the validation objective, and dominance-aware aggregation or switching can reduce that risk under controlled stress.\n"
    )
    figure1 = (
        thesis + "\n"
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=0.92\\textwidth]{figures/dominant-site-figure-1-problem-schematic.png}\n"
        "\\caption{Problem schematic. FedAvg uses sample count as aggregation authority, but a high-volume site can have a training-label process that is less aligned with the declared validation objective.}\n"
        "\\label{fig:problem_schematic}\n"
        "\\end{figure*}\n"
    )
    if "dominant-site-figure-1-problem-schematic.png" not in text and thesis in text:
        text = text.replace(thesis, figure1)

    calib = (
        "This produced 36 detector configurations. A configuration was counted as robust-positive if it preserved clean trigger rate at or below 20\\% and positive global-QWK, macro-F1, and worst-site-QWK deltas at both 35\\% and 45\\% conservative shift. In total, 29 of 36 configurations were robust positive.\n"
    )
    figure4 = (
        calib + "\n"
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=0.92\\textwidth]{figures/dominant-site-figure-4-detector-ablation.png}\n"
        "\\caption{Detector interpretability, ablation, and calibration robustness. The transfer result is not a one-diagnostic or one-threshold artifact in the conservative threshold-shift setting.}\n"
        "\\label{fig:detector_ablation}\n"
        "\\end{figure*}\n"
    )
    if "dominant-site-figure-4-detector-ablation.png" not in text and calib in text:
        text = text.replace(calib, figure4)

    path.write_text(text, encoding="utf-8")


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    FIG_BUILD.mkdir(parents=True)

    for src in FILES:
        copy_required(src, BUILD / src.name)

    apply_classic_paper_format(BUILD / "main.tex")

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
