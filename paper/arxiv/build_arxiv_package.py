#!/usr/bin/env python3
"""Prepare a self-contained umbrella-platform LaTeX source folder.

The build copy preserves the PathoAlign-first paper while explicitly retaining
PANDA/TransnnMIL and CAMELYON17/PathologyFL as substantive secondary research
lines. It applies compact single-column formatting without injecting retired
legacy figures or rewriting the central scientific claims.

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
    ARXIV / "broader_research_program.tex",
    ARXIV / "identifiability_calculations.tex",
    ARXIV / "identifiability_calculations_part1.tex",
    ARXIV / "identifiability_calculations_part2a.tex",
    ARXIV / "identifiability_calculations_part2b.tex",
    ARXIV / "identifiability_calculations_part3a.tex",
]

GENERIC_SECONDARY_BLOCK = r"""\section{Secondary research components}
The repository also contains whole-slide multiple-instance learning and federated-pathology experiments. TransnnMIL combines correlated transformer aggregation with gated attention for slide-level modeling. PathologyFL studies sample-weighted and contribution-aware institutional aggregation under controlled corruption and external-center validation. These projects are scientifically related because they address later stages of the pathology learning pipeline, but they are not the central evidence in this paper and are not used to strengthen the PathoAlign representation claim.
"""

BROADER_RESEARCH_INCLUDE = r"\input{broader_research_program.tex}"


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required paper source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def apply_compact_single_column_format(path: Path) -> None:
    """Apply layout and public-PDF composition transformations."""
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

    # Older source copies contained a short generic secondary-research section
    # that the public build replaced with the full broader-program include.
    # Current main.tex already contains the include directly, so the build must
    # accept that state instead of failing while searching for retired text.
    if BROADER_RESEARCH_INCLUDE not in text:
        if GENERIC_SECONDARY_BLOCK not in text:
            raise RuntimeError(
                "Could not locate either the broader-research include or the "
                "legacy generic secondary-research block in main.tex"
            )
        text = text.replace(
            GENERIC_SECONDARY_BLOCK,
            BROADER_RESEARCH_INCLUDE + "\n",
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

    required_terms = (
        "PathoAlign",
        "SCORPION",
        "broader_research_program.tex",
    )
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise RuntimeError(f"The build copy lost required paper sections: {missing}")

    broader = (BUILD / "broader_research_program.tex").read_text(encoding="utf-8")
    broader_required = ("PANDA", "CAMELYON17", "TransnnMIL", "PathologyFL")
    broader_missing = [term for term in broader_required if term not in broader]
    if broader_missing:
        raise RuntimeError(
            f"The public PDF lost broader research components: {broader_missing}"
        )

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
