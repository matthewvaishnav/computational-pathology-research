#!/usr/bin/env python3
"""Prepare a self-contained LaTeX source folder for the public research PDF.

The source is organized as a compact, result-first main paper followed by a
complete empirical appendix. The build preserves the PathoAlign architecture
and matched-budget resource-allocation figures together with the detailed
PANDA, CAMELYON17, PCam, federated, and identifiability evidence.

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
    ARXIV / "pathoalign_architecture_diagram.tex",
    ARXIV / "pathoalign_resource_allocation_figure.tex",
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
ARCHITECTURE_INCLUDE = r"\input{pathoalign_architecture_diagram.tex}"
ALLOCATION_INCLUDE = r"\input{pathoalign_resource_allocation_figure.tex}"


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required paper source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_and_normalize_main(path: Path) -> None:
    """Apply only idempotent compatibility transforms and validate composition."""
    text = path.read_text(encoding="utf-8")

    # Compatibility with older source copies. The current source already uses
    # the intended compact single-column format and explicit geometry.
    text = text.replace(
        r"\documentclass[11pt]{article}",
        r"\documentclass[10pt]{article}",
    )
    text = text.replace(
        r"\documentclass[10pt,twocolumn]{article}",
        r"\documentclass[10pt]{article}",
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

    if BROADER_RESEARCH_INCLUDE not in text:
        if GENERIC_SECONDARY_BLOCK not in text:
            raise RuntimeError(
                "Could not locate either the complete-study appendix include or "
                "the legacy generic secondary-research block in main.tex"
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
        "Paired-Acquisition Neural Factorization",
        "PathoAlign",
        "SCORPION",
        "PANDA",
        "CAMELYON17",
        "PCam",
        "broader_research_program.tex",
        "pathoalign_architecture_diagram.tex",
        "pathoalign_resource_allocation_figure.tex",
    )
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise RuntimeError(f"The build copy lost required paper content: {missing}")

    if ARCHITECTURE_INCLUDE not in text or ALLOCATION_INCLUDE not in text:
        raise RuntimeError("The compact main paper lost one or both primary figures")

    broader = (BUILD / "broader_research_program.tex").read_text(encoding="utf-8")
    broader_required = (
        "PANDA",
        "CAMELYON17",
        "TransnnMIL",
        "PathologyFL",
        "PCam",
        "Matched-budget",
    )
    broader_missing = [term for term in broader_required if term not in broader]
    if broader_missing:
        raise RuntimeError(
            f"The complete empirical appendix lost study families: {broader_missing}"
        )

    architecture = (BUILD / "pathoalign_architecture_diagram.tex").read_text(
        encoding="utf-8"
    )
    if "biological branch" not in architecture or "acquisition branch" not in architecture:
        raise RuntimeError("The PathoAlign architecture figure is incomplete")

    allocation = (BUILD / "pathoalign_resource_allocation_figure.tex").read_text(
        encoding="utf-8"
    )
    if "6,400" not in allocation or "12,800" not in allocation:
        raise RuntimeError("The matched-budget allocation figure is incomplete")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    for source in FILES:
        copy_required(source, BUILD / source.name)

    validate_and_normalize_main(BUILD / "main.tex")

    print(f"Prepared {BUILD.relative_to(ROOT)}")
    print("Build with:")
    print("  cd paper/arxiv/build")
    print("  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex")


if __name__ == "__main__":
    main()
