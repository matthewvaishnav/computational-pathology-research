#!/usr/bin/env python3
"""Prepare a self-contained LaTeX source folder for the public research PDF.

The source is organized as a compact, result-first main paper followed by a
complete empirical appendix. The build preserves the complete mathematical
PathoAlign specification, its full-page 3D architecture figure, and the
matched-budget resource-allocation figure together with the detailed PANDA,
CAMELYON17, PCam, federated, and identifiability evidence.

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
    ARXIV / "pathoalign_model_math.tex",
    ARXIV / "pathoalign_architecture_fullpage.tex",
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
MODEL_MATH_INCLUDE = r"\input{pathoalign_model_math.tex}"
FULLPAGE_ARCHITECTURE_INCLUDE = r"\input{pathoalign_architecture_fullpage.tex}"
ALLOCATION_INCLUDE = r"\input{pathoalign_resource_allocation_figure.tex}"

TITLE_REPOSITORY_LINK = (
    r"\url{https://github.com/matthewvaishnav/computational-pathology-research}"
)
TITLE_REPOSITORY_TEXT = (
    r"{\ttfamily github.com/matthewvaishnav/computational-pathology-research}"
)

APPENDIX_WRAPPER = r"""\appendix
\section{Complete Empirical Study Record}
The main text is deliberately narrow and result-first. The following appendix preserves the complete study-by-study research record, including methods, secondary experiments, negative results, systems probes, and claim boundaries.

\input{broader_research_program.tex}

\section{Supporting Identifiability Calculations}
\input{identifiability_calculations.tex}
"""

APPENDIX_COMPACT = r"""\appendix
\input{broader_research_program.tex}
\input{identifiability_calculations.tex}
"""


ARCHITECTURE_START = r"\subsection{Architecture}"
ARCHITECTURE_END = r"\section{Preventing Representation Failure}"


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required paper source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def restore_model_math(text: str) -> str:
    """Replace the compact prose-only model block with the canonical math include."""
    if MODEL_MATH_INCLUDE in text:
        return text
    start = text.find(ARCHITECTURE_START)
    end = text.find(ARCHITECTURE_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "Could not locate the PathoAlign architecture/objective block in main.tex"
        )
    return text[:start] + MODEL_MATH_INCLUDE + "\n\n" + text[end:]


def validate_and_normalize_main(path: Path) -> None:
    """Apply idempotent composition transforms and validate the public build."""
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

    # Keep the classic paper appearance black-and-white, including references.
    text = text.replace("blue!55!black", "black")

    # Avoid a large colored URL annotation in the title block while preserving
    # the public repository address in a classic monospace author line.
    text = text.replace(TITLE_REPOSITORY_LINK, TITLE_REPOSITORY_TEXT, 1)

    # Restore the complete, implementation-faithful model specification before
    # applying the remaining composition transforms.
    text = restore_model_math(text)

    # The included appendix files already define their own top-level sections.
    # Remove the temporary wrapper headings to keep appendix lettering clean.
    text = text.replace(APPENDIX_WRAPPER, APPENDIX_COMPACT, 1)

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
        "pathoalign_model_math.tex",
        "pathoalign_resource_allocation_figure.tex",
    )
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise RuntimeError(f"The build copy lost required paper content: {missing}")

    if MODEL_MATH_INCLUDE not in text or ALLOCATION_INCLUDE not in text:
        raise RuntimeError("The compact main paper lost the model math or allocation figure")

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

    model_math = (BUILD / "pathoalign_model_math.tex").read_text(encoding="utf-8")
    model_math_required = (
        r"\mathcal{L}_{\mathrm{pair}}",
        r"\mathcal{L}_{\mathrm{recon}}",
        r"\mathcal{L}_{\mathrm{var},a}",
        r"\mathcal{L}_{\mathrm{cov},b}",
        r"\mathcal{L}_{\mathrm{scan},b}",
        r"\mathcal{L}_{\mathrm{scan},a}",
        r"\mathcal{L}_{\mathrm{dep}}",
        r"\mathcal{L}_{\mathrm{xcov}}",
        r"\operatorname{GRL}_{\gamma}",
        r"0.25\mathcal{L}_{\mathrm{var},a}",
        r"20\mathcal{L}_{\mathrm{dep}}",
        "same-region agreement and scanner suppression",
        "scanner prediction and acquisition retention",
        FULLPAGE_ARCHITECTURE_INCLUDE,
        r"\pdfpageattr{/Rotate 90}",
        r"\captionof{figure}",
    )
    model_math_missing = [term for term in model_math_required if term not in model_math]
    if model_math_missing:
        raise RuntimeError(
            f"The PathoAlign mathematical specification is incomplete: {model_math_missing}"
        )
    if "pathoalign_architecture_diagram" in model_math:
        raise RuntimeError("The retired compact PathoAlign flowchart remains in the model section")

    architecture = (BUILD / "pathoalign_architecture_fullpage.tex").read_text(
        encoding="utf-8"
    )
    architecture_required = (
        "PathoAlign: paired-acquisition neural factorization",
        "Biological factor",
        "Acquisition factor",
        "decoder MLP",
        r"\mathcal L_{\mathrm{xcov}}",
        r"\mathcal L_{\mathrm{recon}}",
    )
    architecture_missing = [
        term for term in architecture_required if term not in architecture
    ]
    if architecture_missing:
        raise RuntimeError(
            f"The full-page PathoAlign architecture is incomplete: {architecture_missing}"
        )

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
