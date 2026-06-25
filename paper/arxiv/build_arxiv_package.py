#!/usr/bin/env python3
"""Build an arXiv-ready source package for the paired-acquisition paper.

The script copies the paper source into ``paper/arxiv/build`` and performs a few
lightweight source-normalization checks before optionally compiling with
``latexmk`` if it is available on the local machine.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARXIV = ROOT / "paper" / "arxiv"
BUILD = ARXIV / "build"
PACKAGE = ARXIV / "paired_acquisition_neural_factorization_arxiv_source.zip"

MAIN_SOURCE = ARXIV / "main.tex"

MODEL_MATH_BASENAME = "paired_acquisition_model_math"
ALLOCATION_BASENAME = "paired_acquisition_resource_allocation_figure"
FIGURE1_BASENAME = "paired_acquisition_figure1_benchmark_table"
BROADER_BASENAME = "broader_research_program"

MODEL_MATH_INCLUDE = rf"\input{{{MODEL_MATH_BASENAME}}}"
ALLOCATION_INCLUDE = rf"\input{{{ALLOCATION_BASENAME}}}"
FIGURE1_INCLUDE = rf"\input{{{FIGURE1_BASENAME}}}"
BROADER_INCLUDE = rf"\input{{{BROADER_BASENAME}}}"

RETIRE_BEGIN = "% BEGIN retired legacy platform framing"
RETIRE_END = "% END retired legacy platform framing"


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def include_present(text: str, basename: str) -> bool:
    """Accept both \input{name} and \input{name.tex}."""
    return rf"\input{{{basename}}}" in text or rf"\input{{{basename}.tex}}" in text


def first_present(text: str, markers: tuple[str, ...]) -> str | None:
    return next((marker for marker in markers if marker in text), None)


def remove_retired_block(text: str) -> str:
    start = text.find(RETIRE_BEGIN)
    end = text.find(RETIRE_END)
    if start == -1 and end == -1:
        return text
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Malformed retired legacy block markers in main.tex")
    return text[:start] + text[end + len(RETIRE_END) :]


def normalize_main_source(text: str) -> str:
    text = remove_retired_block(text)

    if not include_present(text, BROADER_BASENAME):
        marker = r"\end{document}"
        if marker not in text:
            raise RuntimeError("main.tex is missing \\end{document}")
        text = text.replace(marker, f"\n{BROADER_INCLUDE}\n\n{marker}")

    if not include_present(text, MODEL_MATH_BASENAME):
        marker = first_present(
            text,
            (
                r"\section{The Paired-Acquisition Neural Factorization Model}",
                r"\section{The Paired-Acquisition Neural Factorization Model}",
                r"\section{Method}",
            ),
        )
        if marker is None:
            raise RuntimeError("main.tex is missing the model/method section marker")
        text = text.replace(marker, f"{marker}\n{MODEL_MATH_INCLUDE}\n", 1)

    if not include_present(text, ALLOCATION_BASENAME):
        marker = first_present(
            text,
            (
                r"\subsection{Resource allocation under matched compute}",
                r"\section{Synthetic resource-allocation experiment}",
            ),
        )
        if marker is None:
            raise RuntimeError("main.tex is missing the allocation insertion marker")
        text = text.replace(marker, f"{marker}\n{ALLOCATION_INCLUDE}\n", 1)

    if not include_present(text, FIGURE1_BASENAME):
        marker = first_present(
            text,
            (
                r"\subsection{External paired-scanner validation package}",
                r"\section{External validation evidence}",
            ),
        )
        if marker is None:
            raise RuntimeError("main.tex is missing the external-validation insertion marker")
        text = text.replace(marker, f"{marker}\n{FIGURE1_INCLUDE}\n", 1)

    return text


def copy_sources() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    for path in ARXIV.iterdir():
        if path == BUILD or path == PACKAGE:
            continue
        if path.is_file():
            shutil.copy2(path, BUILD / path.name)

    normalized = normalize_main_source(MAIN_SOURCE.read_text(encoding="utf-8"))
    (BUILD / "main.tex").write_text(normalized, encoding="utf-8")


def find_pdflatex() -> str | None:
    return shutil.which("pdflatex")


def find_latexmk() -> str | None:
    return shutil.which("latexmk")


def compile_pdf() -> None:
    latexmk = find_latexmk()
    if latexmk:
        try:
            run([latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=BUILD)
            return
        except subprocess.CalledProcessError:
            print("latexmk failed; falling back to pdflatex", flush=True)

    pdflatex = find_pdflatex()
    if not pdflatex:
        print("latexmk/pdflatex not found; skipping PDF compilation", flush=True)
        return

    for _ in range(3):
        run([pdflatex, "-interaction=nonstopmode", "main.tex"], cwd=BUILD)


def zip_sources() -> None:
    if PACKAGE.exists():
        PACKAGE.unlink()

    allowed_suffixes = {
        ".tex",
        ".bib",
        ".bst",
        ".cls",
        ".sty",
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
    }

    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(BUILD.iterdir()):
            if path.suffix.lower() in allowed_suffixes:
                zf.write(path, path.name)

    print(f"Wrote {PACKAGE}", flush=True)


def validate_and_normalize_main(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if not include_present(text, BROADER_BASENAME):
        raise RuntimeError("The broader research-program appendix include was not injected")
    if r"\onecolumn" not in text:
        raise RuntimeError("main.tex should switch to one-column layout before the appendix")

    if r"\begin{figure*}" not in text or r"\begin{table*}" not in text:
        raise RuntimeError("Wide result floats must use two-column-spanning environments")

    if "federated_pathology_pipeline_diagram" in text:
        raise RuntimeError("A retired legacy pipeline figure was unexpectedly injected")

    required_terms = (
        "Paired-Acquisition Neural Factorization",
        "SCORPION",
        "PANDA",
        "CAMELYON17",
        "PCam",
    )
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise RuntimeError(f"The build copy lost required paper content: {missing}")

    if not include_present(text, MODEL_MATH_BASENAME):
        raise RuntimeError("The compact main paper lost the model math include")
    if not include_present(text, ALLOCATION_BASENAME):
        raise RuntimeError("The compact main paper lost the allocation figure include")
    if not include_present(text, FIGURE1_BASENAME):
        raise RuntimeError("The compact main paper lost the Figure 1 benchmark table include")

    broader = (BUILD / f"{BROADER_BASENAME}.tex").read_text(encoding="utf-8")
    if r"\section{Broader computational-pathology study record}" not in broader:
        raise RuntimeError("The broader research-program appendix file lost its section header")

    broader_required = (
        "PANDA",
        "CAMELYON17",
        "TransnnMIL",
        "PCam",
        "Pair-repeat",
    )
    broader_missing = [term for term in broader_required if term not in broader]
    if broader_missing:
        raise RuntimeError(
            f"The complete empirical appendix lost study families: {broader_missing}"
        )

    model_math = (BUILD / f"{MODEL_MATH_BASENAME}.tex").read_text(encoding="utf-8")
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
    )
    model_math_missing = [term for term in model_math_required if term not in model_math]
    if model_math_missing:
        raise RuntimeError(f"The model math include lost objective terms: {model_math_missing}")

    allocation = (BUILD / f"{ALLOCATION_BASENAME}.tex").read_text(encoding="utf-8")
    allocation_required = (
        "0.4081",
        "0.4259",
        "0.4489",
        "0.4619",
        "100",
        "200",
        "6,400 presentations",
        "12,800 presentations",
    )
    allocation_missing = [term for term in allocation_required if term not in allocation]
    if allocation_missing:
        raise RuntimeError(f"The allocation include lost frozen numeric results: {allocation_missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    copy_sources()
    validate_and_normalize_main(BUILD / "main.tex")

    if not args.no_compile:
        compile_pdf()

    zip_sources()


if __name__ == "__main__":
    main()
