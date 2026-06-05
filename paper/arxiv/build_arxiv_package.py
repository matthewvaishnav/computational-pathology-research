#!/usr/bin/env python3
"""Prepare a local arXiv submission source folder.

The script copies the focused LaTeX preprint and its references into
paper/arxiv/build/, copies the generated dominant-site figures, and applies a
classic compact single-column paper format similar to early NIPS proceedings.

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
    ARXIV / "figures" / "federated_pathology_pipeline_diagram.tex",
]

FILES = [
    ARXIV / "main.tex",
    ARXIV / "references.bib",
]


REPRO_OLD = r"""\section{Reproducibility artifacts}
Primary result files include:
\begin{itemize}[leftmargin=*]
  \item \texttt{results/threshold\_shift\_detector\_conservative\_fixed\_labelnoise\_rule\_15seed/best\_detector\_summary.csv}
  \item \texttt{results/threshold\_shift\_detector\_conservative\_fixed\_labelnoise\_rule\_15seed/best\_detector\_run\_diagnostics.csv}
  \item \texttt{results/threshold\_shift\_detector\_conservative\_fixed\_labelnoise\_rule\_15seed\_diagnostic\_summary/diagnostic\_frequency\_by\_stress.csv}
  \item \texttt{results/threshold\_shift\_detector\_conservative\_fixed\_labelnoise\_rule\_15seed\_leave\_one\_out/diagnostic\_ablation\_headline\_35\_45.csv}
  \item \texttt{results/threshold\_shift\_detector\_conservative\_fixed\_labelnoise\_rule\_15seed\_calibration\_sensitivity/calibration\_sensitivity\_headline.csv}
\end{itemize}

Figure-generation scripts include \texttt{scripts/figures/make\_dominant\_site\_schematic.py} and \texttt{scripts/figures/make\_dominant\_site\_paper\_figures.py}.
"""

REPRO_NEW = r"""\section{Reproducibility artifacts}
The result tables, run diagnostics, diagnostic summaries, ablation outputs, calibration-sensitivity outputs, and figure-generation scripts are released in the project repository. Long filesystem paths are shortened in the paper body; exact paths are available from the repository's reproducibility documentation and source tree.

\begin{itemize}[leftmargin=*]
  \item Detector summary CSV and per-run diagnostics.
  \item Diagnostic-frequency summary by stress regime.
  \item Leave-one-diagnostic-family-out ablation table.
  \item Calibration-sensitivity sweep table.
  \item Figure-generation scripts for the schematic and paper figures.
\end{itemize}

Repository: \url{https://github.com/matthewvaishnav/computational-pathology-research}.
"""

PIPELINE_FIGURE = r"""\begin{figure}[!htbp]
\centering
\resizebox{\textwidth}{!}{%
\input{figures/federated_pathology_pipeline_diagram.tex}%
}
\caption{Dominant-site detector-switch federated pathology pipeline. Site-level whole-slide data are converted into Phikon feature representations, local slide-level models are trained per simulated client, and validation diagnostics drive a switch between sample-size-weighted FedAvg and a dominance-aware alternative when dominant-site shift is detected.}
\label{fig:pipeline_diagram}
\end{figure}
\FloatBarrier
"""

PROBLEM_SCHEMATIC_FIGURE = r"""\begin{figure}[!htbp]
\centering
\includegraphics[width=0.86\textwidth]{figures/dominant-site-figure-1-problem-schematic.png}
\caption{Problem schematic. FedAvg uses sample count as aggregation authority, but a high-volume site can have a training-label process that is less aligned with the declared validation objective.}
\label{fig:problem_schematic}
\end{figure}
\FloatBarrier
"""

MATH_START = r"\section{Mathematical notation}"
MATH_END = r"\section{Fixed detector transfer result}"

MATH_NEW = r"""\section{Mathematical formulation}

\subsection{Client objectives and validation alignment}

Let client $k\in\{1,\ldots,K\}$ have sample count $n_k$, local empirical objective
\begin{equation}
  \mathcal{L}_k(\theta)=\frac{1}{n_k}\sum_{i=1}^{n_k}\ell\!\left(f_\theta(x_{ki}),y_{ki}\right),
\end{equation}
and local update $\Delta\theta_k=\theta_k-\theta$. Let the clean validation objective be
\begin{equation}
  \mathcal{V}(\theta)=\mathbb{E}_{(x,y)\sim P_{\mathrm{val}}}\!\left[\ell\!\left(f_\theta(x),y\right)\right].
\end{equation}
FedAvg assigns
\begin{equation}
  w_k=\frac{n_k}{\sum_{j=1}^{K}n_j},\qquad \theta^{+}_{\mathrm{FA}}=\theta+\sum_{k=1}^{K}w_k\Delta\theta_k.
\end{equation}

Let $g_{\mathrm{val}}=\nabla_\theta\mathcal{V}(\theta)$. Define the task-specific update alignment
\begin{equation}
  A_k=-\frac{g_{\mathrm{val}}^{\top}\Delta\theta_k}{\lVert g_{\mathrm{val}}\rVert_2\lVert\Delta\theta_k\rVert_2+\varepsilon}.
\end{equation}
Positive $A_k$ means that the client update is, to first order, a descent direction for the clean validation objective. A Taylor expansion yields
\begin{align}
  \mathcal{V}(\theta^{+}_{\mathrm{FA}})-\mathcal{V}(\theta)
  &\approx g_{\mathrm{val}}^{\top}\sum_{k=1}^{K}w_k\Delta\theta_k \\
  &=-\sum_{k=1}^{K}w_k\lVert g_{\mathrm{val}}\rVert_2\lVert\Delta\theta_k\rVert_2A_k.
\end{align}
Thus a dominant client $D$ with $n_D\gg n_k$ receives large $w_D$ even when $A_D$ is reduced or negative. The quantity $A_k$ is an analytical, run-specific notion of alignment, not an intrinsic property of an institution and not a quantity directly estimated by the deployed detector.

\subsection{Dominance-aware reweighting}

Let $\widetilde{w}_k\geq0$ with $\sum_k\widetilde{w}_k=1$ denote an alternative aggregation rule,
\begin{equation}
  \theta^{+}_{\mathrm{DA}}=\theta+\sum_{k=1}^{K}\widetilde{w}_k\Delta\theta_k.
\end{equation}
Its first-order difference from FedAvg is
\begin{align}
  \mathcal{V}(\theta^{+}_{\mathrm{DA}})-\mathcal{V}(\theta^{+}_{\mathrm{FA}})
  &\approx g_{\mathrm{val}}^{\top}\sum_{k=1}^{K}(\widetilde{w}_k-w_k)\Delta\theta_k \\
  &=-\sum_{k=1}^{K}(\widetilde{w}_k-w_k)\lVert g_{\mathrm{val}}\rVert_2\lVert\Delta\theta_k\rVert_2A_k.
\end{align}
Reducing the weight of a negatively aligned dominant update can therefore improve validation risk, whereas the same reweighting can be harmful when all client updates remain aligned. This motivates conditional switching rather than unconditional replacement of FedAvg.

\subsection{Ordinal validation diagnostics}

For ordinal labels $\mathcal{Y}=\{0,\ldots,C-1\}$, define the normalized confusion matrix
\begin{equation}
  O_{ab}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[y_i=a,\widehat{y}_i=b],
\end{equation}
with marginals $p_a=\sum_bO_{ab}$, $q_b=\sum_aO_{ab}$ and independence matrix $E_{ab}=p_aq_b$. Let
\begin{equation}
  W_{ab}=\frac{(a-b)^2}{(C-1)^2}.
\end{equation}
Quadratic weighted kappa is
\begin{equation}
  \kappa=1-\frac{\sum_{a,b}W_{ab}O_{ab}}{\sum_{a,b}W_{ab}E_{ab}}.
\end{equation}
For validation site $s$, let $\kappa_s$ be site-specific QWK. The detector uses
\begin{align}
  \kappa_{\min}&=\min_s\kappa_s, \\
  S_\kappa&=\max_s\kappa_s-\min_s\kappa_s, \\
  \operatorname{MAOE}&=\frac{1}{N}\sum_{i=1}^{N}|\widehat{y}_i-y_i|, \\
  R_{\mathrm{sev}}&=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[|\widehat{y}_i-y_i|\geq3].
\end{align}
The run-level diagnostic vector is
\begin{equation}
  d(r)=\left(\kappa_{\mathrm{global}}(r),\kappa_{\min}(r),S_\kappa(r),\operatorname{MAOE}(r),R_{\mathrm{sev}}(r)\right).
\end{equation}

\subsection{Clean-calibrated detector}

Let $\mathcal{R}_0$ be the clean calibration runs. Lower-tail thresholds are
\begin{equation}
  \tau_j^{-}=Q_{q_{\mathrm{low}}}\!\left(\{d_j(r):r\in\mathcal{R}_0\}\right),
\end{equation}
and upper-tail thresholds are
\begin{equation}
  \tau_j^{+}=Q_{q_{\mathrm{high}}}\!\left(\{d_j(r):r\in\mathcal{R}_0\}\right).
\end{equation}
For the fixed transferred rule, $q_{\mathrm{low}}=0.10$ and $q_{\mathrm{high}}=0.80$. Define
\begin{align}
  I_1(r)&=\mathbf{1}[\kappa_{\mathrm{global}}(r)<\tau_{\mathrm{global}}^{-}], \\
  I_2(r)&=\mathbf{1}[\kappa_{\min}(r)<\tau_{\min}^{-}], \\
  I_3(r)&=\mathbf{1}[S_\kappa(r)>\tau_{\mathrm{spread}}^{+}], \\
  I_4(r)&=\mathbf{1}[\operatorname{MAOE}(r)>\tau_{\mathrm{MAOE}}^{+}], \\
  I_5(r)&=\mathbf{1}[R_{\mathrm{sev}}(r)>\tau_{\mathrm{sev}}^{+}].
\end{align}
The detector count and trigger are
\begin{equation}
  C(r)=\sum_{j=1}^{5}I_j(r),\qquad T(r)=\mathbf{1}[C(r)\geq3].
\end{equation}
Prediction entropy was available as a sixth diagnostic but was excluded from the fixed headline rule.

\subsection{Switching policy and expected gain}

Let $M_{\mathrm{FA}}(r)$ and $M_{\mathrm{DA}}(r)$ denote the same evaluation metric under FedAvg and the dominance-aware candidate. The detector-controlled metric is
\begin{equation}
  M_{\pi}(r)=[1-T(r)]M_{\mathrm{FA}}(r)+T(r)M_{\mathrm{DA}}(r),
\end{equation}
so the exact per-run gain over always using FedAvg is
\begin{equation}
  \Delta M(r)=T(r)\left[M_{\mathrm{DA}}(r)-M_{\mathrm{FA}}(r)\right].
\end{equation}
Let $H$ denote harmful sample-volume dominance. Then
\begin{align}
  \mathbb{E}[\Delta M]
  &=\Pr(H)\Pr(T=1\mid H)\mathbb{E}[M_{\mathrm{DA}}-M_{\mathrm{FA}}\mid H,T=1] \\
  &\quad+\Pr(H^c)\Pr(T=1\mid H^c)\mathbb{E}[M_{\mathrm{DA}}-M_{\mathrm{FA}}\mid H^c,T=1].
\end{align}
The first term is benefit from correct switching; the second is generally the cost of false-positive switching. Across $R$ paired seeds,
\begin{equation}
  \overline{\Delta M}=\frac{1}{R}\sum_{r=1}^{R}\Delta M(r),
\end{equation}
with paired confidence interval
\begin{equation}
  \overline{\Delta M}\pm t_{0.975,R-1}\frac{s_{\Delta M}}{\sqrt{R}}.
\end{equation}
This separates sample-volume influence $w_k$, latent validation alignment $A_k$, and observable detector evidence $d(r)$. The experiments do not claim to identify $A_k$ directly; they test whether clean-calibrated ordinal diagnostics provide enough evidence to conditionally reduce unsafe sample-volume dominance.

"""


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required arXiv source asset not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def replace_section(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    """Replace a complete LaTeX section between two exact section markers."""
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"Could not replace section between {start_marker!r} and {end_marker!r}"
        )
    return text[:start] + replacement + text[end:]


def apply_classic_paper_format(path: Path) -> None:
    """Apply compact single-column formatting to the build copy of main.tex."""
    text = path.read_text(encoding="utf-8")

    text = text.replace(r"\documentclass[11pt]{article}", r"\documentclass[10pt]{article}")
    text = text.replace(r"\documentclass[10pt,twocolumn]{article}", r"\documentclass[10pt]{article}")
    text = text.replace(r"\usepackage[margin=1in]{geometry}", r"\usepackage[margin=0.82in]{geometry}")
    text = text.replace(r"\usepackage[margin=0.75in]{geometry}", r"\usepackage[margin=0.82in]{geometry}")

    if r"\usepackage{times}" not in text:
        text = text.replace(r"\usepackage{microtype}", r"\usepackage{microtype}" + "\n" + r"\usepackage{times}")

    if r"\PassOptionsToPackage{hyphens}{url}" not in text:
        text = text.replace(r"\usepackage{hyperref}", r"\PassOptionsToPackage{hyphens}{url}" + "\n" + r"\usepackage{hyperref}")

    if r"\usepackage{tikz}" not in text:
        tikz = "\n".join([
            r"\usepackage{tikz}",
            r"\usetikzlibrary{arrows.meta,calc,fit,positioning}",
        ])
        text = text.replace(r"\usepackage{xcolor}", r"\usepackage{xcolor}" + "\n" + tikz)

    if r"\setlength{\parskip}" not in text:
        insert = "\n".join([
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

    text = text.replace(r"\begin{table*}[t]", r"\begin{table}[t]")
    text = text.replace(r"\end{table*}", r"\end{table}")
    text = text.replace(r"\begin{figure*}[t]", r"\begin{figure}[t]")
    text = text.replace(r"\end{figure*}", r"\end{figure}")
    text = text.replace(r"width=0.92\textwidth", r"width=0.88\textwidth")
    text = text.replace(r"width=0.95\textwidth", r"width=0.88\textwidth")

    text = text.replace(PROBLEM_SCHEMATIC_FIGURE, PIPELINE_FIGURE)
    text = text.replace(REPRO_OLD, REPRO_NEW)
    text = replace_section(text, MATH_START, MATH_END, MATH_NEW)

    thesis = (
        "\\paragraph{Working thesis.}\n"
        "In federated computational pathology, raw sample count is not the same as task-specific site-signal alignment. FedAvg can become less safe when the largest simulated pathology client has a training-label process that is misaligned with the validation objective, and dominance-aware aggregation or switching can reduce that risk under controlled stress.\n"
    )
    figure1 = thesis + "\n" + PIPELINE_FIGURE
    if "federated_pathology_pipeline_diagram.tex" not in text and thesis in text:
        text = text.replace(thesis, figure1)

    calib = (
        "This produced 36 detector configurations. A configuration was counted as robust-positive if it preserved clean trigger rate at or below 20\\% and positive global-QWK, macro-F1, and worst-site-QWK deltas at both 35\\% and 45\\% conservative shift. In total, 29 of 36 configurations were robust positive.\n"
    )
    figure4 = (
        calib + "\n"
        "\\begin{figure}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=0.88\\textwidth]{figures/dominant-site-figure-4-detector-ablation.png}\n"
        "\\caption{Detector interpretability, ablation, and calibration robustness. The transfer result is not a one-diagnostic or one-threshold artifact in the conservative threshold-shift setting.}\n"
        "\\label{fig:detector_ablation}\n"
        "\\end{figure}\n"
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
