# Umbrella technical report package

This folder contains the single, sectioned technical report for the computational pathology research platform.

## Paper

```text
Computational Pathology Research Platform for Federated Oncology:
PathoAlign Representation Alignment, TransnnMIL Whole-Slide Modeling,
and PathologyFL Federated Training
```

Canonical public PDF:

```text
https://matthewvaishnav.github.io/computational-pathology-research/computational-pathology-research-platform-for-federated-oncology.pdf
```

The repository-root Pages URL redirects to this canonical document. Previously shared PathoAlign and *When More Data Is Less Trustworthy* PDF URLs remain compatibility aliases.

The report treats the three systems as separate components rather than aliases:

- **PathoAlign** — neural representation alignment and identifiability.
- **TransnnMIL** — custom whole-slide multiple-instance learning.
- **PathologyFL** — custom federated learning, including FedAvg baselines, FAIR-WEIGHTS-H mechanisms, contribution-aware weighting, and dominance-aware switching.

The current PDF is an umbrella technical report. Each component has its own problem statement, methods, evidence, limitations, and interface with the other components. The complete PathoAlign → TransnnMIL → PathologyFL pipeline is presented as an integration agenda until a locked end-to-end experiment is completed.

## Files

```text
paper/arxiv/main.tex
paper/arxiv/references.bib
paper/arxiv/identifiability_calculations.tex
paper/arxiv/identifiability_calculations_part*.tex
paper/arxiv/build_arxiv_package.py
```

The build script creates the local submission folder:

```text
paper/arxiv/build/
```

## Build

From the repository root:

```bash
python paper/arxiv/build_arxiv_package.py
cd paper/arxiv/build
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Equivalent manual build:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Expected output:

```text
paper/arxiv/build/main.pdf
```

## Before publishing or submitting

Check all of the following:

1. The PDF builds cleanly from source.
2. No private data, logs, credentials, local paths, or hidden drafts are included.
3. References are real and appropriate.
4. Tables and claims match committed result artifacts.
5. Component boundaries remain explicit.
6. Privacy-aware architecture is not described as a formal privacy guarantee without accounting and attack evaluation.
7. The integrated stack is not described as validated before a locked end-to-end experiment exists.
8. The author metadata and date are correct.
9. The arXiv category choice is reviewed.

Likely categories include `cs.CV`, `cs.LG`, `stat.ML`, and `q-bio.QM`. The strongest fit depends on whether the report is framed primarily around computational pathology vision, representation learning, or federated-learning methodology.
