# arXiv preprint package

This folder contains the focused arXiv-style preprint for the dominant-site federated pathology result.

## Paper

```text
When More Data Is Less Trustworthy:
Site-Signal Alignment Failure Modes in Federated Computational Pathology
```

The arXiv paper is intentionally narrower than the full project website/report. It focuses on the dominant-site federated pathology result, with PCam, PANDA/TransnnMIL, PathologyFL, and FAIR-WEIGHTS-H included only as context.

## Files

```text
paper/arxiv/main.tex
paper/arxiv/references.bib
paper/arxiv/README.md
paper/arxiv/build_arxiv_package.py
```

The build script creates the local arXiv submission folder:

```text
paper/arxiv/build/
```

and copies generated figures into:

```text
paper/arxiv/build/figures/
```

## Build

From the repository root:

```bash
python paper/arxiv/build_arxiv_package.py
cd paper/arxiv/build
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Expected output:

```text
paper/arxiv/build/main.pdf
```

## Before submitting to arXiv

Check all of the following:

1. The PDF builds cleanly from source.
2. No private data, logs, credentials, local paths, or hidden drafts are included.
3. References are real and appropriate.
4. Figures are generated from committed scripts/results.
5. Claim boundaries remain explicit: research-only, not clinical validation, not diagnostic software, not deployment evidence.
6. The author metadata is correct.
7. The arXiv category choice is reviewed.

Likely category options:

```text
cs.CV
cs.LG
stat.ML
q-bio.QM
```

The strongest fit is likely `cs.CV` or `cs.LG`, depending on whether the submission is framed more as computational pathology vision or federated-learning methodology.
