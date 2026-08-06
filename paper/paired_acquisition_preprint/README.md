# Paired-Acquisition Neural Factorization preprint

This directory contains the corrected, focused PA-NF manuscript. The canonical repository source is split into three readable files under `source_parts/`; the publication workflow concatenates them into `main.tex` without transformation. It supersedes the earlier focused PDF and does not restore or retroactively validate claims withdrawn during the 2026 scientific audit.

## Current public paper

- PDF: https://matthewvaishnav.github.io/computational-pathology-research/paired-acquisition-neural-factorization.pdf
- arXiv source package: https://matthewvaishnav.github.io/computational-pathology-research/paired-acquisition-neural-factorization-arxiv-source.zip
- Repository claim boundary: ../../CLAIM_BOUNDARY.md

## Build

```bash
cat source_parts/part*.tex > main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected output: `main.pdf`.

The GitHub Pages workflow builds this source in a clean Ubuntu environment and publishes the resulting PDF directly. The root Pages URL redirects to the PDF rather than to a manuscript-status landing page.

## Scientific scope

The manuscript reports corrected representation-level evidence from SCORPION and the independent multi-scanner canine SCC audit. It supports partial structured separation under the tested paired-acquisition protocols. It does not claim pure biological factors, complete scanner invariance, clinical utility, diagnostic improvement, or superiority over strong simple scanner-removal baselines.

## External submission files

- `arxiv_metadata.md`: title, abstract, categories, and comments for an authenticated arXiv submission.
- `zenodo_metadata.json`: draft metadata for an authenticated Zenodo deposit or GitHub-release archive.

Creating an arXiv identifier or Zenodo DOI still requires the author's authenticated account and final confirmation of submission metadata.
