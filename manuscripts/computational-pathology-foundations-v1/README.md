# Paired-Acquisition Neural Factorization: An End-to-End Computational Pathology Pipeline

**Canonical PA-NF preprint package — 2026-09-04.**

This directory contains the single public PA-NF manuscript for the full computational pathology pipeline. The paper follows the system from paired scanner-aware representation learning through whole-slide neural aggregation and into multi-institutional learning. Corrections and evidence updates are incorporated directly into this manuscript rather than published as a separate "corrected" paper.

## Scientific structure

1. **Paired-acquisition representation learning** — SCORPION and independent multi-scanner canine SCC; tissue/acquisition factorization; strong simple scanner-removal controls; cross-backbone transfer.
2. **Whole-slide modeling** — TransnnMIL and the 10,611-slide PANDA Phikon feature pipeline; optimization-stability studies; spatial/tissue-dynamics extensions.
3. **Multi-institutional learning** — PathologyFL, FAIR-WEIGHTS-H, dominance-aware aggregation, PANDA simulated-site stress, and detector transfer to ordinal threshold shift.
4. **Natural center shift** — CAMELYON17/WILDS source-weighting and center-subspace studies over 455,954 examples from five centers.
5. **Patch and mechanism foundations** — PCam patch evaluation and controlled synthetic identifiability/resource-allocation experiments.

## Main files

- `main.tex` — canonical manuscript.
- `supplement.tex` — implementation, testing, result-pointer, and reproducibility supplement.
- `references.bib` — bibliography.
- `sections/` — scientific manuscript sections.
- `figures/` — figures and equations used by the manuscript.
- `tables/` — supporting tables.
- `evidence/`, `claims/`, and `validation/` — supporting reproducibility and historical research records; these remain available without defining the narrative of the main paper.

## Build

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error supplement.tex
```

The GitHub publication workflow builds this manuscript and supplement, creates the source archive, and publishes `paired-acquisition-neural-factorization-pipeline.pdf` as the sole public main-paper PDF.
