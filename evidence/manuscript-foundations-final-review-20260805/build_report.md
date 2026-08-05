# Final Manuscript Review — Build Report

**Date:** 2026-08-05
**Branch:** `research/foundations-manuscript-final-review-20260805`
**Source commit:** `793bb6d3f3de7d7a99ff56582dee82423654c1df`

## Clean-checkout build (isolated worktree)

A fresh git worktree was created at the committed source and built without any
pre-existing aux files.

- **OS:** Windows 11
- **LaTeX distribution:** MiKTeX
- **Engine:** pdflatex (latexmk 4.88)
- **Commands:** `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  and `latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex`
- **Build duration:** 13 s
- **Warnings:** 0 overfull hbox, 0 undefined references, 0 missing figures

## Outputs (committed package PDFs)

| Artifact | Pages | SHA-256 |
| --- | --- | --- |
| `main.pdf` | 12 | `8d01305b2047c567803c5c50aaa08c575f09aaf622533e789695d3328d84b40f` |
| `supplement.pdf` | 3 | `d8525a9ce49fa38145cf5e211d34f75d42290789ff239f9b9fa42b6d9794f672` |

## Rebuilt-PDF hashes (fresh worktree build)

| Artifact | SHA-256 |
| --- | --- |
| `main.pdf` | `265cfff0ca74582255f21cc9bb636fcf319dee152121375750330c92931deaa0` |
| `supplement.pdf` | `31ef3f461964f2a4b999e6fcccd977e16f93e57ecfeb31138eba8572969844e0` |

The rebuilt-PDF hashes differ from the committed PDFs only in embedded LaTeX
metadata (creation timestamps/IDs); page counts and content are identical. The
committed PDFs are the canonical artifacts.

## Requirements

- Zero build errors: PASS
- Zero unresolved citations: PASS
- Zero undefined references: PASS
- Zero missing figures: PASS
