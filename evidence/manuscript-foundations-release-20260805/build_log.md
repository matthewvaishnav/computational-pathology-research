# Build Log

**Date:** 2026-08-05
**Branch:** `research/foundations-manuscript-release-hardening-20260805`
**Package:** `manuscripts/computational-pathology-foundations-v1/`

## Command

```
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

## Environment

- MiKTeX `latexmk` 4.88; `pdflatex` engine.
- Windows 11.
- Build runs from the package sources in a clean checkout; no external data
  required.

## Output

| Artifact | Pages | SHA-256 |
| --- | --- | --- |
| `main.pdf` | 12 | `66c36796c794dba7eb13d102e3bc5c43b3ff02b6be5671f65f2994e1f6d8e1d1` |
| `supplement.pdf` | 3 | `bc1328f04364813d7d4159554a31f128fc307f1e2a7268bc64489baa6c5a3d8e` |

## Log summary

- Unresolved citations: 0
- Undefined references: 0
- Missing figures: 0
- Overfull hbox warnings: cosmetic only (no errors)

## Status

`full_foundations_manuscript_release_candidate` after final review. Every active numerical claim is
bound to a real verified artifact hash; no placeholder hashes remain; all cited
files exist; all references verified; PathologyFL/FAIR-WEIGHTS-H status matches
executable reality; PDFs build cleanly; all validation tests pass; the public
site is not modified.

The PDFs are marked internal review and are not published.
