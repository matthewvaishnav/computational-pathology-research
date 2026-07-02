# Paired-Acquisition Neural Factorization Publication Readiness

Audit date: 2026-07-02

Scope: final publication-readiness verification for the main repository and the three public child repositories:

- `paired-acquisition-factorization-caninescc`
- `paired-acquisition-factorization-allocation`
- `paired-acquisition-factorization-scorpion`

## Done

- Main repository public index files have current child-package links:
  - `README.md`
  - `docs/studies/index.md`
  - `paper/arxiv/main.tex`
  - `paper/arxiv/study_specific_packages.tex`
- Canine SCC, allocation, and SCORPION child repositories exist, have public PDFs, and have claim boundaries.
- The main arXiv source package exists at `paper/arxiv/paired_acquisition_neural_factorization_arxiv_source.zip`.
- The main arXiv PDF builds locally and the live GitHub Pages PDF returns `200 application/pdf`.
- The main repository has `CITATION.cff`, `LICENSE`, `CHANGELOG.md`, a clean top-level README, the arXiv source zip, and a manuscript PDF build path.
- Public/semi-public old-name prose was cleaned from outreach docs, benchmark summaries, selected research notes, the Figure 1 table snippet, and the canine SCC scaffold script.
- SCORPION pair-integrity falsification completed across 5 seeds, 5 folds, and 3 pair-construction conditions. Broken-pair controls reduced scanner-probe accuracy similarly or more than true pairs, but degraded paired-tissue consistency and same-region retrieval, strengthening peer-review readiness without expanding clinical claims.

## Missing Blockers

1. Allocation child repo still appears DOI-incomplete compared with canine SCC and SCORPION because the inspected local checkout does not contain root `CITATION.cff` or `LICENSE`.
2. Zenodo archive scope is not finalized. The release should explicitly enumerate which evidence tables, PDFs, source zips, and child-package artifacts are included.

## Optional Improvements

- Update stale wording in `paper/arxiv/study_specific_packages.tex` that still describes child-package URLs as pending or repository-target slugs.
- Add `CITATION.cff` and `LICENSE` to the allocation child repository if it will receive a standalone DOI.
- Repeat the child-repo DOI metadata check for canine SCC and SCORPION if those child packages will receive standalone DOI releases.
- Curate a short evidence manifest for the main DOI release so reviewers can map claims to exact result tables and PDFs.
- Consider a future documentation-file rename pass for old lower-case historical slugs in `docs/`, but do that separately because it can create link churn.

## Preprint-Ready Checklist

- [x] Main paper PDF builds.
- [x] Main arXiv source zip exists.
- [x] Main public index points to canine SCC, allocation, and SCORPION repositories and PDFs.
- [x] Canine SCC public PDF URL works.
- [x] Allocation public PDF URL works.
- [x] SCORPION public PDF URL works.
- [x] Child packages include explicit claim boundaries.
- [x] Public/semi-public old-name prose has been cleaned from the main preprint surface.
- [x] Strict preprint surface check has zero tracked hits in `README.md`, `docs/studies`, `paper/arxiv`, and `paper/figures`.

## Zenodo-Ready Checklist

- [x] Main repository has `CITATION.cff`.
- [x] Main repository has `LICENSE`.
- [x] Main repository has release notes in `CHANGELOG.md`.
- [x] Main manuscript PDF and arXiv source zip exist.
- [x] Current public URL docs no longer point at old public PDF aliases.
- [ ] Finalize DOI release title, version tag, creators, and related identifiers.
- [ ] Curate exact evidence artifacts to include in the archive.
- [ ] Confirm allocation child package metadata if it will be cited or archived independently.
- [ ] Confirm canine SCC and SCORPION metadata if they will receive standalone DOI releases.

## Peer-Review-Ready Checklist

- [x] Claim boundaries are explicit in the main paper and child packages.
- [x] External canine SCC validation package is live.
- [x] Allocation study package is live.
- [x] SCORPION core package is live.
- [x] Public-facing naming cleanup is acceptable for arXiv/preprint.
- [x] Pair-integrity falsification completed and documented as a falsification control.
- [x] Pair-integrity falsification integrated into the main manuscript and evidence documentation.

## Naming-scope cleanup result

- PathoAlign hits before cleanup: 204 tracked hits from `git grep -n -i "pathoalign" -- README.md docs paper scripts/release`.
- PathoAlign hits after cleanup: 66 tracked hits from the same command.
- Strict public/preprint surface after cleanup: 0 tracked hits in `README.md`, `docs/studies`, `paper/arxiv`, and `paper/figures`.
- Remaining docs/paper/README hits: 42. These are lower-case legacy run directories, script names, file names, or run IDs such as `pathoalign_dep20`; they are retained for reproducibility and are not presented as public branding.
- Remaining `scripts/release` hits: 24. These are source-path references or destination-cleanup replacement rules in scaffold scripts; the canine SCC scaffold now rewrites generated child-repo filenames/content to paired-acquisition names, matching the SCORPION scaffold pattern.
- Public-facing blockers remaining: none found in the preprint/public-index surface.
- Naming-scope cleanup remains a blocker: no, not for arXiv/preprint submission.

Files changed in this cleanup pass:

- Public/semi-public docs: `docs/outreach/outreach-list.md`, `docs/outreach/research-package-index.md`, `docs/repository-cleanup/VALIDATION.md`, `docs/repository-cleanup/INVENTORY.md`, `docs/REPO_ORGANIZATION.md`.
- Benchmark and research notes: `docs/benchmarks/*` files with old method prose; `docs/research/pathoalign-external-multiscanner-caninescc-protocol.md`; `docs/research/scorpion-pathoalign-crossbackbone-protocol.md`; `docs/research/scorpion-pathoalign-crossbackbone-results.md`; `docs/research/scorpion-pathoalign-plan.md`.
- Paper/release artifacts: `paper/figures/pathoalign_figure1_benchmark_table.tex`; `scripts/release/build_paired_acquisition_factorization_caninescc_repo.ps1`.
- Control document: `docs/research/paired-acquisition-factorization-publication-readiness.md`.
- Pair-integrity integration: `docs/research/paired-acquisition-factorization-pair-integrity-falsification.md`, `experiments/scorpion/run_pair_integrity_falsification.py`, `paper/arxiv/main.tex`, and `paper/arxiv/study_specific_packages.tex`.

## Next 5 Actions In Exact Order

1. Review and commit the naming-scope cleanup and pair-integrity integration after confirming the diff is acceptable.
2. Add root `CITATION.cff` and `LICENSE` to `paired-acquisition-factorization-allocation` if that child package will be archived or cited independently.
3. Create a concise evidence manifest for the main DOI release.
4. Decide the exact evidence artifacts to archive, including the SCORPION pair-integrity result tables if they are in scope.
5. Create the final release tag and Zenodo draft after the evidence manifest and child-repo metadata decision are complete.

## Bottom Line

SCORPION is no longer the main remaining blocker: the repository exists, the PDF exists, GitHub Pages serves the PDF directly, and claim boundaries are present.

Pair-integrity falsification is now completed and integrated. It strengthens peer-review readiness by showing that broken-pair controls can suppress scanner signal while damaging tissue preservation, but it does not expand the clinical or deployment claim boundary.
