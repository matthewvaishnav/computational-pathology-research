# Paired-Acquisition Neural Factorization — Final Package Audit

Audit date: 2026-07-03

## Current Thesis

Paired-Acquisition Neural Factorization reduces linearly recoverable scanner/acquisition
signal while preserving tissue identity. This tradeoff is not explained by broken-pair
training, DINOv2-specific behavior, simple linear scanner projection, or PCA component
removal.

## Evidence Stack

| # | Evidence Line | Dataset(s) | Backbone(s) | Status |
|---|---|---|---|---|
| 1 | SCORPION core result | SCORPION | DINOv2 | Committed |
| 2 | External canine SCC validation | Canine SCC | DINOv2 | Committed |
| 3 | Pair-repeat allocation | Synthetic | — | Committed |
| 4 | SCORPION pair-integrity falsification | SCORPION | DINOv2 | Committed |
| 5 | Canine SCC pair-integrity falsification | Canine SCC | DINOv2 | Committed |
| 6 | Cross-backbone pair-integrity | SCORPION | Phikon, ResNet50 | Committed |
| 7 | SCORPION baseline murder test | SCORPION | DINOv2 | Committed |
| 8 | Canine SCC baseline murder test | Canine SCC | DINOv2 | Committed |

## What Is Complete

### Main repository
- [x] README.md with correct public naming, research program table, and child repo links
- [x] docs/studies/index.md with verified child repo links
- [x] arXiv manuscript with all evidence lines integrated
- [x] arXiv PDF builds successfully (14 pages, 400 KB)
- [x] arXiv source zip exists (367 KB)
- [x] CITATION.cff present
- [x] LICENSE present
- [x] CHANGELOG.md present
- [x] CLAIM_BOUNDARY.md present
- [x] Explicit limitations section in manuscript
- [x] All 6 result artifact directories populated with verified outputs

### Child repositories
- [x] SCORPION: clean, v0.1.0 published, has CITATION.cff + LICENSE
- [x] Canine SCC: clean, v0.1.0 published, has CITATION.cff + LICENSE
- [x] Allocation: clean, v0.1.0 published, has CITATION.cff + LICENSE
- [x] All child PDFs accessible via GitHub Pages

### Public naming
- [x] Zero "PathoAlign" hits in README.md, docs/studies, paper/arxiv, paper/figures
- [x] One minor fix applied during this audit (baseline research note table)

### Result artifacts (all verified present)
- [x] results/paired_acquisition_factorization_pair_integrity_scorpion/
- [x] results/paired_acquisition_factorization_pair_integrity_caninescc/
- [x] results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/
- [x] results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/
- [x] results/paired_acquisition_factorization_baseline_murder_test/
- [x] results/paired_acquisition_factorization_baseline_murder_test_caninescc/

### Claim boundaries
- [x] Manuscript explicitly states limitations (lines 340-341):
  - "Tissue retrieval is not a clinical endpoint"
  - "No experiment establishes treatment benefit, prospective workflow safety, regulatory readiness, or universal biological--acquisition identifiability"
  - "Companion PDFs provide exact study-specific boundaries"
- [x] No clinical/diagnostic/deployment overclaiming detected
- [x] CLAIM_BOUNDARY.md exists at repo root

## What Still Requires Action (Pre-Zenodo/DOI)

1. **Zenodo archive scope**: Decide exact evidence artifacts to include (manuscript PDF, arXiv source zip, result CSVs, child repo PDFs).
2. **DOI metadata finalization**: Title, version tag, creators, related identifiers.
3. **Release tag**: Create a final release tag (e.g., `v0.2.0` or `v1.0.0-preprint`) after this audit commit.
4. **Allocation child repo**: Has CITATION.cff and LICENSE; confirm if it will receive a standalone DOI.
5. **Zenodo draft**: Create the Zenodo draft and link it to the GitHub release.

## Issues Found During Audit

### 1. `paper/arxiv/build/main.tex` is tracked (minor)
The build script regenerates this file from the source `main.tex`, but it's tracked in git (committed before the `build/` gitignore rule). It shows as modified after every build. Fix with:
```bash
git rm --cached paper/arxiv/build/main.tex
```
This is cosmetic — doesn't affect the source or the build.

### 2. Untracked CAMELYON17/pathoalign files (pre-existing)
Many untracked files in `scripts/camelyon17/`, `scripts/pathoalign_identifiability*/`, etc. These predate this work and are unrelated to the current package. Not a blocker.

### 3. Legacy doc filenames still contain "pathoalign"
Files like `docs/research/scorpion-pathoalign-crossbackbone-protocol.md` retain legacy names. These are internal docs, not public-facing. Renaming them would cause link churn across multiple docs. Not a blocker for arXiv/preprint.

## What Should NOT Be Touched

- Zenodo / DOI metadata (manual action needed, not automated)
- GitHub releases (manual action needed)
- CITATION.cff / LICENSE (already correct)
- Child repos (already clean with published releases)
- Unrelated CAMELYON17/pathoalign files (pre-existing, out of scope)
- Result artifacts (committed evidence, do not modify)

## Next Scientific Experiment (After Packaging)

**Acquisition branch audit.** The next reviewer attack will be:
"Are you really factorizing acquisition information into a separate branch, or just suppressing scanner signal in the biological branch?"

The paper already reports acquisition-branch scanner probe (0.86 DINOv2, 0.97 Phikon, 0.78 ResNet50) and tissue retrieval (0.10, 0.07, 0.17), but a systematic audit would:
- Test whether acquisition-branch tissue retrieval is statistically below biological-branch retrieval across all folds/seeds
- Test whether the gap widens under stronger probes (RF, kNN, MLP)
- Test whether the separation pattern holds on canine SCC
- Test whether the acquisition branch carries more acquisition signal than linear baselines

## Explicit Claim Boundary

This package does NOT claim:
- Clinical validation
- Diagnostic performance
- Disease biology discovery
- Human clinical generalization
- Complete scanner invariance
- Perfect disentanglement
- Deployment readiness
- Regulatory readiness

It DOES claim:
- Paired-acquisition supervision can factor scanner signal from tissue identity in frozen embeddings
- The tradeoff transfers across backbones without retuning
- Broken-pair controls and simple baselines cannot reproduce the tradeoff
- The result is representation-identifiability evidence, not clinical evidence

## Exact Unresolved Manual Actions

1. Create Zenodo draft with evidence artifacts
2. Finalize DOI metadata (title, creators, version)
3. Create GitHub release tag
4. Confirm allocation child repo DOI status
5. Optionally: `git rm --cached paper/arxiv/build/main.tex` to clean tracked build artifact

## Files Changed During This Audit

- `docs/research/paired-acquisition-factorization-baseline-murder-tests.md` — fixed one "PathoAlign" → "Paired-Acquisition Neural Factorization"

## Verification

- `git diff --check`: clean
- arXiv build: succeeds, 14 pages, no errors
- Public naming check: zero hits in README.md, docs/studies, paper/arxiv, paper/figures
- All child repos: clean, published releases
- All result artifacts: present and verified
