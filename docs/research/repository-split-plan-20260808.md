# Repository Split Plan — 2026-08-08

## Goal

Keep `computational-pathology-research` as the program-level manuscript, evidence,
claim-boundary, and provenance hub while extracting major authored methods into
standalone repositories with preserved Git history.

## Target topology

1. `computational-pathology-research`
   - foundations manuscript and supplement;
   - program-level claim ledger and evidence summaries;
   - cross-project provenance/audit machinery;
   - links to standalone method/study repositories.

2. `paired-acquisition-neural-factorization`
   - reusable PA-NF model/core and generic paired-acquisition objectives;
   - shared tests and reusable evaluation utilities;
   - links to the existing SCORPION, canine SCC, and allocation study repositories.

3. `transnnmil`
   - `src/models/transnnmil/`;
   - required `src/models/mil/` dependencies after import audit;
   - TransnnMIL-specific tests, PANDA trainers, stability/ablation scripts, and
     architecture documentation.

4. `pathologyfl`
   - `src/features/federated/pathology_fl/`;
   - federated integration/smoke tests;
   - PathologyFL experiment/configuration documentation;
   - FAIR-WEIGHTS-H remains inside initially so extraction does not break imports.

5. `fair-weights-h` (optional second-stage split)
   - protocol implementation, PathoAlign bridge, tests, and specification docs;
   - only after PathologyFL extraction is stable and the dependency surface is explicit.

6. WSI-NCA / Factorized Tissue Dynamics
   - remain on the experimental branch until a real-pathology Phase A result exists;
   - do not create a public standalone method repository solely from synthetic evidence.

## Existing PA-NF study repositories

These are already separate and should remain study-specific satellites rather than
being folded back into the hub:

- `matthewvaishnav/paired-acquisition-factorization-scorpion`
- `matthewvaishnav/paired-acquisition-factorization-caninescc`
- `matthewvaishnav/paired-acquisition-factorization-allocation`

## Why filtered history instead of copy/paste

A copy into a fresh repository would make the extracted method look as if it was
created on the extraction date and would sever the implementation/repair history.
The standalone repositories should therefore be generated with `git filter-repo`
from a full clone of this repository.

## Extraction procedure

Install once:

```bash
python -m pip install git-filter-repo
```

Always operate on a disposable fresh clone. The commands below rewrite the clone
only; they must never be run in the working copy used for ordinary research.

### TransnnMIL

Start conservatively with the complete model family and the known dedicated
training/docs surfaces:

```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git transnnmil-export
cd transnnmil-export

git filter-repo \
  --path src/models/transnnmil/ \
  --path src/models/mil/ \
  --path scripts/training/train_panda_transnnmil_baseline.py \
  --path scripts/experiments/aggregate_transnnmil_stability.py \
  --path docs/TRANSNNMIL_IMPLEMENTATION.md \
  --path docs/TRANSNNMIL_V2_API.md \
  --path docs/TRANSNNMIL_V2_ARCHITECTURE.md \
  --path docs/TRANSNNMIL_V2_TRAINING.md \
  --path docs/models/transnnmil-v2.md \
  --path docs/results/panda-transnnmil-stability.md \
  --path docs/results/panda-transnnmil-ablation-plan.md \
  --path tests/models/
```

Before the first standalone release, run an import audit and add any shared data
loader/config paths that are genuinely required. Do not drag unrelated PA-NF or
federated code into the repository merely to satisfy historical imports.

### PathologyFL

```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git pathologyfl-export
cd pathologyfl-export

git filter-repo \
  --path src/features/federated/pathology_fl/ \
  --path tests/federated/ \
  --path experiments/FEDERATED_ABLATION_PROTOCOL.md \
  --path docs/results/panda-centralized-vs-federated.md
```

Then perform an import audit for shared federated utilities outside
`pathology_fl/`. FAIR-WEIGHTS-H stays with PathologyFL during this first extraction.

### Reusable PA-NF core

The existing study repositories already cover three concrete study surfaces. A
new reusable PA-NF core repository should be extracted only from paths that are
method-generic (factorizer, paired objectives, shared tests/evaluation) rather
than by duplicating full SCORPION/canine study histories. The exact path list must
be frozen after an import graph audit because the current paired-acquisition code
is distributed across `src/`, `experiments/paired_acquisition/`, provenance, and
shared evaluators.

## First-commit requirements in every extracted repository

Each standalone repository should immediately add a new top-level README that
states:

- method identity and architecture;
- exact evidence status;
- dataset-specific claims versus general method claims;
- provenance back-link to this program hub;
- research-only / not-clinical-software boundary;
- historical-result withdrawals relevant to that method;
- reproducibility instructions.

## Monorepo removal policy

Do **not** delete implementation paths from this hub the moment a new repository
is created. First:

1. extract history;
2. make the standalone tests pass;
3. tag the first standalone release;
4. update all program/manuscript links;
5. leave a migration stub in the old path for one release cycle;
6. then remove duplicated implementation from the hub.

This prevents broken citations, dead manuscript paths, and accidental loss of
reproducibility while the split is in progress.

## Claim-language rule after the split

Claims remain endpoint- and comparator-specific. In particular:

- PA-NF **does** have a supported controlled comparative advantage over the
  equal-capacity two-branch neural control on the registered SCORPION structured-
  separation objective;
- the corrected canine fixed-estimand audit **does not** establish an additional
  neural feature-space increment over every strong simple scanner-removal baseline;
- neither statement implies universal superiority over all harmonization methods.
