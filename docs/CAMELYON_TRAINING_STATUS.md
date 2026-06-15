# CAMELYON data and training status

## Correction

CAMELYON16 and CAMELYON17 are **real human histopathology datasets** containing digitized breast-cancer lymph-node tissue. The active CAMELYON17/WILDS experiments in this repository use real pathology images and features derived from those images.

An older version of this page incorrectly treated an early generated smoke-test fixture as if it described the project's CAMELYON research. That was false and has been removed.

## Current real-data evidence

The repository contains an audited CAMELYON17/WILDS multi-center workflow with:

- 455,954 labeled image examples;
- five acquisition centers;
- source-center training and validation;
- out-of-distribution validation on center 1;
- held-out out-of-distribution testing on center 2;
- frozen ResNet18 image-feature baselines;
- CAMELYON17-trained supervised ResNet18 features;
- repeated-seed center-weighting comparisons;
- validation-aware policy selection without using held-out test performance for policy choice.

The primary evidence note is:

```text
docs/research/camelyon17-external-center-validation-note.md
```

That note reports the dataset audit, split structure, real-image pipeline checks, repeated-seed results, claim boundaries, scripts, and result artifacts.

## Implemented CAMELYON components

- `experiments/train_camelyon.py`: slide-level training path over pathology feature bags.
- `experiments/evaluate_camelyon.py`: slide-level evaluation and metric export.
- `src/data/camelyon_dataset.py`: slide metadata, feature-bag loading, collation, and aggregation helpers.
- `scripts/camelyon17/`: real CAMELYON17/WILDS audit, feature, weighting, and detector experiments.
- `experiments/camelyon17_federated_audit.py`: multi-center audit and federated-analysis support.

## Test-only fixtures

Legacy utilities such as `scripts/generate_synthetic_camelyon.py` create generated arrays solely for isolated unit or smoke testing. They validate software plumbing; they are not CAMELYON benchmark data and are not the basis of the reported scientific results.

## Submission boundary

The Grand Challenge leaderboard requires an official patient-level prediction file and evaluates pathological nodal staging with Cohen's kappa. Preparing that submission artifact is a separate packaging and inference task. It does not change the provenance of the repository's existing CAMELYON17 research, which uses real human pathology data.

## Clinical boundary

The work is research-only. Use of real human benchmark tissue does not imply prospective clinical validation, regulatory approval, or readiness for patient care.

See also: `docs/DATA_PROVENANCE.md`.
