# CAMELYON slide-level implementation

## Data provenance

CAMELYON16 and CAMELYON17 are public pathology benchmarks derived from real human
lymph-node histopathology. Active CAMELYON17/WILDS experiments in this repository
use real pathology images or features extracted from those images.

Generated bags used by isolated tests are software fixtures only. They are not the
research dataset and are not the basis of reported CAMELYON results.

## Implemented slide-level path

The CAMELYON slide-level path includes:

- complete slide feature bags;
- variable-length collation with padding and masking;
- slide and patient metadata preservation;
- mean and max aggregation support;
- masked aggregation that excludes padded patches;
- slide-level training and evaluation consistency;
- checkpoint, metric, and artifact export;
- validation of missing files and malformed feature caches.

Core files:

```text
src/data/camelyon_dataset.py
experiments/train_camelyon.py
experiments/evaluate_camelyon.py
experiments/configs/camelyon.yaml
```

## Real CAMELYON17 multi-center work

The current CAMELYON17/WILDS evidence is documented in:

```text
docs/research/camelyon17-external-center-validation-note.md
```

That work audits 455,954 real pathology image examples across five centers and
evaluates source-domain and held-out-center behavior using frozen ResNet18 features
and CAMELYON17-trained supervised ResNet18 features.

## Test-only validation utilities

The repository retains small generated fixtures for deterministic software tests.
A validation script may create mock bags to verify collation, masking, serialization,
or tensor shapes. Such fixtures must be described as test-only and never as the data
source for CAMELYON research.

## Claim boundary

The implementation supports research on real human pathology data. It remains
research-only and is not a clinical diagnostic system.

See `docs/DATA_PROVENANCE.md` for the canonical wording.
