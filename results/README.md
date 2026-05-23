# Results directory policy

This directory contains **small, curated, reproducible result artifacts**:

Tracked examples:
- benchmark summaries
- metrics JSON files
- markdown reports
- confidence intervals
- comparison tables
- validation summaries

Not tracked:
- extracted features
- WSI tiles
- raw predictions
- checkpoints
- embeddings
- model weights
- temporary experiment outputs
- large tensors

Reason:

The repository is intended to preserve **methods, validation summaries, and reproducibility**, not serve as a storage backend for datasets or generated feature stores.

For PANDA, PCam, Camelyon, or future WSI datasets:

```text
raw data → local only
feature extraction → local only
checkpoints → local only
summary metrics/reports → tracked
```

Before committing:

```bash
git status
```

Verify no `.pt`, `.pth`, `.h5`, `.npy`, checkpoints, or extracted feature directories are staged.
