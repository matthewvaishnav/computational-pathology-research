# Paired-acquisition dimensionality × cross-covariance evidence

This separately versioned package promotes the validated summaries of the locked
450-cell factorial. It binds the reviewed Gate 2 execution source, all registered
cell identities and hashes, frozen inputs and configuration, the preregistered
fold-aware analysis, and a conservative claim-boundary snapshot.

The package intentionally excludes checkpoints, feature archives, projections,
raw per-slide analysis rows, and slide-level contrasts. Those local source
artifacts remain bound by hash and are not committed.

Validate with:

```powershell
python scripts\provenance\validate_paired_acquisition_factorial_evidence.py `
  evidence\paired_acquisition\dimensionality-xcov-factorial-20260726
```
