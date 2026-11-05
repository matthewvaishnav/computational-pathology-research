# Repaired TransnnMIL matched PANDA rerun — 2026-09-23

Status: **protocol frozen; full real-data execution not yet completed**.

This experiment repairs the missing evidence path identified by the repository claim audit. Historical pre-remediation TransnnMIL QWK values remain historical execution records and are not relabelled as evidence for the repaired canonical fusion.

## Frozen question

Under one shared PANDA train/selection/confirmation split and one shared optimization budget, how does the repaired canonical TransnnMIL compare with standalone and simple-fusion controls?

The complete matrix is:

- models: AttentionMIL, nnMIL, TransMIL, repaired TransnnMIL, concat fusion, gate fusion, learned branch-attention fusion;
- seeds: 7, 19, 42, 123, 2025;
- total required full cells: **35**;
- train / selection / confirmation: 70 / 15 / 15;
- confirmation is not evaluated until selection-only checkpoint choice has completed;
- primary endpoint: confirmation QWK;
- paired hierarchical bootstrap: prespecified seeds, then identical confirmation slides within seed;
- primary architecture-success gate: repaired TransnnMIL must beat nnMIL, TransMIL, concat, and gate on mean confirmation QWK, be positive in at least 4/5 paired seeds, have a positive 95% paired-bootstrap lower bound for every required contrast, and show no practical one-branch collapse.

The exact protocol is machine-readable in:

`experiments/transnnmil/transnnmil_matched_panda_rerun_spec_20260923.json`

The protocol is bound to the canonical PANDA manifest Git blob:

`c928627705929ef7981934c05b9f294572671aad`

and requires 10,611 valid 768-dimensional Phikon feature bags.

## Execution

From the repository root on a machine containing the historical feature directory `D:\panda\features_phikon`:

```powershell
git switch research/transnnmil-matched-rerun-20260923
python scripts/experiments/run_panda_transnnmil_matched_matrix.py --device cuda --verify-all-hdf5
```

If the HDF5 files live elsewhere, rebase only their filesystem location before the locked split is created:

```powershell
python scripts/experiments/run_panda_transnnmil_matched_matrix.py --device cuda --verify-all-hdf5 --feature-root "E:\path\to\features_phikon"
```

The matrix launcher is resumable. It skips a cell only when its result is marked `full_evidence_candidate` and its model, seed, frozen-spec hash, and locked-split hash all match.

To check data availability without training:

```powershell
python scripts/experiments/run_panda_transnnmil_matched_matrix.py --preflight-only --verify-all-hdf5
```

## Interpretation boundary

A completed matched rerun can establish the actual performance of the repaired architecture on the frozen internal PANDA development protocol. A stronger architecture-superiority statement is promoted only if the preregistered comparison gates pass. A failed gate is retained; confirmation outcomes must not be used to retune the protocol.

This experiment is not blinded external validation and is not clinical evidence.
