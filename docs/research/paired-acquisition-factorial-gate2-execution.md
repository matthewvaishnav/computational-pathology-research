# Paired-acquisition Gate 2 execution

## Authorization

Gate 2 is authorized only by a provenance-valid Gate 1 release whose frozen plan matches the committed factorial design. The first real authorization release is:

```text
parelease-v1-49d5fce02b8c033e2b100ae5db522b240d8172b22024ac25e9305fbdfa58fba5
```

This authorization permits execution of the locked full grid. It is not a positive scientific result.

## Locked workload

Gate 2 executes all combinations of:

- acquisition dimensions `[2, 4, 8, 16, 32, 64]`;
- cross-covariance weights `[0.0, 0.05, 0.20]`;
- folds `[0, 1, 2, 3, 4]`;
- seeds `[911, 912, 913, 914, 915]`;
- `75` epochs per run.

The total is `450` provenance-bound runs. Scientific parameters cannot change during resume.

## Resumable execution

The runner stores every validated cell under a persistent work directory. It records an immutable execution-state document, per-attempt logs, failures, completed run IDs, and record checksums. A resumed invocation validates every existing cell before skipping it. A changed commit, feature archive, split manifest, device, frozen plan, or Gate 1 authorization fails closed.

Local factorial artifacts are ignored by Git through `results/paired_acquisition_factorial/.gitignore` so the producer can continue to enforce a clean checkout.

Run a bounded first batch:

```powershell
python experiments/paired_acquisition/run_provenance_bound_factorial_full.py `
  --work-dir results/paired_acquisition_factorial/full-gate-v1-work `
  --release-dir results/paired_acquisition_factorial/full-gate-v1 `
  --smoke-manifest results/paired_acquisition_factorial/smoke-gate-v1/release_manifest.json `
  --device cuda `
  --max-new-runs 5
```

Rerun the identical command to resume. The operational `--max-new-runs` limit may be changed or omitted because it does not alter any scientific configuration. Omit it to continue until all remaining cells finish.

The final aggregate release is exposed only after all 450 cells validate. Source cell files are hard-linked into the aggregate release rather than duplicated, so the work and release directories must remain on the same hard-link-capable filesystem.

## Independent validation

After the aggregate release is published:

```powershell
python scripts/provenance/validate_paired_acquisition_factorial_full_release.py `
  results/paired_acquisition_factorial/full-gate-v1/release_manifest.json
```

A valid execution release contains the complete cell-level table, the frozen plan, the Gate 1 authorization binding, all 450 run records, and the full-gate checksum bindings.

## Claim boundary

Completing Gate 2 establishes a complete execution record. It does not by itself establish capacity, regularization, or interaction effects. Those claims require the preregistered aggregate analysis, uncertainty blocked at the biological-sample level, and review of every retained failure. The 350 unresolved historical artifacts remain excluded.
