# SCORPION capacity-matched ablation evidence

This separately versioned package promotes the validated 175-fit SCORPION
capacity-matched ablation campaign executed at `0adea50f1ef22865969109f1834a3c175e3f8b43`.

It contains small manifests, the complete append-only ledger, artifact hashes,
the preregistered fold-aware aggregate outputs, exact command records, and a
claim-boundary snapshot. It intentionally excludes checkpoints, projected
features, raw feature arrays, per-slide rows, and durable terminal logs.

Validate the package:

```powershell
python scripts/provenance/validate_scorpion_capacity_matched_evidence.py `
  evidence/paired_acquisition/scorpion-capacity-matched-20260726/release_manifest.json
```

When the local external artifacts are present, rehash them too:

```powershell
python scripts/provenance/validate_scorpion_capacity_matched_evidence.py `
  evidence/paired_acquisition/scorpion-capacity-matched-20260726/release_manifest.json `
  --require-external-artifacts
```

This package does not modify or supersede
`evidence/paired_acquisition/corrected-20260726`.
