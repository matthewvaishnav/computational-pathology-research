# Forward paired-acquisition provenance v1

## Boundary

This contract applies only to runs created after adoption. It does **not**
reconstruct, correct, or validate the historical origin of the 426 audited NPZ
archives, including the 350 archives whose producer/configuration lineage
remains unresolved. Those artifacts retain their existing evidence status.

## Release structure

Each release contains a top-level `release_manifest.json` and one directory per
run:

```text
release_manifest.json
runs/<run_id>/
  run_record.json
  config.json
  dataset_manifest.json
  dataset_source.txt
  environment.json
  feature_metadata.json
  features.csv
  metrics.json
  run_log.json
  split_manifest.csv
```

Production artifacts may use different data formats, but the record must retain
the same logical roles. Large tensors and checkpoints may stay outside Git; a
release is valid only when every declared file is present at validation time
and its recorded SHA-256 matches.

## Identity and lineage

`run_id` is `parun-v1-` followed by the SHA-256 of a canonical JSON identity
containing:

- schema version;
- exact 40-character source commit;
- producer command as an argument array;
- seed;
- dataset name, source hash, and split-manifest hash;
- semantic configuration and environment hashes; and
- parent run identifiers and exact parent-record hashes.

Configuration, dataset manifest, environment, metrics, run log, and
feature/checkpoint metadata must all carry the same `run_id`. The release
manifest binds each run ID to the exact checksum and repository-relative path
of its run record. Parent identifiers and parent-record hashes must resolve
within the release and form an acyclic graph.

## Fail-closed rules

Validation rejects:

- missing roles or files;
- absolute, noncanonical, symlinked, or run-directory-escaping paths;
- content or record checksum mismatches;
- duplicate run identifiers, artifact roles, or artifact paths;
- a run ID that does not match its immutable identity;
- a component carrying a different run ID;
- inconsistent dataset, split, configuration, environment, or feature hashes;
- missing parent links or parent cycles; and
- any run whose status is not `completed`.

No path, filename, nearby log, or current-state hash is used to infer historical
lineage.

## Smoke gate

Validate the tracked deterministic fixture:

```bash
python scripts/provenance/validate_paired_acquisition_release.py
python scripts/provenance/create_paired_acquisition_smoke_release.py --check
pytest -q -o addopts='' tests/test_paired_acquisition_provenance.py
```

Create a fresh self-contained fixture in an empty directory:

```bash
python scripts/provenance/create_paired_acquisition_smoke_release.py \
  --out-dir /tmp/paired-acquisition-provenance-smoke
```

The smoke fixture is synthetic evidence about the validator, not scientific
evidence. A scientific release tag may be created only after the real release
directory passes this validator from a clean checkout and the release manifest
is published with the tag.
