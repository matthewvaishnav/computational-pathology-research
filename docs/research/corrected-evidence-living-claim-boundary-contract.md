# Corrected-Evidence Living Claim-Boundary Contract

**Branch:** `research/real-bottleneck-artifact-recovery-and-adjudication-v2-20260804`
**Validator:** `scripts/provenance/validate_corrected_paired_acquisition_evidence.py`
**Tests:** `tests/test_corrected_paired_acquisition_evidence.py`
**Release:** `evidence/paired_acquisition/corrected-20260726/`

## The provenance defect

The July 26 corrected-evidence validator bound the **current living**
`CLAIM_BOUNDARY.md` to the release-time `publication_sha256`:

```python
if sha256_canonical_text(current_claim) != claim.get("publication_sha256"):
    raise EvidenceValidationError("authoritative claim boundary checksum mismatch")
```

The release manifest binds two values:

- `snapshot_sha256` — the raw SHA-256 of the immutable
  `claim_boundary_snapshot.md` artifact;
- `publication_sha256` — the canonical-text SHA-256 of that same immutable
  snapshot at release time (for this release both equal
  `cb06886a6050a66f6471b2468d9f8586be993d5d45f9f8c3b27259404c5bc91b`).

The authoritative repository `CLAIM_BOUNDARY.md` is a **living document** and has
legitimately evolved since the July 26 release. Because the old validator
required the living file to retain the release-time checksum, the tracked
release failed `test_tracked_corrected_evidence_is_current` on the base commit.
Updating the historical manifest hash to match today's file is prohibited: the
historical release is immutable.

## The repair

The validator now keeps the historical release cryptographically bound to its
immutable snapshot and treats the living file's hash as informational:

1. `claim_boundary.snapshot_sha256` must equal the snapshot artifact's recorded
   `sha256` (unchanged binding).
2. The immutable snapshot artifact must exist, its byte checksum and size must
   match the manifest, and its **canonical text hash must equal
   `publication_sha256`** — this is the true publication commitment.
3. The current authoritative repository claim-boundary file must exist at the
   declared `authoritative_repository_path`.
4. The current living file's canonical-text hash is compared to the immutable
   publication hash and reported in the summary under `claim_boundary_report`:

```json
{
  "immutable_publication_hash": "cb06886a…bc91b",
  "current_authoritative_hash": "03896af1…684f1",
  "hashes_match": false
}
```

A mismatch between the living file and the immutable snapshot is
**informational, not release invalidation**. The historical release remains valid
when the current authoritative claim boundary is subsequently edited.

## Unchanged guarantees

The repair does not weaken any other validator guarantee:

- source-code binding (`source_code.commit`, `tree`,
  `equivalent_execution_commit`, script blobs at the bound commit);
- promoted-artifact hashes, sizes, and row counts;
- corrected canine estimand checks (fixed categories, support, key metrics);
- SCORPION fold-aware checks (design, contrasts, key metrics);
- historical-evidence withdrawal checks;
- optional external-input revalidation.

## Files changed

- `scripts/provenance/validate_corrected_paired_acquisition_evidence.py`
- `tests/test_corrected_paired_acquisition_evidence.py`
- `docs/research/corrected-evidence-living-claim-boundary-contract.md` (this file)

No historical evidence artifact — the release manifest, the immutable
claim-boundary snapshot, or any promoted evidence file — was modified.
