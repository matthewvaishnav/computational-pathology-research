# Archived Performance-Comparison Notice

**Retired:** August 2, 2026

The former page ranked the repository’s PCam result against published systems
using numbers collected from different papers, hardware, preprocessing paths,
model-selection procedures, and experimental protocols. That comparison was not
a controlled benchmark and is withdrawn.

## Current supported PCam record

The repository records one official patch-level PCam test-split result:

- ROC AUC: `0.9394`
- accuracy: `0.8526`

See [`PCAM_REAL_RESULTS.md`](PCAM_REAL_RESULTS.md) for the bounded result record.

## What is no longer claimed

The repository does not claim:

- “#1” performance;
- state-of-the-art performance;
- statistical superiority over published systems;
- valid improvement percentages computed across incompatible studies;
- universal speed or cost advantages;
- production suitability from training time or throughput; or
- clinical value from a patch-classification benchmark.

## Requirements for a future comparison

A defensible comparative study would need, at minimum:

- identical data and split definitions;
- identical preprocessing and augmentation boundaries;
- matched tuning and model-selection budgets;
- controlled hardware and software environments when timing is compared;
- repeated seeds where optimization variability matters;
- a prespecified primary endpoint;
- uncertainty at the correct independent unit; and
- explicit separation of engineering throughput from scientific performance.

Until such a study is completed and promoted through a forward-valid evidence
package, the old leaderboard and hardware-value recommendations must not be
cited.

The repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) is authoritative.
