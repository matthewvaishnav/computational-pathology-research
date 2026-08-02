# Historical Model Card Notice

**Retired:** August 2, 2026

The former “AttentionMIL + TransnnMIL v2.0” model card is obsolete and must not
be interpreted as a current validated model card.

## Why it was retired

The earlier page combined multiple architectures and software components into a
single product-style description. It also contained fixed parameter counts,
speed claims, intended clinical use cases, dataset descriptions, and performance
statements that do not represent a single currently validated model artifact.

More importantly, the historical TransnnMIL record contains two established
implementation defects:

- the one-query/one-key attention path was invariant to the TransMIL query; and
- historical topology mode created an unregistered projection during each
  forward pass.

Historical QWK or architecture-performance values therefore do not validate
genuine branch fusion or a trained topology contribution.

## Current status

TransnnMIL remains an active whole-slide neural-network research line. The
canonical implementation has been repaired, but new matched evaluations are
required against:

- standalone nnMIL;
- standalone TransMIL;
- historical TransnnMIL;
- concatenation fusion;
- gated fusion; and
- explicit branch-token attention.

Those comparisons must use identical splits, seeds, tuning budgets, feature
inputs, and evaluation rules before a performance claim is promoted.

## Current use boundary

No model in this repository is currently represented by this page as:

- clinically validated;
- suitable for diagnosis or patient care;
- production or hospital ready;
- FDA approved, CE marked, or otherwise regulated;
- universally superior to other MIL architectures; or
- supported by the former model-card metrics.

Use the repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) and
[`CURRENT_STATUS.md`](CURRENT_STATUS.md) for the current public record.
