# Historical PCam Cross-Paper Comparison

> **Status: withdrawn as a current leaderboard.** Earlier versions of this page ranked the repository's PCam result against metrics copied from unrelated publications and described the result as "#1" or "state of the art." That interpretation is not admissible because the compared studies differ in protocol, model selection, hardware, preprocessing, splits, and reporting conventions.

The historical comparison is preserved in Git history for auditability, but it is **not current claim evidence**.

## Current PCam evidence

Use the bounded result record instead:

- [PCAM_REAL_RESULTS.md](../PCAM_REAL_RESULTS.md)
- [THRESHOLD_OPTIMIZATION.md](../THRESHOLD_OPTIMIZATION.md)
- [CLAIM_BOUNDARY.md](../../CLAIM_BOUNDARY.md)

The current public PCam record supports a patch-level engineering benchmark on the official PCam test split. It does not support:

- cross-paper statistical superiority;
- a state-of-the-art ranking;
- clinical validation or readiness;
- patient-, slide-, or workflow-level benefit;
- diagnoses/lives-saved claims;
- claims that different published metrics form one controlled leaderboard.

## Why the old comparison was withdrawn

Published PCam numbers are useful literature context, but they are not paired observations from one experiment. Treating them as though they were produced under a common estimand creates false precision and can turn hardware/protocol differences into a fictitious model ranking.

Future comparative claims require a matched benchmark in which candidate methods share the same data, split units, preprocessing, training/tuning budget, evaluation code, and uncertainty procedure.

## Historical value

The old page remains important as part of the program's scientific self-correction: it documents a pre-audit mode of benchmarking that the current evidence system explicitly forbids.
