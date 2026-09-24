# PCam Historical Numerical Comparison

## Current result

The repository's recorded full-test PCam result is:

- **ROC AUC: 0.9394**
- **accuracy: 0.8526**
- **F1: 0.8507**
- official test split: **32,768 patches**

See [PCAM_REAL_RESULTS.md](../PCAM_REAL_RESULTS.md) for the bounded result record.

## What the historical table shows

The earlier repository comparison assembled 10 external PCam AUC values from
published or externally attributed sources. Within that collected table, the
repository's **0.9394 AUC was numerically higher than every external value**.

That numerical ordering is a legitimate descriptive fact about the table and is
retained.

## What it does not show

The external rows were not generated inside one matched experiment. They differ
in study design, preprocessing, splits, architecture selection, tuning budget,
hardware, and reporting conventions. Some legacy source/metric attributions also
require primary-source revalidation.

Therefore the historical table does **not** by itself establish:

- statistical superiority;
- a protocol-controlled state-of-the-art result;
- a significant effect size over another method;
- universal speed, cost, or parameter-efficiency superiority;
- clinical validation or readiness.

The correct wording is:

> The repository model achieved 0.9394 ROC AUC on the official PCam test split,
> which was numerically higher than every external PCam AUC collected in the
> historical comparison table. Because those values were reported under
> different protocols, the table is descriptive rather than a matched
> superiority test.

## What would establish the stronger claim

A direct superiority study should run candidate methods under the same:

- dataset and split definitions;
- preprocessing and augmentation boundaries;
- tuning/model-selection budgets;
- hardware/software environment when timing is compared;
- repeated-seed policy;
- primary endpoint;
- evaluation code; and
- uncertainty procedure.

That matched study is the route to a statistically controlled superiority claim.

## Authority

The repository-root [CLAIM_BOUNDARY.md](../../CLAIM_BOUNDARY.md) is authoritative.
