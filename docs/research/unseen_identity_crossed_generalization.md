# Unseen-Identity Crossed Generalization

## Status

Post-confirmatory exploratory research path. This experiment does not alter the
frozen private pathology campaign or either preceding synthetic campaign.

## Motivation

The successful crossed-target scanner-prototype experiment withheld one scanner
cell per biological identity, but every evaluated identity still appeared in
training under the remaining scanners. That establishes missing-cell completion
for known identities. It does not establish generalization to new biology.

This experiment removes that loophole.

## Primary question

Does Crossed-Target Scanner-Prototype Factorization preserve biological identity
and transfer acquisition state when the biological identity was never present
during optimization?

## Identity-disjoint design

For every independently generated synthetic dataset:

- training identities and test identities are disjoint;
- all scanner views of a biological identity belong to exactly one partition;
- training identities are observed under all scanners;
- unseen test identities are also rendered under all scanners, solely for
  evaluation;
- standardization statistics are computed only from training identities; and
- representation probes are fit on training identities and evaluated on unseen
  identities.

The smoke campaign uses two dataset-generation seeds. The full campaign uses
three.

## Intervention evaluation

For each unseen biological identity and every ordered scanner pair
`source_scanner != target_scanner`, evaluate

```text
D(C(x[b_new, source_scanner]), acquisition[target_scanner])
    -> x[b_new, target_scanner]
```

For PA-NF and the supervised oracle, the target acquisition code comes from a
different unseen donor identity observed under the requested target scanner. For
the scanner-prototype models, the requested scanner prototype is used directly.

With five scanners, this creates 20 ordered interventions per unseen identity.

Two independent contrasts are retained:

```text
biology retention =
    MSE(swapped, different_biology_same_target_scanner)
    - MSE(swapped, correct_biology_correct_target_scanner)

acquisition transfer =
    MSE(swapped, correct_biology_source_scanner)
    - MSE(swapped, correct_biology_correct_target_scanner)
```

Bootstrap confidence intervals operate on per-identity means across all ordered
scanner transfers, preventing the 20 correlated transfers from being treated as
independent biological samples.

## Frozen model families

1. Unchanged PA-NF.
2. Scanner prototypes with reconstruction and biological consistency, but no
   crossed-target loss.
3. Crossed-Target Scanner-Prototype Factorization.
4. Supervised known-factor oracle.

No family is removed from control validation.

## Frozen factorization thresholds

The thresholds are unchanged from the preceding experiment:

- biology-retention 95% bootstrap lower bound greater than zero;
- acquisition-transfer 95% bootstrap lower bound greater than zero;
- more than half of unseen identities pass both counterfactual axes;
- biological-to-biological linear-probe R² at least 0.80;
- biological-to-acquisition linear-probe R² at most 0.10;
- acquisition-to-acquisition linear-probe R² at least 0.80;
- acquisition-to-biological linear-probe R² at most 0.10; and
- combined representation to joint factors R² at least 0.80.

The path-forward gate opens only if every dataset-seed/renderer condition has:

- every oracle optimization seed pass;
- every unchanged PA-NF optimization seed fail; and
- every crossed-target prototype optimization seed pass.

## Additional diagnostics

These diagnostics are reported without changing the frozen gate:

- success across all ordered intervention pairs;
- worst unseen identity across its 20 transfers;
- worst ordered scanner-pair biology-retention margin;
- worst ordered scanner-pair acquisition-transfer margin;
- worst ordered scanner-pair two-axis success rate;
- cross-scanner identity retrieval among unseen identities;
- scanner classification from the biological representation;
- acquisition-code variation within scanner across donor identities; and
- branch-ablation penalties.

## Campaign sizes

### Smoke

```text
2 dataset seeds
2 renderers
4 model families
2 optimization seeds
32 fits
40 training identities
20 unseen test identities
100 requested epochs
500 identity-bootstrap replicates
```

### Full

```text
3 dataset seeds
2 renderers
4 model families
10 optimization seeds
240 fits
256 training identities
128 unseen test identities
250 requested epochs
5,000 identity-bootstrap replicates
```

The supervised oracle retains the existing minimum-control epoch safeguard.

## Interpretation

A pass establishes empirical intervention-consistent generalization to unseen
synthetic biological identities under the tested renderer distributions. It does
not establish pathology-domain validity, scanner generalization to unseen
devices, or a mathematical identifiability theorem.

A failure is informative:

- oracle failure invalidates the benchmark or control training;
- PA-NF success requires inspection for a changed failure mode or an overly weak
  test;
- crossed-target prototype failure indicates that the previous result depended
  on identity overlap; and
- reconstruction-only success across all conditions would show that the crossed
  loss is no longer incrementally necessary under this stronger split.
