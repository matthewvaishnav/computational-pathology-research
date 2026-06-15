# SCORPION paired-scanner PathoAlign plan

**Status:** active next-dataset workstream  
**Tracking:** GitHub issue #10  
**Reference:** Ryu et al., *SCORPION: Addressing Scanner-Induced Variability in Histopathology*, arXiv:2507.20907

## Decision

CAMELYON17 is frozen as external-center and future patient-level leaderboard evidence. New representation-identifiability development moves to SCORPION because it provides matched views of the same real human tissue under five scanner acquisition systems.

This creates a substantially cleaner intervention:

```text
same tissue region + changed scanner
```

The central test is whether a model can retain tissue information while suppressing scanner information in its biological/content representation.

## Dataset structure

The published SCORPION release is described as:

- 48 H&E-stained source slides;
- five scans per slide;
- scanners: Leica Aperio AT2, Leica Aperio GT450, Roche Ventana DP200, 3DHistech P1000, and Philips UFS B300;
- 10 aligned regions from every source slide;
- 480 aligned tissue regions;
- five scanner views per region;
- 2,400 total 1024 x 1024 patches;
- each patch covers approximately 800 micrometers x 800 micrometers.

## Scientific mapping

| PathoAlign concept | SCORPION variable |
|---|---|
| biological/content identity | aligned tissue region |
| acquisition factor | scanner |
| paired intervention | two scanners imaging the same region |
| grouping unit | original source slide |
| paired views per region | five |
| unique scanner pairs | ten |

## Non-negotiable leakage rule

The original source slide is the split, bootstrap, and permutation unit.

All 10 regions and all five scanner views from an original slide must remain in the same train, validation, or test partition. Random patch-level or region-level splitting is prohibited because it would expose nearly identical tissue content across partitions.

The first executable guardrail is:

```text
scripts/scorpion/audit_scorpion_manifest.py
```

It validates complete five-scanner groups, the five expected scanners, original-slide split integrity, duplicate rows and paths, optional image readability, optional SHA-256 duplication, and the full 48/480/2400 release counts.

## Stage A: access and audit

1. Obtain the official dataset archive and record its license and access terms.
2. Keep raw images outside Git.
3. Build `data/scorpion/manifest.csv` with:

```text
slide_id,region_id,scanner_id,path,split
```

4. Run:

```powershell
python scripts/scorpion/audit_scorpion_manifest.py `
  --manifest data/scorpion/manifest.csv `
  --data-root data/scorpion `
  --out-dir results/scorpion/audit `
  --strict-release-counts `
  --check-images `
  --checksums
```

No feature extraction or training starts until this audit passes.

## Stage B: frozen representation audit

Run the same slide-grouped partitions through:

1. ImageNet-pretrained ResNet50, matching the paper's initial feature analysis;
2. at least one pathology encoder already supported by this repository.

For every embedding space, report:

- same-region cross-scanner cosine similarity;
- same-region versus different-region distance margin;
- tissue-region retrieval across scanners;
- scanner-probe accuracy;
- scanner-pair-specific consistency for all ten pairs;
- average scanner-pair consistency;
- worst scanner-pair consistency;
- embedding variance and collapse diagnostics.

The paired analysis is primary. Unpaired scanner UMAPs are descriptive only and cannot substitute for same-tissue comparisons.

## Stage C: controlled model comparison

Use identical slide-grouped splits and seeds for:

1. frozen encoder / no adaptation;
2. ordinary task baseline where labels permit;
3. style augmentation only;
4. SimCons-style prediction consistency;
5. PathoAlign paired consistency;
6. PathoAlign biological/scanner factor separation.

Do not tune against the final held-out slides.

## PathoAlign objectives

Let `x_{r,s}` be tissue region `r` scanned by scanner `s`. The model produces a biological/content representation `z_b` and an acquisition representation `z_a`.

Desired properties:

- `z_b(x_{r,s1})` and `z_b(x_{r,s2})` agree for the same region;
- `z_b` retains enough information to identify or retrieve tissue content;
- scanner identity is difficult to decode from `z_b`;
- scanner identity remains decodable from `z_a`;
- neither representation collapses;
- task performance is preserved where task labels are available.

## Primary confirmatory contrast

Primary hypothesis:

> PathoAlign factor separation improves worst scanner-pair biological/content agreement relative to an augmentation-only baseline.

Independent block:

```text
original source slide
```

Primary statistic:

```text
within-slide difference in worst scanner-pair consistency
```

Use a slide-block bootstrap interval and an exact or Monte Carlo sign-flip test, depending on the final number of held-out slides.

## Secondary constraints

A result is not considered successful if consistency improves through collapse. Therefore require:

- tissue-region retrieval does not materially decrease;
- embedding variance remains above a frozen threshold;
- scanner-probe reduction occurs specifically in `z_b`;
- scanner information is retained or strengthened in `z_a`;
- available task performance stays within a predefined non-inferiority margin.

## CAMELYON17 freeze rule

During SCORPION development:

- do not tune new detector thresholds on CAMELYON17 center 2;
- do not select SCORPION hyperparameters from CAMELYON17 results;
- do not add repeated CAMELYON17 variants to rescue a failed SCORPION result.

After the SCORPION method, hyperparameters, and analysis are frozen, CAMELYON17 may be used once as a transfer/generalization evaluation and separately for the official patient-level Grand Challenge pipeline.

## Deliverables

- dataset access and license record;
- validated manifest and audit report;
- frozen-feature paired analysis;
- baseline comparison;
- PathoAlign comparison;
- slide-block statistical analysis;
- compact evidence tables;
- paper-ready methods and results sections;
- a frozen transfer protocol for CAMELYON17.

## Claim boundary

SCORPION supplies real human tissue with controlled scanner variation. It supports claims about scanner consistency and representation separation under this benchmark. It does not by itself establish clinical safety, diagnostic equivalence, or prospective deployment performance.
