# Research Engineering Portfolio Summary

## Overview

This repository documents independent computational-pathology research with an
emphasis on neural networks, paired experimental structure, leakage-resistant
evaluation, reproducible execution, and explicit claim boundaries.

The strongest current line is **Paired-Acquisition Neural Factorization**, which
uses matched scans of the same tissue regions to study how frozen pathology
embeddings allocate tissue-associated and scanner-associated information.

This is a research portfolio, not a clinically validated product or a claim of
production hospital deployment.

## What the repository demonstrates

### Neural representation learning

- multi-branch neural factorization of frozen pathology embeddings;
- pair agreement and scanner-suppression objectives;
- an explicit scanner-retaining acquisition branch;
- joint reconstruction and bottleneck controls;
- covariance, cross-covariance, variance, and dependence regularization; and
- controlled latent-factor and crossed-target synthetic diagnostics.

### Experimental design

- exact same-region multi-scanner pairing where available;
- biological-sample- or slide-blocked train/test separation;
- fit-only preprocessing and probe fitting;
- same-region and same-sample neighbourhood exclusions;
- equal-capacity controls and objective ablations;
- broken-pair and pair-integrity checks; and
- preregistered noninferiority and falsification boundaries.

### Reproducibility engineering

- deterministic source- and configuration-bound run identities;
- frozen input and split hashes;
- append-only execution ledgers;
- resumable fail-closed campaigns;
- unique attempt directories and atomic records;
- checkpoint, corruption, and finite-gradient validation;
- prospective analysis specifications; and
- forward-valid evidence packages with independent validators.

### Scientific self-correction

The repository records corrections rather than hiding them. Examples include:

- withdrawal of slide-independent inference that overstated independent sample
  size;
- replacement of leakage-prone category and neighbourhood estimands;
- identification of capacity-mismatched controls;
- withdrawal of historical TransnnMIL fusion and topology interpretations after
  implementation defects were found;
- removal of clinical, superiority, and deployment claims unsupported by the
  evidence; and
- publication of negative factorial results instead of selecting a favourable
  configuration post hoc.

## Current promoted evidence

### SCORPION capacity-matched campaign

- 175/175 registered fits completed;
- seven variants, five original-slide-blocked folds, and five seeds;
- tissue-branch scanner balanced accuracy reduced relative to an equal-capacity
  control;
- same-region retrieval preserved within the registered margin; and
- acquisition-branch scanner information retained.

### External canine SCC audit and factorial

- corrected biological-sample-blocked fixed-estimand evaluation;
- fit-only probes and fit-pool neighbourhoods;
- 450/450 dimensionality × cross-covariance cells completed; and
- no stable universal bottleneck or regularization optimum found.

These findings support partial structured separation under the tested protocols.
They do not prove pure biological factors or clinical value.

## Active but unpromoted work

- paired affine and orthogonal-Procrustes comparison on frozen SCORPION folds;
- crossed-target scanner-prototype synthetic diagnostics;
- identity-disjoint synthetic generalization;
- repaired TransnnMIL matched reruns; and
- stronger external human-tissue validation.

Smoke results and open PRs are engineering or exploratory records, not promoted
claim evidence.

## Secondary research lines

- TransnnMIL and controlled whole-slide aggregation studies;
- institutional weighting and source-influence mechanisms under simulated or
  centralized settings;
- PCam patch-level engineering benchmarks; and
- provenance, metadata-readiness, and identifiability audits.

Each line has a separate evidence boundary. Results are not pooled into a single
cross-protocol leaderboard.

## Professional profile demonstrated

The repository shows ability to:

- formulate neural-network research questions from biomedical confounding
  problems;
- implement and debug multi-objective PyTorch experiments;
- design statistically defensible validation units;
- distinguish harmonization, factorization, representation geometry, and task
  performance;
- build reproducible experiment infrastructure;
- audit leakage, pseudoreplication, confounding, and provenance;
- report negative results and narrower conclusions; and
- maintain public documentation that matches the actual evidence.

## Explicit limitations

This repository does not currently establish:

- novelty, priority, or patentability;
- complete disentanglement or scanner invariance;
- diagnostic or patient benefit;
- hospital, PACS, privacy, regulatory, or clinical deployment validation;
- full federated-learning validation; or
- universal superiority over other architectures or correction methods.

The authoritative public boundary is `CLAIM_BOUNDARY.md` at the repository root.
