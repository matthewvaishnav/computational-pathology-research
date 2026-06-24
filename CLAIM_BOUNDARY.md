# Claim Boundary

This document defines the public claim boundary for this repository.

## Current safe claim

This repository is a research-focused computational pathology framework centered on **Paired-Acquisition Neural Factorization**, representation identifiability, whole-slide histopathology modeling, multiple-instance learning, and external-center validation experiments.

The central method claim is:

> Paired-Acquisition Neural Factorization is a neural factorization framework for auditing whether frozen pathology embeddings can separate tissue identity from acquisition provenance. In paired-scanner benchmarks, the current evidence supports a scanner-suppressed tissue factor that preserves same-region retrieval and cross-scanner agreement while reducing linearly recoverable scanner identity relative to paired-consistency baselines.

Current validated research evidence includes:

- SCORPION paired-acquisition experiments using 48 original H&E slides, 480 aligned regions, five scanners, and 2,400 real-human-tissue patches.
- Cross-backbone transfer of the locked paired-acquisition neural factorization objective across DINOv2-Base, Phikon, and ImageNet ResNet50 frozen representations.
- Independent Multi-Scanner Canine SCC external paired-scanner validation using locked hyperparameters.
- Pair-repeat allocation controls showing that unique biological pair diversity improves biological consistency and factor separation under matched total pair-presentation budgets of 6,400 and 12,800 pair presentations.
- CAMELYON17 center-subspace projection mechanism experiments for attenuating center leakage while preserving tumor signal.
- PANDA grading experiments using Phikon patch features.
- PANDA tuned TransnnMIL repeated-seed QWK values of 0.8155, 0.8225, and 0.8086.

## Boundary

Research-only at this stage. Not currently used for patient care.

Do not describe the current repository as deployed, clinically validated, ready for patient care, proven to improve patient outcomes, proof that a model has learned disease biology, proof of complete biological/acquisition factor separation, or proof that scanner-suppressed tissue factors are causal disease factors.

## Paired-Acquisition Neural Factorization wording boundary

Safe:

> Paired-Acquisition Neural Factorization is a neural factorization framework for auditing tissue and acquisition signal in frozen computational-pathology representations.

Safe:

> The method learns a scanner-suppressed tissue factor and an acquisition-specific factor in paired-scanner settings, then audits whether each factor contains the intended information.

Safe:

> The current evidence supports representation-identifiability and scanner-suppression claims on paired-acquisition benchmarks.

Not safe:

> This proves disease biology.

Not safe:

> This completely removes all dataset artifacts.

Not safe:

> This is ready for diagnostic deployment.

## Safer language

Use:

> Paired-acquisition neural factorization for representation-identifiability auditing.

Use:

> Scanner-suppressed tissue factor and acquisition-specific factor.

Use:

> A step toward biological accountability in computational pathology representations.

Do not use:

> Learns disease biology instead of dataset artifacts.

## Model-performance boundary

Safe:

> Tuned TransnnMIL is competitive with gated AttentionMIL and slightly favorable across the current repeated-seed PANDA experiments, beating AttentionMIL on 2 of 3 tested seeds.

Not safe:

> TransnnMIL is conclusively superior to AttentionMIL.

Safe:

> TransnnMIL appears highly optimization-sensitive in the current PANDA setup; lowering learning rate from 1e-3 to 3e-4 was a major contributor to performance.

Not safe:

> TransnnMIL solves PANDA grading.
