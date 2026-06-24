# Paired-Acquisition Neural Factorization positioning

## One-sentence position

I use **Paired-Acquisition Neural Factorization** as the logical method name for a neural factorization framework that audits how much frozen computational-pathology representations encode tissue identity versus acquisition provenance.

## Why this matters

Whole-slide and patch-level pathology embeddings can entangle biologically meaningful tissue morphology with nuisance or provenance variables such as scanner, stain, center, preparation date, acquisition workflow, and cohort construction. A high downstream score alone does not prove that a representation is using stable tissue signal rather than acquisition-specific shortcuts.

I therefore treat robustness as a source-separation problem rather than only a performance problem.

## What I claim

The safe current claim is:

> In paired-acquisition benchmarks where multiple scanner views of the same underlying tissue region are available, Paired-Acquisition Neural Factorization learns a scanner-suppressed tissue factor that preserves same-region retrieval and cross-scanner agreement while reducing linearly recoverable scanner identity relative to paired-consistency baselines.

The supported evidence includes:

- SCORPION five-fold paired-scanner experiments over 48 original H&E slides, 480 aligned tissue regions, five scanners, and 2,400 real-human-tissue patches.
- Cross-backbone transfer across DINOv2-Base, Phikon, and ImageNet ResNet50 frozen representations.
- Independent external validation on the Multi-Scanner Canine SCC paired-scanner benchmark using locked hyperparameters.
- Pair-repeat allocation controls showing that unique biological pair diversity matters under matched total pair-presentation budgets.

## What I do not claim

I do not claim that Paired-Acquisition Neural Factorization proves disease biology or complete biological/acquisition factor separation.

I do not claim that scanner-suppressed tissue factors are automatically causal disease factors.

I do not claim that good AUC, good retrieval, or reduced scanner-probe accuracy alone establishes biological understanding.

## Related-work distinction

I position Paired-Acquisition Neural Factorization near, but not identical to, the following families:

- scanner-invariant and domain-generalized histopathology representation learning;
- stain normalization and stain/content disentanglement;
- technical-attribute augmentation methods;
- adversarial domain removal;
- pathology foundation-model robustness and domain-shift auditing;
- shared/private or invariant/specific latent-factor models.

The distinction is that Paired-Acquisition Neural Factorization does not only suppress or augment away acquisition signal. It explicitly factorizes frozen pathology embeddings into a scanner-suppressed tissue factor and an acquisition-specific factor, then audits both factors.

## Evaluation contract

A run should report at least the following:

| Factor | Expected behavior | Required audit |
|---|---|---|
| Tissue factor | Preserve same-region tissue identity across scanner views | pair cosine, top-1 retrieval, effective rank |
| Tissue factor | Suppress linearly recoverable scanner identity | tissue-factor scanner probe |
| Acquisition factor | Retain acquisition/scanner information | acquisition-factor scanner probe |
| Acquisition factor | Avoid becoming the tissue-identity embedding | acquisition-factor tissue retrieval |
| Joint representation | Avoid trivial collapse or uncontrolled leakage | reconstruction, variance, cross-covariance, rank |

The win condition is not downstream task performance alone. The win condition is a representation-audit profile showing that tissue identity and acquisition provenance move into different measurable factors.

## Preferred wording

Use:

> paired-acquisition neural factorization

Use:

> scanner-suppressed tissue factor

Use:

> acquisition-specific factor

Use:

> representation-identifiability audit

Use:

> a step toward biological accountability in computational pathology representations

Avoid:

> proven biological factor

Avoid:

> branded method names as the primary public framing

Avoid:

> complete disentanglement

Avoid:

> learns disease biology instead of dataset artifacts

## Short abstract-style paragraph

I use Paired-Acquisition Neural Factorization to address a missing layer in computational pathology robustness: not only making representations invariant to acquisition variation, but explicitly factorizing and auditing the tissue and acquisition components of frozen pathology embeddings. Using paired scanner views of the same underlying tissue, the method learns a scanner-suppressed tissue factor and an acquisition-specific factor, then evaluates whether scanner identity is reduced in the tissue factor while same-region retrieval and cross-scanner tissue agreement are preserved. This frames paired-acquisition learning as a representation-identifiability problem rather than a conventional downstream-performance benchmark.
