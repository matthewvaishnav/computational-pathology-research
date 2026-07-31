# PathoROB and encoder robustification: related-work positioning

**Status:** literature and prospective-protocol analysis only. This document
introduces no new experimental result and does not rank one research program
above another.

**Source lock:**

- PathoROB benchmark: Alber et al., *Towards robust foundation models for
  digital pathology*, Nature Communications (2026), DOI
  `10.1038/s41467-026-73923-2`.
- Encoder-robustification paper: Filiot et al., *Robustifying pathology
  foundation models via fine-tuning*, arXiv `2607.22861v1`, posted
  2026-07-24.
- Fine-tuned weights named in that preprint: Phaet, derived from Phikon-v2, and
  Mascaret, derived from Midnight-12k.

## Purpose

This review positions PathoROB, encoder robustification, and
Paired-Acquisition Neural Factorization as related but non-equivalent parts of
an evolving pathology-representation research landscape.

None of these objects should be treated as a final answer:

- PathoROB is an evaluation framework, not a trainable method;
- Filiot et al. study how pathology foundation-model encoders can be adapted
  toward more robust single representations;
- Paired-Acquisition Neural Factorization studies whether exact matched
  acquisitions can support an explicit, reconstructive separation between a
  tissue-oriented representation and an acquisition representation.

The useful question is therefore not which project “wins.” It is what each line
of work reveals, what remains unidentified, and which next experiments can
connect them without collapsing distinct scientific questions into one
leaderboard.

## Shared research direction

All three efforts respond to the same broad problem: pathology representations
can encode acquisition, center, staining, processing, and workflow effects in
addition to tissue information.

They intervene at different levels:

1. **Benchmarking:** characterize whether representation geometry follows
   biological or confounding structure.
2. **Encoder evolution:** modify a foundation-model backbone so its single
   output representation behaves more robustly across benchmark shifts.
3. **Explicit factorization:** preserve a frozen encoder and learn separate
   neural branches from exact paired acquisitions, including an acquisition
   branch and joint reconstruction.

These are complementary levels of investigation. A benchmark can expose a
problem without specifying the mechanism that solves it. A robustified encoder
can improve a single representation without identifying where acquisition
information went. A factorization model can make acquisition information
explicit without yet establishing broad downstream utility.

## What PathoROB contributes

PathoROB evaluates whether local representation geometry is organized more by
biological class or by non-biological medical-center differences. Its public
implementation includes:

- a local-neighborhood robustness index;
- average performance drop under center shift;
- a clustering score;
- multi-center datasets derived from Camelyon, TCGA, and Tolkach ESCA.

The robustness index is a useful representation-level diagnostic, but it is not
an information-theoretic proof of independence from scanner or site. It may
improve because biological classes become more separated, because center
effects weaken, or because both occur.

PathoROB uses medical center as its confounding label. Center can combine
scanner hardware, staining, tissue processing, sectioning, surgical workflow,
and other laboratory differences. That breadth is valuable for external
robustness assessment, while co-registered multi-scanner tissue offers a more
controlled view of scanner-specific variation. Neither replaces the other.

## What the Filiot et al. work contributes

The preprint reports a model-agnostic, label-free fine-tuning strategy applied
to ten pathology foundation models. It evaluates base and fine-tuned encoders
on:

- PathoROB robustness;
- HEST spatial gene-expression prediction;
- THUNDER tile-level tasks;
- Patho-Bench slide-level tasks.

Reported aggregate changes include:

- average PathoROB robustness index increasing from `0.72` to `0.87`;
- mean HEST Pearson correlation increasing from `0.398` to `0.411`;
- mean THUNDER rank sum decreasing from `62` to `49`;
- mean Patho-Bench grand average increasing from `54.9` to `56.6`;
- released Phaet and Mascaret weights.

The paper also analyzes PLISM and SCORPION to argue that scanner effects appear
approximately affine at the encoder output while robustification is distributed
through deeper transformer blocks. On SCORPION, it reports stronger
cross-scanner retrieval after fine-tuning.

The headline “43%” value is derived from a normalized cross-benchmark ranking
construction. It should not be restated as a 43% raw accuracy increase. The raw
aggregate changes reported for HEST and Patho-Bench are smaller.

## What Paired-Acquisition Neural Factorization contributes

The current paired-acquisition program uses exact co-registered views of the
same tissue region. It keeps the upstream feature encoder frozen and learns:

- a tissue-oriented branch trained for cross-acquisition agreement and reduced
  scanner recoverability;
- an acquisition branch trained to retain scanner information;
- a joint decoder constrained to reconstruct the original frozen feature.

Its scientific object is not simply a more robust embedding. It is an explicit
partition with separately auditable branches, together with capacity-matched
controls and paired evaluation.

The current evidence is intentionally local. It supports partial structured
separation under registered SCORPION and canine paired-acquisition protocols. It
does not establish biological purity, universal robustness, downstream clinical
benefit, or superiority over adapted foundation models.

## How the research objects differ

| Dimension | PathoROB | Encoder robustification | Paired-Acquisition Neural Factorization |
|---|---|---|---|
| Primary role | Evaluation framework | Evolution of a foundation-model encoder | Post-hoc neural factorization of frozen features |
| Main output | Robustness and shift measurements | One adapted representation | Tissue-oriented branch, acquisition branch, reconstruction |
| Main supervision | Biological and center labels for evaluation | Training recipe not implementably disclosed in arXiv v1 | Exact matched acquisitions of the same tissue region |
| Acquisition representation | Not learned | Not separately exposed | Explicit and separately probed |
| Intervention location | Evaluation only | Internal encoder parameters | Adapter branches after a frozen encoder |
| Breadth currently demonstrated | Multiple public multi-center datasets | Ten encoders and several downstream suites | SCORPION and canine paired-acquisition studies |
| Mechanistic auditability | Diagnoses geometry | Intermediate-layer analysis | Branch probes, reconstruction, capacity controls, ablations |

The table describes different research objects rather than a ranking. The lines
can be combined prospectively: PathoROB can evaluate factorized features,
robustified encoders can become inputs to the factorization model, and exact
paired acquisitions can reveal what encoder adaptation changed.

## Reproducibility notes for arXiv v1

The current preprint identifies the evaluated backbones and benchmark suites but
does not yet disclose the central fine-tuning method in implementable detail.
The missing or incomplete elements include:

- the fine-tuning objective;
- the fine-tuning dataset or exact sampling frame;
- whether matched acquisitions are used during training;
- image transformations or positive-pair construction;
- optimizer, learning rate, schedule, batch size, and number of updates;
- trainable layers or parameter-efficient modules;
- checkpoint-selection criteria;
- random seeds and training-run replication;
- a public implementation of the fine-tuning recipe.

The released Phaet and Mascaret weights support black-box evaluation now. They
do not yet permit a complete method-level reproduction across all ten
backbones. This is a maturity boundary of the current release, not evidence that
the direction is invalid.

## Statistical and reporting cautions

### Wilcoxon statement

The paper reports `p < 10^-4` from one-sided Wilcoxon signed-rank tests over ten
base/fine-tuned model pairs. For exactly ten independent non-zero paired
observations, the smallest possible exact one-sided signed-rank probability is
`1 / 2^10 = 0.0009765625`.

Therefore, the reported value cannot be the exact ten-pair result as stated. It
may derive from an asymptotic calculation, a larger unit of analysis, or pooled
model-by-dataset cells. The unit and dependence structure should be made
explicit before the value is reused.

This caution concerns inferential specification. It does not erase the reported
direction of the aggregate changes.

### Aggregate improvement is not universal improvement

The paper's aggregate results are favorable, while some encoder-task cells
regress. Its results are best interpreted as movement of an aggregate frontier,
not a guarantee that every downstream task improves for every backbone.

### Training uncertainty

The downstream evaluations include repeated tasks, but arXiv v1 does not clearly
separate downstream resampling uncertainty from uncertainty across independently
repeated robustification runs. Future releases could clarify this by reporting
multiple training seeds for the adaptation stage.

## What this work changes about the paired-acquisition roadmap

The literature does not make Paired-Acquisition Neural Factorization obsolete.
It makes the next questions more precise.

The project should no longer motivate itself around the generic proposition that
scanner or center information can be reduced. That direction already includes
normalization, harmonization, adversarial adaptation, contrastive learning,
benchmark-driven robustification, and other approaches.

The paired-acquisition program should instead continue evolving around:

- explicit retention of acquisition information in a designated branch;
- exact physical-tissue controls;
- reconstruction as a constraint against unmeasured deletion;
- equal-capacity controls;
- testing whether robustification and factorization are complementary;
- identifying when an explicit branch provides information that a robust single
  representation cannot expose.

## Prospective integration plan

The following campaigns are designed as integration experiments, not a contest.

### Campaign A: released-encoder audit on frozen SCORPION tiles

Evaluate four encoders on the exact 2,400 SCORPION observations and original
slide identities:

- Phikon-v2;
- Phaet;
- Midnight-12k;
- Mascaret.

For each base/adapted encoder pair, report:

- held-out original-slide linear scanner balanced accuracy;
- cross-scanner same-region Recall@1 and mean average precision;
- average and worst scanner-pair retrieval;
- paired cosine agreement;
- local scanner-neighborhood mixing;
- feature dimension, inference cost, and storage cost.

This campaign asks whether the released weights reproduce the reported direction
on the project's frozen SCORPION archive. It is not a reproduction of the
undisclosed training recipe.

### Campaign B: factorization after encoder adaptation

For each encoder with an implementable frozen feature archive, fit the same
registered adapter family:

- identity-standardized representation;
- ridge affine harmonization;
- equal-capacity two-branch control;
- full Paired-Acquisition Neural Factorization.

Use the registered original-slide-blocked five folds, train/validation-only
fitting, five optimization seeds, and fold-then-slide inference.

The primary questions are:

1. How much scanner information remains after encoder adaptation?
2. Does an explicit acquisition branch remain learnable?
3. Does factorization preserve same-region retrieval after robustification?
4. Are encoder adaptation and explicit factorization redundant,
   complementary, or conditional on backbone family?
5. Does reconstruction reveal information discarded by a single robustified
   output?

Comparisons should be within encoder family first: Phaet with Phikon-v2, and
Mascaret with Midnight-12k. Cross-family rankings remain secondary because
architecture, pretraining data, dimensionality, and compute differ.

### Campaign C: prospective PathoROB transfer

Apply a SCORPION-trained, frozen DINOv2 factorization module to DINOv2 features
from public PathoROB datasets without fitting on PathoROB test observations.
Report:

- PathoROB robustness index;
- center balanced accuracy;
- biological-class balanced accuracy;
- average performance drop under center shift;
- changes relative to unmodified DINOv2 features.

This asks whether a factorization learned from exact scanner pairs transfers to
broader center variation. Success would extend the scope of the learned
projection. Failure would identify a boundary between scanner-specific paired
structure and multi-center laboratory variation; it would not invalidate the
within-SCORPION result.

## Inference requirements

- Original slide or patient/case remains the independent evaluation unit.
- Hyperparameters and checkpoint selection use training and validation data
  only.
- Repeated optimization seeds are averaged before fold-level inference.
- Report paired effect sizes and fold-aware intervals, not only pooled patch
  p-values.
- Preserve the registered `-0.02` retrieval noninferiority margin where the
  endpoint is comparable.
- Do not compress benchmark robustness, harmonization, explicit factorization,
  and downstream utility into one scalar leaderboard.

## Updated claim boundary

Supported positioning:

> Paired-Acquisition Neural Factorization is an evolving post-hoc neural
> factorization of frozen pathology features using co-registered acquisition
> views. It produces a separately auditable acquisition representation and a
> tissue-oriented representation constrained by joint reconstruction. It is
> related to, but scientifically distinct from, benchmark-based robustness
> evaluation and single-output encoder adaptation. Current evidence is limited
> to registered paired-acquisition protocols and does not establish universal
> robustness, biological purity, diagnostic benefit, or superiority to adapted
> foundation-model encoders.

Unsupported language remains:

- first method to address scanner sensitivity;
- first use of matched scanner acquisitions;
- proof that the tissue branch contains only biology;
- proof that encoder fine-tuning cannot achieve similar scanner suppression;
- superiority to Phaet, Mascaret, or an undisclosed fine-tuning recipe;
- clinical or deployment benefit.

## Decision

Treat PathoROB, Filiot et al., and Paired-Acquisition Neural Factorization as
parts of an evolving research program around acquisition-sensitive pathology
representations.

The immediate next step is to evaluate released Phaet and Mascaret weights on
the frozen SCORPION protocol, then test whether explicit factorization still
adds an auditable acquisition branch or other measurable structure. The goal is
not to defeat another method. It is to learn which components combine, which
questions remain distinct, and how the paired-acquisition model should evolve.