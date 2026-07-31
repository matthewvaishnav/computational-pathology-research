# PathoROB and encoder robustification: competitor analysis

**Status:** literature and protocol analysis only; no new experimental result is
introduced by this document.

**Source lock:**

- PathoROB benchmark: Alber et al., *Towards robust foundation models for
  digital pathology*, Nature Communications (2026), DOI
  `10.1038/s41467-026-73923-2`.
- Encoder-robustification paper: Filiot et al., *Robustifying pathology
  foundation models via fine-tuning*, arXiv `2607.22861v1`, posted
  2026-07-24.
- Fine-tuned weights named in that preprint: Phaet, derived from Phikon-v2, and
  Mascaret, derived from Midnight-12k.

This analysis separates the **PathoROB benchmark** from the **Filiot et al.
fine-tuning method**. PathoROB is an evaluation framework. It is not itself a
trainable competitor to Paired-Acquisition Neural Factorization.

## Executive verdict

The Filiot et al. work is a serious adjacent competitor for the broad claim
that acquisition-sensitive pathology representations can be made more robust.
It is not a direct substitute for the object learned by Paired-Acquisition
Neural Factorization.

The two approaches answer different primary questions:

1. **Encoder robustification:** can a pathology foundation-model backbone be
   fine-tuned into a single, task-agnostic representation with better aggregate
   robustness and downstream benchmark performance?
2. **Explicit factorization:** can matched acquisitions support a separately
   auditable tissue-oriented representation and acquisition representation,
   with joint reconstruction and capacity-matched controls?

The fine-tuning paper is stronger in breadth. It reports ten backbones and broad
external benchmark suites. The current paired-acquisition evidence is stronger
in local mechanistic auditability: exact matched tissue, an explicit acquisition
branch, reconstruction, equal-capacity controls, held-out original-slide folds,
and fold-aware inference.

The competitor therefore narrows the safe novelty boundary. The project should
not claim that it is the first method to reduce scanner or center information,
that paired acquisitions are uniquely capable of producing robust features, or
that a lower scanner probe alone proves factorization. The defensible
contribution remains the explicit neural separation object and the controls
that make that object inspectable.

## What PathoROB actually is

PathoROB is a benchmark for measuring whether foundation-model representation
geometry is organized more strongly by biological class or by non-biological
medical-center differences. Its public implementation includes:

- a local-neighborhood robustness index;
- average performance drop under center shift;
- a clustering score;
- multi-center datasets derived from Camelyon, TCGA, and Tolkach ESCA.

The robustness index compares the prevalence of neighbors sharing biological
class but not confounding class against neighbors sharing confounding class but
not biological class. It is a useful representation-level diagnostic, but it
is not equivalent to information-theoretic independence from scanner or site.
An index can improve because biological classes become more separated, because
center effects become weaker, or because both occur.

PathoROB also uses **medical center** as the confounding label. Center can bundle
scanner hardware, staining, tissue processing, sectioning, surgical workflow,
and other laboratory differences. It therefore does not isolate scanner effects
as cleanly as co-registered multi-scanner tissue.

## What the Filiot et al. paper claims

The preprint reports a model-agnostic, label-free fine-tuning strategy applied
to ten pathology foundation models. It evaluates the base and fine-tuned
encoders on:

- PathoROB robustness;
- HEST spatial gene-expression prediction;
- THUNDER tile-level tasks;
- Patho-Bench slide-level tasks.

The reported aggregate results include:

- average PathoROB robustness index increasing from `0.72` to `0.87`;
- mean HEST Pearson correlation increasing from `0.398` to `0.411`;
- mean THUNDER rank sum decreasing from `62` to `49`;
- mean Patho-Bench grand average increasing from `54.9` to `56.6`;
- released Phaet and Mascaret weights.

The paper additionally analyzes PLISM and SCORPION to argue that scanner effects
look approximately affine at the output but that robustification is distributed
through deeper transformer blocks. On SCORPION, the paper reports stronger
cross-scanner retrieval after fine-tuning.

The paper's headline “43%” performance improvement is based on its normalized
cross-benchmark ranking construction. It must not be paraphrased as a 43%
increase in raw predictive accuracy. The raw aggregate changes reported for
HEST and Patho-Bench are much smaller.

## Reproducibility audit of arXiv v1

The main weakness of the current preprint is not its name. It is that the method
needed to reproduce the central result is not disclosed in implementable form.

The experimental-setup section specifies the ten backbones and the evaluation
benchmarks, then moves directly into results. In arXiv `2607.22861v1`, the paper
does not provide a complete account of:

- the fine-tuning objective;
- the fine-tuning dataset or exact sampling frame;
- whether matched acquisitions are used for training;
- image transformations or positive-pair construction;
- optimizer, learning rate, schedule, batch size, or number of updates;
- trainable layers or parameter-efficient modules;
- checkpoint-selection criteria;
- random seeds and training-run replication;
- a public implementation of the fine-tuning recipe.

The released weights allow evaluation of Phaet and Mascaret, but they do not
make the reported recipe independently reproducible across the other eight
backbones.

This gap matters for the competitor analysis. The project can compare against
released robustified encoders now. It cannot honestly claim to have reproduced
or falsified the fine-tuning method until the authors release enough training
information or code.

## Statistical and reporting cautions

### Wilcoxon claim as written

The paper states that all ten models improve in aggregate robustness and reports
`p < 10^-4` from a one-sided Wilcoxon signed-rank test. It similarly reports
`p < 10^-4` for the ten base/fine-tuned cross-benchmark rank pairs.

For exactly ten independent non-zero paired observations, the smallest possible
exact one-sided signed-rank probability is `1 / 2^10 = 0.0009765625`. Therefore,
`p < 10^-4` cannot be the exact ten-pair Wilcoxon result. The reported value may
come from a larger unit of analysis, an asymptotic approximation, or pooled
model-by-dataset observations, but the preprint should specify that unit. Pooled
benchmark cells would also require care because observations sharing a model or
dataset are not independent.

This does not negate the direction of the reported improvements. It means the
inferential statement is under-specified and should not be copied into this
project's claim language.

### “No observed trade-off” is aggregate, not universal

The paper's aggregate results are broadly favorable, but its own tables and
limitations contain regressions, including H0-mini on THUNDER and
GenBio-PathFM on HEST. “No observed trade-off” should therefore be interpreted
as an aggregate frontier claim, not as a guarantee that every downstream task
improves for every encoder.

### Training uncertainty is not reported clearly

Patho-Bench tasks are repeated, but the preprint does not expose uncertainty
from independently repeated robustification runs. Base-versus-fine-tuned
comparisons may therefore mix downstream resampling uncertainty with one fixed
fine-tuned checkpoint per backbone.

## Direct comparison with Paired-Acquisition Neural Factorization

| Dimension | Filiot et al. robustification | Paired-Acquisition Neural Factorization |
|---|---|---|
| Intervention point | Fine-tunes the foundation-model encoder | Keeps the encoder frozen and learns post-hoc neural branches |
| Primary output | One robustified representation | Tissue-oriented branch, acquisition branch, and joint reconstruction |
| Training signal | Not implementably disclosed in arXiv v1 | Exact co-registered acquisitions of the same tissue region |
| Acquisition branch | None | Explicit and separately probed |
| Reconstruction | Not part of the disclosed output | Registered joint reconstruction objective |
| Mechanistic intervention | Changes internal transformer representations | Can audit bottleneck, leakage, branch removal, and latent swapping |
| Deployment object | Replacement encoder | Adapter/factorization module on frozen features |
| Empirical breadth | Ten encoders and several large benchmarks | Current promoted evidence is bounded to SCORPION and canine paired data |
| Compute profile | Full-backbone adaptation on HPC resources | Much smaller post-hoc training on frozen embeddings |
| Current reproducibility | Released weights for two encoders; training recipe incomplete | Source, manifests, input hashes, registered fits, and evidence packages are public within stated boundaries |

## Where the competitor is genuinely stronger

The project should concede the following strengths directly:

1. **Breadth:** ten backbones and multiple benchmark suites are far broader than
   the current paired-acquisition evidence.
2. **Single-encoder usability:** a robustified backbone is operationally simple;
   downstream users do not need a separate branch architecture.
3. **External utility evidence:** the paper evaluates gene-expression,
   tile-level, and slide-level tasks rather than relying only on scanner probes
   and paired retrieval.
4. **Depth analysis:** intermediate-block SCORPION retrieval suggests the change
   is distributed through the encoder rather than confined to the output layer.
5. **Released artifacts:** Phaet and Mascaret make immediate black-box
   evaluation possible.

## Where Paired-Acquisition Neural Factorization remains distinct

1. **The acquisition representation is retained rather than only suppressed.**
   This allows direct confirmation that scanner information moved into the
   intended branch.
2. **The same physical tissue is the control.** The acquisition contrast does
   not depend on assuming that center labels are independent of biology.
3. **Capacity-matched controls are explicit.** The full model is compared with
   an equal-capacity two-branch network rather than only with a base encoder.
4. **Reconstruction constrains deletion.** The model is asked to preserve enough
   information jointly to reconstruct the frozen feature.
5. **The claim can remain mechanism-focused.** The strongest defensible claim is
   partial structured separation under a paired protocol, not universal
   robustness or clinical utility.

## Threat assessment

| Potential project claim | Threat level | Reason |
|---|---:|---|
| “Our method is the first to reduce scanner or site signal” | Critical | The broader literature already contains stain normalization, batch correction, adversarial adaptation, paired contrastive losses, and now encoder robustification |
| “Paired acquisitions are required for robust pathology features” | High | Filiot et al. report broad robustness gains without disclosing paired training as a requirement |
| “Lower scanner probe proves disentanglement” | Critical | A single robust representation can lower scanner predictability without learning a factorized acquisition branch |
| “Our explicit acquisition branch is a distinct scientific object” | Low-to-moderate | The competitor produces one representation and does not report a separately decoded acquisition factor |
| “Our current evidence is broader or more clinically validated” | Critical | It is not; the competitor's benchmark breadth is much stronger |
| “Our protocol provides unusually strong paired and capacity controls” | Low | The current public evidence supports this narrower methodological distinction |

## Registered comparison plan

The first comparison should use released weights and require no attempt to
reverse-engineer the undisclosed fine-tuning recipe.

### Campaign A: released-encoder audit on frozen SCORPION tiles

Evaluate four encoders on the exact 2,400 SCORPION observations and original
slide identities:

- Phikon-v2;
- Phaet;
- Midnight-12k;
- Mascaret.

For each base/fine-tuned pair, report:

- held-out original-slide linear scanner balanced accuracy;
- cross-scanner same-region Recall@1 and mean average precision;
- average and worst scanner-pair retrieval;
- paired cosine agreement;
- local scanner-neighborhood mixing;
- feature dimension, inference cost, and storage cost.

This campaign answers whether the released robustified encoders reproduce their
claimed direction on this project's exact SCORPION patch archive. It is not a
method-level reproduction.

### Campaign B: factorization compatibility

For each encoder with an implementable frozen feature archive, fit the same
registered adapter family:

- identity-standardized representation;
- ridge affine harmonization;
- equal-capacity two-branch control;
- full Paired-Acquisition Neural Factorization.

Use the same original-slide-blocked five folds, train/validation-only fitting,
five optimization seeds, and fold-then-slide inference already required by the
promoted SCORPION campaign.

The primary questions are:

1. Does robustified-encoder output make the neural factorization unnecessary?
2. Does factorization still reduce scanner recoverability after encoder
   robustification?
3. Does the acquisition branch remain decodable when the backbone has already
   been robustified?
4. Are encoder robustification and explicit factorization complementary?

Comparisons must be within encoder family first. Phaet should be compared with
Phikon-v2, and Mascaret with Midnight-12k. Cross-family rankings are secondary
because architecture, pretraining data, dimensionality, and compute differ.

### Campaign C: prospective PathoROB transfer test

A separate external experiment can apply a SCORPION-trained, frozen DINOv2
factorization module to DINOv2 features from the public PathoROB datasets,
without fitting on PathoROB test observations. Report:

- PathoROB robustness index;
- center balanced accuracy;
- biological-class balanced accuracy;
- average performance drop under center shift;
- change relative to the unmodified DINOv2 representation.

This would test transfer of the learned tissue-oriented projection. It would not
prove that the biological branch is pure, and failure to transfer would not
invalidate the within-SCORPION paired result.

## Inference requirements

- Original slide or patient/case must remain the independent evaluation unit.
- Hyperparameters and checkpoint selection must use training and validation data
  only.
- Repeated optimization seeds must be averaged before fold-level inference.
- Report paired effect sizes and fold-aware intervals, not only pooled patch
  p-values.
- Preserve the registered `-0.02` retrieval noninferiority margin where the
  endpoint is comparable.
- Do not merge encoder-level robustness, harmonization, and explicit
  factorization into one scalar leaderboard.

## Claim boundary after this review

The competitor analysis supports the following language:

> Paired-Acquisition Neural Factorization is a post-hoc neural factorization of
> frozen pathology features using co-registered acquisition views. Unlike
> single-output encoder robustification or affine harmonization, it produces a
> separately auditable acquisition representation and a tissue-oriented
> representation constrained by joint reconstruction. Current evidence is
> limited to the registered paired-acquisition protocols and does not establish
> universal robustness, biological purity, diagnostic benefit, or superiority
> to robustified foundation-model encoders.

The following language remains unsupported:

- first method to address scanner sensitivity;
- first use of matched scanner acquisitions;
- proof that the tissue branch contains only biology;
- proof that encoder fine-tuning cannot achieve the same scanner suppression;
- superiority to Phaet, Mascaret, or the undisclosed fine-tuning recipe;
- clinical or deployment benefit.

## Decision

Treat the Filiot et al. preprint as a **strong adjacent encoder-level
competitor**, not as evidence that the paired-acquisition project is obsolete.
The immediate action is to benchmark the released Phaet and Mascaret weights on
the frozen SCORPION protocol. Do not spend substantial compute attempting a
method reproduction until an implementable training recipe or code is released.

The project's novelty should be stated around explicit, reconstructive,
capacity-controlled factorization—not around the generic idea of removing
scanner or center information.