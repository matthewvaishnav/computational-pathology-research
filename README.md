# Computational Pathology Research

Independent research program spanning representation learning, whole-slide neural aggregation, institutional aggregation, and scientific provenance in computational pathology.

This repository is the **program-level hub and evidence ledger**. Distinct methods are being maintained or extracted as standalone repositories so that each contribution has its own implementation, tests, experiments, and claim boundary without losing the cross-project research narrative.

## Primary public preprint

**Accountable Neural Aggregation in Computational Pathology: From Paired-Acquisition Representations to Whole-Slide and Institutional Learning** is the program-level foundations manuscript.

- [Open the primary paper PDF directly](https://matthewvaishnav.github.io/computational-pathology-research/accountable-neural-aggregation-in-computational-pathology.pdf)
- [Open the supplement](https://matthewvaishnav.github.io/computational-pathology-research/accountable-neural-aggregation-in-computational-pathology-supplement.pdf)
- [Download the primary arXiv source package](https://matthewvaishnav.github.io/computational-pathology-research/accountable-neural-aggregation-arxiv-source.zip)
- [Read the manuscript source and release metadata](manuscripts/computational-pathology-foundations-v1/README.md)
- [Read the authoritative claim boundary](CLAIM_BOUNDARY.md)

The root GitHub Pages URL opens the primary PDF directly.

## Research lines

| Research line | Repository / status | Current evidence boundary |
|---|---|---|
| **Paired-Acquisition Neural Factorization (PA-NF)** | Study-specific repositories already split; reusable method extraction planned | Registered SCORPION capacity-matched campaign establishes a controlled comparative advantage over the equal-capacity two-branch neural control on the structured-separation objective. The corrected canine fixed-estimand audit does **not** establish an additional neural feature-space increment over every strong simple scanner-removal baseline. |
| **TransnnMIL** | Standalone repository extraction prepared | Authored multibranch whole-slide architecture is implemented; historical fusion/topology scores remain withdrawn pending repaired matched reruns. |
| **PathologyFL** | Standalone repository extraction prepared | Pathology-specific federated-learning research infrastructure is implemented and integration/smoke tested; this is not a real multi-center deployment validation. |
| **FAIR-WEIGHTS-H** | Currently inside PathologyFL; standalone extraction remains optional | Auditable hybrid institutional-weighting protocol with implemented stability/safety mechanisms; universal fairness or performance superiority is not claimed. |
| **WSI-NCA / Factorized Tissue Dynamics** | Experimental branch only | Architecture research remains pre-pathology-validation and is intentionally not promoted as an established research line yet. |
| **Scientific provenance** | Remains in this hub | Immutable evidence packages, claim ledgers, hostile review, exact artifact recovery, and fail-closed validation. |

## PA-NF: what is actually established

Paired-Acquisition Neural Factorization uses matched acquisitions of the same tissue region to learn a tissue-oriented branch and an explicit acquisition branch.

The separately versioned SCORPION capacity-matched campaign completed all **175/175 registered fits**. Against an equal-capacity two-branch neural control without scanner objectives, PA-NF reduced tissue-branch scanner balanced accuracy by **0.3108** with a fold-aware 95% interval of **[-0.3346, -0.2858]**, while preserving average and worst same-region retrieval within the registered **0.02 noninferiority margin** and retaining strong acquisition-branch scanner information. This is a supported **controlled comparative advantage on the registered SCORPION structured-separation objective**.

The corrected canine fixed-estimand audit answers a different question. On that comparison, PA-NF B32/B64 did not establish a feature-space increment over the strongest simple centroid/QR and paired-linear scanner-removal baselines. Increasing the tissue bottleneck from 32 to 64 dimensions increased retrieval and scanner recoverability without a supported corrected-category gain.

These results are complementary rather than contradictory: a bounded controlled advantage is established where the registered SCORPION comparator supports it, while a broader claim of superiority over every simple scanner-removal baseline is not supported by the corrected canine experiment.

The results support partial structured separation under tested protocols. They do not establish pure biological factors, complete scanner invariance, diagnostic improvement, clinical utility, deployment readiness, or universal superiority over all harmonization methods.

## Study-specific PA-NF repositories

- [SCORPION paired-acquisition study](https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion)
- [External canine SCC study](https://github.com/matthewvaishnav/paired-acquisition-factorization-caninescc)
- [Pair-repeat allocation study](https://github.com/matthewvaishnav/paired-acquisition-factorization-allocation)

## Hugging Face release layer

Hugging Face is used only for curated, versioned scientific objects; GitHub
remains the authoritative laboratory, engineering, evidence, and claim-history
record.

- [Retrospective release audit](docs/releases/huggingface-retrospective-audit-20260812.md)
- [Machine-readable release registry](docs/releases/huggingface-release-registry.yaml)
- [Fail-closed publishing and checksum tooling](tools/huggingface/README.md)
- [Public PA-NF evidence release](https://huggingface.co/datasets/MatthewVaishnav/paired-acquisition-factorization-evidence)
  at immutable revision `a9853bd32e3b446a97608002f7e5ea12f68f88e1`
- [Public PA-NF model release](https://huggingface.co/MatthewVaishnav/paired-acquisition-neural-factorization)
  at immutable revision `b5de3cf9c062cb0d5623628165098c26923309c7`

The model release is the complete registered 25-checkpoint SCORPION
`pathoalign_dep20` family plus five fold-specific standardizers, exact model and
inference code, and provenance manifests. It consumes raw 768-dimensional
frozen `facebook/dinov2-base` features under the documented fold-specific input
contract; it does not accept raw pathology images. The separate evidence
release carries registered metrics, analyses, manifests, and retained negative
results.

A registry entry marked `prepared` or `deferred` is not a released Hub
artifact. Public/private status and immutable HF revisions are recorded only
after remote verification.

## Repository split

The intended repository topology is:

```text
computational-pathology-research      # program hub, manuscripts, claim/evidence ledger
paired-acquisition-neural-factorization # reusable PA-NF method/core
transnnmil                            # whole-slide architecture and PANDA evaluations
pathologyfl                           # federated pathology infrastructure
fair-weights-h                        # optional standalone protocol repository
```

The detailed extraction boundaries and history-preserving commands are documented in [`docs/research/repository-split-plan-20260808.md`](docs/research/repository-split-plan-20260808.md).

The split is intentionally **history preserving**. New repositories should be created from filtered history rather than by copying current source trees into unrelated initial commits.

## Evidence restrictions

Do not use the following as current claim evidence:

- historical TransnnMIL fusion or topology interpretations;
- withdrawn canine analyses that predate the corrected fixed-estimand audit;
- unified cross-protocol scoreboard rankings;
- claims that cosine differences prove biological preservation or tissue damage;
- historical slide-independent SCORPION p-values as exact inference;
- PCam claims about diagnoses, lives, clinical benefit, readiness, or state-of-the-art performance;
- any superseded PDF that conflicts with the current primary manuscript or claim boundary.

## Start here

- [Primary foundations manuscript](manuscripts/computational-pathology-foundations-v1/README.md)
- [Focused PA-NF manuscript](paper/paired_acquisition_preprint/README.md)
- [Current claim boundary](CLAIM_BOUNDARY.md)
- [Repository split plan](docs/research/repository-split-plan-20260808.md)
- [Scientific audit remediation ledger](docs/research/scientific-audit-remediation-20260725.md)
- [Research-engineering brief](docs/research/paired-acquisition-research-engineering-brief.md)

## Reproducibility

Raw whole-slide images, large feature archives, checkpoints, and generated run directories remain outside Git.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```
