# Computational Pathology Research

Independent research spanning representation learning, whole-slide neural aggregation, institutional aggregation, and scientific provenance in computational pathology.

## Primary public preprint

**Accountable Neural Aggregation in Computational Pathology: From Paired-Acquisition Representations to Whole-Slide and Institutional Learning** is the primary program-level manuscript.

- [Open the primary paper PDF directly](https://matthewvaishnav.github.io/computational-pathology-research/accountable-neural-aggregation-in-computational-pathology.pdf)
- [Open the supplement](https://matthewvaishnav.github.io/computational-pathology-research/accountable-neural-aggregation-in-computational-pathology-supplement.pdf)
- [Download the primary arXiv source package](https://matthewvaishnav.github.io/computational-pathology-research/accountable-neural-aggregation-arxiv-source.zip)
- [Read the manuscript source and release metadata](manuscripts/computational-pathology-foundations-v1/README.md)
- [Read the authoritative claim boundary](CLAIM_BOUNDARY.md)

The root GitHub Pages URL opens the primary PDF directly.

## Secondary focused manuscript

**Paired-Acquisition Neural Factorization for Computational Pathology** remains available as a narrower supporting paper:

- [Open the focused PA-NF PDF](https://matthewvaishnav.github.io/computational-pathology-research/paired-acquisition-neural-factorization.pdf)
- [Read the focused manuscript source](paper/paired_acquisition_preprint/README.md)

Earlier PA-NF PDFs remain superseded. The focused manuscript does not replace the program-level foundations paper.

## Research program

| Research line | Dataset / setting | Current status |
|---|---|---|
| **Paired-Acquisition Neural Factorization** | SCORPION and multi-scanner canine SCC | Corrected fold-aware and fixed-estimand evidence is promoted; results support partial structured separation, including negative findings against strong simple baselines |
| **TransnnMIL** | PANDA | Authored multibranch whole-slide architecture is implemented; historical scores do not validate repaired fusion and matched controlled reruns remain pending |
| **PathologyFL** | Federated-learning research infrastructure | Implemented infrastructure and integration behavior; not a real multi-center deployment validation |
| **FAIR-WEIGHTS-H** | Institutional weighting protocol | Proposed and partially implemented protocol; prospective fairness and performance superiority remain unestablished |
| **Institutional mechanism studies** | PANDA and CAMELYON17/WILDS | Simulated stress tests and centralized held-out-center source-weighting proxies, not complete federated clinical validation |
| **PCam** | PatchCamelyon | Single-split patch-classification and engineering benchmark only |
| **Scientific provenance** | Repository-wide | Immutable evidence packages, claim ledgers, hostile review, exact artifact recovery, and fail-closed validation |

## Current PA-NF evidence boundary

Paired-Acquisition Neural Factorization uses matched acquisitions of the same tissue region to learn:

- a tissue-oriented branch with reduced scanner recoverability;
- an explicit acquisition branch that retains scanner information.

The separately versioned SCORPION capacity-matched campaign completed all **175/175 registered fits**. Against an equal-capacity two-branch control without scanner objectives, the full model reduced tissue-branch scanner balanced accuracy while preserving same-region retrieval within the registered noninferiority margin and retaining strong acquisition-branch scanner information.

The corrected canine fixed-estimand audit records the essential negative result: PA-NF B32/B64 did not establish a feature-space increment over the strongest simple centroid/QR and paired-linear scanner-removal baselines. Increasing the tissue bottleneck from 32 to 64 dimensions increased retrieval and scanner recoverability without a supported corrected-category gain.

These results support partial structured separation under tested protocols. They do not establish pure biological factors, complete scanner invariance, diagnostic improvement, clinical utility, or deployment readiness.

## Evidence restrictions

Do not use the following as current claim evidence:

- historical TransnnMIL fusion or topology interpretations;
- withdrawn canine analyses that predate the corrected fixed-estimand audit;
- unified cross-protocol scoreboard rankings;
- claims that cosine differences prove biological preservation or tissue damage;
- historical slide-independent SCORPION p-values as exact inference;
- PCam claims about diagnoses, lives, clinical benefit, readiness, or state-of-the-art performance;
- any superseded PDF that conflicts with the current primary manuscript or claim boundary.

## Study-specific repositories

- [SCORPION paired-acquisition study](https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion)
- [External canine SCC study](https://github.com/matthewvaishnav/paired-acquisition-factorization-caninescc)
- [Pair-repeat allocation study](https://github.com/matthewvaishnav/paired-acquisition-factorization-allocation)

## Start here

- [Primary foundations manuscript](manuscripts/computational-pathology-foundations-v1/README.md)
- [Focused PA-NF manuscript](paper/paired_acquisition_preprint/README.md)
- [Current claim boundary](CLAIM_BOUNDARY.md)
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
