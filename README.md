# Computational Pathology Research

Independent research on **Paired-Acquisition Neural Factorization** and related computational-pathology neural-network studies.

## Current focused preprint

**Paired-Acquisition Neural Factorization for Computational Pathology** is the current corrected focused manuscript.

- [Open the paper PDF directly](https://matthewvaishnav.github.io/computational-pathology-research/paired-acquisition-neural-factorization.pdf)
- [Download the arXiv source package](https://matthewvaishnav.github.io/computational-pathology-research/paired-acquisition-neural-factorization-arxiv-source.zip)
- [Read the manuscript source and release metadata](paper/paired_acquisition_preprint/README.md)
- [Read the authoritative claim boundary](CLAIM_BOUNDARY.md)

> **Corrected focused-preprint status — updated 2026-08-06:** the new manuscript is built from the promoted fold-aware SCORPION and corrected canine fixed-estimand evidence. Earlier PA-NF PDFs remain superseded and are not restored or retroactively validated.

## Central framing

**Paired-Acquisition Neural Factorization** uses matched acquisitions of the same underlying tissue to learn:

- a tissue-oriented branch with reduced scanner recoverability;
- an explicit acquisition branch that retains scanner information.

The bounded question is whether matched acquisitions can support partial structured separation while preserving same-region tissue identity. The work does not claim pure biological factors, complete disentanglement, disease biology, diagnostic improvement, clinical utility, or deployment readiness.

## Current headline evidence

The separately versioned SCORPION capacity-matched campaign completed all **175/175 registered fits**: seven variants, five original-slide-blocked folds, and five seeds. Against the equal-capacity two-branch control, the full model:

- reduced tissue-branch scanner balanced accuracy by `0.3108` (fold-aware 95% interval `[-0.3346, -0.2858]`);
- preserved average and worst same-region retrieval within the registered `0.02` noninferiority margin;
- retained strong acquisition-branch scanner information (`0.8565` accuracy; `+0.6565` above chance).

The corrected canine fixed-estimand audit also records the essential negative result: PA-NF B32/B64 did not establish a feature-space increment over the strongest simple centroid/QR and paired-linear scanner-removal baselines. Increasing the tissue bottleneck from 32 to 64 dimensions increased retrieval and scanner recoverability without a supported corrected-category gain.

These results support partial structured separation under the tested protocols. They do not prove biological purity or clinical value. See the [versioned SCORPION evidence package](evidence/paired_acquisition/scorpion-capacity-matched-20260726/README.md).

## Research program

| Research line | Dataset | Current status |
|---|---|---|
| **Paired-Acquisition Neural Factorization** | SCORPION | Corrected fold-aware and 175-fit capacity-matched evidence is promoted; the focused preprint is current |
| **External paired-acquisition validation** | Multi-Scanner Canine SCC | Corrected fixed-estimand evidence and the complete 450-fit dimensionality × cross-covariance factorial are promoted within their documented boundaries |
| **Pair-repeat allocation** | Controlled latent-factor experiments | Matched-budget allocation evidence remains active under its synthetic-data boundary |
| **TransnnMIL** | PANDA | Canonical fusion and topology code were repaired; historical QWK values do not validate genuine branch fusion and require matched reruns |
| **Institutional weighting** | PANDA and CAMELYON17/WILDS | PANDA stress tests remain simulated institutional experiments; CAMELYON17 weighted logistic models are centralized source-weighting proxies on one held-out center, not full federated-learning validation |
| **PCam** | PatchCamelyon | Single-split patch-classification benchmark only; historical clinical and cross-paper superiority claims were removed |

## Current evidence restrictions

Do not use the following as current claim evidence:

- historical TransnnMIL fusion or topology interpretations;
- withdrawn canine analyses that predate the corrected fixed-estimand audit;
- unified cross-protocol scoreboard rankings;
- claims that cosine differences prove biological preservation or tissue damage;
- historical slide-independent SCORPION p-values as exact inference;
- PCam claims about diagnoses, lives, clinical benefit, readiness, or state-of-the-art performance;
- any superseded PA-NF PDF that conflicts with the current focused preprint or claim boundary.

## Study-specific repositories

The study-specific repositories remain available as historical and technical records. Their older PDFs and summaries may predate the 2026-07-25 audit and are not authoritative when they conflict with this repository's claim boundary.

- [SCORPION paired-acquisition study](https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion)
- [External canine SCC study](https://github.com/matthewvaishnav/paired-acquisition-factorization-caninescc)
- [Pair-repeat allocation study](https://github.com/matthewvaishnav/paired-acquisition-factorization-allocation)

## Start here

- [Current focused preprint](paper/paired_acquisition_preprint/README.md)
- [Current claim boundary](CLAIM_BOUNDARY.md)
- [Research-engineering brief](docs/research/paired-acquisition-research-engineering-brief.md)
- [Scientific audit remediation ledger](docs/research/scientific-audit-remediation-20260725.md)
- [Paired-Acquisition Neural Factorization positioning](docs/research/paired-acquisition-neural-factorization-positioning.md)
- [SCORPION core paired-acquisition results](docs/research/paired-acquisition-factorization-scorpion-results.md)
- [External canine SCC paired-scanner results](docs/research/paired-acquisition-factorization-caninescc-results.md)

## Required next evidence

- execute and promote the prospective paired affine/Procrustes comparison before making FEATMAP-style harmonization claims;
- run repaired TransnnMIL controlled PANDA comparisons;
- create forward-valid releases for every newly promoted numerical claim;
- obtain stronger external human-tissue validation for broader biological claims.

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
