# Computational Pathology Research

Independent research on **Paired-Acquisition Neural Factorization** and related
computational-pathology neural-network studies.

> **Scientific audit status — updated 2026-08-02:** several historical analyses,
> architectural interpretations, and public claims were withdrawn or corrected.
> Start with the [current claim boundary](CLAIM_BOUNDARY.md) and the
> [remediation ledger](docs/research/scientific-audit-remediation-20260725.md).
> Older manuscripts, PDFs, result summaries, website pages, and child packages
> are subordinate to those documents whenever they conflict.

## Central framing

**Paired-Acquisition Neural Factorization** uses matched acquisitions of the same
underlying tissue to learn:

- a tissue-oriented branch with reduced scanner recoverability;
- an explicit acquisition branch that retains scanner information; and
- a joint reconstruction path that discourages simple information deletion.

The bounded question is whether matched acquisitions support partial structured
separation while retaining same-region identity. The work does not claim pure
biological factors, complete disentanglement, novelty or priority over the
broader literature, disease biology, diagnostic improvement, clinical utility,
or deployment readiness.

## Current headline evidence

The separately versioned SCORPION capacity-matched campaign completed all
**175/175 registered fits**: seven variants, five original-slide-blocked folds,
and five seeds. Against the equal-capacity two-branch control, the full model:

- reduced tissue-branch scanner balanced accuracy by `0.3108`
  (fold-aware 95% interval `[-0.3346, -0.2858]`);
- preserved average and worst same-region retrieval within the registered
  `0.02` noninferiority margin; and
- retained strong acquisition-branch scanner information
  (`0.8565` accuracy; `+0.6565` above chance).

These results support partial structured separation under the tested protocol.
They do not prove biological purity, causal factor recovery, or clinical value.
See the
[versioned evidence package](evidence/paired_acquisition/scorpion-capacity-matched-20260726/README.md).

## Research program

| Research line | Dataset | Current status |
|---|---|---|
| **Paired-Acquisition Neural Factorization** | SCORPION | Corrected fold-aware and 175-fit capacity-matched evidence is promoted within the claim boundary |
| **External paired-acquisition validation** | Multi-Scanner Canine SCC | Corrected fixed-estimand evidence and the complete 450-fit dimensionality × cross-covariance factorial are promoted within documented limits |
| **Paired affine comparison** | SCORPION | Prospective translation, Procrustes, affine, and ridge-affine protocol is specified but has not been promoted as numerical evidence |
| **Crossed-target synthetic diagnostics** | Controlled known-factor data | Draft exploratory PRs test scanner-prototype and unseen-identity intervention behavior; these are not pathology-domain evidence and are not current public claim evidence |
| **TransnnMIL** | PANDA | Canonical fusion and topology code were repaired; historical QWK values do not validate genuine branch fusion and require matched reruns |
| **Institutional weighting** | PANDA and CAMELYON17/WILDS | Simulated or centralized mechanism studies only; not full federated-learning or clinical validation |
| **PCam** | PatchCamelyon | Single-split patch-classification engineering benchmark only; historical clinical and cross-paper superiority claims remain withdrawn |

## Current evidence restrictions

Do not use the following as current claim evidence:

- historical TransnnMIL fusion or topology interpretations;
- withdrawn canine analyses that predate the corrected fixed-estimand audit;
- unified cross-protocol scoreboard rankings;
- claims that cosine differences prove biological preservation or tissue damage;
- historical slide-independent SCORPION p-values as exact inference;
- unpromoted smoke runs or draft-PR outputs;
- PCam claims about diagnoses, lives, clinical benefit, readiness, or
  state-of-the-art performance; or
- claims that the repository establishes novelty, patentability, or priority.

## Start here

- [Current claim boundary](CLAIM_BOUNDARY.md)
- [Current status](docs/CURRENT_STATUS.md)
- [Research-engineering brief](docs/research/paired-acquisition-research-engineering-brief.md)
- [Prospective paired affine comparison](docs/research/paired-affine-comparison-protocol.md)
- [Scientific audit remediation ledger](docs/research/scientific-audit-remediation-20260725.md)
- [SCORPION core paired-acquisition results](docs/research/paired-acquisition-factorization-scorpion-results.md)
- [External canine SCC paired-scanner results](docs/research/paired-acquisition-factorization-caninescc-results.md)

## Required next evidence

- execute and independently validate the prospective paired affine/Procrustes
  comparison before any comparative harmonization claim;
- complete the crossed-target synthetic studies before interpreting their
  intervention behavior, while retaining the synthetic-only boundary;
- run repaired TransnnMIL controlled PANDA comparisons;
- create forward-valid releases for every additional numerical claim; and
- obtain stronger external human-tissue validation before broader biological
  claims.

## Reproducibility

Raw whole-slide images, large feature archives, checkpoints, and generated run
directories remain outside Git.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```
