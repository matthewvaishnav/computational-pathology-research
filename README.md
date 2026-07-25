# Computational Pathology Research

Independent research on **Paired-Acquisition Neural Factorization** and related
computational-pathology neural-network studies.

> **Scientific audit hold — 2026-07-25:** several secondary analyses,
> architectural interpretations, and public claims were withdrawn or corrected.
> Start with the [current claim boundary](CLAIM_BOUNDARY.md) and the
> [remediation ledger](docs/research/scientific-audit-remediation-20260725.md).
> Older manuscripts, PDFs, result summaries, and child packages are subordinate
> to those documents until corrected reruns are published.

## Central framing

**Paired-Acquisition Neural Factorization** uses matched acquisitions of the same
underlying tissue to learn:

- a tissue-oriented branch with reduced scanner recoverability;
- an explicit acquisition branch that retains scanner information.

The current bounded question is whether matched acquisitions can support partial
structured separation while preserving same-region tissue identity. The work
does not claim pure biological factors, complete disentanglement, disease
biology, diagnostic improvement, clinical utility, or deployment readiness.

## Research program

| Research line | Dataset | Current status |
|---|---|---|
| **Paired-Acquisition Neural Factorization** | SCORPION | Slide-blocked paired-scanner and frozen cross-backbone evidence remains active; fold-aware inference and capacity-matched ablations are pending |
| **External paired-acquisition validation** | Multi-Scanner Canine SCC | Sample-blocked scanner, agreement, retrieval, and pair-integrity evidence remains active; historical category metrics are withdrawn pending the fixed-estimand audit |
| **Pair-repeat allocation** | Controlled latent-factor experiments | Matched-budget allocation evidence remains active under its synthetic-data boundary |
| **TransnnMIL** | PANDA | Canonical fusion and topology code were repaired; historical QWK values do not validate genuine branch fusion and require matched reruns |
| **Institutional weighting** | PANDA and CAMELYON17/WILDS | PANDA stress tests remain simulated institutional experiments; CAMELYON17 weighted logistic models are centralized source-weighting proxies on one held-out center, not full federated-learning validation |
| **PCam** | PatchCamelyon | Single-split patch-classification benchmark only; historical clinical and cross-paper superiority claims were removed |

## Current evidence restrictions

Do not use the following as current claim evidence:

- historical TransnnMIL fusion or topology interpretations;
- historical canine category probe, neighbourhood purity, or category/scanner
  trade-off numbers;
- unified cross-protocol scoreboard rankings;
- claims that cosine differences prove biological preservation or tissue damage;
- historical slide-independent SCORPION p-values as exact inference;
- PCam claims about diagnoses, lives, clinical benefit, readiness, or
  state-of-the-art performance.

## Study-specific repositories

The study-specific repositories remain available as historical and technical
records. Their older PDFs and summaries may predate the 2026-07-25 audit and are
not authoritative when they conflict with this repository's claim boundary.

- [SCORPION paired-acquisition study](https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion)
- [External canine SCC study](https://github.com/matthewvaishnav/paired-acquisition-factorization-caninescc)
- [Pair-repeat allocation study](https://github.com/matthewvaishnav/paired-acquisition-factorization-allocation)

## Start here

- [Current claim boundary](CLAIM_BOUNDARY.md)
- [Scientific audit remediation ledger](docs/research/scientific-audit-remediation-20260725.md)
- [Paired-Acquisition Neural Factorization positioning](docs/research/paired-acquisition-neural-factorization-positioning.md)
- [SCORPION core paired-acquisition results](docs/research/paired-acquisition-factorization-scorpion-results.md)
- [External canine SCC paired-scanner results](docs/research/paired-acquisition-factorization-caninescc-results.md)

## Required reruns

- fixed-estimand canine biological-label audit;
- fold-aware SCORPION inference;
- 175-fit capacity-matched SCORPION objective ablations;
- repaired TransnnMIL controlled PANDA comparisons;
- locked dimension-by-cross-covariance factorial;
- forward-valid releases for all newly promoted claims.

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
