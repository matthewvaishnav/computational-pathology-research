# Computational Pathology Research

Independent research on representation identifiability, paired-acquisition neural factorization, whole-slide modeling, and external-center validation in computational pathology.

**Public research PDF:** [Paired-Acquisition Neural Factorization and a Broader Computational-Pathology Research Program](https://matthewvaishnav.github.io/computational-pathology-research/pathoalign-computational-pathology-research.pdf)

> Research only. This repository is not clinically validated, diagnostic software, or intended for patient-care use.

## Central framing

I use PathoAlign to treat computational pathology robustness as a representation-audit problem: frozen pathology embeddings can entangle tissue morphology with acquisition provenance, including scanner, stain, center, preparation, and workflow signals. The paired-acquisition neural factorization work asks whether matched views of the same underlying tissue can separate a scanner-suppressed tissue factor from an acquisition-specific factor, and then audits whether each branch contains the intended information.

I am not claiming that this repository proves disease biology. The supported claim is narrower: PathoAlign provides a paired-acquisition framework for testing and improving whether pathology representations preserve tissue identity while suppressing linearly recoverable scanner signal in the tissue branch.

## Research program

| Research line | Dataset | Question | Current evidence |
|---|---|---|---|
| **Paired-Acquisition Neural Factorization** | SCORPION | Can paired acquisitions separate tissue identity from scanner signal? | Frozen five-fold and cross-backbone transfer across DINOv2, Phikon, and ResNet50 |
| **Paired-Acquisition Neural Factorization external validation** | Multi-Scanner Canine SCC | Does the locked paired-acquisition neural factorization objective transfer to an independent paired-scanner benchmark? | DINOv2 five-fold sample-blocked external test: scanner probe 0.7529 to 0.3614, cosine improved, retrieval preserved |
| **Pair-repeat allocation** | Synthetic paired-acquisition allocation controls | Under matched pair-presentation budget, is representation quality driven more by unique biological pair diversity or repeated anchors? | More unique biological pairs improved biological consistency and factor separation at matched budgets of 6,400 and 12,800 pair presentations |
| **CAMELYON17 center-subspace projection mechanism branch** | CAMELYON17 / WILDS | Can source-center leakage be attenuated while preserving tumor signal? | v3/v4/v5 adversarial removal failed; v7 supervised center-subspace projection reduced center accuracy 0.8946 to 0.7636 while preserving tumor AUC near 0.9903 |
| **TransnnMIL** | PANDA | How should pathology foundation-model feature bags be aggregated for whole-slide grading? | Stabilized multi-seed PANDA validation against mean pooling and AttentionMIL |
| **Federated Learning for Computational Pathology** | CAMELYON17 / WILDS | How do institutional weighting rules affect held-out-center generalization? | Feature-level external-center comparisons and controlled aggregation stress tests |

## Study-specific packages

The main PDF is the research-program overview. Focused study repositories and PDFs carry the detailed protocols, frozen result tables, and reproduction scripts.

| Study package | Scope | Links |
|---|---|---|
| **Paired-Acquisition Neural Factorization external canine SCC validation** | Independent five-scanner canine SCC paired-acquisition validation of the locked paired-acquisition neural factorization objective | [Repository](https://github.com/matthewvaishnav/pathoalign-external-caninescc) · [Study PDF](https://matthewvaishnav.github.io/pathoalign-external-caninescc/pathoalign-external-caninescc.pdf) |
| **PathoAlign pair-repeat allocation study** | Matched-budget test of whether unique biological pair diversity improves factor separation more than repeated-anchor allocation | [Repository](https://github.com/matthewvaishnav/pathoalign-pair-repeat-allocation) |

## Start here

- [PathoAlign representation-audit positioning](docs/research/pathoalign-representation-audit-positioning.md)
- [Paired-Acquisition Neural Factorization cross-backbone results](docs/research/scorpion-pathoalign-crossbackbone-results.md)
- [Paired-Acquisition Neural Factorization external canine SCC validation results](docs/research/pathoalign-external-caninescc-results.md)
- [CAMELYON17 center-projection mechanism result](docs/benchmarks/camelyon17_center_projection_negative_to_positive_mechanism.md)
- [PANDA TransnnMIL stability results](docs/results/panda-transnnmil-stability.md)
- [CAMELYON17 external-center validation](docs/research/camelyon17-external-center-validation-note.md)
- [Plain-English research summary](docs/outreach/plain-english-summary.md)
- [Technical research summary](docs/outreach/technical-summary.md)
- [Claim boundary](CLAIM_BOUNDARY.md)

## Repository map

```text
src/          maintained Python package and model code
experiments/  reproducible experiment runners and fixed protocols
scripts/      data preparation, evaluation, auditing, and utilities
tests/        automated tests
data/         small manifests and split metadata; no raw datasets
results/      compact, reviewable research evidence
paper/        LaTeX paper source and build tooling
docs/         canonical documentation, results notes, outreach, and archive
configs/      active experiment configurations
```

Legacy product, deployment, business, and dated status material belongs under `docs/archive/`, not at the repository root. See [repository organization](docs/REPO_ORGANIZATION.md).

## Reproducibility

The repository keeps code, split metadata, compact summaries, and claim-supporting artifacts together. Raw whole-slide images, large feature archives, checkpoints, and generated run directories stay outside Git.

Typical development setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```

Dataset-specific commands and frozen protocols are documented beside their corresponding research notes.

## Citation and license

- [Citation metadata](CITATION.cff)
- [License](LICENSE)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
