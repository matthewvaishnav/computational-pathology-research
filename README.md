# Computational Pathology Research

Independent research on representation identifiability, whole-slide modeling, and external-center validation in computational pathology.

**Public research PDF:** [PathoAlign and a Broader Computational-Pathology Research Program](https://matthewvaishnav.github.io/computational-pathology-research/pathoalign-computational-pathology-research.pdf)

> Research only. This repository is not clinically validated, diagnostic software, or intended for patient-care use.

## Research program

| Research line | Dataset | Question | Current evidence |
|---|---|---|---|
| **PathoAlign** | SCORPION | Can paired acquisitions separate tissue identity from scanner signal? | Frozen five-fold and cross-backbone transfer across DINOv2, Phikon, and ResNet50 |
| **TransnnMIL** | PANDA | How should pathology foundation-model feature bags be aggregated for whole-slide grading? | Stabilized multi-seed PANDA validation against mean pooling and AttentionMIL |
| **PathologyFL** | CAMELYON17 / WILDS | How do institutional weighting rules affect held-out-center generalization? | Feature-level external-center comparisons and controlled aggregation stress tests |

## Start here

- [PathoAlign cross-backbone results](docs/research/scorpion-pathoalign-crossbackbone-results.md)
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
