# Computational Pathology Research

Independent research on **Paired-Acquisition Neural Factorization** for computational pathology representation identifiability.

## Central framing

I use **Paired-Acquisition Neural Factorization** as the logical method name. It is not a brand name. The method uses paired acquisitions of the same underlying tissue to factor frozen pathology embeddings into:

- a scanner-suppressed tissue factor;
- an acquisition-specific factor.

The research question is whether paired acquisitions can preserve tissue identity while reducing linearly recoverable scanner signal in the tissue factor.

I am not claiming that this proves disease biology. I am making a narrower representation-identifiability claim.

## Research program

| Research line | Dataset | Question | Current evidence |
|---|---|---|---|
| **Paired-Acquisition Neural Factorization** | SCORPION | Can paired acquisitions separate tissue identity from scanner signal? | Frozen five-fold and cross-backbone transfer across DINOv2, Phikon, and ResNet50 |
| **External paired-acquisition validation** | Multi-Scanner Canine SCC | Does the locked objective transfer to an independent paired-scanner benchmark? | DINOv2 five-fold sample-blocked external test: scanner probe 0.7529 to 0.3614, cosine improved, retrieval preserved |
| **Pair-repeat allocation** | Matched-budget controls | Is representation quality driven more by unique biological pair diversity or repeated anchors? | More unique biological pairs improved biological consistency and factor separation at matched budgets of 6,400 and 12,800 pair presentations |
| **CAMELYON17 center-subspace projection mechanism branch** | CAMELYON17 / WILDS | Can source-center leakage be attenuated while preserving tumor signal? | v7 supervised center-subspace projection reduced center accuracy while preserving tumor AUC near 0.9903 |
| **TransnnMIL** | PANDA | How should foundation-model feature bags be aggregated for whole-slide grading? | Stabilized multi-seed PANDA validation against mean pooling and AttentionMIL |
| **Federated Learning for Computational Pathology** | CAMELYON17 / WILDS | How do institutional weighting rules affect held-out-center generalization? | Feature-level external-center comparisons and controlled aggregation stress tests |

## Study-specific packages

| Study package | Scope | Repository status |
|---|---|---|
| **SCORPION core paired-acquisition factorization** | Primary paired-scanner method study on 48 original human H&E slides, five scanners, and three feature families | Repository target: `paired-acquisition-factorization-scorpion` |
| **External canine SCC validation** | Independent five-scanner paired-acquisition validation of the locked neural factorization objective | Repository target: `paired-acquisition-factorization-caninescc` |
| **Pair-repeat allocation study** | Matched-budget test of whether unique biological pair diversity improves factor separation more than repeated-anchor allocation | Repository target: `paired-acquisition-factorization-allocation` |

Child-package repository links stay unlinked here until the package URLs are verified from the public view.

## Start here

- [Paired-Acquisition Neural Factorization positioning](docs/research/paired-acquisition-neural-factorization-positioning.md)
- [SCORPION core paired-acquisition factorization results](docs/research/paired-acquisition-factorization-scorpion-results.md)
- [External canine SCC validation results](docs/research/paired-acquisition-factorization-caninescc-results.md)
- [Claim boundary](CLAIM_BOUNDARY.md)

## Reproducibility

Raw whole-slide images, large feature archives, checkpoints, and generated run directories stay outside Git.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```
