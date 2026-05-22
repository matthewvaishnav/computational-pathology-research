# Getting Started

This page is the MkDocs entrypoint for installing and exploring the computational pathology research platform.

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version before running experiments.

## First checks

Run the focused FAIR-WEIGHTS-H tests:

```bash
pytest tests/federated/test_fair_weights_h.py
pytest tests/federated/test_weighted_aggregator.py
```

Run the core federated integration checks:

```bash
pytest tests/federated/test_fl_integration.py
```

## Documentation map

- [Repository overview](repository-overview.md)
- [Platform overview](platform-overview.md)
- [Performance comparison](results/performance-comparison.md)
- [FAIR-WEIGHTS-H theory](theory/fair-weights-h.md)
- [Federated test status](engineering/testing-status.md)
