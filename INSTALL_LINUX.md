# Linux installation guide

This repository is easiest to run on Linux. The commands below target Ubuntu/Debian-style systems.

## 1. Clone and enter the repo

```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research
```

## 2. Install dependencies

Run the bootstrap script:

```bash
bash scripts/setup/install_linux_dependencies.sh
```

Then activate the virtual environment:

```bash
source .venv/bin/activate
```

The script installs:

- system packages for Python builds, OpenSlide, OpenGL runtime libraries, and Redis
- editable Python package install with `dev`, `federated`, and `ml` extras
- common full-test-suite dependencies such as `sqlalchemy`, `hypothesis`, and `pytest-cov`

## CUDA / GPU note

The default script does **not** force-install PyTorch because CUDA wheels depend on your NVIDIA driver and CUDA runtime.

For CPU-only PyTorch, run:

```bash
INSTALL_TORCH_CPU=1 bash scripts/setup/install_linux_dependencies.sh
```

For CUDA, install the PyTorch wheel recommended by the official PyTorch selector, then run the dependency script normally:

```bash
# example only; choose the command for your CUDA version from pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
bash scripts/setup/install_linux_dependencies.sh
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')
PY
```

## 3. Quick checks

Run focused tests without coverage first:

```bash
python -m pytest tests/test_pathology_fl_privacy_regressions.py -q -o addopts=''
python -m pytest tests/test_secure_aggregation_contract.py -q -o addopts=''
python -m pytest tests/test_federated_learning.py -q -o addopts=''
```

Check the PANDA FAIR-WEIGHTS-H runner:

```bash
python scripts/experiments/run_fair_weights_h_panda_feature_stress.py --help
```

## 4. PANDA feature cache path

On Linux, put large PANDA caches outside the git repository, for example:

```bash
mkdir -p /data/panda_cache
```

Example run using a cached feature file:

```bash
python scripts/experiments/run_fair_weights_h_panda_feature_stress.py \
  --feature-cache /data/panda_cache/panda_phikon_mean_features_3000.npz \
  --output-dir results/ordinal_harm_smoke \
  --rounds 1 \
  --large-site-label-flip 0.25 \
  --seed 42 \
  --device cuda \
  --strategies fedavg cross_site_blend_50 ordinal_harm_blend_50 adaptive_ordinal_harm \
  --save-predictions
```

Do not commit large `.npz`, `.h5`, or full `metrics.json` artifacts unless intentionally documenting a small result summary.

## 5. Full test-suite warning

The repository contains a broad legacy/platform test suite. A full `pytest tests` run may require optional services and dependencies, including Redis, SQLAlchemy-backed database setup, PACS/DICOM mocks, and integration fixtures.

For active research work, prefer focused tests and experiment smoke runs first.
