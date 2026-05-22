# Reproducing Experiments

This page gives a clean path for reproducing the main research checks.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Run focused tests

```bash
pytest tests/federated/test_fair_weights_h.py
pytest tests/federated/test_weighted_aggregator.py
pytest tests/federated/test_fl_integration.py
```

## 3. Run smoke validation

Synthetic and PCam smoke tests validate execution, logging, aggregation, and numerical stability.

```bash
python scripts/federated/run_pcam_federated_smoke.py --weighting equal --rounds 5 --num-sites 5
python scripts/federated/run_pcam_federated_smoke.py --weighting volume --rounds 5 --num-sites 5
python scripts/federated/run_pcam_federated_smoke.py --weighting prestige --rounds 5 --num-sites 5
python scripts/federated/run_pcam_federated_smoke.py --weighting fair_weights_h --rounds 5 --num-sites 5
```

## 4. Interpret correctly

Smoke tests prove pipeline execution. They do not prove clinical performance or FAIR-WEIGHTS-H superiority.
