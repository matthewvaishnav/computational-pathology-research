# Track A: Camelyon17 external validation plan

## Goal

Test whether the dominant-site / site-signal-alignment detector result survives a naturally multi-site pathology setting, rather than only simulated site splits over PANDA-derived features.

The target validation question is:

> Does a site-dominance detector-switch policy improve robustness when hospital/site structure is natural, observable, and clinically plausible?

## Dataset target

Primary target: **Camelyon17 via WILDS**.

Why this dataset:

- It is pathology/histology-based.
- It has hospital/domain structure.
- It is already framed as a distribution-shift benchmark.
- It is a better match for site-signal-alignment validation than random simulated client splits.

## Claim boundary

This validation does not turn the project into clinical software. The goal is to test whether the current failure-mode framing and detector-switch idea remain useful when site shift is naturally present.

Supported if successful:

- The detector-switch idea has external support beyond PANDA simulated-site stress.
- Site structure can be used to audit aggregation assumptions.
- Sample-volume dominance should be compared against site-aware alternatives.

Not supported:

- Real hospital deployment readiness.
- Diagnostic safety.
- Universal detector calibration.
- Institution-level reliability ranking.

## Minimum viable experiment

### Stage 1: dataset loader smoke test

- Install WILDS.
- Load Camelyon17 labeled dataset.
- Confirm metadata fields, especially hospital/domain labels.
- Print train/val/test split counts by hospital/domain.
- Save a metadata summary CSV.

Expected artifact:

```text
docs/results/camelyon17_dataset_audit.md
```

### Stage 2: centralized baseline

Train or evaluate a simple centralized baseline using the WILDS loader.

Minimum options:

1. Use WILDS default model/evaluation if available.
2. Use a small ResNet-style baseline.
3. If compute is limited, first build a feature-cache pipeline and use linear/MLP heads.

Expected artifact:

```text
results/camelyon17/centralized_baseline.csv
```

### Stage 3: federated client construction

Use hospital/domain metadata as simulated federated clients.

Client definition:

```text
client_id = hospital/domain metadata field
```

Record:

- client sample counts
- class balance per client
- train/validation split membership
- dominant client by sample count
- per-client validation metrics

Expected artifact:

```text
results/camelyon17/client_audit.csv
```

### Stage 4: federated baselines

Compare at least:

| Strategy | Purpose |
|---|---|
| FedAvg | sample-volume dominance baseline |
| Equal client weighting | removes sample-volume dominance |
| FedProx | standard heterogeneity-aware FL baseline |
| q-FedAvg if feasible | fairness/heterogeneity-oriented baseline |
| Robust aggregation if feasible | median or trimmed-mean robustness comparison |
| Oracle balanced/site-aware strategy | upper-bound diagnostic, not deployable claim |
| Detector-switch strategy | current project hypothesis |

Expected artifact:

```text
results/camelyon17/federated_baselines.csv
```

### Stage 5: detector transfer test

Start with the existing detector logic:

- global metric degradation
- worst-site metric degradation
- site-metric spread
- absolute/severe error signals where applicable

Because Camelyon17 is binary tumor classification rather than ordinal ISUP grading, ordinal-error diagnostics do not transfer directly. Replace ordinal-specific diagnostics with binary analogues:

| PANDA ordinal diagnostic | Camelyon17 binary analogue |
|---|---|
| global QWK low | global AUROC / accuracy / F1 low |
| worst-site QWK low | worst-site AUROC / accuracy / F1 low |
| site-QWK spread high | site metric spread high |
| mean absolute error high | cross-entropy / calibration error / error rate high |
| severe ordinal error high | high-confidence wrong predictions / false-negative rate high |

Expected artifact:

```text
results/camelyon17/detector_transfer.csv
```

## Success criteria

A successful external validation does not require the detector to dominate every baseline.

Minimum meaningful success:

1. The code runs end-to-end on natural hospital/domain splits.
2. FedAvg, equal weighting, and detector-switch results are all reported.
3. The detector has low clean/ID over-switching.
4. The detector improves or protects worst-site performance under natural domain shift.
5. Failure cases are documented honestly.

Strong success:

1. Detector-switch improves worst-site metric without large global-metric cost.
2. Result is stable across at least 5 seeds or repeated splits where feasible.
3. It outperforms both FedAvg and equal weighting in the key stress setting.
4. Results survive a nearby calibration-sensitivity sweep.

## First implementation tasks

1. Add WILDS dependency or optional install instructions.
2. Create a loader script:

```text
scripts/camelyon17/audit_camelyon17_wilds.py
```

3. Create a metadata audit output:

```text
results/camelyon17/client_audit.csv
```

4. Add a smoke test that can run without downloading the full dataset by mocking WILDS metadata.
5. Add a README section explaining how Camelyon17 validation relates to the PANDA simulated-site result.

## First command sequence

```powershell
python -m pip install wilds
mkdir scripts\camelyon17
mkdir results\camelyon17
python scripts\camelyon17\audit_camelyon17_wilds.py --download false
```

If the dataset is not present locally, rerun with:

```powershell
python scripts\camelyon17\audit_camelyon17_wilds.py --download true
```

## Expert-review framing

When sending this to an expert, do not claim clinical readiness. Say:

> I have a simulated-site PANDA result showing a sample-volume / site-signal-alignment failure mode in federated pathology. I am now validating whether the same detector-switch idea survives on naturally multi-site Camelyon17/WILDS hospital-domain structure. I would value critique on whether the validation design is meaningful and what baselines are missing.
