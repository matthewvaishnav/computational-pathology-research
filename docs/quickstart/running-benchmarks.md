# Running Benchmarks

This page documents the benchmark execution path.

## PCam federated benchmark

The current benchmark plan compares:

- equal weighting
- volume weighting
- prestige weighting
- FAIR-WEIGHTS-H

Recommended first full benchmark:

```text
4 strategies x 3 seeds x 30 rounds
```

## Windows

```cmd
scripts\federated\run_pcam_benchmark.bat
```

## Linux or macOS

```bash
bash scripts/federated/run_pcam_benchmark.sh
```

## Metrics

Current benchmark metrics include:

- global accuracy
- site-wise accuracy
- weight entropy
- effective institution count
- convergence curves

Publication-quality validation should also add:

- AUC
- calibration / ECE
- worst-site sensitivity
