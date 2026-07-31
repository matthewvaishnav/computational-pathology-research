# Synthetic crossed-factor identifiability diagnostic

## Status

Post-confirmatory exploratory diagnosis only. This work does not modify or reinterpret the frozen private crossed-target campaign.

## Why v2 supersedes the first smoke analysis

The first smoke run revealed that the original counterfactual delta measured biological identity retention but did not independently establish acquisition transfer. An unconstrained joint autoencoder could score positively by retaining the source biological identity while largely ignoring the acquisition donor.

The corrected v2 diagnostic therefore requires two independent counterfactual contrasts:

1. **Biology retention:** the crossed output must be closer to the correct target than to the donor-identity target.
2. **Acquisition transfer:** the crossed output must be closer to the correct target than to the same biological identity under its original source scanner.

A run passes crossed factorization only when both bootstrap intervals are above zero, a majority of identities succeed on both axes, and the known biological/acquisition factors are allocated to the intended branches.

## Controls

- `joint_autoencoder`: negative control; arbitrary code split should not pass the complete semantic gate.
- `oracle_supervised`: positive control; explicitly supervised factor branches and decoder. In smoke mode it receives at least 250 epochs so control under-optimization cannot block interpretation.

The full-grid execution gate remains closed unless every oracle seed passes factor allocation and two-axis transfer while every joint-autoencoder seed is rejected by the complete crossed-factorization gate.

## Entry point

```powershell
py experiments/paired_acquisition/run_synthetic_crossed_factor_identifiability_v2.py `
    --mode smoke `
    --device cuda `
    --output-root results/synthetic_crossed_factor_identifiability_v2_smoke
```

The original v1 output remains useful as a diagnostic artifact documenting why a single biology-retention contrast was insufficient, but it must not be treated as the final experiment.
