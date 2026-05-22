# FAIR-WEIGHTS-H Synthetic Perturbation Report

**Status:** Synthetic engineering check; not clinical validation.

This report compares equal, volume, prestige, and FAIR-WEIGHTS-H weighting under deterministic perturbation scenarios.

| Scenario                         |       Strategy | Baseline rural w | Perturbed rural w | Delta rural w | Delta entropy | Delta effective N |
| -------------------------------- | -------------: | ---------------: | ----------------: | ------------: | ------------: | ----------------: |
| rural_uncertainty_spike          |          equal |           0.2500 |            0.2500 |       +0.0000 |       +0.0000 |           +0.0000 |
| rural_uncertainty_spike          |         volume |           0.0600 |            0.0600 |       +0.0000 |       +0.0000 |           +0.0000 |
| rural_uncertainty_spike          |       prestige |           0.1509 |            0.1509 |       +0.0000 |       +0.0000 |           +0.0000 |
| rural_uncertainty_spike          | fair_weights_h |           0.3498 |            0.2460 |       -0.1038 |       +0.0177 |           +0.2005 |
| rural_rare_population_enrichment |          equal |           0.2500 |            0.2500 |       +0.0000 |       +0.0000 |           +0.0000 |
| rural_rare_population_enrichment |         volume |           0.0600 |            0.0600 |       +0.0000 |       +0.0000 |           +0.0000 |
| rural_rare_population_enrichment |       prestige |           0.1509 |            0.1509 |       +0.0000 |       +0.0000 |           +0.0000 |
| rural_rare_population_enrichment | fair_weights_h |           0.3498 |            0.3966 |       +0.0467 |       -0.0195 |           -0.2088 |
| cancer_center_scanner_shift      |          equal |           0.2500 |            0.2500 |       +0.0000 |       +0.0000 |           +0.0000 |
| cancer_center_scanner_shift      |         volume |           0.0600 |            0.0600 |       +0.0000 |       +0.0000 |           +0.0000 |
| cancer_center_scanner_shift      |       prestige |           0.1509 |            0.1509 |       +0.0000 |       +0.0000 |           +0.0000 |
| cancer_center_scanner_shift      | fair_weights_h |           0.3498 |            0.3573 |       +0.0074 |       -0.0030 |           -0.0326 |
| community_quality_degradation    |          equal |           0.2500 |            0.2500 |       +0.0000 |       +0.0000 |           +0.0000 |
| community_quality_degradation    |         volume |           0.0600 |            0.0600 |       +0.0000 |       +0.0000 |           +0.0000 |
| community_quality_degradation    |       prestige |           0.1509 |            0.1509 |       +0.0000 |       +0.0000 |           +0.0000 |
| community_quality_degradation    | fair_weights_h |           0.3498 |            0.3837 |       +0.0339 |       -0.0221 |           -0.2072 |

## Interpretation Guardrail

These numbers only describe behavior of the synthetic weighting functions. They do not establish model performance, clinical utility, fairness guarantees, or regulatory readiness.

## Key Observations

### Uncertainty Penalty Response

When the rural hospital experiences an uncertainty spike, FAIR-WEIGHTS-H reduces its weight from 0.3498 to 0.2460 (-0.1038), while equal, volume, and prestige weights remain unchanged. This demonstrates the uncertainty penalty mechanism is working as designed.

### Rare Population Enrichment

When the rural hospital gains rare population coverage, FAIR-WEIGHTS-H increases its weight from 0.3498 to 0.3966 (+0.0467), rewarding useful uniqueness. Other strategies ignore this signal.

### Scanner Shift Detection

When the cancer center experiences a scanner shift (increasing uniqueness and uncertainty), FAIR-WEIGHTS-H slightly increases its weight (+0.0074), suggesting the uniqueness signal outweighs the uncertainty penalty in this scenario.

### Quality Degradation Response

When the community hospital's quality degrades, FAIR-WEIGHTS-H increases the rural hospital's weight from 0.3498 to 0.3837 (+0.0339), redistributing influence away from the degraded institution.

## Limitations

- **Synthetic data only**: These scenarios use deterministic test inputs, not real institutional data.
- **No model training**: Weights are computed but not used to train actual federated models.
- **No clinical outcomes**: No patient-level or diagnostic performance metrics are measured.
- **No fairness validation**: Subgroup performance and representation constraints are not empirically validated.
- **Simplified signals**: Real institutional signals would be noisier and more complex.

## Next Steps

1. **Multi-institutional validation**: Test on real federated pathology datasets (Camelyon17, PANDA, TCGA).
2. **Subgroup analysis**: Measure worst-group sensitivity, calibration error, and false-negative-rate disparity.
3. **Robustness testing**: Evaluate under missingness, gaming, and adversarial scenarios.
4. **Comparison study**: Benchmark against equal, volume, prestige, and Shapley-only weighting.
5. **Regulatory review**: Ensure claims align with validation evidence before any clinical deployment.
