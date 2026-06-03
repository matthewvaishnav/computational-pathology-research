# Plain-English summary: When more data is less trustworthy

**Project:** Site-signal alignment failure modes in federated computational pathology  
**Author:** Matthew Vaishnav  
**Status:** Independent research artifact; research-only, not clinical software.

## The simple idea

Federated learning is often used when hospitals or institutions cannot share raw data. Instead of pooling data in one place, each site trains locally and contributes model updates. A common baseline, FedAvg, gives more influence to sites with more samples.

That sounds reasonable, but in pathology it can be risky. A site can have many samples while also having a training-label process that is less aligned with the validation objective. This could happen because of grading-threshold differences, staining/scanning workflows, case mix, label-source differences, reporting practices, or pathologist disagreement. More samples do not automatically mean a site's signal should dominate the shared model.

## What I tested

I simulated federated pathology experiments using PANDA-derived Phikon feature representations for prostate cancer grading. The largest simulated site was treated as the dominant client. Validation labels were kept clean, while the dominant site's training labels were perturbed in controlled ways.

The main question was:

> Does FedAvg become less safe when the largest simulated site has a misaligned training signal, and can a detector-switch rule identify when sample-size dominance has become unsafe?

## Main result

A detector rule calibrated under dominant-site label-noise stress transferred to conservative ordinal threshold-shift stress. In clean conditions, it switched rarely. At stronger threshold-shift levels, it switched more often and improved key metrics.

Headline conservative threshold-shift results:

| Stress level | Trigger rate | Global QWK delta | Macro-F1 delta | Worst-site QWK delta |
|---:|---:|---:|---:|---:|
| 0% | 13.3% | -0.00025 | +0.00009 | +0.00151 |
| 35% | 60.0% | +0.00542 | +0.00838 | +0.00991 |
| 45% | 73.3% | +0.01053 | +0.01512 | +0.01290 |

Interpretation: the detector mostly preserved FedAvg when the clean regime looked safe, but became useful when the dominant site's training signal was systematically shifted.

## Why this matters

The result supports a narrow but important point: sample count is not the same as task-specific site-signal alignment. In federated medical AI, the largest client should not automatically be treated as the most trustworthy source of training influence. Its influence should be auditable.

## Robustness checks

The result was not only a one-metric artifact:

- Diagnostic-frequency analysis showed triggers were mainly driven by ordinal-error increases and QWK degradation.
- Removing the most frequent diagnostic, `mean_abs_error_high`, reduced trigger rate but did not collapse the 35% / 45% positive transfer result.
- Calibration-sensitivity analysis found that 29 of 36 nearby detector settings preserved low clean-regime switching and positive 35% / 45% gains across global QWK, macro-F1, and worst-site QWK.

## What this does not claim

This is not clinical validation. It is not diagnostic software. It is not evidence that any hospital, institution, or pathologist is better or worse than another. It is not a deployment-ready system.

The supported claim is narrower:

> In these simulated federated pathology experiments over real pathology-derived features, raw sample count is not equivalent to task-specific site-signal alignment, and sample-size dominance should be treated as an auditable modeling assumption.

## Best next validation steps

1. Run more seeds for the conservative threshold-shift detector result.
2. Test on naturally multi-center pathology data such as Camelyon17 or another center-labeled dataset.
3. Compare against simpler detector and aggregation baselines.
4. Add prospective governance framing for how validation diagnostics would be computed and shared in real federated settings.
