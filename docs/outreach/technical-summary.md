# Technical summary: site-signal alignment failure modes in federated pathology

**Repository:** `matthewvaishnav/computational-pathology-research`  
**Main artifact:** arXiv-style PDF and reproducibility package  
**Scope:** simulated federated pathology over PANDA-derived Phikon features; research-only.

## Research question

FedAvg weights client updates by sample count:

\[
w_k = \frac{n_k}{\sum_j n_j}, \qquad \Delta\theta = \sum_k w_k \Delta\theta_k.
\]

This assumes sample volume is a valid proxy for aggregation authority. In computational pathology, this can fail when a high-volume site's training-label process is less aligned with the declared validation objective. The project tests whether sample-size dominance becomes unsafe under controlled dominant-site label misalignment, and whether a clean-calibrated detector-switch rule can identify when to move away from FedAvg.

## Experimental setting

- Dataset/features: PANDA-derived prostate pathology feature bags.
- Feature extractor: Phikon.
- Readable slide-level feature files: 10,611.
- Feature dimension: 768.
- Task: ISUP grade prediction, classes 0-5.
- Federation: simulated multi-site setting.
- Perturbed client: largest simulated client.
- Validation labels: kept clean.
- Metrics: global QWK, macro-F1, worst-site QWK, mean-site QWK, accuracy.
- Stress studies: 15 seeds.

## Stress modes

1. **Dominant-site label corruption:** controlled label-noise stress at the largest simulated client.
2. **Systematic ordinal threshold shift:** selected labels at the dominant site are shifted by one ISUP grade. Conservative shift is treated as the headline systematic-bias result; aggressive shift is supplementary and weaker.

## Detector-switch rule

The detector monitors clean-calibrated validation diagnostics. It triggers when enough diagnostics leave their clean safe ranges. The fixed headline rule was calibrated under dominant-site label-noise stress and evaluated on conservative threshold-shift stress:

| Setting | Value |
|---|---:|
| low quantile | 0.10 |
| high quantile | 0.80 |
| min trigger count | 3 |
| entropy used | false |

Diagnostics included global QWK, worst-site QWK, site-QWK spread, mean absolute ordinal error, and severe ordinal error rate.

## Headline fixed-detector transfer result

Deltas are detector-switch minus clean-strategy baseline.

| Conservative shift | Trigger rate | Global QWK delta | Macro-F1 delta | Worst-site QWK delta |
|---:|---:|---:|---:|---:|
| 0% | 13.3% | -0.00025 | +0.00009 | +0.00151 |
| 25% | 33.3% | +0.00129 | +0.00290 | +0.00353 |
| 35% | 60.0% | +0.00542 | +0.00838 | +0.00991 |
| 45% | 73.3% | +0.01053 | +0.01512 | +0.01290 |

The 35% and 45% regimes are the core positive transfer result. The 0% clean regime has low switching and near-zero global-QWK cost.

## Interpretability and robustness

Top diagnostic frequencies under conservative threshold shift:

| Diagnostic | Count |
|---|---:|
| mean_abs_error_high | 44 |
| worst_site_qwk_low | 31 |
| global_qwk_low | 27 |
| severe_error_rate_high | 22 |
| site_qwk_spread_high | 12 |

Interpretation: triggers are mainly driven by ordinal-error and QWK degradation, not only by site-spread.

Leave-one-diagnostic-family-out ablation over the 35% / 45% regimes showed that removing `mean_abs_error_high` reduced trigger rate but preserved positive global-QWK, macro-F1, and worst-site-QWK deltas. Calibration sensitivity over 36 nearby detector configurations found 29 robust-positive configurations.

## Claim boundary

Supported:

- FedAvg can encode a sample-volume / site-signal alignment failure mode in these simulated federated pathology experiments.
- A label-noise-calibrated detector-switch rule transfers to conservative ordinal threshold shift with low clean-regime switching and positive 35% / 45% gains.
- The detector signal is interpretable and not solely dependent on one diagnostic or one exact threshold setting.

Not supported:

- Clinical readiness.
- Diagnostic safety.
- Real hospital federated deployment performance.
- Universal detector calibration.
- Institutional reliability ranking.

## Next technical steps

1. Extend the conservative detector transfer result to 30-50 seeds.
2. Validate on center-labeled multi-site pathology data, ideally Camelyon17 or a similar naturally multi-center dataset.
3. Add baseline detector comparisons and additional aggregation baselines.
4. Package all result tables with exact reproducibility scripts and stable commit tags.
