# Figure Plan: Dominant-Site Reliability in Federated Pathology

This page defines the figures needed to turn the dominant-site result into a preprint-style research package.

---

## Figure 1 — Core mechanism

**Title:** When more data is less trustworthy

**Purpose:** Explain the fundamental failure mode without equations.

**Panels:**

1. **Clean federation**
   - Five sites.
   - Site 0 is largest and reliable.
   - FedAvg sample weighting is reasonable.

2. **Dominant-site reliability failure**
   - Site 0 remains largest but labels are unreliable or systematically shifted.
   - FedAvg still gives Site 0 dominant influence.

3. **Dominance-aware switch**
   - Validation diagnostics flag unsafe dominance.
   - Aggregation switches from FedAvg to cross-site blending.

**Caption draft:**

> FedAvg treats sample count as aggregation authority. In clinical federations, sample count and reliability can diverge. A large pathology site with systematic label bias can dominate the shared model even when its training signal is less reliable. Dominance-aware switching uses validation diagnostics to preserve FedAvg under clean conditions and reduce raw sample-size dominance under unsafe conditions.

---

## Figure 2 — Label-noise stress result

**Title:** Cross-site blending improves robustness under dominant-site label corruption

**Data source:**

```text
results/cross_site_fedavg_panda_all_15seed_combined/paired_seed_delta_summary.csv
```

**Plot type:**

- x-axis: dominant-site label noise percentage
- y-axis: cross-site blend delta vs FedAvg
- lines: global QWK, worst-site QWK, macro-F1
- error bars: 95% CI
- horizontal zero line

**Message:**

- Clean regime is not a universal win.
- 25% and 35% label-noise regimes show significant global-QWK gains.
- 45% label-noise regime shows significant worst-site-QWK gain.

---

## Figure 3 — Conservative ordinal threshold-shift transfer

**Title:** The effect transfers to systematic conservative grading bias

**Data source:**

```text
results/threshold_shift_panda_all_conservative_25_15seed_aggregate/aggregate_summary.csv
results/threshold_shift_panda_all_conservative_35_15seed_aggregate/aggregate_summary.csv
results/threshold_shift_panda_all_conservative_45_15seed_aggregate/aggregate_summary.csv
```

or a derived paired-delta table.

**Plot type:**

- x-axis: conservative threshold-shift percentage
- y-axis: cross-site blend delta vs FedAvg
- lines: global QWK, worst-site QWK, macro-F1
- error bars if paired seed delta table is available

**Message:**

- Cross-site blending improves all major metrics under conservative dominant-site grading bias.
- 45% shift has the strongest effect.

---

## Figure 4 — Fixed detector transfer

**Title:** A fixed detector rule transfers to conservative threshold shift

**Data source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
```

**Plot type:**

1. **Trigger-rate plot**
   - x-axis: threshold-shift percentage
   - y-axis: detector trigger rate
   - show 0%, 25%, 35%, 45%

2. **Performance delta plot**
   - x-axis: threshold-shift percentage
   - y-axis: detector delta vs FedAvg
   - lines: global QWK, worst-site QWK, macro-F1
   - error bars: 95% CI

**Message:**

- Clean false-switching is low.
- Detector gains are significant at 35% and 45% conservative shift.
- The fixed detector does not require retuning for this transfer test.

---

## Figure 5 — Claim boundary figure

**Title:** What is validated and what remains open

**Format:** table or checklist diagram.

**Validated:**

- 10,611 readable PANDA-derived Phikon slide features
- simulated multi-site federations
- 15-seed stress studies
- dominant-site label corruption
- conservative ordinal threshold shift
- fixed detector transfer to conservative threshold shift

**Not yet validated:**

- real hospital federation
- clinical deployment
- diagnostic use
- universal detector calibration
- natural multi-center benchmark transfer

**Message:**

> The result is a robust simulated-federation research finding, not a clinical claim.

---

## Needed derived tables

To support publication-quality figures, create these derived CSVs:

```text
results/dominant_site_research_package/label_noise_paired_deltas.csv
results/dominant_site_research_package/conservative_threshold_shift_paired_deltas.csv
results/dominant_site_research_package/fixed_detector_threshold_shift_summary.csv
results/dominant_site_research_package/main_result_summary.csv
```

---

## Visual style

Use simple plots:

- white background
- black zero line
- clear metric labels
- no excessive decoration
- one figure per claim
- captions that state the claim and the limitation

The figures should make the public-interest idea obvious:

> More data is not always more trustworthy data.
