# When More Data Is Less Trustworthy

## Dominant-site reliability failure modes in federated computational pathology

**Status:** research note / working draft  
**Scope:** simulated federations over pathology-derived feature vectors  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## One-sentence claim

In federated computational pathology, raw sample count is not the same as institutional reliability: FedAvg can become unsafe when the largest simulated pathology site is systematically unreliable, and dominance-aware aggregation/switching can reduce that risk.

---

## Fundamental question

Federated learning is attractive in medicine because hospitals may want to collaborate without sharing raw patient data. Standard FedAvg makes a simple assumption:

> the client with more samples should receive more aggregation influence.

That assumption is convenient, but it is not automatically safe in clinical settings.

The fundamental research question is:

> **When should a federated pathology model trust one institution more than another?**

This leads to a sharper question:

> **Does having more patient data mean a hospital should receive more model influence, even when its labels are less reliable?**

---

## Why this matters

Pathology labels are not purely mechanical ground truth. They can reflect institutional practice, grading thresholds, scanner/preparation differences, annotation policies, and pathologist disagreement. In this setting, a large institution may contribute many samples while also contributing systematic bias.

FedAvg does not distinguish between:

- a large, reliable site
- a large, noisy site
- a large, systematically biased site

It only sees sample count.

That creates a possible failure mode:

> If the largest client becomes unreliable, FedAvg may amplify that unreliability because the client still receives dominant aggregation weight.

---

## Experimental setup

The current experiments use PANDA-derived Phikon slide features and simulate multi-site federated learning over real pathology-derived feature vectors.

```text
Data and task
├── Dataset: PANDA prostate cancer grading
├── Feature extractor: Phikon
├── Feature cache: 10,611 readable slide-level feature vectors
├── Feature dimension: 768
├── Task: ISUP grade prediction, classes 0–5
├── Federation: simulated multi-site setting
├── Metrics: global QWK, worst-site QWK, mean-site QWK, accuracy, macro-F1
└── Seeds: 15-seed stress studies
```

Validation labels are kept clean in stress experiments. Perturbations are applied to the largest simulated site's training labels so the experiments test whether aggregation methods are robust to unreliable dominant-client training signal.

---

## Stress mode 1: dominant-site label corruption

The first stress mode corrupts labels at the largest simulated site. This tests whether a high-volume site can damage FedAvg when its training labels become unreliable.

Observed pattern:

- Clean setting: FedAvg remains strong and should not be replaced automatically.
- Corrupted dominant-site setting: cross-site blending improves robustness.
- Detector switch setting: a clean-calibrated rule can switch away from FedAvg in unsafe regimes.

Representative result summary:

```text
15-seed full-PANDA label-noise stress

25% dominant-site label corruption:
  cross-site global QWK improves vs FedAvg

35% dominant-site label corruption:
  cross-site global QWK improves vs FedAvg

45% dominant-site label corruption:
  cross-site worst-site QWK improves vs FedAvg
```

Interpretation:

> The result is not that cross-site blending is always better. The result is conditional: when the dominant site becomes unreliable, pure sample-size weighting becomes less safe.

---

## Stress mode 2: systematic ordinal threshold shift

Random label corruption exposes the mechanism, but pathology disagreements can be systematic rather than random. The second stress mode therefore applies ordinal threshold shift to the dominant site's training labels.

Two directions are tested:

```text
Aggressive shift:
  selected labels move upward by one ISUP grade when possible

Conservative shift:
  selected labels move downward by one ISUP grade when possible
```

The conservative threshold-shift result is the strongest transfer setting.

Representative 15-seed result pattern:

```text
Conservative dominant-site threshold shift

25% shift:
  cross-site blending improves global QWK, worst-site QWK, and macro-F1

35% shift:
  cross-site blending improves global QWK, worst-site QWK, and macro-F1

45% shift:
  cross-site blending improves global QWK, worst-site QWK, and macro-F1 more strongly
```

Interpretation:

> The dominant-site reliability effect transfers from random label corruption to a more pathology-plausible systematic ordinal grading bias.

---

## Dominance-aware detector switch

Cross-site blending can help in corrupted regimes, but using it unconditionally can impose clean-regime costs. A more defensible method is a detector switch:

```text
1. Calibrate normal FedAvg validation behavior on clean runs.
2. Monitor validation diagnostics.
3. If enough diagnostics leave the clean-calibrated safe range, switch away from sample-size dominance.
4. Otherwise, keep FedAvg.
```

The detector is intended to answer this question:

> Can we observe when sample-size dominance has become unsafe?

A fixed label-noise-calibrated detector rule was evaluated on conservative threshold-shift stress:

```text
Fixed detector rule
├── low_quantile = 0.10
├── high_quantile = 0.80
├── min_trigger_count = 3
└── use_entropy = false
```

Observed pattern:

```text
Clean 0% conservative shift:
  low clean-regime switching

35% conservative shift:
  statistically positive improvements across global QWK, macro-F1, and worst-site QWK

45% conservative shift:
  statistically positive improvements across global QWK, macro-F1, and worst-site QWK
```

Interpretation:

> A single stricter detector rule calibrated in the label-noise setting also transfers to systematic conservative ordinal threshold shift. Aggressive threshold shift is weaker and should not be the headline claim.

---

## Mechanistic interpretation

FedAvg encodes a statistical assumption:

> more samples = more authority.

In computational pathology, this assumption can fail:

> more samples can coexist with systematic institutional label bias.

Dominance-aware weighting changes the question from:

> How many samples does this site have?

into:

> How much influence should this site receive given both its contribution and its reliability signal?

---

## Supported claims

The current evidence supports these claims:

1. FedAvg has a dominant-site reliability failure mode in simulated federated pathology experiments.
2. The failure mode appears when the largest simulated site is made less reliable while validation labels remain clean.
3. Cross-site blending improves robustness under dominant-site label corruption.
4. The effect transfers to systematic conservative ordinal threshold shift.
5. A fixed label-noise-calibrated detector rule transfers to conservative threshold-shift stress with low clean-regime switching and positive 35–45% shift gains.

---

## Claim boundaries

The current evidence does **not** prove:

1. clinical readiness
2. diagnostic safety
3. real hospital federated deployment performance
4. universal detector calibration across all site-shift types
5. that the same effect will hold unchanged on every pathology dataset

This is simulated-federation research over real pathology-derived feature vectors, not a clinical deployment.

---

## What would make the result stronger

The next validation steps are:

1. **Camelyon17 or another real multi-center pathology benchmark**  
   Test whether the dominant-site reliability mechanism appears when site identity is naturally defined by center.

2. **External expert review**  
   Have a computational pathology, federated learning, or pathology-AI researcher review the methods and claim boundaries.

3. **Preprint-style write-up**  
   Convert this note into a compact methods/results paper with figures, tables, and reproducibility commands.

4. **Ablation of detector diagnostics**  
   Identify which FedAvg validation diagnostics drive the detector and which are redundant.

5. **Training-stability evidence**  
   Show that TransnnMIL performance is not a single learning-rate accident by testing warmup, cosine decay, gradient clipping, and repeated-seed stability.

---

## Public-interest framing

The broader message is:

> In medical AI, more data is not always more trustworthy data.

Federated learning promises privacy-preserving collaboration across institutions. But if aggregation blindly follows sample count, a large unreliable site can dominate the shared model. Computational pathology therefore needs reliability-aware validation infrastructure before federated medical AI can be trusted in high-stakes settings.

---

## Short pitch

I am studying a fundamental safety problem in federated computational pathology: when hospitals train models together, how should the system decide which institution to trust? My current PANDA/Phikon simulated-federation results show that FedAvg can become vulnerable when the largest site is unreliable, and that cross-site blending plus dominance-aware detector switching improves robustness under dominant-site label corruption and conservative ordinal grading bias.
