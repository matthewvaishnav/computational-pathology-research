# When More Data Is Less Trustworthy

## Site-signal alignment failure modes in federated computational pathology

**Status:** research note / working draft  
**Scope:** simulated federations over pathology-derived feature vectors  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## One-sentence claim

In federated computational pathology, raw sample count is not the same as task-specific site-signal alignment: FedAvg can become unsafe when the largest simulated pathology client has a label process that is misaligned with the validation objective, and dominance-aware aggregation/switching can reduce that risk.

---

## Fundamental question

Federated learning is attractive in medicine because hospitals may want to collaborate without sharing raw patient data. Standard FedAvg makes a simple assumption:

> the client with more samples should receive more aggregation influence.

That assumption is convenient, but it is not automatically safe in clinical settings.

The fundamental research question is:

> **When should a federated pathology model give one client more aggregation influence than another?**

This leads to a sharper question:

> **Does having more patient data mean a client should receive more model influence, even when its label process is less aligned with the declared validation objective?**

---

## Ethical framing: site-signal alignment, not institutional worth

This work does **not** claim that some hospitals, pathologists, or institutions are inherently more reliable, competent, or trustworthy than others.

The term alignment is used in a narrow, task-specific modeling sense:

> whether a simulated client's training signal appears aligned with the declared validation objective under a given experimental setup.

A client may appear misaligned for many non-blameworthy reasons, including differences in grading thresholds, staining/scanning protocols, case mix, patient population, annotation workflow, label source, historical reporting practice, or local clinical policy.

Therefore, dominance-aware aggregation should **not** be interpreted as an institutional ranking mechanism. It is an audit mechanism for a modeling assumption that FedAvg already makes silently:

> larger client = more influence.

The ethical purpose of this work is to make that assumption visible, stress-testable, and contestable. Any real deployment would require transparent governance, local clinical review, pathologist input, bias auditing, and agreement on what validation objective is appropriate.

---

## Why this matters

Pathology labels are not purely mechanical ground truth. They can reflect institutional practice, grading thresholds, scanner/preparation differences, annotation policies, and pathologist disagreement. In this setting, a high-volume client may contribute many samples while also having a label process that differs from the target objective.

FedAvg does not distinguish between:

- a large client whose training signal is aligned with the validation objective
- a large client with random label noise
- a large client with systematic ordinal grading shift

It only sees sample count.

That creates a possible failure mode:

> If the largest client's training signal is misaligned with the target validation objective, FedAvg may amplify that misalignment because the client still receives dominant aggregation weight.

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

Validation labels are kept clean in stress experiments. Perturbations are applied to the largest simulated site's training labels so the experiments test whether aggregation methods are robust to dominant-client training-signal misalignment.

---

## Stress mode 1: dominant-site label corruption

The first stress mode corrupts labels at the largest simulated site. This tests whether a high-volume client can damage FedAvg when its training labels become misaligned with the validation objective.

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

> The result is not that cross-site blending is always better. The result is conditional: when the dominant client's training signal becomes misaligned, pure sample-size weighting becomes less safe.

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

> The dominant-site alignment effect transfers from random label corruption to a more pathology-plausible systematic ordinal grading bias.

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

> Can we observe when sample-size dominance has become unsafe under site-specific training-signal shift?

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

> more samples can coexist with systematic site-specific label-process shift.

Dominance-aware weighting changes the question from:

> How many samples does this client have?

into:

> How much influence should this client receive given both its contribution and its observed task-specific site-signal alignment?

---

## Supported claims

The current evidence supports these claims:

1. FedAvg has a dominant-site sample-volume/alignment failure mode in simulated federated pathology experiments.
2. The failure mode appears when the largest simulated client's training signal is made less aligned with the validation objective while validation labels remain clean.
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
6. that any real institution, hospital, or pathologist is inherently more or less reliable than another

This is simulated-federation research over real pathology-derived feature vectors, not a clinical deployment or institutional ranking system.

---

## What would make the result stronger

The next validation steps are:

1. **Camelyon17 or another real multi-center pathology benchmark**  
   Test whether the sample-volume/alignment mechanism appears when site identity is naturally defined by center.

2. **External expert review**  
   Have a computational pathology, federated learning, or pathology-AI researcher review the methods, ethical framing, and claim boundaries.

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

Federated learning promises privacy-preserving collaboration across institutions. But if aggregation blindly follows sample count, a high-volume client with a misaligned training signal can dominate the shared model. Computational pathology therefore needs transparent, task-specific, alignment-aware validation infrastructure before federated medical AI can be trusted in high-stakes settings.

---

## Short pitch

I am studying a fundamental safety problem in federated computational pathology: when hospitals train models together, how should the system decide how much influence each client receives? My current PANDA/Phikon simulated-federation results show that FedAvg can become vulnerable when the largest site's training signal is misaligned with the validation objective, and that cross-site blending plus dominance-aware detector switching improves robustness under dominant-site label corruption and conservative ordinal grading bias.
