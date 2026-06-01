# External Review Outreach Templates

This page contains concise outreach templates for requesting expert feedback on the dominant-site federated pathology result.

---

## Target reviewers

Prioritize people in these groups:

1. computational pathology researchers
2. federated learning researchers
3. medical imaging AI researchers
4. digital pathology startup research engineers
5. pathologists familiar with prostate grading or multi-site label variation
6. professors/labs working on robust medical AI validation

---

## Short email

**Subject:** Request for feedback on federated pathology robustness result

```text
Hi [Name],

I am an independent computational pathology AI research engineer working on federated oncology validation.

I recently built a PANDA/Phikon simulated-federation study around a dominant-site reliability failure mode in FedAvg. The core question is: when hospitals train pathology models together, should the largest site always receive the most aggregation influence?

Using 10,611 readable PANDA-derived Phikon slide feature vectors across 15-seed stress studies, I found that FedAvg remains strong in clean settings but becomes vulnerable when the largest simulated site is unreliable. Cross-site blending and a dominance-aware detector switch improve robustness under dominant-site label corruption and systematic conservative ordinal grading bias.

I am careful about the claim boundary: this is research-only simulated-federation evidence, not clinical validation or diagnostic software.

Would you be willing to skim a short research note and tell me whether the mechanism and claim boundary are scientifically reasonable?

Research note:
[link]

Results page:
https://matthewvaishnav.github.io/computational-pathology-research/results/dominance-aware-switch-full-panda

Repository:
https://github.com/matthewvaishnav/computational-pathology-research

Thank you,
Matthew Vaishnav
```

---

## LinkedIn message

```text
Hi [Name], I am an independent computational pathology AI research engineer working on federated oncology validation.

I recently found a FedAvg dominant-site reliability failure mode in PANDA/Phikon simulated federations: when the largest simulated site becomes unreliable, sample-size-weighted aggregation becomes unsafe. Cross-site blending and a dominance-aware detector switch improved robustness under label corruption and conservative ordinal grading bias.

Would you be open to giving brief feedback on the mechanism and claim boundary? It is research-only, not clinical validation.
```

---

## Research-engineer hiring message

```text
Hi [Name],

I am looking for research engineer / applied ML roles in computational pathology, medical imaging AI, federated learning, and biomedical ML infrastructure.

Over the past two months I built a reproducible computational pathology research framework around PANDA/Phikon slide features, PCam validation, TransnnMIL, and simulated federated oncology learning.

My strongest current result identifies a FedAvg dominant-site reliability failure mode across 15-seed full-PANDA stress studies and evaluates cross-site blending plus detector switching under label noise and conservative ordinal grading bias.

I would be very interested in contributing to [company/lab] as a research engineer. Would you be open to a short conversation?

Portfolio:
[portfolio link]

Repository:
https://github.com/matthewvaishnav/computational-pathology-research
```

---

## What to ask reviewers specifically

Do not ask, "Is this genius?" Ask precise questions:

```text
1. Is the dominant-site reliability failure mode a reasonable federated medical-AI concern?
2. Is the stress-test design scientifically defensible?
3. Are the claim boundaries conservative enough?
4. Which additional benchmark would make the result more convincing?
5. Is the detector-switch framing useful, or should it be framed only as an analysis tool?
6. Would Camelyon17 be a reasonable next validation target?
```

---

## What not to claim in outreach

Avoid:

```text
I solved medical AI.
This is clinically ready.
This proves hospitals should use my method.
This is AGI-level work.
This is guaranteed novel.
```

Use:

```text
This is a simulated-federation robustness result.
I am seeking expert feedback.
The claim boundary is research-only.
The mechanism appears meaningful and needs external validation.
```
