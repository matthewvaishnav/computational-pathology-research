# Current Project Status

**Last updated:** May 2026  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not currently used for patient care

---

## Summary

This repository is a research-focused computational pathology AI framework for whole-slide histopathology modeling, multiple-instance learning, benchmark validation, federated oncology validation experiments, and reproducible research infrastructure.

The strongest current evidence is:

- PCam validation AUC: **95.37%**
- PANDA slide-level prostate cancer grading with Phikon features
- PANDA readable slide-level feature files after HDF5 verification: **10,611**
- PANDA mean-pooled Phikon + MLP: **QWK 0.7274**
- PANDA gated AttentionMIL: **QWK 0.8100**
- PANDA tuned TransnnMIL repeated-seed QWK: **0.8155 / 0.8225 / 0.8086**

---

## PANDA status

The PANDA feature pipeline and slide-level baseline suite are complete for the current research pass.

Current data integrity status:

| Item | Count |
|---|---:|
| PANDA labels | 10,616 |
| HDF5 feature files | 10,615 |
| Missing feature files | 1 |
| Feature files selected after manifest filtering | 10,614 |
| Additional unreadable compressed HDF5 files dropped during read verification | 3 |
| Readable slides used for full baselines | 10,611 |

Current PANDA results:

| Model | Best validation QWK |
|---|---:|
| Mean-pooled Phikon + MLP | 0.7274 |
| Gated AttentionMIL | 0.8100 |
| Tuned TransnnMIL, seed 42 | 0.8155 |
| Tuned TransnnMIL, seed 123 | 0.8225 |
| Tuned TransnnMIL, seed 2025 | 0.8086 |

Interpretation:

> Tuned TransnnMIL is competitive with gated AttentionMIL and slightly favorable across the current repeated-seed PANDA experiments, beating AttentionMIL on 2 of 3 tested seeds. The advantage is small and should not be described as conclusive superiority.

---

## TransnnMIL ablation status

| Run | Best validation QWK | Interpretation |
|---|---:|---|
| lr=3e-4, dropout=0.15, patch cap 600 | 0.8155 | tuned reference setting |
| lr=1e-3, dropout=0.15, patch cap 600 | 0.7403 | high learning rate unstable |
| lr=3e-4, dropout=0.25, patch cap 600 | 0.8015 | higher dropout mildly hurts |

Conclusion:

> In the current PANDA setup, TransnnMIL is highly optimization-sensitive. Lowering learning rate from 1e-3 to 3e-4 was a major contributor to competitive performance. Higher dropout mildly reduced performance. Patch cap increase was not the main driver in the seed-42 comparison.

---

## FAIR-WEIGHTS-H status

FAIR-WEIGHTS-H is a research hypothesis for contribution-aware institutional weighting in federated oncology learning.

Current honest status:

- execution stability and aggregation behavior have been explored,
- a performance or fairness advantage over simpler baselines has **not** yet been demonstrated,
- the next step is a controlled five-site simulation comparing equal weighting, FedAvg/sample-size weighting, inverse-loss weighting, uncertainty weighting, leave-one-site-out contribution weighting, and simplified FAIR-WEIGHTS-H.

See: `docs/research/fair-weights-h-experiment-plan.md`

---

## Claim boundary

Research-only at this stage. Not clinically validated, not diagnostic software, and not currently used for patient care.

The long-term goal is responsible clinical translation after proper validation, regulatory review, security review, usability testing, and deployment testing.
