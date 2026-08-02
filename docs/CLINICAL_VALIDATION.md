# Synthetic Site Simulation — Not Clinical Validation

**Reclassified:** August 2, 2026

The code referenced by the former page generates synthetic site profiles,
demographics, data-quality settings, prevalence assumptions, and operational
characteristics for software and mechanism testing.

It is **not a clinical-validation framework** and must not be described as one.

## Appropriate uses

The synthetic-site utilities may support:

- software tests;
- stress testing of data-loading or aggregation logic;
- demonstrations of how algorithms respond to controlled simulated shifts;
- generation of fixtures for robustness experiments; and
- development of fail-closed validation code.

## What synthetic sites cannot establish

They do not establish:

- performance at a real hospital;
- generalization across actual institutions, populations, scanners, staining
  workflows, or clinical practice;
- demographic fairness;
- calibrated disease prevalence;
- clinical safety or utility;
- regulatory readiness;
- patient-level outcomes; or
- valid claims about named hospital categories or geographic populations.

Synthetic demographic or institutional labels are model assumptions, not
observational evidence. They should not be interpreted as realistic patient
cohorts merely because the generated values look plausible.

## Requirements for actual external validation

A real validation study would require independently sourced data, documented
provenance, appropriate consent and governance, clinically meaningful endpoints,
patient- or specimen-grouped evaluation, prespecified analysis, correct
hierarchical inference, and transparent reporting of missingness, selection,
label uncertainty, and domain shift.

No such validation is supplied by this synthetic module.

The repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) is authoritative.
