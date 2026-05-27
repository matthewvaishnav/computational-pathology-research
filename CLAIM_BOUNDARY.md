# Claim Boundary

This document defines the public claim boundary for this repository.

## Current safe claim

This repository is a research-focused computational pathology and oncology AI engineering framework for whole-slide histopathology modeling, multiple-instance learning, benchmark validation, and federated oncology validation experiments.

Current validated research evidence includes:

- PCam validation AUC of 95.37%
- PANDA prostate cancer grading experiments using Phikon patch features
- 10,611 readable PANDA slide-level feature files after HDF5 read verification
- PANDA mean-pooled Phikon + MLP baseline QWK 0.7274
- PANDA gated AttentionMIL baseline QWK 0.8100
- PANDA tuned TransnnMIL repeated-seed QWK values of 0.8155, 0.8225, and 0.8086
- controlled TransnnMIL ablations showing learning-rate sensitivity

## Clinical and regulatory boundary

Research-only at this stage. Not clinically validated, not diagnostic software, and not currently used for patient care.

The long-term goal is responsible clinical translation after proper validation, regulatory review, security review, usability testing, and deployment testing.

## Do not claim

Do not describe the current repository as:

- clinically validated
- diagnostic software
- deployed in hospitals
- FDA-cleared
- CE-marked
- HIPAA-certified
- a medical device platform
- ready for patient care
- ready for clinical deployment
- proven to improve patient outcomes

## Safer language

Use:

> Research-focused computational pathology AI framework.

Use:

> DICOM/PACS-style research prototype or workflow integration experiment.

Use:

> Audit logging patterns designed with healthcare constraints in mind.

Use:

> Long-term goal is responsible clinical translation after proper validation and regulatory review.

Do not use:

> Production-ready clinical PACS integration.

Do not use:

> HIPAA-compliant hospital deployment.

Do not use:

> FDA-ready medical device platform.

## Model-performance boundary

Safe:

> Tuned TransnnMIL is competitive with gated AttentionMIL and slightly favorable across the current repeated-seed PANDA experiments, beating AttentionMIL on 2 of 3 tested seeds.

Not safe:

> TransnnMIL is conclusively superior to AttentionMIL.

Safe:

> TransnnMIL appears highly optimization-sensitive in the current PANDA setup; lowering learning rate from 1e-3 to 3e-4 was a major contributor to performance.

Not safe:

> TransnnMIL solves PANDA grading.
