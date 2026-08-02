---
layout: default
title: Clinical Workflow Prototype Notice
---

# Clinical Workflow Prototype Notice

**Reclassified:** August 2, 2026

The repository contains historical software prototypes for multi-class outputs,
risk-score objects, DICOM/FHIR data structures, longitudinal records, reporting,
and workflow orchestration.

The former page described these components as a clinically viable diagnostic
platform and listed physicians, hospital laboratories, and clinical systems as
target users. That framing is withdrawn.

## Current interpretation

These modules may be used for software development, interface experiments,
synthetic fixtures, and research demonstrations. They are not validated medical
software and must not be used to diagnose, grade, triage, monitor, or recommend
treatment for a patient.

## Not established

The repository does not currently establish:

- calibrated multi-class disease probabilities;
- clinically meaningful risk scores;
- validated disease taxonomies or clinical endpoints;
- safe use of patient metadata or longitudinal records;
- DICOM, HL7, or FHIR interoperability in a live environment;
- inference within a clinically required time limit;
- explainability sufficient for medical decisions;
- regulatory compliance or medical-device status;
- prospective clinical validation; or
- patient or workflow benefit.

Historical code examples may use medical terminology or realistic object names.
Those examples are API illustrations, not clinical evidence.

## Requirements before clinical consideration

Clinical use would require a clearly defined intended use, governed datasets,
representative external validation, locked models, calibrated endpoints,
prospective workflow testing, human-factors evaluation, safety and failure-mode
analysis, cybersecurity review, quality systems, regulatory strategy, and
continuous post-deployment monitoring.

None of those requirements is satisfied merely by repository code or unit tests.

The current public research boundary is defined in
[`../CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md).
