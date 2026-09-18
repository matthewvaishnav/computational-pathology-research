---
layout: doc
aside: false
---

# Computational Pathology Research

This documentation belongs to the program-level research repository and evidence ledger.

## Canonical public record

- [Repository overview](../README.md)
- [Authoritative claim boundary](../CLAIM_BOUNDARY.md)
- [Canonical PA-NF manuscript source and release metadata](../manuscripts/computational-pathology-foundations-v1/README.md)
- [Current research status](./CURRENT_STATUS.md)
- [Hugging Face release registry](./releases/huggingface-release-registry.yaml)

The canonical public paper is **Paired-Acquisition Neural Factorization: An End-to-End Computational Pathology Pipeline**. The publication workflow builds and publishes the manuscript PDF, supplement, source archive, and checksums.

## Research lines

The repository contains several distinct research lines with separate evidence boundaries:

- Paired-Acquisition Neural Factorization (PA-NF);
- TransnnMIL whole-slide modeling;
- PathologyFL;
- FAIR-WEIGHTS-H;
- experimental WSI-NCA / factorized tissue dynamics;
- scientific provenance and evidence tooling.

Do not infer a unified end-to-end validated clinical system from the presence of these components in one repository.

> **Research-only boundary:** this repository does not establish clinical validation, diagnostic utility, hospital deployment, regulatory approval, or universal superiority. When older documentation conflicts with the current claim boundary, `CLAIM_BOUNDARY.md` controls.
