---
layout: default
title: Current Status
---

# Current Research Status

**Last reviewed:** September 18, 2026

The authoritative public interpretation is the repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md). This page is intentionally concise so that status information is not duplicated across multiple drifting documents.

## Public record

The program-level canonical manuscript is:

**Paired-Acquisition Neural Factorization: An End-to-End Computational Pathology Pipeline**

Current entry points:

- [repository overview](../README.md);
- [canonical manuscript source and release metadata](../manuscripts/computational-pathology-foundations-v1/README.md);
- [authoritative claim boundary](../CLAIM_BOUNDARY.md);
- [public release registry](releases/huggingface-release-registry.yaml);
- [repository split plan](research/repository-split-plan-20260808.md).

## Evidence posture

The root README and claim boundary distinguish promoted evidence from withdrawn, historical, exploratory, and engineering-only results.

In particular:

- the registered SCORPION capacity-matched campaign supports the bounded structured-separation comparison stated in the claim boundary;
- the canine fixed-estimand audit retains its negative comparison against strong simple scanner-removal baselines;
- historical TransnnMIL fusion/topology results remain withdrawn pending repaired matched reruns;
- PathologyFL/PANDA institutional studies remain simulated or centralized-proxy research rather than real multi-center deployment validation;
- PCam remains a patch-level engineering benchmark, not clinical evidence.

## Repository identity

This is an independent computational-pathology **research and research-engineering repository**. It contains historical platform/deployment scaffolding as well as active research code.

The presence of API, cloud, Docker, Kubernetes, PACS, monitoring, or regulatory-oriented code does not establish that those systems were clinically validated or deployed in patient care.

## Status-document rule

Dated status snapshots and archived reports are historical records. They must not override the current root README, the canonical manuscript release metadata, or `CLAIM_BOUNDARY.md`.

When a current numerical or scientific statement is needed, cite the promoted evidence artifact or release that supports it rather than copying the value into another status page.
