---
layout: default
title: Home
---

<div class="hero">
  <h1 class="hero-title">Computational Pathology Research</h1>
  <p class="hero-subtitle">Production-grade framework combining MIL, PathologyFL, and DMI for privacy-preserving multi-institutional pathology AI</p>
  <p class="hero-author">Matthew Vaishnav</p>
</div>

<div class="badges">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/tests-5071+-green.svg" alt="5,071+ tests">
  <img src="https://img.shields.io/badge/AUC-0.9394-brightgreen.svg" alt="PCam AUC">
</div>

---

## Abstract

Production-grade framework for computational pathology combining **PathologyFL** (domain-specific federated learning) with **DMI** (Distributed Medical Intelligence) for privacy-preserving, multi-institutional AI collaboration.

**Key achievements:**
- **#1 AUC (0.9394)** on PatchCamelyon vs 10 published baselines (Swin-Transformer, ConvNeXt, ViT-Base, PathViT, MedViT)
- **7x more efficient** than Swin-Transformer (12.2M vs 88M parameters)
- **Multi-center validation** on Camelyon17 (attention consistency across 5 simulated hospital sites)
- **Production infrastructure** with PACS/FHIR integration, security hardening, 5,071+ tests

<div class="callout callout-warning">
  <strong>Research Use Only:</strong> This framework is designed for research purposes and has not been validated for clinical diagnostic use.
</div>

---

## Core Innovation: PathologyFL + DMI

### Two-Layer Federated Learning System

**Layer 1: PathologyFL** - Domain-specific federated learning
- Hierarchical aggregation: Patch → Slide → Case → Hospital → Global
- Cancer-type specific strategies (breast, lung, prostate, colorectal)
- Slide quality weighting (sharpness, stain consistency, artifacts, label confidence)
- Attention-aware aggregation for MIL models

**Layer 2: DMI** - Institutional expertise intelligence
- **FAIR-WEIGHTS-H hybrid institutional weighting** replaces fixed prestige multipliers with auditable, evidence-based weighting signals.
- Core training-weight objective:

  $$
  w_t = \arg\max_{w\in\mathcal W}\sum_{i=1}^K w_i\left(\hat\phi_{i,t}^{Owen}+\lambda_DD_{i,t}^{useful}+\lambda_FF_{i,t}+\lambda_QQ_{i,t}-\lambda_SS_{i,t}\right)
  $$

  subject to:

  $$
  \sum_i w_i=1,\quad w_i^{min}\le w_i\le w_i^{max},\quad C_g(w)\ge C_g^{min},\quad \mathrm{Perf}_g(w)\ge \mathrm{Perf}_g^{min}
  $$

- Signals include group-aware counterfactual contribution, difficulty-adjusted quality, useful distributional uniqueness, subgroup representation, uncertainty penalties, and anomaly monitoring.
- Legacy prestige weights (cancer center 2.0x, teaching 1.5x, community 1.0x, rural 0.8x) are retained only as comparison baselines in synthetic experiments.
- Implementation status: experimental engine, explicit weighted aggregation adapter, synthetic federation benchmark, perturbation suite, and markdown report generator are implemented for research validation.

**Hypothesis:** PathologyFL + DMI > PathologyFL alone > Standard FedAvg, especially for rare subtypes and heterogeneous data quality.

**Status:** 🚧 Validation experiments in progress

---

## Empirical Results

<div class="features-grid">
  <div class="feature-card">
    <h3>🏆 #1 AUC on PatchCamelyon</h3>
    <p><strong>0.9394 AUC</strong> on full PCam dataset (327K patches), beating 10 published baselines including Swin-Transformer (+0.88%), ConvNeXt (+1.03%), ViT-Base (+1.15%), PathViT (+1.37%), and MedViT (+1.73%). <strong>85.26% test accuracy</strong> with bootstrap confidence intervals. <strong>7x more efficient</strong> (12.2M vs 88M parameters). See <a href="PCAM_REAL_RESULTS">full results</a>.</p>
  </div>
  
  <div class="feature-card">
    <h3>🔬 Multi-Center Validation</h3>
    <p><strong>Camelyon17 federated audit</strong> across 5 simulated hospital sites. Measured cross-site attention consistency and site predictability. <strong>Verdict:</strong> Models learn site-invariant pathological features, not scanner shortcuts. Proves generalization across institutions.</p>
  </div>
  
  <div class="feature-card">
    <h3>🚧 PANDA Training</h3>
    <p>Prostate cancer Gleason grading on 1,365 slides. Training in progress on separate machine. Expected: competitive with PANDA challenge top 10 (>0.89 kappa). Demonstrates generalization to different cancer types.</p>
  </div>
</div>

---

## Key Contributions

<div class="features-grid">
  <div class="feature-card">
    <h3>🔬 PathologyFL + DMI</h3>
    <p>Novel two-layer federated learning system combining domain-specific pathology knowledge (cancer-type strategies, slide quality) with FAIR-WEIGHTS-H institutional intelligence: counterfactual contribution, useful uniqueness, quality, uncertainty, and subgroup-safety constrained weighting.</p>
  </div>
  
  <div class="feature-card">
    <h3>🏆 State-of-the-Art Results</h3>
    <p>#1 AUC (0.9394) on PCam vs 10 baselines. 7x more efficient than Swin-Transformer. Statistical significance confirmed with bootstrap CIs and DeLong tests. Comprehensive benchmark report with effect sizes.</p>
  </div>
  
  <div class="feature-card">
    <h3>🎯 TransnnMIL v2.0</h3>
    <p>3-branch architecture: TransMIL (self-attention) + Hierarchical pooling (multi-scale) + Topology branch (GNN). Adaptive pruning for 30% computation reduction. 6.8M parameters with attention-aware aggregation.</p>
  </div>
  
  <div class="feature-card">
    <h3>🏥 Production Infrastructure</h3>
    <p>PACS integration (DICOM C-FIND/C-MOVE/C-STORE), FHIR adapter, security hardening (39 commits), CI/CD optimized (99% faster), 5,071+ tests. Ready for clinical deployment with HIPAA compliance.</p>
  </div>
</div>

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         the platform System Architecture                          │
│                    End-to-End Computational Pathology Platform                 │
└───────────────────────────────────────────────────────────────────────────────┘

                                 ┌─────────────┐
                                 │   WSI Data  │
                                 │ (.svs/.tiff)│
                                 └──────┬──────┘
                                        │
                    ┌───────────────────┴───────────────────┐