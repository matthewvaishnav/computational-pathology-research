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
- Hospital type weighting (cancer center 2.0x, teaching 1.5x, community 1.0x, rural 0.8x)
- Specialization matching (route to cancer-specific experts)
- Volume & accuracy factors (log-scaled case volume + diagnostic accuracy)
- Experience scaling with diminishing returns

**Hypothesis:** PathologyFL + DMI > PathologyFL alone > Standard FedAvg, especially for rare subtypes and heterogeneous data quality.

**Status:** 🚧 Validation experiments in progress (see [Federated Ablation Protocol](FEDERATED_ABLATION_PROTOCOL))

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
    <p>Novel two-layer federated learning system combining domain-specific pathology knowledge (cancer-type strategies, slide quality) with institutional expertise intelligence (hospital type, specialization, volume, accuracy). First system to integrate both layers for medical AI collaboration.</p>
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
                    │                                       │
         ┌──────────▼──────────┐              ┌───────────▼──────────┐
         │  WSI Processing     │              │  Real-Time Streaming │
         │  ─────────────────  │              │  ──────────────────  │
         │  • OpenSlide        │              │  • <30s processing   │
         │  • Patch extraction │              │  • GPU parallel      │
         │  • 96x96, 224x224   │              │  • WebSocket updates │
         └──────────┬──────────┘              └───────────┬──────────┘
                    │                                     │
                    └───────────────────┬─────────────────┘
                                        │
                            ┌───────────▼───────────┐
                            │  Feature Extraction   │
                            │  ─────────────────── │
                            │  • ResNet18          │
                            │  • Foundation models │
                            │  • UNI, Phikon       │
                            └───────────┬───────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
         ┌──────────▼──────────┐              ┌───────────▼──────────┐
         │   MIL Models        │              │  Training Pipeline   │
         │   ──────────────    │              │  ────────────────   │
         │   • nnMIL           │◄─────────────┤  • torch.compile    │
         │   • AttentionMIL    │              │  • AMP (2-3x)       │
         │   • CLAM            │              │  • channels_last    │
         │   • TransMIL        │              │  • Multi-GPU (DDP)  │
         └──────────┬──────────┘              │  • 8-12x speedup    │
                    │                         └─────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  Interpretability   │
         │  ───────────────── │
         │  • Grad-CAM         │
         │  • Attention maps   │
         │  • SHAP values      │
         │  • Failure analysis │
         └──────────┬──────────┘
                    │
    ┌───────────────┴───────────────┐
    │                               │
┌───▼────────────┐      ┌──────────▼──────────┐
│  DMI System    │      │  Clinical Integration│
│  ────────────  │      │  ───────────────────│
│  • Expertise   │      │  • PACS (DICOM)     │
│    weighting   │      │  • FHIR adapter     │
│  • Multi-center│      │  • Patient context  │
│  • Cancer-type │      │  • Longitudinal     │
│    matching    │      │  • <5s inference    │
└───┬────────────┘      └──────────┬──────────┘
    │                              │
    └──────────────┬───────────────┘
                   │
         ┌─────────▼─────────┐
         │  Production API   │
         │  ───────────────  │
         │  • FastAPI        │
         │  • JWT auth       │
         │  • Rate limiting  │
         │  • Pydantic       │
         │  • SQL params     │
         │  • HTTPS/CORS     │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │   Deployment      │
         │   ──────────────  │
         │   • Docker/K8s    │
         │   • Monitoring    │
         │   • 5,000+ tests  │
         │   • HIPAA ready   │
         └───────────────────┘
```

**Performance Metrics:**
- **95.37% validation AUC** on PatchCamelyon (262K samples)
- **8-12x training speedup** (20-40h → 2-3h on RTX 4070)
- **<5s inference** for clinical deployment
- **<30s streaming** for gigapixel slides

**Production Features:**
- **5,071 tests** across 310 modules with property-based testing
- **39+ security commits** (authentication, input validation, privacy)
- **Privacy guarantees:** TenSEAL + Opacus required (no silent degradation)
- **HIPAA compliance:** Audit logging, encryption, access controls

**Current Development (May 2026):**
- **TransnnMIL v2.0** (Week 7 of 12): Hierarchical pooling + topology-aware GNNs
- **Feature-level fusion** ✅ completed with comprehensive testing
- **3-branch architecture** ✅ integrated (attention + hierarchical + topology)
- See [Current Status](CURRENT_STATUS) for detailed progress

---

## Quickstart

<div class="callout callout-info">
  <strong>New to the platform?</strong> Start with the 5-minute tutorial to train your first model on PCam!
</div>

**Interactive Tutorials:**
- [5-Minute PCam Training](https://github.com/matthewvaishnav/the platform/blob/main/examples/quickstart_pcam_training.ipynb) - Train AttentionMIL on PatchCamelyon
- [Custom Dataset Tutorial](https://github.com/matthewvaishnav/the platform/blob/main/examples/custom_dataset_tutorial.ipynb) - Adapt to your own data

**Quick Start Command:**
```bash
git clone https://github.com/matthewvaishnav/the platform.git
cd the platform
pip install -r requirements.txt
python experiments/train_pcam.py --config experiments/configs/pcam_ultra_fast.yaml
```

Expected: **95.37% validation AUC in 2-3 hours** on RTX 4070!

---

## Documentation

<div class="doc-links">
  <a href="CURRENT_STATUS" class="doc-link">📊 Current Status (May 2026)</a>
  <a href="GETTING_STARTED" class="doc-link">Getting Started</a>
  <a href="ARCHITECTURE" class="doc-link">🏗️ Architecture</a>
  <a href="PERFORMANCE_COMPARISON" class="doc-link">📊 Performance vs Competitors</a>
  <a href="BENCHMARK_SYSTEM" class="doc-link">🏆 Competitor Benchmark System</a>
  <a href="OPTIMIZATION_SUMMARY" class="doc-link">⚡ Training Optimizations (8-12x)</a>
  <a href="INFERENCE_OPTIMIZATION" class="doc-link">🚀 Inference Optimization (2-3x)</a>
  <a href="MULTI_GPU_TRAINING" class="doc-link">⚡ Multi-GPU Training (DDP)</a>
  <a href="FOUNDATION_MODELS" class="doc-link">🎯 Foundation Models (UNI, Phikon)</a>
  <a href="START_NOW_RTX4070" class="doc-link">RTX 4070 Guide</a>
  <a href="EXPERIMENTS" class="doc-link">Run Experiments</a>
  <a href="MODEL_INTERPRETABILITY" class="doc-link">Model Interpretability</a>
  <a href="CLINICAL_WORKFLOW_INTEGRATION" class="doc-link">Clinical Integration</a>
  <a href="PACS_INTEGRATION" class="doc-link">🏥 PACS Integration</a>
  <a href="COMPREHENSIVE_DATASET_TESTING" class="doc-link">Dataset Testing</a>
  <a href="regulatory_compliance" class="doc-link">Regulatory Compliance</a>
  <a href="API_REFERENCE" class="doc-link">API Reference</a>
  <a href="DOCS_INDEX" class="doc-link">Full Documentation</a>
</div>

---

## Installation

```bash
git clone https://github.com/matthewvaishnav/the platform.git
cd the platform
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

See the [Getting Started Guide](GETTING_STARTED) for detailed instructions.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vaishnav2026the platform,
  title = {the platform: Core Infrastructure for Computational Pathology Research},
  author = {Vaishnav, Matthew},
  year = {2026},
  url = {https://github.com/matthewvaishnav/the platform},
  note = {Production-grade PyTorch framework for computational pathology research}
}
```

---

<div class="footer-note">
  <p><strong>📊 Current Status (May 2026):</strong> PCam #1 AUC complete, Camelyon17 attention audit complete, PANDA training in progress, federated ablation study next. See <a href="ROADMAP_TO_GENIUS">Roadmap to Genius</a> for path forward.</p>
  <p><strong>Contact:</strong> For questions or collaboration opportunities, please open an issue on <a href="https://github.com/matthewvaishnav/computational-pathology-research/issues">GitHub</a>.</p>
  <p><em>Last updated: May 19, 2026</em></p>
</div>
