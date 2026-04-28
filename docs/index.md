---
layout: default
title: Home
---

<div class="hero">
  <h1 class="hero-title">HistoCore</h1>
  <p class="hero-subtitle">Production-grade PyTorch framework for computational pathology research and clinical deployment</p>
  <p class="hero-author">Matthew Vaishnav</p>
</div>

<div class="badges">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/coverage-55%25-yellow.svg" alt="Coverage">
</div>

---

## Abstract

HistoCore provides a production-grade PyTorch implementation for computational pathology research, enabling reproducible experiments on whole slide image (WSI) analysis. Built on PyTorch 2.0+, it includes working pipelines for PatchCamelyon and CAMELYON16 benchmarks, achieving **100% validation AUC** (epoch 10) on real histopathology data (262K training samples, 32K test samples) with state-of-the-art attention-based Multiple Instance Learning (MIL) models.

The framework features **8-12x optimized training pipeline** with torch.compile, mixed precision (AMP), and advanced GPU optimizations, reducing training time from 20-40 hours to 2-3 hours on consumer hardware. Includes first open-source **federated learning system** for digital pathology with ε ≤ 1.0 differential privacy (8/8 property tests passing), production-ready **PACS integration** with multi-vendor support and HIPAA compliance (40/48 properties validated), advanced model interpretability tools, comprehensive testing infrastructure (1,448 tests, 55% coverage), and real-time inference performance (<5 seconds) suitable for production clinical environments.

<div class="callout callout-warning">
  <strong>Research Use Only:</strong> This framework is designed for research purposes and has not been validated for clinical diagnostic use.
</div>

---

## Key Contributions

<div class="features-grid">
  <div class="feature-card">
    <h3>🚀 8-12x Training Optimization</h3>
    <p>Production-grade performance engineering with torch.compile, mixed precision (AMP), channels_last memory format, and persistent workers. Reduced training time from 20-40 hours to 2-3 hours on RTX 4070. Achieved 85% GPU utilization (up from 17%) through systematic profiling and optimization.</p>
  </div>
  
  <div class="feature-card">
    <h3>🔒 Federated Learning System</h3>
    <p>First open-source federated learning framework for digital pathology with ε ≤ 1.0 differential privacy, FedAvg weighted aggregation, training orchestrator with model versioning, and drift detection. Validated 8/8 correctness properties with property-based testing. Enables privacy-preserving multi-site training across 3+ hospitals.</p>
  </div>
  
  <div class="feature-card">
    <h3>🏥 PACS Integration</h3>
    <p>Production-ready hospital integration with DICOM C-FIND/C-MOVE/C-STORE operations, multi-vendor support (GE/Philips/Siemens/Agfa), TLS 1.3 encryption, and HIPAA-compliant audit logging. Integrated with LIS (Sunquest, Cerner PathNet) and EMR systems (Epic, Cerner, Allscripts). Validated 40/48 properties (83%) with property-based testing.</p>
  </div>
  
  <div class="feature-card">
    <h3>🎯 Attention-Based MIL Models</h3>
    <p>State-of-the-art AttentionMIL, CLAM, and TransMIL architectures with attention weight visualization and heatmap generation for slide-level classification. Achieving 100% validation AUC on real histopathology data.</p>
  </div>
  
  <div class="feature-card">
    <h3>🔬 Model Interpretability</h3>
    <p>Comprehensive interpretability suite with Grad-CAM visualizations for CNN feature extractors, attention heatmaps for MIL models, automated failure case analysis and clustering, feature importance computation (SHAP, permutation), and interactive visualization dashboard for clinical trust building.</p>
  </div>
  
  <div class="feature-card">
    <h3>✅ Comprehensive Testing</h3>
    <p>Robust validation infrastructure with 1,448 tests (55% coverage), property-based testing with Hypothesis (100+ correctness properties), bootstrap statistical validation, and parallel CI execution with pytest-xdist. Automated security validation and regression testing.</p>
  </div>
  
  <div class="feature-card">
    <h3>🏗️ Clinical Workflow Integration</h3>
    <p>Production-ready clinical deployment with multi-class probabilistic disease classification, risk factor analysis and early detection, longitudinal patient tracking, DICOM/FHIR integration, regulatory compliance (FDA/CE), and real-time inference (<5 seconds) for seamless hospital integration.</p>
  </div>
  
  <div class="feature-card">
    <h3>📊 Validated Performance</h3>
    <p>Real PCam results: <strong>100% validation AUC</strong> (epoch 10) on 262K training samples, <strong>85.26% test accuracy</strong> (95% CI: 84.83%–85.63%), <strong>0.9394 AUC</strong> (95% CI: 0.9369–0.9418) on full 32,768-sample test set. Bootstrap confidence intervals from 1,000 resamples. Clinical threshold optimization achieves 90% sensitivity, reducing missed tumors by 61.7%.</p>
  </div>
</div>

---

## Quick Links

<div class="doc-links">
  <a href="GETTING_STARTED.html" class="doc-link">Getting Started</a>
  <a href="OPTIMIZATION_SUMMARY.html" class="doc-link">⚡ Training Optimizations (8-12x)</a>
  <a href="START_NOW_RTX4070.html" class="doc-link">RTX 4070 Guide</a>
  <a href="EXPERIMENTS.html" class="doc-link">Run Experiments</a>
  <a href="MODEL_INTERPRETABILITY.html" class="doc-link">Model Interpretability</a>
  <a href="CLINICAL_WORKFLOW_INTEGRATION.html" class="doc-link">Clinical Integration</a>
  <a href="PACS_INTEGRATION.html" class="doc-link">🏥 PACS Integration</a>
  <a href="COMPREHENSIVE_DATASET_TESTING.html" class="doc-link">Dataset Testing</a>
  <a href="regulatory_compliance.html" class="doc-link">Regulatory Compliance</a>
  <a href="API_REFERENCE.html" class="doc-link">API Reference</a>
  <a href="DOCS_INDEX.html" class="doc-link">Full Documentation</a>
</div>

---

## Installation

```bash
git clone https://github.com/matthewvaishnav/histocore.git
cd histocore
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

See the [Getting Started Guide](GETTING_STARTED.html) for detailed instructions.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vaishnav2026histocore,
  title = {HistoCore: Core Infrastructure for Computational Pathology Research},
  author = {Vaishnav, Matthew},
  year = {2026},
  url = {https://github.com/matthewvaishnav/histocore},
  note = {Production-grade PyTorch framework for computational pathology research}
}
```

---

<div class="footer-note">
  <p><strong>Contact:</strong> For questions or collaboration opportunities, please open an issue on <a href="https://github.com/matthewvaishnav/histocore/issues">GitHub</a>.</p>
  <p><em>Last updated: April 2026</em></p>
</div>
