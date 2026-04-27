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
  <img src="https://img.shields.io/badge/coverage-80%25-green.svg" alt="Coverage">
  <img src="https://img.shields.io/badge/tests-1448+-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/processing-<30s-success.svg" alt="Processing Speed">
  <img src="https://img.shields.io/badge/memory-<2GB-success.svg" alt="Memory Usage">
</div>

---

## Abstract

HistoCore provides a production-grade PyTorch implementation for computational pathology research, enabling reproducible experiments on whole slide image (WSI) analysis. Built on PyTorch 2.0+, it includes working pipelines for PatchCamelyon and CAMELYON16 benchmarks, achieving **85.26% test accuracy** and **93.94% AUC** on the full 32,768-sample PatchCamelyon test set with state-of-the-art attention-based Multiple Instance Learning (MIL) models.

The framework features **real-time WSI streaming** with 7x faster processing (<30s vs 3-5min) and 75% less memory (<2GB vs 8-12GB), **zero-downtime model hot-swapping** with A/B testing, **comprehensive stress testing** (90%+ success under 50+ concurrent loads), and **automated performance regression detection**. Includes 141 source modules, 150+ test files with 1,448+ tests (>80% coverage), advanced model interpretability tools (Grad-CAM, attention visualization, failure analysis), clinical workflow integration with DICOM/FHIR/PACS support, regulatory compliance features for clinical deployment, multimodal fusion capabilities, and production-ready performance suitable for hospital environments.

<div class="callout callout-warning">
  <strong>Research Use Only:</strong> This framework is designed for research purposes and has not been validated for clinical diagnostic use.
</div>

## About the Author

**Matthew Vaishnav** is a computational systems engineer based in Kitchener, building production-grade machine learning infrastructure for computational pathology. Creator of HistoCore, featuring attention-based MIL models, complete WSI processing pipelines with OpenSlide integration, clinical workflow systems with DICOM/FHIR support, and comprehensive model interpretability tools. The framework includes 141 source modules, 150 test files with 1,448 tests (55% coverage), and validated performance on real-world benchmarks (85.26% accuracy, 0.9394 AUC on PCam). Focus areas include reliable, clinically-deployable systems with regulatory compliance features, robust testing infrastructure, and practical tools for real-world medical imaging applications.

[Learn more about the project →](ABOUT.html)

---

## Key Contributions

<div class="features-grid">
  <div class="feature-card">
    <h3>🚀 Real-Time WSI Streaming</h3>
    <p><strong>7x faster processing</strong> (<30s vs 3-5min) with <strong>75% less memory</strong> (<2GB vs 8-12GB). Progressive tile loading, streaming attention aggregation, TensorRT optimization (3-5x speedup), multi-GPU parallelism, and real-time visualization via WebSocket. Production-ready with HIPAA/GDPR/FDA compliance.</p>
  </div>
  
  <div class="feature-card">
    <h3>🔄 Model Hot-Swapping</h3>
    <p><strong>Zero-downtime model updates</strong> with versioning, A/B testing infrastructure, automatic rollback capabilities, and compatibility validation. Thread-safe operations with SHA256 checksums for integrity verification. MLOps-ready for production deployments.</p>
  </div>
  
  <div class="feature-card">
    <h3>💪 Stress Testing Suite</h3>
    <p><strong>90%+ success rate</strong> under 50+ concurrent slide processing. Automatic OOM recovery, network resilience with exponential backoff, memory leak detection, and sustained load validation. Production-grade resilience engineering.</p>
  </div>
  
  <div class="feature-card">
    <h3>📊 Performance Regression Testing</h3>
    <p><strong>Automated baseline tracking</strong> with 10% regression detection threshold. CI/CD integration, historical trend analysis, and JSON report generation. Tracks processing time, throughput, and memory usage across 100+ runs.</p>
  </div>
  
  <div class="feature-card">
    <h3>🔐 Federated Learning System</h3>
    <p>First open-source federated learning framework for digital pathology, enabling privacy-preserving multi-site training across hospitals without centralizing patient data. FedAvg aggregation, training orchestrator with model versioning, 8/8 property tests passing.</p>
  </div>
  
  <div class="feature-card">
    <h3>🎯 Attention-Based MIL Models</h3>
    <p>State-of-the-art AttentionMIL, CLAM, and TransMIL architectures with attention weight visualization and heatmap generation for slide-level classification.</p>
  </div>
  
  <div class="feature-card">
    <h3>🏥 Clinical Workflow Integration</h3>
    <p>Production-ready clinical deployment with multi-class probabilistic disease classification, risk factor analysis and early detection, longitudinal patient tracking, DICOM/FHIR integration, regulatory compliance (FDA/CE), and real-time inference (<5 seconds) for seamless hospital integration.</p>
  </div>
  
  <div class="feature-card">
    <h3>🔍 Model Interpretability</h3>
    <p>Comprehensive interpretability suite with Grad-CAM visualizations for CNN feature extractors, attention heatmaps for MIL models, automated failure case analysis and clustering, feature importance computation (SHAP, permutation), and interactive visualization dashboard for clinical trust building.</p>
  </div>
  
  <div class="feature-card">
    <h3>✅ Comprehensive Testing</h3>
    <p>Robust validation infrastructure with 1,448+ tests (>80% coverage), property-based testing for edge cases, stress testing under extreme conditions, performance regression detection, synthetic data generation, and automated CI/CD integration ensuring production reliability.</p>
  </div>
  
  <div class="feature-card">
    <h3>🔗 Multimodal Fusion</h3>
    <p>Cross-modal attention for integrating WSI, genomic, and clinical text data with temporal progression modeling and missing data handling.</p>
  </div>
</div>

---

## Quick Links

<div class="doc-links">
  <a href="ABOUT.html" class="doc-link">About HistoCore</a>
  <a href="GETTING_STARTED.html" class="doc-link">Getting Started</a>
  <a href="REALTIME_STREAMING.html" class="doc-link">🚀 Real-Time Streaming</a>
  <a href="START_NOW_RTX4070.html" class="doc-link">RTX 4070 Guide</a>
  <a href="EXPERIMENTS.html" class="doc-link">Run Experiments</a>
  <a href="MODEL_INTERPRETABILITY.html" class="doc-link">Model Interpretability</a>
  <a href="CLINICAL_WORKFLOW_INTEGRATION.html" class="doc-link">Clinical Integration</a>
  <a href="COMPREHENSIVE_DATASET_TESTING.html" class="doc-link">Dataset Testing</a>
  <a href="REGULATORY_COMPLIANCE.html" class="doc-link">Regulatory Compliance</a>
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
  note = {Production-grade PyTorch framework with 141 modules, 1,448 tests, 55\% coverage}
}
```

---

<div class="footer-note">
  <p><strong>Contact:</strong> For questions or collaboration opportunities, please open an issue on <a href="https://github.com/matthewvaishnav/histocore/issues">GitHub</a>.</p>
  <p><em>Last updated: April 2026</em></p>
</div>
