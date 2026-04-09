---
layout: default
title: Home
---

<div class="hero">
  <h1 class="hero-title">Computational Pathology Research Framework</h1>
  <p class="hero-subtitle">A PyTorch-based framework for whole slide image analysis and deep learning in digital pathology</p>
  <p class="hero-author">Matthew Vaishnav</p>
</div>

<div class="badges">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/coverage-62%25-yellow.svg" alt="Coverage">
</div>

---

## Abstract

This framework provides tested implementations for computational pathology research, enabling reproducible experiments on whole slide image (WSI) analysis. Built on PyTorch 2.0+, it includes working pipelines for PatchCamelyon and CAMELYON16 benchmarks, achieving 94% accuracy on patch-level classification and functional slide-level aggregation with multiple pooling strategies.

The codebase emphasizes research reproducibility with comprehensive unit testing (62% coverage), modular architecture, and extensive documentation.

<div class="callout callout-warning">
  <strong>Research Use Only:</strong> This framework is designed for research purposes and has not been validated for clinical diagnostic use.
</div>

---

## Key Contributions

<div class="features-grid">
  <div class="feature-card">
    <h3>Benchmark Implementations</h3>
    <p>Complete pipelines for PatchCamelyon (94% accuracy) and CAMELYON16 slide-level classification with aggregation strategies.</p>
  </div>
  
  <div class="feature-card">
    <h3>Pretrained Models</h3>
    <p>Integration with 1000+ models from torchvision and timm, featuring automatic extraction and dimension detection.</p>
  </div>
  
  <div class="feature-card">
    <h3>Analysis Tools</h3>
    <p>Model profiling, ONNX export, prediction CSV generation, and visualization utilities for publication.</p>
  </div>
  
  <div class="feature-card">
    <h3>Tested and Documented</h3>
    <p>62% code coverage with 500+ tests, comprehensive documentation, and reproducible configurations.</p>
  </div>
</div>

---

## Quick Links

<div class="doc-links">
  <a href="GETTING_STARTED.html" class="doc-link">Getting Started</a>
  <a href="START_NOW_RTX4070.html" class="doc-link">RTX 4070 Guide</a>
  <a href="EXPERIMENTS.html" class="doc-link">Run Experiments</a>
  <a href="API_REFERENCE.html" class="doc-link">API Reference</a>
  <a href="DOCS_INDEX.html" class="doc-link">Documentation</a>
</div>

---

## Installation

```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research
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
@software{vaishnav2026sentinel,
  title = {Computational Pathology Research Framework},
  author = {Vaishnav, Matthew},
  year = {2026},
  url = {https://github.com/matthewvaishnav/computational-pathology-research},
  note = {A PyTorch framework for whole slide image analysis}
}
```

---

<div class="footer-note">
  <p><strong>Contact:</strong> For questions or collaboration opportunities, please open an issue on <a href="https://github.com/matthewvaishnav/computational-pathology-research/issues">GitHub</a>.</p>
  <p><em>Last updated: April 2026</em></p>
</div>
