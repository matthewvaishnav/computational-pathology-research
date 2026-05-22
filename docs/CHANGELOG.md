# Changelog

All notable changes to the platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **FAIR-WEIGHTS-H institutional weighting system** (experimental)
  - Hybrid weighting framework combining quality, uniqueness, fairness, contribution, volume, and uncertainty
  - Stable softmax normalization with integrity gates
  - Conservative mode for high-uncertainty scenarios
  - Synthetic federation utilities with volume and prestige baselines
  - Perturbation scenarios: uncertainty spike, quality degradation, rare population enrichment, scanner shift
  - Benchmark system comparing equal, volume, prestige, and FAIR-WEIGHTS-H strategies
  - Experiment runner and canonical perturbation suite
  - Markdown reporting with interpretation guardrails
  - Weighted aggregator integration with PathologyFL
  - Comprehensive test suite (27/27 tests passing, 87-100% coverage)
  - Documentation: protocol specification, implementation status, synthetic report
  - **Status:** Experimental research implementation; requires empirical validation before clinical use

### Security

- Fixed Bandit security findings (0 HIGH, 0 MEDIUM remaining)
  - Added timeout parameter to requests.post() calls (B113)
  - Replaced direct pickle.loads() with safe_pickle wrapper (B301)
  - Verified Jinja2 autoescape enabled for XSS prevention
  - Added comprehensive security documentation in SECURITY.md
  - All nosec markers now include justification comments

### Added

- Cross-validation infrastructure for robust model evaluation
  - Stratified K-fold splitting with 5 folds
  - Bootstrap confidence intervals per fold
  - Aggregated statistics across all folds
  - Quick test mode for pipeline validation
- Training metrics analysis tool (`experiments/analyze_metrics.py`)
  - Automated training curve generation
  - Checkpoint metrics extraction
  - Comprehensive markdown reports
- Baseline model comparison framework
  - Multi-model comparison tables
  - Efficiency analysis (accuracy vs parameters)
  - Training time comparisons
  - Publication-quality visualizations

### Changed

- Updated PCam dataset to use memory-mapped loading for large .npy files
  - Reduces RAM usage from 6.9GB to minimal overhead
  - Enables training on systems with limited memory
- Improved Windows compatibility for DataLoader
  - Set `num_workers=0` to avoid multiprocessing issues
  - Validated on Windows 11 with RTX 4070 Laptop

### Fixed

- Fixed macOS CI timeout issues in property-based tests
  - Implemented CI-aware test configuration
  - Reduced test parameters for resource-constrained environments
  - Maintained comprehensive coverage in local development

## [0.1.0] - 2026-12-15

### Added

- Initial release of the platform framework
- Attention-based MIL models (AttentionMIL, CLAM, TransMIL)
- Clinical workflow integration (DICOM/FHIR support)
- Model interpretability tools (Grad-CAM, attention heatmaps)
- Comprehensive testing infrastructure (3,171 tests, 55% coverage)
- PCam benchmark training and evaluation
- CAMELYON16 slide-level classification support
- Multi-GPU distributed training
- Docker/Kubernetes deployment configurations
- Regulatory compliance features (FDA/CE marking support)

### Benchmarks

- **PCam Real Dataset**: 85.26% test accuracy, 0.9394 AUC
- **Clinical Optimization**: 90% sensitivity at threshold=0.051
- **Training Time**: ~18 min/epoch on RTX 4070 Laptop (8GB VRAM)

[Unreleased]: https://github.com/matthewvaishnav/the platform/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/matthewvaishnav/the platform/releases/tag/v0.1.0
