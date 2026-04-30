# Changelog

All notable changes to HistoCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.0] - 2024-12-15

### Added
- Initial release of HistoCore framework
- Attention-based MIL models (AttentionMIL, CLAM, TransMIL)
- Clinical workflow integration (DICOM/FHIR support)
- Model interpretability tools (Grad-CAM, attention heatmaps)
- Comprehensive testing infrastructure (3,006 tests, 55% coverage)
- PCam benchmark training and evaluation
- CAMELYON16 slide-level classification support
- Multi-GPU distributed training
- Docker/Kubernetes deployment configurations
- Regulatory compliance features (FDA/CE marking support)

### Benchmarks
- **PCam Real Dataset**: 85.26% test accuracy, 0.9394 AUC
- **Clinical Optimization**: 90% sensitivity at threshold=0.051
- **Training Time**: ~18 min/epoch on RTX 4070 Laptop (8GB VRAM)

[Unreleased]: https://github.com/matthewvaishnav/histocore/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/matthewvaishnav/histocore/releases/tag/v0.1.0
