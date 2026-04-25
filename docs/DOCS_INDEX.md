---
layout: default
title: Documentation Index
---

# Documentation Index

Comprehensive documentation for HistoCore - the production-grade computational pathology framework.

---

## Getting Started

### Installation and Setup
- [Getting Started Guide](GETTING_STARTED.html) - Complete installation and setup guide
- [System Requirements](GETTING_STARTED.html#system-requirements) - Hardware and software prerequisites
- [Installation](GETTING_STARTED.html#installation) - Step-by-step installation instructions

### Tutorials
- [Your First Model](GETTING_STARTED.html#your-first-model) - Step-by-step PCam tutorial
- [Working with Real Data](GETTING_STARTED.html#working-with-real-data) - PatchCamelyon and CAMELYON16 workflows
- [API Reference](API_REFERENCE.html) - Complete API documentation

---

## Training and Evaluation

### PatchCamelyon (PCam)
- [PCAM_REAL_RESULTS.md](PCAM_REAL_RESULTS.html) - **Real dataset results**: 85.26% accuracy, 0.9394 AUC with bootstrap CIs
- [PCAM_FAILURE_ANALYSIS.md](PCAM_FAILURE_ANALYSIS.html) - **Failure analysis**: 26.11% false negative rate, clinical implications
- [THRESHOLD_OPTIMIZATION.md](THRESHOLD_OPTIMIZATION.html) - **Clinical optimization**: 90% sensitivity for cancer screening
- [PCAM_CROSS_VALIDATION.md](PCAM_CROSS_VALIDATION.html) - **Cross-validation**: K-fold validation for robustness assessment
- [PCAM_BENCHMARK_RESULTS.md](PCAM_BENCHMARK_RESULTS.html) - Synthetic benchmark results and analysis
- [PCAM_COMPARISON_GUIDE.md](PCAM_COMPARISON_GUIDE.html) - Comparing model architectures
- **Topics**: Patch-level classification, data augmentation, baseline comparisons, clinical deployment, robustness validation

### CAMELYON16 Slide-Level
- [CAMELYON_TRAINING_STATUS.md](CAMELYON_TRAINING_STATUS.html) - Training guide and best practices
- [CAMELYON_SLIDE_LEVEL_IMPLEMENTATION.md](CAMELYON_SLIDE_LEVEL_IMPLEMENTATION.html) - Implementation details
- **Topics**: Slide-level aggregation, attention mechanisms, feature extraction

### WSI Processing Pipeline
- [WSI_PROCESSING_PIPELINE.md](WSI_PROCESSING_PIPELINE.html) - **Complete WSI processing pipeline**: OpenSlide integration, CLI tools, production deployment
- **Topics**: Multi-format WSI support (.svs, .tiff, .ndpi, DICOM), streaming processing, CNN feature extraction, HDF5 caching, clinical deployment

### Advanced Features
- [Model Interpretability Guide](MODEL_INTERPRETABILITY.html) - Grad-CAM, attention visualization, failure analysis
- [Clinical Workflow Integration](CLINICAL_WORKFLOW_INTEGRATION.html) - Multi-class classification, DICOM/FHIR support
- [PACS Integration System](PACS_INTEGRATION.html) - **Production-ready PACS integration**: DICOM C-FIND/C-MOVE/C-STORE, multi-vendor support, TLS 1.3, HIPAA audit logging (40/48 properties tested, 83%)
- [Comprehensive Dataset Testing](COMPREHENSIVE_DATASET_TESTING.html) - 1,448 tests with property-based testing

### Evaluation Metrics
- Model performance analysis with bootstrap confidence intervals
- ROC curves and confusion matrices
- CSV export for downstream analysis
- Attention weight visualization and heatmap generation

---

## Architecture and Design

### System Overview
- [ARCHITECTURE.md](ARCHITECTURE.html) - Complete system architecture
- **Components**: Data loaders, model architectures, training loops, evaluation pipelines

### Model Architectures
- **Attention-Based MIL Models**: AttentionMIL, CLAM, TransMIL with attention weight visualization
- **Multimodal Fusion**: Cross-modal attention for WSI, genomic, and clinical text integration
- **Temporal Models**: Disease progression prediction with positional encoding
- **Baseline Models**: ResNet, DenseNet, EfficientNet with pretrained weights
- **Slide Classifiers**: Attention-based aggregation, pooling strategies
- **Pretrained Integration**: torchvision and timm model loading (1000+ architectures)

### Data Pipeline
- **WSI Processing Pipeline**: Complete production-ready pipeline with OpenSlide integration
- **PCam Dataset**: Patch extraction and preprocessing
- **CAMELYON Dataset**: HDF5 feature storage, slide-level batching
- **Augmentation**: Standard transforms, normalization

---

## Deployment

### Docker Deployment
- [DOCKER.md](DOCKER.html) - Complete Docker guide
- **Topics**: Container setup, GPU support, production deployment

### REST API
- [deploy/README.md](../deploy/README.html) - API deployment instructions
- **Endpoints**: Model inference, batch processing, health checks

### Model Export
- **ONNX Export**: Cross-platform deployment
- **TorchScript**: Production optimization
- **Quantization**: Model compression

---

## Development

### Contributing
- [CONTRIBUTING.md](../CONTRIBUTING.html) - Contribution guidelines
- **Topics**: Code style, pull requests, issue reporting

### Testing
- [TESTING_SUMMARY.md](TESTING_SUMMARY.html) - Test suite documentation
- **Coverage**: 55% code coverage, 1,448 unit tests
- **Topics**: Unit tests, integration tests, property-based testing, edge case handling, performance benchmarks

### Build System
- [MAKEFILE.md](MAKEFILE.html) - Makefile usage guide
- **Commands**: Build, test, lint, format, clean

---

## Performance and Optimization

### Performance Analysis
- [PERFORMANCE.md](PERFORMANCE.html) - Optimization guide
- **Topics**: Profiling, bottleneck analysis, GPU utilization

### Model Profiling
- Inference latency measurement
- Memory usage analysis
- Throughput optimization

---

## Results and Analysis

### Benchmark Results
- [DEMO_RESULTS.md](DEMO_RESULTS.html) - Demo training results
- **Metrics**: Accuracy, AUC, training time, convergence analysis

### Project Portfolio
- [PORTFOLIO_SUMMARY.md](PORTFOLIO_SUMMARY.html) - Complete project overview
- **Status**: Current capabilities, limitations, future work

### Roadmap
- [ROADMAP_TO_REAL_DATASETS.md](ROADMAP_TO_REAL_DATASETS.html) - Real dataset integration plan
- **Timeline**: Short-term goals, long-term vision

---

## API Reference

### Core Modules

#### Data Loading
```python
from src.data import PatchCamelyonDataset, CAMELYONSlideDataset
from src.data.wsi_pipeline import BatchProcessor, ProcessingConfig
```
- `PatchCamelyonDataset`: Patch-level image loading
- `CAMELYONSlideDataset`: Slide-level feature loading
- `collate_slide_bags`: Variable-length batch collation
- `BatchProcessor`: WSI processing pipeline orchestration
- `ProcessingConfig`: WSI pipeline configuration management

#### Models
```python
from src.models import SimpleClassifier, SimpleSlideClassifier
from src.models.pretrained import load_pretrained_encoder
```
- `SimpleClassifier`: Patch-level classifier
- `SimpleSlideClassifier`: Slide-level aggregation
- `load_pretrained_encoder`: Pretrained model loading

#### Training
```python
from src.training import train_epoch, evaluate
```
- `train_epoch`: Single epoch training loop
- `evaluate`: Model evaluation with metrics

#### Utilities
```python
from src.utils import set_seed, save_checkpoint, load_checkpoint
```
- `set_seed`: Reproducibility utilities
- `save_checkpoint`: Model checkpointing
- `load_checkpoint`: Checkpoint loading

---

## Quick Reference

### Common Commands

#### PACS Integration
```bash
# Start PACS service with production config
python -c "from src.clinical.pacs import PACSService; \
  with PACSService(config_path='.kiro/pacs/config.production.yaml') as s: \
    print(s.health_check())"

# Query PACS for studies
python -c "from src.clinical.pacs import PACSService; \
  with PACSService(config_path='.kiro/pacs/config.yaml') as s: \
    studies = s.query_studies(patient_id='12345678', modality='SM'); \
    print(f'Found {len(studies)} studies')"

# Get PACS statistics
python -c "from src.clinical.pacs import PACSService; \
  with PACSService(config_path='.kiro/pacs/config.yaml') as s: \
    print(s.get_statistics())"
```

#### WSI Processing
```bash
# Process WSI files
python -m src.data.wsi_pipeline.cli process slide.svs --output-dir ./features

# Batch processing
python -m src.data.wsi_pipeline.cli process *.svs --config config.yaml

# Validate installation
python -m src.data.wsi_pipeline.cli validate

# Performance benchmarks
python -m src.data.wsi_pipeline.cli benchmark --quick
```

#### Training
```bash
# PCam training
python experiments/train_pcam.py --config experiments/configs/pcam.yaml

# CAMELYON training
python experiments/train_camelyon.py --config experiments/configs/camelyon.yaml
```

#### Evaluation
```bash
# PCam evaluation
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam/best_model.pth \
  --data-root data/pcam

# CAMELYON evaluation with CSV export
python experiments/evaluate_camelyon.py \
  --checkpoint checkpoints/camelyon/best_model.pth \
  --save-predictions-csv
```

#### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

#### Model Profiling
```bash
# Profile inference time
python scripts/model_profiler.py \
  --checkpoint models/best_model.pth \
  --profile-type time

# Export to ONNX
python scripts/export_onnx.py \
  --checkpoint models/best_model.pth \
  --output models/model.onnx
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size in config file
- Use gradient accumulation
- Enable mixed precision training

#### Slow Training
- Check GPU utilization with `nvidia-smi`
- Increase number of data loader workers
- Use faster data augmentation

#### Poor Model Performance
- Verify data preprocessing
- Check learning rate schedule
- Increase training epochs
- Try different architectures

---

## Additional Resources

### External Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
- [timm Documentation](https://huggingface.co/docs/timm/)

### Research Papers
- **PatchCamelyon**: Veeling et al. (2018) - "Rotation Equivariant CNNs for Digital Pathology"
- **CAMELYON16**: Bejnordi et al. (2017) - "Diagnostic Assessment of Deep Learning Algorithms"

### Community
- [GitHub Issues](https://github.com/matthewvaishnav/histocore/issues)
- [Discussions](https://github.com/matthewvaishnav/histocore/discussions)

---

## Archived Documentation

Historical documentation and implementation notes are available in [archive/](archive/).

---

<div class="footer-note">
  <p><em>Last updated: April 2026</em></p>
  <p>For questions or suggestions, please <a href="https://github.com/matthewvaishnav/histocore/issues">open an issue</a>.</p>
</div>
