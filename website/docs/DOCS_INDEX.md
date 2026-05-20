---
title: Documentation Index
description: Navigation guide for HistoCore platform, modeling, deployment, and validation documentation.
---


# Documentation Index

Comprehensive documentation for HistoCore - the production-grade computational pathology framework.

---

## Project Status

- [**Current Status (May 2026)**](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/CURRENT_STATUS.md) - Latest development progress, benchmarks, and roadmap

## Getting Started

### Installation and Setup
- [Getting Started Guide](/docs/GETTING_STARTED) - Complete installation and setup guide
- [System Requirements](/docs/GETTING_STARTED#system-requirements) - Hardware and software prerequisites
- [Installation](/docs/GETTING_STARTED#installation) - Step-by-step installation instructions

### Tutorials
- [Your First Model](/docs/GETTING_STARTED#your-first-model) - Step-by-step PCam tutorial
- [Working with Real Data](/docs/GETTING_STARTED#working-with-real-data) - PatchCamelyon and CAMELYON16 workflows
- [API Reference](/docs/API_REFERENCE) - Complete API documentation

---

## Training and Evaluation

### PatchCamelyon (PCam)
- [PCAM_REAL_RESULTS.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/PCAM_REAL_RESULTS.md) - **Real dataset results**: 85.26% accuracy, 0.9394 AUC with bootstrap CIs
- [PCAM_FAILURE_ANALYSIS.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/PCAM_FAILURE_ANALYSIS.md) - **Failure analysis**: 26.11% false negative rate, clinical implications
- [THRESHOLD_OPTIMIZATION.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/THRESHOLD_OPTIMIZATION.md) - **Clinical optimization**: 90% sensitivity for cancer screening
- [PCAM_CROSS_VALIDATION.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/PCAM_CROSS_VALIDATION.md) - **Cross-validation**: K-fold validation for robustness assessment
- [PCAM_BENCHMARK_RESULTS.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/PCAM_BENCHMARK_RESULTS.md) - Synthetic benchmark results and analysis
- [PCAM_COMPARISON_GUIDE.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/PCAM_COMPARISON_GUIDE.md) - Comparing model architectures
- **Topics**: Patch-level classification, data augmentation, baseline comparisons, clinical deployment, robustness validation

### CAMELYON16 Slide-Level
- [CAMELYON_TRAINING_STATUS.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/CAMELYON_TRAINING_STATUS.md) - Training guide and best practices
- [CAMELYON_SLIDE_LEVEL_IMPLEMENTATION.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/CAMELYON_SLIDE_LEVEL_IMPLEMENTATION.md) - Implementation details
- **Topics**: Slide-level aggregation, attention mechanisms, feature extraction

### WSI Processing Pipeline
- [WSI_PROCESSING_PIPELINE.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/WSI_PROCESSING_PIPELINE.md) - **Complete WSI processing pipeline**: OpenSlide integration, CLI tools, production deployment
- **Topics**: Multi-format WSI support (.svs, .tiff, .ndpi, DICOM), streaming processing, CNN feature extraction, HDF5 caching, clinical deployment

### Advanced Features
- [Model Interpretability Guide](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/MODEL_INTERPRETABILITY.md) - Grad-CAM, attention visualization, failure analysis
- [Clinical Workflow Integration](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/CLINICAL_WORKFLOW_INTEGRATION.md) - Multi-class classification, DICOM/FHIR support
- [Comprehensive Dataset Testing](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/COMPREHENSIVE_DATASET_TESTING.md) - 3,171 tests with property-based testing
- [Inference Optimization](/docs/INFERENCE_OPTIMIZATION) - Sliding-window inference, TorchScript, uncertainty-aware aggregation
- [Foundation Models](/docs/FOUNDATION_MODELS) - UNI, CONCH, Phikon, GigaPath, ResNet50 compatibility

### Evaluation Metrics
- Model performance analysis with bootstrap confidence intervals
- ROC curves and confusion matrices
- CSV export for downstream analysis
- Attention weight visualization and heatmap generation

---

## Architecture and Design

### System Overview
- [ARCHITECTURE.md](/docs/ARCHITECTURE) - Complete system architecture
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
- [DOCKER.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/DOCKER.md) - Complete Docker guide
- **Topics**: Container setup, GPU support, production deployment

### REST API
- [deploy/README.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/deploy/README.md) - API deployment instructions
- **Endpoints**: Model inference, batch processing, health checks

### Model Export
- **ONNX Export**: Cross-platform deployment
- **TorchScript**: Production optimization
- **Quantization**: Model compression

---

## Development

### Contributing
- [CONTRIBUTING.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/CONTRIBUTING.md) - Contribution guidelines
- **Topics**: Code style, pull requests, issue reporting

### Testing
- [TESTING.md](/docs/TESTING) - **Comprehensive testing documentation**: 3,171 tests, 55% coverage, CI/CD pipeline
- [TESTING_SUMMARY.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/TESTING_SUMMARY.md) - Historical test suite documentation
- **Coverage**: Unit tests, integration tests, property-based testing, clinical validation
- **Topics**: Test execution, coverage reports, benchmarks, quality assurance, reproducibility, nnMIL migration properties, sliding-window inference validation, foundation-model compatibility

### Build System
- [MAKEFILE.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/MAKEFILE.md) - Makefile usage guide
- **Commands**: Build, test, lint, format, clean

---

## Performance and Optimization

### Performance Analysis
- [PERFORMANCE.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/PERFORMANCE.md) - Optimization guide
- [PERFORMANCE_COMPARISON.md](/docs/PERFORMANCE_COMPARISON) - **8-12x speedup**: Benchmark vs PathML, CLAM
- [INFERENCE_OPTIMIZATION.md](/docs/INFERENCE_OPTIMIZATION) - **2-3x faster inference**: TorchScript, quantization
- **Topics**: Profiling, bottleneck analysis, GPU utilization, training optimization

### Model Optimization
- [QUANTIZATION.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/QUANTIZATION.md) - **INT8/FP16 quantization**: 4x compression, 2-3x speedup
- [MULTI_GPU_TRAINING.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/MULTI_GPU_TRAINING.md) - **DistributedDataParallel**: Linear GPU scaling
- [FOUNDATION_MODELS.md](/docs/FOUNDATION_MODELS) - **Pretrained models**: UNI, Phikon, CONCH
- [INFERENCE_OPTIMIZATION.md](/docs/INFERENCE_OPTIMIZATION) - **Sliding-window inference**: large-bag support, attention compatibility, uncertainty aggregation
- **Topics**: Model compression, distributed training, transfer learning, large-slide inference

### Model Profiling
- Inference latency measurement
- Memory usage analysis
- Throughput optimization

---

## Results and Analysis

### Benchmark Results
- [DEMO_RESULTS.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/DEMO_RESULTS.md) - Demo training results
- **Metrics**: Accuracy, AUC, training time, convergence analysis

### Project Portfolio
- [PORTFOLIO_SUMMARY.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/PORTFOLIO_SUMMARY.md) - Complete project overview
- **Status**: Current capabilities, limitations, future work

### Roadmap
- [ROADMAP_TO_REAL_DATASETS.md](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/docs/ROADMAP_TO_REAL_DATASETS.md) - Real dataset integration plan
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

Historical documentation and implementation notes are available in [archive/](https://github.com/matthewvaishnav/computational-pathology-research/tree/main/docs/archive).

---

&lt;div class="footer-note">
  &lt;p&gt;&lt;strong&gt;📊 Current Status:&lt;/strong&gt; Week 7 of 12 for TransnnMIL v2.0 development. See &lt;a href="CURRENT_STATUS">Current Status&lt;/a&gt; for detailed progress.&lt;/p&gt;
  &lt;p&gt;&lt;em&gt;Last updated: May 14, 2026&lt;/em&gt;&lt;/p&gt;
  &lt;p&gt;For questions or suggestions, please &lt;a href="https://github.com/matthewvaishnav/histocore/issues">open an issue&lt;/a&gt;.&lt;/p&gt;
&lt;/div&gt;
