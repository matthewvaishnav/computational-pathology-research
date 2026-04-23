# WSI Processing Pipeline

A comprehensive pipeline for processing Whole Slide Images (WSI) in clinical formats for computational pathology research and applications.

## 🚀 Features

### Core Capabilities
- **Multi-format Support**: .svs, .tiff, .ndpi, DICOM WSI files
- **Efficient Processing**: Streaming patch extraction with memory optimization
- **Feature Extraction**: Multiple CNN encoders (ResNet-50, DenseNet-121, EfficientNet-B0)
- **Tissue Detection**: Fast Otsu thresholding with optional deep learning enhancement
- **HDF5 Caching**: Optimized storage with compression and chunking
- **Batch Processing**: Parallel processing with GPU acceleration
- **Quality Control**: Comprehensive quality metrics and validation

### Advanced Features
- **Memory Management**: Automatic batch size optimization and memory monitoring
- **Progress Tracking**: Real-time progress with ETA calculation
- **Error Recovery**: Robust error handling with retry logic
- **Configuration Management**: Flexible YAML/JSON configuration with validation
- **CLI Interface**: Command-line tools for easy deployment
- **Comprehensive Testing**: Validation suite and performance benchmarks

## 📦 Installation

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision
pip install openslide-python wsidicom h5py
pip install scikit-image numpy pandas
pip install tqdm pyyaml psutil

# Optional: Additional encoders
pip install timm
```

### Install from Source
```bash
# Clone repository
git clone <repository-url>
cd computational-pathology-research

# Install in development mode
pip install -e .
```

## 🔧 Quick Start

### Basic Usage
```python
from data.wsi_pipeline import BatchProcessor, ProcessingConfig

# Configure pipeline
config = ProcessingConfig(
    patch_size=256,
    encoder_name="resnet50",
    batch_size=32,
    tissue_threshold=0.5
)

# Process single slide
processor = BatchProcessor(config, num_workers=4)
result = processor.process_slide("slide.svs")

print(f"Processed {result.num_patches} patches")
print(f"Features saved to: {result.cache_path}")
```

### Command Line Interface
```bash
# Process WSI files
python -m data.wsi_pipeline.cli process *.svs --output-dir ./features

# Run performance benchmarks
python -m data.wsi_pipeline.cli benchmark --quick

# Validate installation
python -m data.wsi_pipeline.cli validate

# Generate configuration template
python -m data.wsi_pipeline.cli config --create-template high_throughput --output config.yaml
```

### Batch Processing
```python
from data.wsi_pipeline import BatchProcessor, ProcessingConfig

# Configure for high-throughput processing
config = ProcessingConfig(
    patch_size=224,
    encoder_name="efficientnet_b0",
    batch_size=64,
    num_workers=8,
    max_memory_gb=16.0
)

# Process multiple slides
processor = BatchProcessor(config, num_workers=8, gpu_ids=[0, 1])
results = processor.process_batch([
    "slide1.svs",
    "slide2.svs", 
    "slide3.svs"
])

# Check results
for result in results:
    if result.success:
        print(f"✅ {result.slide_path}: {result.num_patches} patches")
    else:
        print(f"❌ {result.slide_path}: {result.error}")
```

## ⚙️ Configuration

### Configuration File (YAML)
```yaml
# config.yaml
patch_size: 256
stride: 256
level: 0
encoder_name: "resnet50"
tissue_method: "otsu"
tissue_threshold: 0.5
batch_size: 32
num_workers: 4
max_memory_gb: 8.0
cache_dir: "./wsi_features"
compression: "gzip"
```

### Programmatic Configuration
```python
from data.wsi_pipeline import ProcessingConfig, get_recommended_config

# Use recommended configuration for specific use case
config = get_recommended_config(
    use_case="high_quality",  # Options: general, high_throughput, high_quality, memory_limited
    hardware="gpu"            # Options: auto, gpu, cpu, high_memory, low_memory
)

# Custom configuration
config = ProcessingConfig(
    patch_size=512,
    encoder_name="resnet50",
    tissue_method="hybrid",
    batch_size=16,
    use_mixed_precision=True,
    compile_model=True
)
```

### Configuration Validation
```python
from data.wsi_pipeline import validate_config, ConfigValidator

# Validate configuration
try:
    validate_config(config)
    print("✅ Configuration is valid")
except ProcessingError as e:
    print(f"❌ Configuration error: {e}")

# Get validation details
validator = ConfigValidator()
is_valid, errors = validator.validate_config(config)
if not is_valid:
    for error in errors:
        print(f"  - {error}")
```

## 🏗️ Architecture

### Core Components

```
WSI Processing Pipeline
├── WSIReader          # Multi-format WSI file reading
├── PatchExtractor     # Efficient patch sampling
├── TissueDetector     # Tissue region detection
├── FeatureGenerator   # CNN-based feature extraction
├── FeatureCache       # HDF5 storage with optimization
├── BatchProcessor     # Orchestration and parallelization
└── QualityControl     # Quality metrics and validation
```

### Data Flow
```
WSI File → WSIReader → PatchExtractor → TissueDetector → FeatureGenerator → FeatureCache
                                    ↓
                              QualityControl ← BatchProcessor
```

### Output Format
The pipeline generates HDF5 files compatible with existing datasets:

```python
# HDF5 structure
slide.h5
├── features/          # [num_patches, feature_dim] float32
├── coordinates/       # [num_patches, 2] int32
└── attributes/        # Metadata (slide_id, encoder, etc.)
```

## 📊 Performance

### Benchmarks (CPU)
- **Patch Extraction**: >2000 patches/sec
- **Tissue Detection**: >1000 patches/sec  
- **HDF5 Write Speed**: >18 MB/sec
- **Feature Extraction**: ~22 patches/sec (CPU), >500 patches/sec (GPU)

### Memory Efficiency
- **Streaming Processing**: <1GB RAM for typical slides
- **Automatic Optimization**: Dynamic batch size adjustment
- **Resource Monitoring**: Memory usage tracking and limits

### Storage Optimization
- **HDF5 Compression**: 1.2-2.7x size reduction
- **Chunked Storage**: Efficient partial reads
- **Metadata Preservation**: Complete processing provenance

## 🧪 Testing and Validation

### Run Tests
```bash
# Basic functionality test
python scripts/test_wsi_pipeline.py --skip-benchmarks

# Full validation suite
python -m data.wsi_pipeline.cli validate

# Performance benchmarks
python -m data.wsi_pipeline.cli benchmark --quick
```

### Validation Results
```
✅ Component Initialization: 7/7 components
✅ End-to-End Pipeline: Synthetic data processing
✅ Memory Efficiency: Resource management
✅ HDF5 Compatibility: CAMELYONSlideDataset integration
```

## 🔧 Advanced Usage

### Custom Encoders
```python
from data.wsi_pipeline import FeatureGenerator

# Use custom encoder from timm
generator = FeatureGenerator(
    encoder_name="vit_base_patch16_224",  # Vision Transformer
    pretrained=True,
    device="cuda"
)
```

### Progress Monitoring
```python
from data.wsi_pipeline import ProgressTracker

# Track processing progress
tracker = ProgressTracker(
    total_items=len(slide_paths),
    description="Processing WSI files",
    show_progress_bar=True
)

tracker.start()
for slide_path in slide_paths:
    tracker.start_item(slide_path)
    # ... process slide ...
    tracker.complete_item(slide_path, patches_processed=1000)

stats = tracker.finish()
print(f"Processed {stats.patches_processed} patches in {stats.elapsed_time}")
```

### Memory Optimization
```python
from data.wsi_pipeline import BatchProcessor

# Configure memory limits
processor = BatchProcessor(
    config=config,
    max_memory_gb=8.0,  # Limit memory usage
    num_workers=4
)

# Automatic batch size optimization
optimal_batch = processor.memory_monitor.get_optimal_batch_size(
    target_batch_size=64
)
```

### Quality Control
```python
from data.wsi_pipeline import QualityControl

qc = QualityControl(
    blur_threshold=100.0,
    min_tissue_coverage=0.1
)

# Generate quality report
qc_report = qc.generate_qc_report(
    slide_id="slide_001",
    patches=patches,
    features=features,
    tissue_coverage=0.85
)

print(f"Quality score: {qc_report['overall_quality']}")
```

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Errors**
```python
# Reduce batch size
config.batch_size = 16

# Enable automatic fallback
generator.fallback_to_cpu()
```

**Slow Processing**
```python
# Use faster encoder
config.encoder_name = "efficientnet_b0"

# Optimize tissue detection
config.tissue_method = "otsu"  # Faster than deep learning

# Increase workers
config.num_workers = 8
```

**File Format Issues**
```python
# Check supported formats
from data.wsi_pipeline import WSIReader

try:
    with WSIReader("slide.svs") as reader:
        print(f"Slide dimensions: {reader.dimensions}")
except FileFormatError as e:
    print(f"Unsupported format: {e}")
```

### Logging and Debugging
```python
from data.wsi_pipeline import setup_logging

# Enable debug logging
setup_logging(level='DEBUG', log_file='wsi_processing.log')

# Component-specific logging
import logging
logging.getLogger('data.wsi_pipeline.reader').setLevel(logging.DEBUG)
```

## 📚 API Reference

### Core Classes
- [`ProcessingConfig`](config.py): Configuration management
- [`BatchProcessor`](batch_processor.py): Main processing orchestrator
- [`WSIReader`](reader.py): WSI file reading
- [`FeatureGenerator`](feature_generator.py): CNN feature extraction
- [`FeatureCache`](cache.py): HDF5 storage management

### Utilities
- [`ConfigValidator`](config_validator.py): Configuration validation
- [`ProgressTracker`](progress_tracker.py): Progress monitoring
- [`PerformanceBenchmark`](benchmarks.py): Performance testing

### CLI Commands
- `process`: Process WSI files
- `benchmark`: Run performance tests
- `validate`: Validate installation
- `config`: Configuration management

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 src/data/wsi_pipeline/
```

### Adding New Features
1. Implement feature in appropriate module
2. Add comprehensive tests
3. Update documentation
4. Run validation suite

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenSlide library for WSI file reading
- PyTorch ecosystem for deep learning
- HDF5 for efficient data storage
- scikit-image for image processing algorithms

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the validation suite: `python -m data.wsi_pipeline.cli validate`
3. Enable debug logging for detailed error information
4. Open an issue with system information and error logs