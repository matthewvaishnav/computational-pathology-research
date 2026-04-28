# HistoCore Examples

This directory contains example scripts, Jupyter notebooks, and demonstrations for HistoCore's capabilities.

## 📚 Interactive Tutorials

### 1. Quickstart: PCam Training (5 minutes)
**File**: `quickstart_pcam_training.ipynb`

Train a state-of-the-art AttentionMIL model on PatchCamelyon in just 5 minutes!

**What you'll learn:**
- Load the PCam dataset
- Configure an AttentionMIL model
- Train with optimized settings (8-12x faster)
- Evaluate performance with bootstrap CI
- Visualize attention maps

**Expected results:** ~93-94% test AUC in 2-3 hours on RTX 4070

```bash
jupyter notebook quickstart_pcam_training.ipynb
```

### 2. Custom Dataset Tutorial
**File**: `custom_dataset_tutorial.ipynb`

Adapt HistoCore to your own histopathology dataset.

**What you'll learn:**
- Prepare your WSI data
- Create custom PyTorch datasets
- Configure data augmentation
- Train on custom data
- Evaluate and interpret results

```bash
jupyter notebook custom_dataset_tutorial.ipynb
```

---

## 🎬 Real-Time WSI Streaming Demo

**File**: `streaming_demo.py`

Comprehensive demonstration of the real-time WSI streaming system showing breakthrough capabilities:
- <30 second processing for gigapixel slides
- <2GB memory footprint
- Real-time confidence updates
- Progressive visualization

### Quick Start

```bash
# Create a synthetic WSI for testing
python streaming_demo.py --create-synthetic test.tiff

# Run all demos
python streaming_demo.py --wsi test.tiff

# Run specific demo
python streaming_demo.py --wsi test.tiff --demo basic
```

### Available Demos

1. **Basic Processing** - Default configuration with full metrics
2. **Convenience Function** - One-line processing API
3. **Custom Configuration** - Optimized for specific requirements
4. **Batch Processing** - Concurrent multi-slide processing
5. **Memory Constrained** - Strict 1GB memory limit
6. **Confidence Tracking** - Progressive confidence building

### Command-Line Options

```
usage: streaming_demo.py [-h] [--wsi WSI] [--demo {all,basic,convenience,custom,batch,memory,confidence}]
                         [--create-synthetic CREATE_SYNTHETIC] [--synthetic-size WIDTH HEIGHT]

Real-Time WSI Streaming Demo

optional arguments:
  -h, --help            show this help message and exit
  --wsi WSI             Path to WSI file (.svs, .tiff, .ndpi, DICOM)
  --demo {all,basic,convenience,custom,batch,memory,confidence}
                        Which demo to run (default: all)
  --create-synthetic CREATE_SYNTHETIC
                        Create a synthetic WSI file for testing
  --synthetic-size WIDTH HEIGHT
                        Size of synthetic WSI (default: 10000 10000)
```

### Examples

```bash
# Run all demos with a real WSI file
python streaming_demo.py --wsi slide.svs

# Run only the basic demo
python streaming_demo.py --wsi slide.svs --demo basic

# Create a large synthetic WSI
python streaming_demo.py --create-synthetic large.tiff --synthetic-size 50000 50000

# Test with synthetic WSI
python streaming_demo.py --create-synthetic test.tiff --wsi test.tiff --demo all
```

### Output

Each demo provides detailed output including:
- Processing time and throughput
- Memory usage (peak and average)
- Confidence progression
- Performance requirements validation
- Attention weight statistics

Example output:
```
================================================================================
DEMO 1: Basic Real-Time Processing
================================================================================

Processing: slide.svs
Configuration:
  - Target time: 30.0s
  - Memory budget: 2.0GB
  - Confidence threshold: 0.95
  - Batch size: 64

================================================================================
RESULTS:
================================================================================
Prediction: 1
Confidence: 0.967
Processing time: 27.34s
Patches processed: 98,432
Throughput: 3,601.2 patches/sec
Peak memory: 1.87GB
Average memory: 1.64GB
Early stopped: True

================================================================================
PERFORMANCE REQUIREMENTS:
================================================================================
✓ Time requirement (<30s): True
✓ Memory requirement (<2GB): True
✓ Confidence requirement (>80%): True
✓ All requirements met: True
```

## Other Examples

### Coming Soon:
- **Model Interpretability Tutorial** - Grad-CAM, SHAP, attention analysis
- **Federated Learning Example** - Multi-site training with differential privacy
- **PACS Integration Example** - Hospital system integration
- **Batch Inference Example** - Production-scale slide processing
- **Custom Model Architecture** - Build your own MIL models

### Training Scripts:
See `experiments/` directory for production training scripts:
- `train_pcam.py` - PCam training with all optimizations
- `evaluate.py` - Model evaluation with bootstrap CI
- `generate_pcam_interpretability.py` - Attention map generation

---

## 🚀 Quick Start

**Option 1: Interactive Notebooks**
```bash
# Install Jupyter
pip install jupyter

# Launch notebook
jupyter notebook quickstart_pcam_training.ipynb
```

**Option 2: Command Line**
```bash
# Train on PCam (ultra-fast config)
python ../experiments/train_pcam.py --config ../experiments/configs/pcam_ultra_fast.yaml

# Expected: 93-94% test AUC in 2.25 hours on RTX 4070
```

---

## 📊 Performance Benchmarks

| Configuration | Training Time | Test AUC | GPU | Parameters |
|---------------|---------------|----------|-----|------------|
| **Ultra Fast** | 2.25 hours | 93.94% | RTX 4070 | 12M |
| Fast Improved | 3.1 hours | 94.2% | RTX 4070 | 18M |
| Full Scale | 5.5 hours | 94.5% | RTX 4070 | 25M |

See [Performance Comparison](../docs/PERFORMANCE_COMPARISON.html) for detailed benchmarks vs competitors.

---

## Requirements

See `requirements.txt` in the project root for dependencies.

## Documentation

For more information, see:
- [Real-Time WSI Streaming README](../src/streaming/README.md)
- [Complete System Status](../STREAMING_COMPLETE.md)
- [Specification](../.kiro/specs/real-time-wsi-streaming/)
