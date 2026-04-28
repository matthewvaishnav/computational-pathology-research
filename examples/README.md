# HistoCore Examples

This directory contains example scripts and demonstrations for HistoCore's capabilities.

## Real-Time WSI Streaming Demo

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

More examples coming soon:
- Training examples
- Evaluation examples
- Visualization examples
- Integration examples

## Requirements

See `requirements.txt` in the project root for dependencies.

## Documentation

For more information, see:
- [Real-Time WSI Streaming README](../src/streaming/README.md)
- [Complete System Status](../STREAMING_COMPLETE.md)
- [Specification](../.kiro/specs/real-time-wsi-streaming/)
