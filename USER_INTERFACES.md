# HistoCore User Interfaces

Three ways to use HistoCore - choose what works best for you.

## 🚀 Quick Start

```bash
# One-click launcher (recommended)
python histocore.py

# Or install first
python install.py
python histocore.py
```

## 1. 🖥️ Desktop GUI (Recommended)

**QuPath-like interface** with drag-and-drop WSI analysis.

### Features
- Drag-and-drop WSI files (.svs, .tiff, .ndpi)
- Real-time progress tracking
- Interactive settings (model, patch size, thresholds)
- Results visualization with attention heatmaps
- Analysis log with detailed status

### Launch
```bash
# From launcher
python histocore.py  # Choose option 1

# Direct launch
python src/gui/main_window.py

# Via CLI
python -m src.cli.main gui
```

### Requirements
- PyQt6 (auto-installed)
- matplotlib
- 4GB+ RAM recommended

## 2. 🌐 Web Interface

**Browser-based** interface accessible from any device.

### Features
- No installation required (just browser)
- Drag-and-drop file upload
- Real-time analysis progress
- Results download as JSON
- Mobile-friendly responsive design

### Launch
```bash
# From launcher
python histocore.py  # Choose option 2

# Direct launch
python src/web/app.py

# Via CLI
python -m src.cli.main web
```

### Access
- Open browser to: http://localhost:5000
- Works on any device with web browser
- Upload limit: 2GB per file

### Requirements
- Flask (auto-installed)
- Any modern web browser

## 3. 💻 Command Line Interface

**Terminal-based** for automation and scripting.

### Features
- Batch processing multiple files
- Scriptable for automation
- Detailed progress output
- JSON results export
- Integration with shell scripts

### Commands
```bash
# Analyze single file
histocore analyze slide.svs --output results/

# Batch analyze
histocore batch-analyze *.svs --model resnet50

# Quick demo
histocore demo --quick

# System info
histocore info

# Launch GUI/web
histocore gui
histocore web
```

### Launch
```bash
# From launcher
python histocore.py  # Choose option 3

# Direct CLI
python -m src.cli.main --help
```

## Installation

### Automatic (Recommended)
```bash
python install.py
```

### Manual
```bash
pip install -r requirements.txt
pip install PyQt6 flask click
```

### Dependencies
- **Core**: torch, numpy, matplotlib, scikit-learn
- **GUI**: PyQt6
- **Web**: flask
- **CLI**: click
- **Optional**: openslide-python (WSI support)

## File Support

All interfaces support:
- **.svs** (Aperio)
- **.tiff/.tif** (Generic TIFF)
- **.ndpi** (Hamamatsu)
- **.vms/.vmu** (Hamamatsu)
- **.scn** (Leica)

## Performance

### GUI Interface
- **Memory**: 2-4GB typical usage
- **Processing**: Real-time progress updates
- **GPU**: Automatic detection and usage

### Web Interface
- **Upload**: Up to 2GB files
- **Concurrent**: Multiple users supported
- **Streaming**: Progressive results display

### CLI Interface
- **Batch**: Process 100+ files
- **Automation**: Shell script integration
- **Logging**: Detailed progress output

## Troubleshooting

### GUI Won't Start
```bash
# Install PyQt6
pip install PyQt6

# Check imports
python -c "import PyQt6; print('OK')"
```

### Web Interface 404
```bash
# Install Flask
pip install flask

# Check port availability
netstat -an | grep 5000
```

### CLI Command Not Found
```bash
# Use full path
python -m src.cli.main --help

# Or add to PATH
export PATH=$PATH:$(pwd)
```

### WSI Files Not Loading
```bash
# Install OpenSlide
pip install openslide-python

# Windows: Download OpenSlide binaries
# macOS: brew install openslide
# Linux: apt-get install openslide-tools
```

## Examples

### GUI Workflow
1. Launch: `python histocore.py` → Option 1
2. Drag WSI file to upload area
3. Adjust settings (model, patch size)
4. Click "Analyze WSI"
5. View results and attention heatmap

### Web Workflow
1. Launch: `python histocore.py` → Option 2
2. Open browser to http://localhost:5000
3. Drag WSI file to upload
4. Configure analysis settings
5. Click "Analyze" and wait for results

### CLI Workflow
```bash
# Single file
histocore analyze tumor_slide.svs --output results/

# Batch processing
histocore batch-analyze slides/*.svs --model densenet121

# Custom settings
histocore analyze slide.svs \
  --model efficientnet_b0 \
  --patch-size 512 \
  --tissue-threshold 0.7 \
  --output results/
```

## Next Steps

- **Clinical deployment**: See PACS integration docs
- **Custom models**: Train your own with experiments/
- **API integration**: Use FastAPI endpoints
- **Scaling**: Deploy with Docker/Kubernetes

Choose the interface that fits your workflow - all provide the same powerful HistoCore analysis capabilities.