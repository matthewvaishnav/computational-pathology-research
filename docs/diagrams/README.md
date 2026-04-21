# Architecture Diagrams

This directory contains generated architecture diagrams for HistoCore.

## Available Diagrams

### System Overview
- **File**: `system_overview.mmd`
- **Description**: High-level system architecture showing data flow from input modalities through processing to predictions
- **Components**: Input layer, encoders, fusion, task heads

### Cross-Modal Attention
- **File**: `cross_modal_attention.mmd`  
- **Description**: Detailed view of the cross-modal attention mechanism
- **Components**: Pairwise attention, fusion, projection

### Training Pipeline
- **File**: `training_pipeline.mmd`
- **Description**: Complete training workflow from data loading to optimization
- **Components**: Data loading, forward pass, loss computation, optimization

### Clinical Deployment
- **File**: `clinical_deployment.mmd`
- **Description**: Production deployment architecture for clinical environments
- **Components**: Data sources, processing pipeline, decision support

## Usage

### Viewing Diagrams
1. **GitHub/GitLab**: Diagrams render automatically in markdown files
2. **VS Code**: Install Mermaid Preview extension
3. **Online**: Copy content to [mermaid.live](https://mermaid.live)

### Exporting to Images
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export to PNG
mmdc -i system_overview.mmd -o system_overview.png -t neutral -b white

# Export to SVG  
mmdc -i system_overview.mmd -o system_overview.svg -t neutral -b white
```

### Regenerating Diagrams
```bash
# Generate all diagrams
python scripts/generate_architecture_diagrams.py

# Generate with PNG export
python scripts/generate_architecture_diagrams.py --format png
```

## Generated Files

- `system_overview.mmd`
- `cross_modal_attention.mmd`
- `training_pipeline.mmd`
- `clinical_deployment.mmd`

---

**Last Generated**: 1776786584.314523  
**Generator**: `scripts/generate_architecture_diagrams.py`  
**Status**: Auto-generated - do not edit manually ⚠️
