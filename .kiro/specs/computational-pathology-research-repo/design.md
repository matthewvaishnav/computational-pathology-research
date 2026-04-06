# Technical Design Document

## Overview

This document specifies the technical design for a computational pathology research repository implementing novel multimodal fusion architectures. The system combines whole-slide images (WSI), genomic features, and clinical text through attention-based fusion, incorporates cross-slide temporal reasoning for disease progression analysis, implements transformer-based stain normalization, and provides self-supervised pretraining objectives tailored to histopathology data.

The repository is designed as a research artifact demonstrating computational innovation in pathology image analysis. It provides modular, extensible implementations of novel architectural components with comprehensive training, evaluation, and ablation study frameworks.

### Design Goals

1. **Modularity**: Each component (fusion, temporal reasoning, stain normalization, pretraining) is independently usable
2. **Reproducibility**: All experiments are fully specified through configuration files with deterministic behavior
3. **Extensibility**: Architecture supports adding new modalities, pretraining objectives, and evaluation metrics
4. **Research-Focused**: Emphasis on algorithmic innovation and experimental rigor rather than production deployment

## Architecture

### High-Level System Architecture

The system consists of eight major subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Repository                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────────────┐    │
│  │  Data Pipeline   │─────▶│  Stain Normalization     │    │
│  │  - Loaders       │      │  Transformer             │    │
│  │  - Preprocessing │      └──────────┬───────────────┘    │
│  └──────────────────┘                 │                     │
│                                        ▼                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Multimodal Fusion Architecture               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │   WSI    │  │ Genomic  │  │  Clinical Text   │  │  │
│  │  │ Encoder  │  │ Encoder  │  │    Encoder       │  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────────────┘  │  │
│  │       │             │              │                 │  │
│  │       └─────────────┴──────────────┘                 │  │
│  │                     │                                 │  │
│  │          ┌──────────▼──────────┐                     │  │
│  │          │  Cross-Modal        │                     │  │
│  │          │  Attention Fusion   │                     │  │
│  │          └──────────┬──────────┘                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                        │                                     │
│  ┌─────────────────────▼──────────────────────────────┐    │
│  │     Cross-Slide Temporal Reasoner                  │    │
│  │  - Temporal Attention                              │    │
│  │  - Progression Features                            │    │
│  └─────────────────────┬──────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────▼──────────────────────────────┐    │
│  │          Task-Specific Heads                       │    │
│  │  - Classification                                  │    │
│  │  - Survival Prediction                             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     Self-Supervised Pretraining Framework           │  │
│  │  - Contrastive Learning                             │  │
│  │  - Reconstruction Objectives                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     Training & Evaluation Pipeline                   │  │
│  │  - Config Management                                 │  │
│  │  - Checkpointing                                     │  │
│  │  - Metrics & Logging                                 │  │
│  │  - Ablation Studies                                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Framework**: PyTorch 2.0+ (for dynamic computation graphs and extensive ecosystem)
- **Configuration**: YAML files with Hydra for hierarchical configuration management
- **Logging**: TensorBoard for training metrics, Python logging for system events
- **Checkpointing**: PyTorch native format (.pth) with full state dict serialization
- **Data Format**: HDF5 for preprocessed WSI features, JSON for metadata, CSV for tabular data
- **Visualization**: Matplotlib/Seaborn for static plots, Plotly for interactive visualizations
- **Notebooks**: Jupyter for exploration and analysis
- **Testing**: pytest for unit tests, property-based testing NOT applicable (see Testing Strategy)

## Components and Interfaces

### 1. Data Pipeline (`src/data/`)

#### 1.1 Dataset Loaders (`src/data/loaders.py`)

**Purpose**: Load and batch multimodal data with support for missing modalities.

**Key Classes**:

```python
class MultimodalDataset(torch.utils.data.Dataset):
    """
    Loads WSI features, genomic data, and clinical text for a patient cohort.
    
    Handles missing modalities by returning None for unavailable data.
    """
    def __init__(self, data_dir: Path, split: str, config: DictConfig):
        """
        Args:
            data_dir: Root directory containing processed data
            split: One of 'train', 'val', 'test'
            config: Configuration dict with modality settings
        """
        
    def __getitem__(self, idx: int) -> Dict[str, Optional[Tensor]]:
        """
        Returns:
            {
                'wsi_features': Tensor [num_patches, feature_dim] or None,
                'genomic': Tensor [num_genes] or None,
                'clinical_text': Tensor [seq_len] or None,
                'label': Tensor (scalar or multi-label),
                'patient_id': str,
                'timestamp': Optional[float]
            }
        """
        
    def __len__(self) -> int:
        """Returns number of samples in split"""


class TemporalDataset(torch.utils.data.Dataset):
    """
    Groups multiple slides from same patient for temporal reasoning.
    
    Returns sequences of slides ordered by timestamp.
    """
    def __init__(self, data_dir: Path, split: str, config: DictConfig):
        """
        Args:
            data_dir: Root directory containing processed data
            split: One of 'train', 'val', 'test'
            config: Configuration dict with temporal settings
        """
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            {
                'slide_sequence': List[Dict] - ordered list of slide data,
                'timestamps': Tensor [num_slides],
                'patient_id': str,
                'label': Tensor
            }
        """
        
    def __len__(self) -> int:
        """Returns number of patients in split"""
```

