# HistoCore Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HistoCore Framework                             │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Clinical Integration Layer                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │ PACS/DICOM   │  │  LIS/EMR     │  │  Regulatory Compliance   │ │ │
│  │  │  Integration │  │  Integration │  │  (FDA/CE/HIPAA)          │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Inference & Deployment                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │  Real-time   │  │  Batch       │  │  Model Interpretability  │ │ │
│  │  │  Inference   │  │  Processing  │  │  (Grad-CAM, Attention)   │ │ │
│  │  │  (<5 sec)    │  │              │  │                          │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Training & Optimization                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │  Optimized   │  │  Federated   │  │  Property-Based Testing  │ │ │
│  │  │  Training    │  │  Learning    │  │  (1,448 tests)           │ │ │
│  │  │  (8-12x)     │  │  (ε ≤ 1.0)   │  │                          │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Model Architecture Layer                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │ AttentionMIL │  │    CLAM      │  │      TransMIL            │ │ │
│  │  │              │  │              │  │                          │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Feature Extraction Layer                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │  ResNet18/50 │  │ EfficientNet │  │  Vision Transformer      │ │ │
│  │  │  (Pretrained)│  │              │  │                          │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Data Processing Layer                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │  WSI Loading │  │  Patch       │  │  Data Augmentation       │ │ │
│  │  │  (OpenSlide) │  │  Extraction  │  │  & Normalization         │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## AttentionMIL Model Architecture

```
Input: Whole Slide Image (WSI)
         │
         ├─> Patch Extraction (96x96 or 224x224)
         │   └─> N patches per slide
         │
         ▼
┌────────────────────────────────────────┐
│     Feature Extractor (ResNet18)       │
│  ┌──────────────────────────────────┐  │
│  │  Conv1 (7x7, 64)                 │  │
│  │  MaxPool                         │  │
│  │  ResBlock1 (64 channels)         │  │
│  │  ResBlock2 (128 channels)        │  │
│  │  ResBlock3 (256 channels)        │  │
│  │  ResBlock4 (512 channels)        │  │
│  │  AdaptiveAvgPool                 │  │
│  └──────────────────────────────────┘  │
│         Output: 512-dim features       │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│    Attention Mechanism (MIL)           │
│  ┌──────────────────────────────────┐  │
│  │  Linear(512 → 256)               │  │
│  │  Tanh Activation                 │  │
│  │  Linear(256 → 1)                 │  │
│  │  Softmax (across patches)        │  │
│  └──────────────────────────────────┘  │
│    Output: Attention weights α_i       │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│    Weighted Aggregation                │
│  ┌──────────────────────────────────┐  │
│  │  h = Σ(α_i * h_i)                │  │
│  │  where h_i are patch features    │  │
│  └──────────────────────────────────┘  │
│    Output: Slide-level embedding       │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│    Classification Head                 │
│  ┌──────────────────────────────────┐  │
│  │  Linear(512 → num_classes)       │  │
│  │  (Optional: Hidden layers)       │  │
│  └──────────────────────────────────┘  │
│    Output: Class logits                │
└────────────────────────────────────────┘
         │
         ▼
    Final Prediction
```

## Training Pipeline Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│                    Optimized Training Pipeline                   │
│                                                                  │
│  Input Data                                                      │
│      │                                                           │
│      ├─> Persistent Workers (avoid reload overhead)             │
│      │   └─> num_workers=4, persistent_workers=True             │
│      │                                                           │
│      ├─> Pin Memory (faster GPU transfer)                       │
│      │   └─> pin_memory=True                                    │
│      │                                                           │
│      ├─> Prefetch Factor (pipeline data loading)                │
│      │   └─> prefetch_factor=2                                  │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              GPU Optimizations                           │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Channels Last Memory Format                       │  │   │
│  │  │  └─> 20-30% faster convolutions                    │  │   │
│  │  ├────────────────────────────────────────────────────┤  │   │
│  │  │  Mixed Precision (AMP)                             │  │   │
│  │  │  └─> 2-3x faster, 50% less memory                  │  │   │
│  │  ├────────────────────────────────────────────────────┤  │   │
│  │  │  torch.compile (PyTorch 2.0+)                      │  │   │
│  │  │  └─> 30-40% faster forward/backward                │  │   │
│  │  ├────────────────────────────────────────────────────┤  │   │
│  │  │  cuDNN Benchmark                                   │  │   │
│  │  │  └─> Auto-tune convolution algorithms              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                           │
│      ▼                                                           │
│  Result: 8-12x faster training (20-40 hours → 2-3 hours)        │
└─────────────────────────────────────────────────────────────────┘
```

## Federated Learning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Learning System                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Central Orchestrator                       │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  • Model versioning                                  │  │ │
│  │  │  • FedAvg weighted aggregation                       │  │ │
│  │  │  • Drift detection                                   │  │ │
│  │  │  • Differential privacy (ε ≤ 1.0)                    │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│           │                  │                  │                │
│           ▼                  ▼                  ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Hospital 1 │    │  Hospital 2 │    │  Hospital 3 │         │
│  │             │    │             │    │             │         │
│  │  Local      │    │  Local      │    │  Local      │         │
│  │  Training   │    │  Training   │    │  Training   │         │
│  │             │    │             │    │             │         │
│  │  Private    │    │  Private    │    │  Private    │         │
│  │  Data       │    │  Data       │    │  Data       │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│           │                  │                  │                │
│           └──────────────────┴──────────────────┘                │
│                              │                                   │
│                              ▼                                   │
│                    Gradient Aggregation                          │
│                    (with DP noise)                               │
└─────────────────────────────────────────────────────────────────┘
```

## PACS Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PACS Integration Workflow                     │
│                                                                  │
│  Hospital PACS                                                   │
│      │                                                           │
│      ├─> DICOM C-FIND (Query for studies)                       │
│      │   └─> Patient ID, Study Date, Modality                   │
│      │                                                           │
│      ├─> DICOM C-MOVE (Retrieve images)                         │
│      │   └─> Transfer to HistoCore                              │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              HistoCore Processing                        │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  1. DICOM Validation                               │  │   │
│  │  │  2. Image Preprocessing                            │  │   │
│  │  │  3. Model Inference                                │  │   │
│  │  │  4. Result Generation                              │  │   │
│  │  │  5. Audit Logging (HIPAA)                          │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                           │
│      ├─> DICOM C-STORE (Send results back)                      │
│      │   └─> Structured reports, annotations                    │
│      │                                                           │
│      ├─> HL7 Messages (to LIS/EMR)                              │
│      │   └─> Patient results, alerts                            │
│      │                                                           │
│      ▼                                                           │
│  Clinical Workflow                                               │
│  (Pathologist review)                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Raw WSI → Patch Extraction → Feature Extraction → Attention Pooling → Classification
   │            │                    │                    │                │
   │            │                    │                    │                └─> Predictions
   │            │                    │                    └─> Attention Maps
   │            │                    └─> Patch Features (512-dim)
   │            └─> N patches (96x96 or 224x224)
   └─> Gigapixel image (10,000+ x 10,000+ pixels)
```

## Key Components

### 1. Data Processing
- **WSI Loading**: OpenSlide backend for multi-format support
- **Patch Extraction**: Sliding window with configurable stride
- **Augmentation**: Rotation, flip, color jitter, stain normalization
- **Normalization**: ImageNet statistics for pretrained models

### 2. Model Architecture
- **Feature Extractor**: ResNet18/50, EfficientNet, ViT
- **Aggregation**: Attention-based MIL, CLAM, TransMIL
- **Classification**: Linear or MLP head with dropout

### 3. Training Optimization
- **Mixed Precision**: torch.cuda.amp for 2-3x speedup
- **Channels Last**: Optimized memory layout for convolutions
- **torch.compile**: JIT compilation for 30-40% speedup
- **Persistent Workers**: Avoid dataloader reload overhead

### 4. Inference
- **Batch Processing**: Efficient multi-sample inference
- **Real-time**: <5 second latency for clinical use
- **Interpretability**: Grad-CAM, attention maps, SHAP

### 5. Clinical Integration
- **PACS**: DICOM C-FIND/C-MOVE/C-STORE operations
- **LIS/EMR**: HL7 messaging for results delivery
- **Compliance**: HIPAA audit logging, FDA/CE validation

## Performance Characteristics

| Component | Metric | Value |
|-----------|--------|-------|
| **Training** | Time (PCam, 15 epochs) | 2.25 hours |
| **Training** | GPU Utilization | 85% |
| **Training** | Speedup vs Baseline | 8-12x |
| **Inference** | Latency (single WSI) | <5 seconds |
| **Inference** | Throughput | 200+ slides/hour |
| **Model** | Parameters (AttentionMIL) | 12M |
| **Model** | Memory (training) | 8GB GPU |
| **Accuracy** | PCam Test AUC | 93.98% |
| **Accuracy** | PCam Test Accuracy | 85.26% |

## Technology Stack

- **Framework**: PyTorch 2.0+
- **WSI Processing**: OpenSlide, Pillow
- **Optimization**: torch.compile, AMP, channels_last
- **Testing**: pytest, Hypothesis (property-based)
- **DICOM**: pydicom, pynetdicom
- **Privacy**: Opacus (differential privacy)
- **Deployment**: Docker, ONNX export

## Scalability

- **Single GPU**: RTX 4070 (12GB) - 256K samples in 2.25 hours
- **Multi-GPU**: Data parallel training (linear scaling)
- **Distributed**: Federated learning across 3+ sites
- **Production**: Batch inference on 1000+ slides/day

---

*For implementation details, see the [API Reference](API_REFERENCE.html)*
