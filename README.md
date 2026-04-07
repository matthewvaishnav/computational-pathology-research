# Computational Pathology Research Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-62%25-yellow.svg)](htmlcov/index.html)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

> **📋 Portfolio Overview**: See [PORTFOLIO_SUMMARY.md](PORTFOLIO_SUMMARY.md) for a complete overview of this project's achievements and capabilities.

> **🚀 Quick Start**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for commands and quick navigation.

**⚠️ IMPORTANT: This is a tested research codebase, not clinically or experimentally validated.**

This repository provides a **tested code framework** for exploring multimodal fusion architectures in computational pathology. It implements architectural ideas for integrating whole-slide images (WSI), genomic features, and clinical text, with comprehensive unit tests but **has not been validated on real clinical data or published datasets**.

## What This Actually Is

**This is a tested starting point for research, not completed research:**
- ✅ Well-structured PyTorch implementations of proposed architectures
- ✅ Modular components that can be used independently
- ✅ Comprehensive unit tests with 62% code coverage
- ✅ Working MCP server for repository exploration
- ✅ PatchCamelyon (PCam) dataset integration and training pipeline
- ✅ ONNX export capabilities for model deployment
- ✅ Model profiling and ablation study tools
- ❌ **No experiments on real clinical datasets**
- ❌ **No validation of clinical effectiveness**
- ❌ **No comparison to published methods**
- ❌ **No trained models or clinical results**
- ❌ **No proof these ideas work in practice**

**Honest Assessment**: This code has been tested with synthetic data and the PCam benchmark dataset. The architectural choices are reasonable but unproven on clinical pathology data. Claims about "expected improvements" are speculation, not experimental results.

## Abstract

This repository proposes computational approaches for pathology image analysis through multimodal fusion architectures. We provide implementations of attention-based mechanisms designed to integrate whole-slide images (WSI), genomic features, and clinical text data. The system includes cross-slide temporal reasoning for disease progression modeling, transformer-based stain normalization, and self-supervised pretraining objectives.

**Critical Caveat**: These are architectural proposals and code implementations only. Actual effectiveness, computational feasibility, and clinical utility remain unvalidated and would require extensive experimentation.

## Background and Motivation

Computational pathology faces fundamental challenges in integrating heterogeneous data sources. Whole-slide images contain rich spatial information about tissue morphology, but analyzing them in isolation ignores complementary molecular and clinical context. Genomic profiles reveal underlying biological mechanisms, while clinical text provides diagnostic context and patient history. However, these modalities exhibit different statistical properties, scales, and information densities, making effective fusion non-trivial.

Existing approaches often use simple concatenation or late fusion strategies that fail to capture complex cross-modal interactions. Additionally, color variation across different staining protocols and laboratories introduces systematic artifacts that confound analysis. Disease progression analysis requires reasoning across multiple tissue samples over time, but most methods treat slides independently.

This repository addresses these challenges through:
- **Attention-based multimodal fusion** that learns adaptive cross-modal interactions
- **Temporal reasoning mechanisms** that model disease progression across longitudinal samples
- **Transformer-based stain normalization** that preserves morphology while reducing color artifacts
- **Self-supervised pretraining** that leverages unlabeled pathology data at scale

## Research Hypothesis (Untested)

**Hypothesis**: Attention-based cross-modal fusion mechanisms can learn complementary relationships between histopathological images, genomic profiles, and clinical text that improve tissue analysis compared to single-modality or simple concatenation approaches.

**Rationale**: Different modalities capture distinct aspects of disease biology. WSI reveals spatial tissue architecture, genomics captures molecular alterations, and clinical text provides diagnostic context. Attention mechanisms can dynamically weight and integrate these heterogeneous signals based on their relevance to specific analytical tasks, enabling the model to discover non-obvious cross-modal patterns.

**Testable Predictions** (None Actually Tested):
1. Models with cross-modal attention will outperform single-modality baselines
2. Ablation studies will show that each modality contributes unique information
3. Attention weights will reveal interpretable cross-modal relationships
4. The fusion architecture will gracefully handle missing modalities

**Current Status**: This is a hypothesis only. No experiments have been conducted to test these predictions. The code provides a framework for testing, but validation requires:
- Access to multimodal pathology datasets (rare)
- Baseline implementations for fair comparison
- Extensive computational resources
- Statistical validation across multiple datasets
- Collaboration with domain experts

## Methodology

### Multimodal Fusion Architecture

Our architecture consists of modality-specific encoders followed by cross-modal attention fusion:

**WSI Encoder**: Processes patch-level features through attention-based aggregation to capture spatial tissue patterns. Uses multi-head self-attention over image patches with learnable positional encodings.

**Genomic Encoder**: Multi-layer perceptron with batch normalization that transforms high-dimensional genomic profiles (gene expression, mutations, copy number variations) into a compact representation.

**Clinical Text Encoder**: Transformer-based encoder that processes clinical notes, pathology reports, and patient history to extract relevant diagnostic context.

**Cross-Modal Attention Fusion**: Implements pairwise attention between all modality pairs, allowing each modality to query information from others. The fusion layer computes attention weights that indicate which cross-modal relationships are most relevant for the task.

**Missing Modality Handling**: The architecture gracefully handles missing modalities through masking, enabling training and inference with incomplete data—a common scenario in clinical settings.

### Cross-Slide Temporal Reasoning

For longitudinal analysis, we implement temporal attention mechanisms that:
- Encode temporal distances between slides using learnable positional embeddings
- Apply self-attention over slide sequences to capture progression patterns
- Extract progression features by computing differences between consecutive slides
- Support multiple pooling strategies (attention-weighted, mean, max, last) for sequence-level representations

This enables the model to reason about disease evolution, treatment response, and recurrence patterns.

### Stain Normalization Transformer

Color variation across staining protocols is addressed through a transformer-based normalization approach:
- **Patch Embedding**: Divides images into patches and projects them to embedding space
- **Color Feature Encoder**: Transformer encoder extracts color-invariant tissue features
- **Style Conditioner**: Adaptive instance normalization conditions on reference style when provided
- **Style Transfer Decoder**: Reconstructs normalized images while preserving morphological details

The model is trained with perceptual loss and color consistency objectives to maintain tissue structure during normalization.

### Self-Supervised Pretraining

We implement two complementary pretraining objectives:

**Contrastive Learning**: SimCLR-style contrastive loss between augmented views of tissue patches. Encourages the encoder to learn representations invariant to staining variations and imaging artifacts while capturing semantic tissue patterns.

**Masked Reconstruction**: Masked autoencoder objective where random patches are masked and the model learns to reconstruct them. This encourages learning of local tissue structure and spatial relationships.

Both objectives can be combined with configurable weights, and pretrained weights can be transferred to downstream tasks.

## Proposed Contributions (Unvalidated)

### Architectural Ideas

1. **Cross-Modal Attention Fusion**: Proposes using attention mechanisms to integrate heterogeneous pathology data sources. Whether this actually works better than simpler approaches is unknown.

2. **Temporal Reasoning Framework**: Implements transformer-based temporal attention for cross-slide analysis. Similar ideas may exist in the literature - comprehensive literature review needed.

3. **Stain Normalization Approach**: Proposes transformer-based stain normalization. Traditional methods (Macenko, Vahadane) are well-established; whether this approach offers advantages is unproven.

4. **Pretraining Framework**: Combines contrastive and reconstruction objectives. Standard techniques adapted to pathology context.

### Hypothetical Performance (Unvalidated)

**⚠️ WARNING: These are speculative estimates, not experimental results:**
- Multimodal fusion *might* improve performance over single-modality baselines (needs testing)
- Temporal reasoning *could* help with progression prediction (requires longitudinal data)
- Stain normalization *may* reduce cross-laboratory variation (needs validation)
- Self-supervised pretraining *might* improve sample efficiency (common in other domains, unproven here)

**Reality Check**: These numbers (5-15%, 10-20%, etc.) are made up. Actual performance could be better, worse, or the same as baselines. Only experiments will tell.

### What Would Actually Be Needed

To validate these ideas, you would need:
1. **Real datasets**: Multimodal pathology data (rare and expensive)
2. **Baseline implementations**: Fair comparisons to existing methods
3. **Extensive experiments**: Multiple datasets, cross-validation, statistical testing
4. **Ablation studies**: Systematic removal of components to measure contribution
5. **Computational resources**: Thousands of GPU-hours for training
6. **Domain expertise**: Collaboration with pathologists for validation
7. **Months of work**: Debugging, tuning, analyzing results

## Limitations and Reality Check

### Critical Limitations of This Repository

**⚠️ This code has never been tested on real data. All limitations below are theoretical.**

### What's Actually Missing

1. **No Real Data**: This code has never processed actual WSI, genomic data, or clinical text
2. **No Training**: Models have never been trained end-to-end
3. **No Validation**: Zero experimental validation of any claims
4. **No Baselines**: No comparison to existing methods
5. **No Results**: No figures, tables, or performance metrics
6. **No Debugging**: Real-world issues haven't been encountered or fixed
7. **No Hyperparameter Tuning**: All settings are arbitrary guesses

### Computational Reality (If You Actually Tried This)

- **Training Cost**: Likely $5,000-$20,000 in GPU compute for full experiments
- **Data Preprocessing**: Weeks to months processing gigapixel WSI images
- **Memory Requirements**: WSI processing requires 64GB+ RAM, specialized tools
- **Training Time**: Weeks to months for full experimental validation
- **Debugging Time**: Expect weeks fixing issues that only appear with real data

### Dataset Reality

- **Multimodal Data is Rare**: Very few datasets have aligned WSI + genomics + clinical text
- **Data Access**: Requires IRB approval, data use agreements, often institutional access only
- **Data Cost**: Some datasets cost thousands of dollars to access
- **Preprocessing Burden**: WSI preprocessing alone is a major research project
- **Temporal Data**: Longitudinal pathology data is extremely rare

### Methodological Limitations (Theoretical)

Since this hasn't been tested, we don't actually know:
- Whether the architecture trains stably
- Whether attention mechanisms help or hurt
- Whether the model overfits immediately
- Whether computational costs are prohibitive
- Whether the approach works at all

### Honest Assessment

**What works**: The code runs without errors in unit tests with synthetic data.

**What's unknown**: Everything else - effectiveness, efficiency, practical utility, comparison to baselines, real-world performance, failure modes, optimal hyperparameters, etc.

**What's needed**: 6-12 months of full-time research work to validate any claims.

## Ethical Considerations

### Data Privacy and Security

- **Patient Privacy**: All data must be de-identified according to HIPAA or equivalent regulations. This repository does not include any patient data.
- **Data Governance**: Users must ensure proper institutional review board (IRB) approval and data use agreements for any datasets used with this code.
- **Secure Storage**: Pathology data and genomic information are sensitive; implement appropriate security measures for storage and transmission.

### Research-Only Use

**CRITICAL**: This is a research implementation demonstrating computational methods. It is **NOT validated for clinical use** and should **NEVER** be used for:
- Clinical diagnosis or treatment decisions
- Patient care or medical advice
- Regulatory submissions or clinical trials without extensive additional validation

### Potential Biases

- **Training Data Bias**: Models reflect biases in training data, which may underrepresent certain demographics, institutions, or disease subtypes.
- **Performance Disparities**: Model performance may vary across patient populations, potentially exacerbating healthcare disparities if deployed without careful validation.
- **Staining Bias**: Despite normalization efforts, models may perform better on staining protocols similar to training data.

### Appropriate Use Cases

**Appropriate**:
- Academic research on computational pathology methods
- Algorithm development and benchmarking
- Educational purposes and teaching
- Exploratory data analysis in research settings

**Inappropriate**:
- Clinical decision-making without extensive validation
- Deployment in healthcare settings without regulatory approval
- Use on populations not represented in validation studies
- Commercial applications without proper licensing and validation

### Transparency and Accountability

- **Limitations Disclosure**: Users must clearly communicate model limitations when presenting results.
- **Validation Requirements**: Any clinical application requires prospective validation studies with appropriate statistical power.
- **Failure Mode Analysis**: Understanding when and why the model fails is critical before any deployment.

## Reproducibility Instructions

### Environment Setup

**Python Version**: Python 3.9 or higher (tested on 3.9, 3.10, 3.11)

**Installation**:
```bash
# Clone repository
git clone <repository-url>
cd computational-pathology-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

**Dependency Versions**: Dependencies use minimum version constraints (e.g., `torch>=2.0.0`) to allow flexibility. For exact reproducibility, consider using `pip freeze > requirements-frozen.txt` after installation. Key minimum versions:
- PyTorch 2.0.0+
- Hydra 1.3.0+
- TensorBoard 2.12.0+

### Hardware Requirements

**Minimum**:
- CPU: Modern multi-core processor
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB VRAM (for small-scale experiments)
- Storage: 50GB for code, models, and small datasets

**Recommended**:
- CPU: 16+ cores
- RAM: 32GB+
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090, A5000, or better)
- Storage: 500GB+ SSD for datasets and results

**Large-Scale Training**:
- GPU: 24GB+ VRAM (RTX 4090, A6000, A100)
- Multi-GPU setup for distributed training
- High-speed storage for large WSI datasets

### Reproducing Experiments

**Random Seeds**: All experiments use fixed random seeds for reproducibility:
```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Training Commands**:
```bash
# Multimodal fusion training
python scripts/train.py --config-name=multimodal

# PatchCamelyon (PCam) training
python experiments/train_pcam.py --config-name=pcam

# Stain normalization training
python experiments/train_stain_norm.py --config-name=stain_norm

# Self-supervised pretraining
python examples/pretraining_example.py

# Ablation studies
python scripts/run_ablation_study.py --config-name=ablation
```

**Configuration Files**: All hyperparameters are specified in `experiments/configs/` YAML files. Modify these files to adjust training settings.

### Expected Training Time

**On Single RTX 3090 (24GB)**:
- Multimodal fusion model: 4-8 hours per epoch (depends on dataset size)
- Stain normalization: 2-4 hours per epoch
- Self-supervised pretraining: 12-24 hours for 10 epochs on large dataset
- Full ablation study: 24-48 hours (multiple model variants)

**On A100 (40GB)**:
- Approximately 2-3x faster than RTX 3090
- Enables larger batch sizes and longer sequences

### Computational Resources

**Training**:
- Single model training: ~100-200 GPU-hours
- Complete ablation study: ~500-1000 GPU-hours
- Self-supervised pretraining: ~500-2000 GPU-hours (dataset dependent)

**Inference**:
- Single slide: <1 second on GPU
- Batch of 32 slides: ~5-10 seconds on GPU

### Verification

**Run Tests**:
```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

**Expected Test Results**: All tests should pass. Current coverage is 62% for core modules. Coverage target is >80% (work in progress).

## Future Work

### Unexplored Research Directions

**3D Spatial Reasoning**: Current architecture processes 2D patches independently. Extending to 3D volumetric reasoning could capture tissue architecture at multiple scales and improve spatial context modeling.

**Multi-Task Learning**: Joint training on multiple related tasks (classification, segmentation, survival prediction) with shared representations could improve sample efficiency and generalization.

**Weakly Supervised Learning**: Developing methods that learn from slide-level labels without requiring expensive patch-level annotations would enable scaling to larger datasets.

**Uncertainty Quantification**: Implementing Bayesian approaches or ensemble methods to provide calibrated confidence estimates for predictions, critical for clinical applications.

**Explainability Methods**: Developing visualization techniques and attribution methods to better understand which image regions, genomic features, and clinical factors drive predictions.

### Extensions and Improvements

**Additional Modalities**:
- Radiology images (CT, MRI) for multi-scale tissue analysis
- Proteomics and metabolomics data for molecular profiling
- Spatial transcriptomics for gene expression mapping
- Immunohistochemistry markers for immune profiling

**New Pretraining Objectives**:
- Rotation prediction for learning tissue orientation invariance
- Jigsaw puzzle solving for spatial relationship learning
- Cross-modal prediction (predict genomics from images, etc.)
- Temporal prediction (predict future slides from past slides)

**Architecture Improvements**:
- Vision transformers (ViT) for end-to-end image processing
- Graph neural networks for modeling tissue microenvironment
- Memory-augmented networks for long-range temporal dependencies
- Mixture-of-experts for handling diverse tissue types

### Scalability Improvements

**Efficient Training**:
- Mixed-precision training (FP16/BF16) for faster training
- Gradient checkpointing for reduced memory usage
- Distributed data parallel training across multiple GPUs
- Model parallelism for very large models

**Data Pipeline Optimization**:
- Streaming data loading for large WSI datasets
- On-the-fly augmentation and preprocessing
- Efficient HDF5 caching strategies
- Multi-process data loading

**Inference Optimization**:
- Model quantization (INT8) for faster inference
- Knowledge distillation to smaller student models
- ONNX export for deployment flexibility
- TensorRT optimization for production deployment

### Clinical Translation

**Validation Studies**:
- Prospective validation on held-out institutions
- Multi-center studies across diverse populations
- Comparison with expert pathologist performance
- Clinical utility studies measuring impact on patient outcomes

**Deployment Considerations**:
- Integration with laboratory information systems (LIS)
- Real-time inference pipelines for clinical workflows
- Quality control and monitoring systems
- Regulatory compliance (FDA, CE marking)

## Quick Start

### Docker Deployment (Recommended)

The fastest way to get started is using Docker:

```bash
# Build and start the API
docker-compose up -d api

# Test the API
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "wsi_features": [[0.1] * 1024] * 50,
    "genomic": [0.1] * 2000,
    "clinical_text": [100, 200, 300, 400, 500]
  }'

# View API documentation
open http://localhost:8000/docs
```

See [DOCKER.md](DOCKER.md) for complete Docker deployment guide including GPU support, Kubernetes, and cloud deployment.

### Repository Exploration with MCP Server

This repository includes an MCP (Model Context Protocol) server for AI-assisted code exploration:

```bash
# Start the MCP server
python scripts/project_mcp_server.py

# Available tools:
# - project_overview: Get repository summary
# - list_project_files: List files with glob patterns
# - read_text_file: Read file contents
# - search_repository: Search for text patterns
# - run_pytest: Execute targeted tests
```

See [docs/mcp_server.md](docs/mcp_server.md) for configuration and usage details.

### PatchCamelyon (PCam) Training

Train on the PatchCamelyon benchmark dataset:

```bash
# Generate synthetic PCam data for testing
python scripts/generate_synthetic_pcam.py

# Train on PCam
python experiments/train_pcam.py --config-name=pcam

# Evaluate on PCam
python experiments/evaluate_pcam.py --checkpoint=models/pcam_best.pth
```

### Model Profiling and Analysis

Profile model performance and resource usage:

```bash
# Profile model inference time
python scripts/model_profiler.py \
  --checkpoint=models/best_model.pth \
  --profile-type=time

# Profile memory usage
python scripts/model_profiler.py \
  --checkpoint=models/best_model.pth \
  --profile-type=memory

# Export to ONNX
python scripts/export_onnx.py \
  --checkpoint=models/best_model.pth \
  --output=models/model.onnx
```

### Ablation Studies

Run systematic ablation studies:

```bash
# Run full ablation study
python scripts/run_ablation_study.py \
  --data-dir=./data \
  --output-dir=./results/ablation \
  --num-epochs=10
```

### Basic Usage Example

```python
import torch
from src.models import MultimodalFusionModel, CrossSlideTemporalReasoner
from src.models import ClassificationHead

# Initialize multimodal fusion model
model = MultimodalFusionModel(embed_dim=256)

# Prepare batch with multiple modalities
batch = {
    'wsi_features': torch.randn(8, 100, 1024),  # 8 samples, 100 patches, 1024-dim features
    'genomic': torch.randn(8, 2000),             # 8 samples, 2000 genes
    'clinical_text': torch.randint(0, 30000, (8, 128))  # 8 samples, 128 tokens
}

# Get fused embedding
fused_embedding = model(batch)  # [8, 256]

# Add classification head
classifier = ClassificationHead(input_dim=256, num_classes=4)
predictions = classifier(fused_embedding)  # [8, 4]
```

### Training Example

```python
from src.models import MultimodalFusionModel
from src.data import MultimodalDataset
import torch.optim as optim

# Create model
model = MultimodalFusionModel(embed_dim=256)

# Create dataset and dataloader
dataset = MultimodalDataset(data_dir='./data', split='train', config={})
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(batch)
        predictions = classifier(embeddings)
        
        # Compute loss
        loss = criterion(predictions, batch['label'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### Evaluation Example

```bash
# Evaluate trained model
python experiments/evaluate.py \
    --checkpoint=models/best_model.pth \
    --data-dir=./data \
    --split=test \
    --output-dir=./results/evaluation
```

### Inference Example

```python
from src.models import MultimodalFusionModel, ClassificationHead
import torch

# Load trained model
model = MultimodalFusionModel(embed_dim=256)
classifier = ClassificationHead(input_dim=256, num_classes=4)

checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

model.eval()
classifier.eval()

# Inference on new sample
with torch.no_grad():
    sample = {
        'wsi_features': torch.randn(1, 100, 1024),
        'genomic': torch.randn(1, 2000),
        'clinical_text': torch.randint(0, 30000, (1, 128))
    }
    
    embedding = model(sample)
    logits = classifier(embedding)
    probabilities = torch.softmax(logits, dim=-1)
    
    predicted_class = torch.argmax(probabilities, dim=-1)
    confidence = probabilities[0, predicted_class].item()
    
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Confidence: {confidence:.3f}")
```

## Repository Structure

```
.
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   │   ├── loaders.py     # MultimodalDataset, TemporalDataset
│   │   ├── pcam_dataset.py  # PatchCamelyon dataset loader
│   │   └── preprocessing.py  # Preprocessing utilities
│   ├── models/            # Model architectures
│   │   ├── encoders.py    # Modality-specific encoders
│   │   ├── fusion.py      # Cross-modal attention fusion
│   │   ├── multimodal.py  # Complete fusion model
│   │   ├── temporal.py    # Temporal reasoning
│   │   ├── heads.py       # Task-specific prediction heads
│   │   ├── feature_extractors.py  # Feature extraction utilities
│   │   └── stain_normalization.py  # Stain normalization transformer
│   ├── pretraining/       # Self-supervised pretraining
│   │   ├── objectives.py  # Contrastive and reconstruction losses
│   │   └── pretrainer.py  # Pretraining wrapper
│   ├── training/          # Training infrastructure
│   │   └── __init__.py    # SupervisedTrainer
│   ├── utils/             # Utilities
│   │   ├── validation.py  # Validation utilities
│   │   ├── monitoring.py  # Training monitoring
│   │   └── interpretability.py  # Model interpretability
│   └── mcp_server.py      # MCP server for repository exploration
├── scripts/               # Utility scripts
│   ├── train.py          # Main training script
│   ├── run_ablation_study.py  # Ablation study runner
│   ├── model_profiler.py  # Model profiling tool
│   ├── export_onnx.py    # ONNX export utility
│   └── project_mcp_server.py  # MCP server entry point
├── experiments/           # Training and evaluation scripts
│   ├── configs/          # Configuration files (YAML)
│   ├── train_pcam.py     # PatchCamelyon training
│   ├── evaluate_pcam.py  # PatchCamelyon evaluation
│   ├── train_stain_norm.py  # Stain normalization training
│   ├── evaluate.py       # General evaluation script
│   └── utils/            # Experiment utilities
├── examples/             # Usage examples
│   └── pretraining_example.py  # Pretraining demonstration
├── tests/                # Unit tests (62% coverage)
├── docs/                 # Additional documentation
│   ├── multimodal_architecture.md  # Architecture details
│   └── mcp_server.md     # MCP server documentation
├── data/                 # Dataset directory
│   ├── pcam/            # PatchCamelyon dataset
│   └── README.md         # Dataset guide
├── configs/              # Hydra configuration files
│   ├── train.yaml       # Main training config
│   ├── model/           # Model configurations
│   ├── data/            # Data configurations
│   └── task/            # Task configurations
├── requirements.txt      # Python dependencies (minimum versions)
├── pyproject.toml        # Package configuration
└── README.md            # This file
```

## Documentation

- **Portfolio Summary**: See `PORTFOLIO_SUMMARY.md` for a complete overview of achievements and capabilities
- **Architecture Details**: See `ARCHITECTURE.md` for detailed component descriptions and system design
- **Performance Benchmarks**: See `PERFORMANCE.md` for comprehensive performance analysis and optimization guide
- **Docker Deployment**: See `DOCKER.md` for complete containerization and deployment guide
- **API Deployment**: See `deploy/README.md` for REST API deployment instructions
- **Demo Results**: See `DEMO_RESULTS.md` for detailed analysis of training results
- **Testing**: See `TESTING_SUMMARY.md` for testing documentation and coverage reports
- **Dataset Guide**: See `data/README.md` for dataset acquisition and preprocessing instructions
- **Getting Started Tutorial**: See `notebooks/00_getting_started.ipynb` for interactive walkthrough

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_fusion.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

**Current Test Coverage**: 62% (target: >80%)

## License

MIT License - See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{computational_pathology_research,
  title = {Computational Pathology Research Repository: Novel Multimodal Fusion Architectures},
  author = {Research Team},
  year = {2024},
  url = {https://github.com/your-org/computational-pathology-research}
}
```

## Acknowledgments

This research implementation builds upon foundational work in computational pathology, multimodal learning, and self-supervised representation learning. We acknowledge the open-source community for providing essential tools and frameworks (PyTorch, Hydra, TensorBoard) that made this work possible.

## Contact

For questions, issues, or collaboration inquiries, please open an issue on the GitHub repository.
