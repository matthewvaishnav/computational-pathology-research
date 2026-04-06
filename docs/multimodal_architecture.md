# Multimodal Fusion Architecture

This document describes the multimodal fusion architecture for computational pathology, including cross-modal attention, temporal reasoning, and task-specific prediction heads.

## Architecture Overview

The complete architecture consists of four main components:

1. **Modality-Specific Encoders** (`src/models/encoders.py`)
2. **Cross-Modal Attention Fusion** (`src/models/fusion.py`)
3. **Temporal Reasoning** (`src/models/temporal.py`)
4. **Task-Specific Prediction Heads** (`src/models/heads.py`)

## Components

### 1. Multimodal Fusion Model

The `MultimodalFusionModel` integrates all modality encoders and fusion mechanisms:

```python
from src.models import MultimodalFusionModel

# Initialize model
model = MultimodalFusionModel(embed_dim=256)

# Prepare batch
batch = {
    'wsi_features': torch.randn(batch_size, num_patches, 1024),
    'genomic': torch.randn(batch_size, num_genes),
    'clinical_text': torch.randint(0, vocab_size, (batch_size, seq_len))
}

# Get fused embedding
fused_embedding = model(batch)  # [batch_size, 256]
```

**Handling Missing Modalities:**

The model gracefully handles missing modalities by setting them to `None`:

```python
batch = {
    'wsi_features': torch.randn(batch_size, num_patches, 1024),
    'genomic': None,  # Missing genomic data
    'clinical_text': torch.randint(0, vocab_size, (batch_size, seq_len))
}

fused_embedding = model(batch)  # Still works!
```

### 2. Cross-Modal Attention Fusion

The fusion layer implements pairwise cross-modal attention:

```python
from src.models import MultiModalFusionLayer

fusion = MultiModalFusionLayer(embed_dim=256, num_heads=8)

embeddings = {
    'wsi': wsi_embeddings,      # [batch_size, 256]
    'genomic': genomic_embeddings,  # [batch_size, 256]
    'clinical': clinical_embeddings  # [batch_size, 256]
}

fused = fusion(embeddings)  # [batch_size, 256]
```

### 3. Temporal Reasoning

For analyzing disease progression across multiple slides:

```python
from src.models import CrossSlideTemporalReasoner

temporal_reasoner = CrossSlideTemporalReasoner(embed_dim=256)

# Slide sequence from same patient
slide_embeddings = torch.randn(batch_size, num_slides, 256)
timestamps = torch.tensor([[0, 30, 90, 180, 365]])  # Days

# Get sequence-level embedding and progression features
sequence_emb, progression_features = temporal_reasoner(
    slide_embeddings,
    timestamps
)
```

**Features:**
- Temporal attention with positional encoding
- Progression feature extraction (differences between consecutive slides)
- Multiple pooling strategies: attention, mean, max, last

### 4. Task-Specific Prediction Heads

#### Classification

```python
from src.models import ClassificationHead

classifier = ClassificationHead(
    input_dim=256,
    num_classes=5,
    dropout=0.3
)

logits = classifier(embeddings)  # [batch_size, 5]
```

#### Survival Prediction

```python
from src.models import SurvivalPredictionHead

# Risk score prediction
risk_head = SurvivalPredictionHead(input_dim=256)
risk_scores = risk_head(embeddings)  # [batch_size, 1]

# Discrete time hazard prediction
hazard_head = SurvivalPredictionHead(
    input_dim=256,
    num_time_bins=12
)
hazards = hazard_head(embeddings, return_hazards=True)  # [batch_size, 12]
survival_curve = hazard_head.compute_survival_curve(embeddings)  # [batch_size, 12]
```

#### Multi-Task Learning

```python
from src.models import MultiTaskHead

multi_head = MultiTaskHead(
    input_dim=256,
    classification_config={'num_classes': 5},
    survival_config={'num_time_bins': 12}
)

class_logits, survival_output = multi_head(embeddings)
```

## Complete End-to-End Example

```python
import torch
from src.models import (
    MultimodalFusionModel,
    CrossSlideTemporalReasoner,
    ClassificationHead,
    SurvivalPredictionHead
)

# 1. Multimodal fusion
fusion_model = MultimodalFusionModel(embed_dim=256)

batch = {
    'wsi_features': torch.randn(8, 100, 1024),
    'genomic': torch.randn(8, 2000),
    'clinical_text': torch.randint(0, 30000, (8, 128))
}

fused_embeddings = fusion_model(batch)

# 2. Temporal reasoning (optional, for longitudinal data)
temporal_reasoner = CrossSlideTemporalReasoner(embed_dim=256)

slide_sequence = torch.randn(8, 5, 256)
timestamps = torch.tensor([[0, 30, 90, 180, 365]] * 8).float()

sequence_emb, progression_features = temporal_reasoner(
    slide_sequence,
    timestamps
)

# 3. Task-specific predictions
classifier = ClassificationHead(input_dim=256, num_classes=4)
class_logits = classifier(fused_embeddings)

survival_head = SurvivalPredictionHead(input_dim=256, num_time_bins=12)
survival_hazards = survival_head(fused_embeddings, return_hazards=True)
```

## Configuration

All components support flexible configuration:

```python
# Custom encoder configurations
wsi_config = {
    'input_dim': 2048,
    'hidden_dim': 1024,
    'output_dim': 512,
    'num_heads': 16,
    'num_layers': 4,
    'dropout': 0.2,
    'pooling': 'attention'
}

genomic_config = {
    'input_dim': 5000,
    'hidden_dims': [2048, 1024, 512],
    'output_dim': 512,
    'dropout': 0.4,
    'use_batch_norm': True
}

model = MultimodalFusionModel(
    wsi_config=wsi_config,
    genomic_config=genomic_config,
    embed_dim=512
)
```

## Key Features

1. **Modular Design**: Each component can be used independently
2. **Missing Modality Handling**: Gracefully handles missing data
3. **Flexible Architecture**: Configurable dimensions, layers, and pooling strategies
4. **Temporal Reasoning**: Built-in support for longitudinal analysis
5. **Multi-Task Learning**: Joint training on multiple objectives
6. **Research-Ready**: Designed for experimentation and ablation studies

## References

- See `src/models/encoders.py` for modality-specific encoder implementations
- See `src/models/fusion.py` for cross-modal attention mechanisms
- See `src/models/multimodal.py` for the complete fusion model
- See `src/models/temporal.py` for temporal reasoning components
- See `src/models/heads.py` for task-specific prediction heads
