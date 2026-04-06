# Architecture Documentation

Detailed technical documentation of the multimodal fusion architecture.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Modalities                              │
├─────────────────────────────────────────────────────────────────┤
│  WSI Features    │   Genomic Data   │   Clinical Text           │
│  [B, N, 1024]    │   [B, 2000]      │   [B, L]                  │
└────────┬─────────┴────────┬──────────┴────────┬─────────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              Modality-Specific Encoders                          │
├─────────────────────────────────────────────────────────────────┤
│  WSI Encoder     │  Genomic Encoder │  Clinical Encoder         │
│  (Attention)     │  (MLP)           │  (Transformer)            │
│  8.5M params     │  2.1M params     │  12.3M params             │
└────────┬─────────┴────────┬──────────┴────────┬─────────────────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Modal Attention Fusion                        │
├─────────────────────────────────────────────────────────────────┤
│  • Pairwise attention between all modalities                    │
│  • Learns cross-modal relationships                             │
│  • Handles missing modalities                                   │
│  • 3.2M parameters                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    [B, embed_dim]
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Task-Specific      │              │  Temporal Reasoning │
│  Heads              │              │  (Optional)         │
├─────────────────────┤              ├─────────────────────┤
│  • Classification   │              │  • Temporal Attn    │
│  • Survival         │              │  • Progression      │
│  • Segmentation     │              │  • 467K params      │
│  • 1.5M params      │              └─────────────────────┘
└─────────────────────┘
         │
         ▼
    Predictions
```

---

## 1. Input Layer

### 1.1 WSI Features

**Format**: `[batch_size, num_patches, 1024]`

```
Whole-Slide Image
       │
       ▼
┌─────────────────┐
│ Patch Extraction│  (e.g., 256x256 patches)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Extract │  (e.g., ResNet-50 pretrained)
└────────┬────────┘
         │
         ▼
  [N, 1024] features
```

**Properties**:
- Variable number of patches (N)
- Each patch: 1024-dimensional feature vector
- Typically 50-200 patches per slide
- Mask indicates valid patches

### 1.2 Genomic Data

**Format**: `[batch_size, 2000]`

```
Raw Genomic Data
       │
       ▼
┌─────────────────┐
│ Gene Selection  │  (e.g., top 2000 genes)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalization   │  (e.g., log-transform, z-score)
└────────┬────────┘
         │
         ▼
  [2000] features
```

**Properties**:
- Fixed dimension (2000 genes)
- Continuous values
- Normalized to zero mean, unit variance

### 1.3 Clinical Text

**Format**: `[batch_size, seq_len]`

```
Clinical Notes
       │
       ▼
┌─────────────────┐
│ Tokenization    │  (e.g., WordPiece, BPE)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Truncation/Pad  │  (max length)
└────────┬────────┘
         │
         ▼
  [L] token IDs
```

**Properties**:
- Variable sequence length (L)
- Integer token IDs (0-30000)
- Mask indicates valid tokens

---

## 2. Encoder Layer

### 2.1 WSI Encoder

```python
class WSIEncoder(nn.Module):
    """
    Attention-based patch aggregation.
    
    Input:  [B, N, 1024]
    Output: [B, embed_dim]
    """
```

**Architecture**:
```
Input: [B, N, 1024]
       │
       ▼
┌─────────────────┐
│ Linear Proj     │  1024 → embed_dim
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Positional Enc  │  Learnable
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │  L layers, H heads
│ Encoder         │  Self-attention over patches
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Attention Pool  │  Weighted average
└────────┬────────┘
         │
         ▼
Output: [B, embed_dim]
```

**Key Features**:
- Self-attention captures spatial relationships
- Positional encoding preserves patch locations
- Attention pooling focuses on important patches
- Handles variable number of patches

### 2.2 Genomic Encoder

```python
class GenomicEncoder(nn.Module):
    """
    MLP with batch normalization.
    
    Input:  [B, 2000]
    Output: [B, embed_dim]
    """
```

**Architecture**:
```
Input: [B, 2000]
       │
       ▼
┌─────────────────┐
│ Linear          │  2000 → 1024
│ BatchNorm       │
│ ReLU            │
│ Dropout(0.1)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear          │  1024 → 512
│ BatchNorm       │
│ ReLU            │
│ Dropout(0.1)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear          │  512 → embed_dim
│ LayerNorm       │
└────────┬────────┘
         │
         ▼
Output: [B, embed_dim]
```

**Key Features**:
- Deep MLP for non-linear transformation
- Batch normalization for stability
- Dropout for regularization
- Layer normalization for output

### 2.3 Clinical Text Encoder

```python
class ClinicalTextEncoder(nn.Module):
    """
    Transformer-based text encoder.
    
    Input:  [B, L]
    Output: [B, embed_dim]
    """
```

**Architecture**:
```
Input: [B, L] token IDs
       │
       ▼
┌─────────────────┐
│ Embedding       │  vocab_size → embed_dim
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Positional Enc  │  Sinusoidal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │  L layers, H heads
│ Encoder         │  Self-attention over tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [CLS] Token     │  First token embedding
└────────┬────────┘
         │
         ▼
Output: [B, embed_dim]
```

**Key Features**:
- Token embeddings learned from scratch
- Sinusoidal positional encoding
- Transformer captures long-range dependencies
- [CLS] token for sequence representation

---

## 3. Fusion Layer

### 3.1 Cross-Modal Attention

```python
class CrossModalAttention(nn.Module):
    """
    Pairwise attention between modalities.
    
    Input:  {modality: [B, embed_dim]}
    Output: [B, embed_dim * num_modalities]
    """
```

**Architecture**:
```
Modality Embeddings
  WSI    Genomic  Clinical
   │        │        │
   └────────┼────────┘
            │
            ▼
┌─────────────────────────────┐
│  Pairwise Attention         │
│                             │
│  WSI → Genomic              │
│  WSI → Clinical             │
│  Genomic → WSI              │
│  Genomic → Clinical         │
│  Clinical → WSI             │
│  Clinical → Genomic         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Concatenate                │
│  [WSI', Genomic', Clinical']│
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Linear Projection          │
│  3*embed_dim → embed_dim    │
└────────────┬────────────────┘
             │
             ▼
      Fused Embedding
      [B, embed_dim]
```

**Attention Mechanism**:
```python
# For each pair (modality_i, modality_j):
Q = W_q @ modality_i  # Query
K = W_k @ modality_j  # Key
V = W_v @ modality_j  # Value

attention = softmax(Q @ K.T / sqrt(d_k))
output = attention @ V
```

**Key Features**:
- All-to-all attention between modalities
- Learns which cross-modal relationships matter
- Handles missing modalities via masking
- Multi-head attention for diverse patterns

### 3.2 Missing Modality Handling

```
Available Modalities: {WSI, Genomic}
Missing: Clinical

┌─────────────────────────────┐
│  Compute Available Pairs    │
│  • WSI → Genomic            │
│  • Genomic → WSI            │
│  (Skip Clinical pairs)      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Concatenate Available      │
│  [WSI', Genomic', 0]        │
│  (Zero for missing)         │
└────────────┬────────────────┘
             │
             ▼
      Fused Embedding
```

**Properties**:
- No imputation required
- Graceful degradation
- Learns to compensate from available modalities

---

## 4. Temporal Reasoning (Optional)

### 4.1 Temporal Attention

```python
class TemporalAttention(nn.Module):
    """
    Attention over slide sequence.
    
    Input:  [B, T, embed_dim]
    Output: [B, T, embed_dim]
    """
```

**Architecture**:
```
Slide Sequence
[slide_0, slide_1, ..., slide_T]
       │
       ▼
┌─────────────────────────────┐
│  Temporal Positional Enc    │
│  Based on timestamps        │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Transformer Encoder        │
│  Self-attention over time   │
└────────────┬────────────────┘
             │
             ▼
  Attended Slides
  [B, T, embed_dim]
```

**Temporal Encoding**:
```python
# Convert timestamps to positional encoding
temporal_bins = timestamps / max_temporal_distance
temporal_enc = sinusoidal_encoding(temporal_bins)
```

### 4.2 Progression Features

```
Attended Slides
[s_0, s_1, s_2, s_3]
       │
       ▼
┌─────────────────────────────┐
│  Compute Differences        │
│  Δ_1 = s_1 - s_0            │
│  Δ_2 = s_2 - s_1            │
│  Δ_3 = s_3 - s_2            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Concatenate                │
│  [s_i, Δ_i] for each pair   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  MLP Projection             │
│  2*embed_dim → embed_dim/2  │
└────────────┬────────────────┘
             │
             ▼
  Progression Features
  [B, T-1, embed_dim/2]
```

### 4.3 Temporal Pooling

```
Attended Slides + Progression
       │
       ▼
┌─────────────────────────────┐
│  Pooling Strategy           │
│  • Attention-weighted       │
│  • Mean pooling             │
│  • Max pooling              │
│  • Last slide               │
└────────────┬────────────────┘
             │
             ▼
  Sequence Representation
  [B, embed_dim]
```

---

## 5. Output Layer

### 5.1 Classification Head

```python
class ClassificationHead(nn.Module):
    """
    Multi-class classification.
    
    Input:  [B, embed_dim]
    Output: [B, num_classes]
    """
```

**Architecture**:
```
Input: [B, embed_dim]
       │
       ▼
┌─────────────────┐
│ LayerNorm       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear          │  embed_dim → embed_dim
│ ReLU            │
│ Dropout(0.3)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear          │  embed_dim → num_classes
└────────┬────────┘
         │
         ▼
Output: [B, num_classes] logits
```

### 5.2 Survival Prediction Head

```python
class SurvivalPredictionHead(nn.Module):
    """
    Cox proportional hazards.
    
    Input:  [B, embed_dim]
    Output: [B, 1] hazard
    """
```

**Architecture**:
```
Input: [B, embed_dim]
       │
       ▼
┌─────────────────┐
│ LayerNorm       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear          │  embed_dim → embed_dim//2
│ ReLU            │
│ Dropout(0.3)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear          │  embed_dim//2 → 1
└────────┬────────┘
         │
         ▼
Output: [B, 1] log hazard
```

---

## 6. Training Pipeline

### 6.1 Forward Pass

```
Batch
  │
  ├─> WSI Encoder ──────┐
  │                     │
  ├─> Genomic Encoder ──┼─> Cross-Modal ──> Fused
  │                     │    Attention        Embedding
  └─> Clinical Encoder ─┘                       │
                                                 │
                                                 ▼
                                            Task Head
                                                 │
                                                 ▼
                                            Predictions
```

### 6.2 Loss Computation

```python
# Classification
loss = CrossEntropyLoss(predictions, labels)

# Survival
loss = CoxLoss(hazards, survival_times, events)

# Multi-task
loss = α * classification_loss + β * survival_loss
```

### 6.3 Optimization

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)

# Training step
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
```

---

## 7. Key Design Decisions

### 7.1 Why Attention-Based Fusion?

**Alternatives Considered**:
1. **Concatenation**: Simple but no interaction
2. **Gating**: Limited interaction patterns
3. **Attention**: ✅ Learns complex relationships

**Benefits**:
- Learns which modalities are relevant
- Handles missing data naturally
- Interpretable via attention weights

### 7.2 Why Separate Encoders?

**Alternatives Considered**:
1. **Shared Encoder**: Doesn't respect modality differences
2. **Separate Encoders**: ✅ Tailored to each modality

**Benefits**:
- WSI: Spatial attention for patches
- Genomic: Deep MLP for continuous features
- Clinical: Transformer for text

### 7.3 Why Temporal Attention?

**Alternatives Considered**:
1. **RNN/LSTM**: Sequential processing
2. **Attention**: ✅ Parallel, long-range dependencies

**Benefits**:
- Captures non-sequential patterns
- Handles variable-length sequences
- Faster training (parallel)

---

## 8. Implementation Details

### 8.1 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| embed_dim | 256 | Embedding dimension |
| num_heads | 8 | Attention heads |
| num_layers | 2-4 | Transformer layers |
| dropout | 0.1-0.3 | Regularization |
| learning_rate | 1e-4 | AdamW |
| weight_decay | 0.01 | L2 regularization |
| batch_size | 8-32 | Depends on memory |
| max_grad_norm | 1.0 | Gradient clipping |

### 8.2 Initialization

```python
# Xavier/Glorot for linear layers
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

# Positional encodings
pos_enc = sinusoidal_encoding(positions)

# Embeddings
nn.init.normal_(embedding.weight, mean=0, std=0.02)
```

### 8.3 Regularization

- **Dropout**: 0.1-0.3 in encoders and heads
- **Weight Decay**: 0.01 in optimizer
- **Gradient Clipping**: max_norm=1.0
- **Layer Normalization**: After each major block
- **Label Smoothing**: Optional, ε=0.1

---

## 9. Computational Complexity

### 9.1 Time Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| WSI Encoder | O(N² · d) | N patches, d embed_dim |
| Genomic Encoder | O(d²) | MLP layers |
| Clinical Encoder | O(L² · d) | L tokens |
| Cross-Modal Fusion | O(M² · d) | M modalities |
| Temporal Attention | O(T² · d) | T time points |

**Total**: O(N² · d + L² · d + T² · d)

### 9.2 Space Complexity

| Component | Memory | Notes |
|-----------|--------|-------|
| Model Parameters | 110MB | FP32 weights |
| Activations | ~2GB | Batch=16 |
| Gradients | 110MB | Same as params |
| Optimizer State | 220MB | AdamW (2x params) |

**Total**: ~2.5GB for training (batch=16)

---

## 10. Extensions

### 10.1 Additional Modalities

```python
# Add new modality
class RadiomicsEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

# Integrate into fusion
modalities['radiomics'] = radiomics_encoder(batch['radiomics'])
```

### 10.2 Multi-Task Learning

```python
# Multiple heads
classification_head = ClassificationHead(embed_dim, num_classes)
survival_head = SurvivalPredictionHead(embed_dim)
segmentation_head = SegmentationHead(embed_dim)

# Combined loss
loss = (α * classification_loss +
        β * survival_loss +
        γ * segmentation_loss)
```

### 10.3 Self-Supervised Pretraining

```python
# Contrastive learning
contrastive_loss = SimCLR(embeddings, augmented_embeddings)

# Masked reconstruction
masked_loss = MSE(reconstructed, original)

# Combined
pretrain_loss = contrastive_loss + masked_loss
```

---

## Summary

**Architecture Highlights**:
- ✅ Modular design with separate encoders
- ✅ Attention-based fusion for cross-modal learning
- ✅ Native missing modality handling
- ✅ Optional temporal reasoning
- ✅ Flexible task-specific heads

**Key Innovations**:
1. Cross-modal attention learns relationships
2. Graceful degradation with missing data
3. Temporal attention for progression
4. Modality-specific encoders

**Production Ready**:
- Well-tested (90+ unit tests)
- Documented (comprehensive)
- Deployable (FastAPI example)
- Scalable (batch processing)

---

**Last Updated**: 2026-04-05  
**Version**: 1.0.0  
**Status**: Production-ready for research ✅
