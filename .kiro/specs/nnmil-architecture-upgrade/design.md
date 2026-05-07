# Design Document: nnMIL Architecture Upgrade

## Overview

This document specifies the design for upgrading the Multiple Instance Learning (MIL) system from TransMIL (2021) to nnMIL (2024, Stanford/NIH). nnMIL represents a paradigm shift in computational pathology MIL training, focusing on **training-centric innovations** rather than architectural complexity. The key insight from the Stanford/NIH research is that systematic optimization of training configuration—including large-batch optimization, fixed-length bag sampling, and task-aware batch construction—delivers superior performance and generalizability across diverse clinical tasks.

### Design Philosophy

nnMIL follows the "no-new-net" philosophy (similar to nnU-Net for segmentation): **distill domain knowledge into explicit heuristic rules that guide model configuration**. Rather than introducing novel architectural components, nnMIL achieves state-of-the-art results through:

1. **Training Strategy**: Large batches (32 vs. 1), balanced sampling, feature subspace regularization
2. **Inference Strategy**: Sliding-window ensemble for uncertainty quantification
3. **Rule-Based Configuration**: Automatic hyperparameter selection from dataset fingerprints

The architecture itself is a **lightweight attention-based aggregator** with gated attention, maintaining parameter efficiency (~12M parameters) while enabling robust slide-level predictions.

### Key Innovations

**From Research ([Content was rephrased for compliance with licensing restrictions](https://arxiv.org/html/2511.14907)):**

- **Fixed-Length Bag Sampling**: Converts variable-length bags (100-10,000 patches) into uniform sub-bags, enabling efficient batching
- **Large-Batch Optimization**: Batch size of 32 (vs. traditional batch=1) with balanced class representation
- **Task-Aware Samplers**: Stratified sampling for classification, binned sampling for regression, event-balanced sampling for survival
- **Sliding-Window Inference**: Overlapping feature subspace predictions aggregated for ensemble uncertainty
- **Feature Subspace Sampling**: Random H-dimensional subsets (H=256) from full D-dimensional embeddings during training

**Performance Achievements:**
- Disease subtyping: 80.7-82.0% BACC across 8 tasks
- Biomarker detection: 77.1-79.5% AUC across 12 tasks  
- Prognosis prediction: 0.640-0.670 C-Index across 16 cancer types
- Consistent superiority over ABMIL, DSMIL, TransMIL, DTFD across 4 foundation models

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    nnMIL Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WSI → Foundation Model → Patch Features [N × D]                │
│                                ↓                                 │
│         ┌──────────────────────────────────────┐               │
│         │   Fixed-Length Bag Sampling          │               │
│         │   • Sample M patches (M = median/2)  │               │
│         │   • Pad if N < M, sample if N > M    │               │
│         └──────────────────────────────────────┘               │
│                                ↓                                 │
│         ┌──────────────────────────────────────┐               │
│         │   Task-Aware Batch Sampler           │               │
│         │   • Balanced classes (classification)│               │
│         │   • Binned targets (regression)      │               │
│         │   • Event-balanced (survival)        │               │
│         └──────────────────────────────────────┘               │
│                                ↓                                 │
│         ┌──────────────────────────────────────┐               │
│         │   Feature Subspace Sampling          │               │
│         │   • Random H dims from D (H=256)     │               │
│         │   • Regularization + efficiency      │               │
│         └──────────────────────────────────────┘               │
│                                ↓                                 │
│         ┌──────────────────────────────────────┐               │
│         │   Gated Attention Aggregator         │               │
│         │   α_i = softmax(w^T(tanh(Vx')⊙σ(Ux')))│               │
│         │   h = Σ α_i x_i                      │               │
│         └──────────────────────────────────────┘               │
│                                ↓                                 │
│         ┌──────────────────────────────────────┐               │
│         │   Classifier Head                    │               │
│         │   Linear(H, H) → ReLU → Linear(H, C) │               │
│         └──────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   nnMIL Inference Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Patch Features [N × D]                                         │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Sliding Window over Feature Dimensions             │       │
│  │  • Divide D dims into K chunks of size H            │       │
│  │  • Stride = H/4 (75% overlap)                       │       │
│  │  • K = (D - H) / stride + 1 predictions             │       │
│  └─────────────────────────────────────────────────────┘       │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Ensemble Aggregation                                │       │
│  │  • Mean prediction: ŷ = (1/K) Σ ŷ_k                 │       │
│  │  • Epistemic uncertainty: Var(ŷ_k)                  │       │
│  │  • Aleatoric uncertainty: Mean entropy               │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### nnMIL Model Architecture

The nnMIL model consists of four main components:

#### 1. Feature Projection (Optional)

```python
# Only used if foundation model feature_dim != hidden_dim
feature_proj = nn.Sequential(
    nn.Linear(feature_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout)
)
```

**Design Decision**: Unlike TransMIL which always projects features, nnMIL operates directly on foundation model embeddings when possible, preserving semantic integrity.

#### 2. Gated Attention Mechanism

```python
# Attention computation on H-dimensional subspace
V, U ∈ R^(H×D)  # Learnable projections
w ∈ R^H          # Scoring vector

α_i = exp(w^T(tanh(V·x'_i) ⊙ σ(U·x'_i))) / Σ_j exp(w^T(tanh(V·x'_j) ⊙ σ(U·x'_j)))
```

Where:
- `x'_i` is a randomly sampled H-dimensional subset of the full D-dimensional embedding during training
- `⊙` denotes element-wise multiplication
- `tanh` and `σ` are hyperbolic tangent and sigmoid activations

**Design Rationale**: Gated attention combines additive and multiplicative interactions, capturing non-linear patch-level importance. The feature subspace sampling (H << D) acts as implicit regularization.

#### 3. Attention-Weighted Aggregation

```python
h = Σ_{i=1}^N α_i · x_i  # Aggregate in FULL D-dimensional space
```

**Critical Design Choice**: Attention is computed in H-dimensional subspace, but aggregation uses the full D-dimensional embeddings. This preserves foundation model semantics while regularizing attention computation.

#### 4. Classifier Head

```python
classifier = nn.Sequential(
    nn.Linear(D, hidden_dim),  # D for single-scale, D*num_scales for multi-scale
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_classes)
)
```

### Multi-Scale Support

nnMIL maintains backward compatibility with TransMIL's multi-scale API:

**Early Fusion**:
```python
# Concatenate features from all scales before attention
fused = concat([scale1, scale2, scale3], dim=-1)  # [B, N, D*num_scales]
h = attention_aggregator(fused)
```

**Late Fusion**:
```python
# Independent attention per scale, concatenate representations
h1 = attention_aggregator(scale1)
h2 = attention_aggregator(scale2)
h3 = attention_aggregator(scale3)
h = concat([h1, h2, h3], dim=-1)  # [B, D*num_scales]
```

## Components and Interfaces

### 1. nnMIL Model Class

```python
class nnMIL(nn.Module):
    """
    nnMIL: No-New-Net Multiple Instance Learning
    
    A training-centric MIL framework that achieves state-of-the-art performance
    through systematic optimization of training configuration rather than
    architectural complexity.
    
    Args:
        feature_dim: Dimension of foundation model embeddings (e.g., 1024 for UNI)
        hidden_dim: Dimension for attention computation (default: 256)
        num_classes: Number of output classes (default: 2)
        dropout: Dropout rate (default: 0.25, higher than TransMIL's 0.1)
        multi_scale: Support multi-scale features (default: False)
        num_scales: Number of magnification scales (default: 1)
        fusion_strategy: 'early' or 'late' fusion (default: 'early')
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.25,
        multi_scale: bool = False,
        num_scales: int = 1,
        fusion_strategy: str = "early"
    ):
        ...
    
    def forward(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        training: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with feature subspace sampling during training.
        
        Args:
            features: [B, N, D] or list of [B, N, D] for multi-scale
            num_patches: [B] actual patch counts for masking
            return_attention: Return attention weights
            training: Enable feature subspace sampling
            
        Returns:
            logits: [B, num_classes]
            attention_weights: (optional) [B, N]
        """
        ...
```

### 2. Fixed-Length Bag Sampler

```python
class FixedLengthBagSampler:
    """
    Converts variable-length bags into fixed-length sub-bags.
    
    Key Design Decisions:
    - Bag length M = median_patches / 2 (rule-based from dataset fingerprint)
    - Training: Random sampling without replacement if N > M
    - Inference: Sliding window with stride for N > M
    - Padding: Zero vectors if N < M
    
    Args:
        bag_length: Fixed length M for all bags
        mode: 'train' (random sample) or 'inference' (sliding window)
        stride: Stride for sliding window (default: bag_length, non-overlapping)
    """
    
    def __init__(
        self,
        bag_length: int,
        mode: str = 'train',
        stride: Optional[int] = None
    ):
        ...
    
    def sample(
        self,
        features: torch.Tensor,  # [N, D]
        num_patches: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample fixed-length bag from variable-length input.
        
        Returns:
            sampled_features: [M, D]
            mask: [M] boolean mask (True for valid patches)
        """
        ...
```

### 3. Task-Aware Batch Samplers

```python
class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler for classification tasks.
    
    Ensures approximately equal representation of each class within batches.
    Addresses class imbalance by oversampling minority classes.
    
    Args:
        labels: [N] class labels
        batch_size: Batch size (default: 32)
        shuffle: Shuffle within classes (default: True)
    """
    ...

class RegressionBatchSampler(Sampler):
    """
    Binned batch sampler for regression tasks.
    
    Divides target range into bins and samples proportionally from each bin
    to achieve uniform coverage of the target distribution.
    
    Args:
        targets: [N] regression targets
        batch_size: Batch size (default: 32)
        num_bins: Number of bins for target discretization (default: 10)
    """
    ...

class SurvivalBatchSampler(Sampler):
    """
    Event-balanced batch sampler for survival analysis.
    
    Maintains balanced event rates and temporal distributions across batches.
    
    Args:
        times: [N] survival times
        events: [N] event indicators (1=event, 0=censored)
        batch_size: Batch size (default: 32)
    """
    ...
```

### 4. Sliding-Window Inference

```python
class SlidingWindowInference:
    """
    Sliding-window inference over feature dimensions for uncertainty estimation.
    
    Key Design:
    - Divide D-dimensional space into overlapping H-dimensional chunks
    - Default stride = H/4 (75% overlap)
    - K = (D - H) / stride + 1 predictions
    - Aggregate via mean pooling for final prediction
    - Compute variance for epistemic uncertainty
    
    Args:
        hidden_dim: Window size H (default: 256)
        stride: Stride for sliding window (default: hidden_dim // 4)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        stride: Optional[int] = None
    ):
        ...
    
    def __call__(
        self,
        model: nn.Module,
        features: torch.Tensor,  # [B, N, D]
        num_patches: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform sliding-window inference.
        
        Returns:
            {
                'logits': [B, num_classes] mean prediction,
                'epistemic_uncertainty': [B] variance across windows,
                'aleatoric_uncertainty': [B] mean entropy,
                'all_predictions': [B, K, num_classes] all window predictions
            }
        """
        ...
```

### 5. Uncertainty Quantification

```python
class UncertaintyEstimator:
    """
    Compute epistemic and aleatoric uncertainty from ensemble predictions.
    
    For classification:
    - Total uncertainty: H(p̄) = -Σ p̄_c log(p̄_c)
    - Aleatoric: (1/K) Σ H(p_k)
    - Epistemic (MI): H(p̄) - Aleatoric
    
    For survival:
    - Epistemic: Var(η_k) where η_k is risk score from chunk k
    - Survival probability uncertainty: SD(S_k(t)) across chunks
    
    Args:
        task_type: 'classification', 'regression', or 'survival'
    """
    
    def __init__(self, task_type: str):
        ...
    
    def compute_uncertainty(
        self,
        predictions: torch.Tensor,  # [B, K, ...] K predictions per sample
        task_specific_args: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty metrics.
        
        Returns:
            {
                'epistemic': [B] model uncertainty,
                'aleatoric': [B] data uncertainty,
                'total': [B] combined uncertainty
            }
        """
        ...
```

### 6. Rule-Based Configuration

```python
class nnMILConfig:
    """
    Rule-based configuration derived from dataset fingerprint.
    
    Dataset Fingerprint:
    - Patch count distribution (median, IQR, 5th/95th percentiles)
    - Magnification level
    - Embedding dimension D
    - Class prevalence (classification)
    - Target range (regression)
    - Event/censoring rates (survival)
    
    Derived Configuration:
    - bag_length: median_patches / 2
    - hidden_dim: 256 (fixed)
    - dropout: 0.25 (fixed)
    - batch_size: 32 (default, adjustable for GPU memory)
    - learning_rate: 3e-4 (1e-4 for survival)
    - inference_stride: hidden_dim / 4
    """
    
    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        task_type: str
    ) -> 'nnMILConfig':
        """
        Automatically derive configuration from dataset characteristics.
        
        Args:
            dataset: Training dataset
            task_type: 'classification', 'regression', or 'survival'
            
        Returns:
            nnMILConfig with rule-based hyperparameters
        """
        ...
```

### 7. Training Interface

```python
class nnMILTrainer:
    """
    Unified training interface for nnMIL.
    
    Key Features:
    - Large-batch optimization (batch_size=32)
    - Task-aware batch sampling
    - Gradient accumulation for memory-constrained GPUs
    - Learning rate scaling: lr_scaled = lr_base * sqrt(batch_size)
    - Cosine annealing with warmup (5 epochs)
    - Early stopping (patience=10)
    
    Args:
        model: nnMIL model
        config: nnMILConfig
        task_type: 'classification', 'regression', or 'survival'
    """
    
    def __init__(
        self,
        model: nnMIL,
        config: nnMILConfig,
        task_type: str
    ):
        ...
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train nnMIL model.
        
        Returns:
            Training history with metrics per epoch
        """
        ...
```

## Data Models

### Bag Representation

```python
@dataclass
class Bag:
    """
    Represents a bag (slide) with patches.
    
    Attributes:
        features: [N, D] patch features from foundation model
        label: Slide-level label (class, target, or survival tuple)
        num_patches: Actual number of patches (before padding)
        slide_id: Unique slide identifier
        metadata: Optional metadata (patient_id, magnification, etc.)
    """
    features: torch.Tensor
    label: Union[int, float, Tuple[float, int]]
    num_patches: int
    slide_id: str
    metadata: Optional[Dict[str, Any]] = None
```

### Training Batch

```python
@dataclass
class TrainingBatch:
    """
    Batch of fixed-length bags for training.
    
    Attributes:
        features: [B, M, D] fixed-length bag features
        labels: [B] slide-level labels
        masks: [B, M] boolean masks (True for valid patches)
        num_patches: [B] actual patch counts
        slide_ids: [B] slide identifiers
    """
    features: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor
    num_patches: torch.Tensor
    slide_ids: List[str]
```

### Inference Output

```python
@dataclass
class InferenceOutput:
    """
    Output from nnMIL inference with uncertainty.
    
    Attributes:
        logits: [B, num_classes] or [B] predictions
        probabilities: [B, num_classes] softmax probabilities (classification only)
        attention_weights: [B, N] attention weights
        epistemic_uncertainty: [B] model uncertainty
        aleatoric_uncertainty: [B] data uncertainty
        total_uncertainty: [B] combined uncertainty
        slide_ids: [B] slide identifiers
    """
    logits: torch.Tensor
    probabilities: Optional[torch.Tensor]
    attention_weights: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor
    slide_ids: List[str]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Prework Analysis

Before writing correctness properties, I'll analyze each acceptance criterion for testability:


### Property Reflection

After analyzing all acceptance criteria, I identified the following redundancies:

- **Properties 3.1 and 3.7** are identical (all bags have length N) - will keep 3.1
- **Properties 3.4 and 5.1** both test sliding window activation - will combine into one property
- **Properties 1.2 and 1.3** can be combined into a single shape invariant property
- **Properties 6.1 and 6.4** both relate to dropout during uncertainty estimation - will combine

The remaining properties provide unique validation value and will be included in the design.

### Property 1: Input/Output Shape Invariance

*For any* valid input tensor with dimensions [batch_size, num_patches, feature_dim], the nnMIL model SHALL accept the input and produce output logits with dimensions [batch_size, num_classes].

**Validates: Requirements 1.2, 1.3**

### Property 2: Configuration Validation

*For any* hidden_dim and num_heads where hidden_dim is not divisible by num_heads, model initialization SHALL raise a ValueError with a descriptive message.

**Validates: Requirements 1.5**

### Property 3: Parameter Efficiency

*For any* valid model configuration, the total parameter count SHALL be within 20% of 12.2M parameters (i.e., between 9.76M and 14.64M).

**Validates: Requirements 1.6**

### Property 4: Batch Size Flexibility

*For any* batch size B in the range [1, 64], the training system SHALL successfully process a batch of B bags without error.

**Validates: Requirements 2.1**

### Property 5: Learning Rate Scaling

*For any* base learning rate lr_base and batch size B, the scaled learning rate SHALL equal lr_base * sqrt(B).

**Validates: Requirements 2.4**

### Property 6: Effective Batch Size Logging

*For any* training step with batch_size B and accumulation_steps A, the logged effective batch size SHALL equal B * A.

**Validates: Requirements 2.6**

### Property 7: Fixed-Length Bag Invariant

*For any* slide with N_actual patches and configured bag length M, the sampled bag SHALL have exactly M patches.

**Validates: Requirements 3.1, 3.7**

### Property 8: Padding Correctness

*For any* slide with N_actual < M patches, the sampled bag SHALL contain N_actual original patches followed by (M - N_actual) zero vectors.

**Validates: Requirements 3.2**

### Property 9: Sampling Without Replacement

*For any* slide with N_actual > M patches during training, the sampled M patches SHALL be unique (no duplicates).

**Validates: Requirements 3.3**

### Property 10: Sliding Window Activation

*For any* slide with N_actual > M patches during inference, the system SHALL apply sliding window processing with overlapping windows.

**Validates: Requirements 3.4, 5.1**

### Property 11: Bag Length Configuration Range

*For any* configured bag length M in the range [100, 10000], the system SHALL successfully create and process bags of length M.

**Validates: Requirements 3.5**

### Property 12: Attention Mask Correctness

*For any* bag with padding, the attention mask SHALL mark padded positions as False and valid positions as True.

**Validates: Requirements 3.6**

### Property 13: Balanced Batch Composition

*For any* batch sampled by the Task_Aware_Sampler on a classification dataset, the class distribution within the batch SHALL be approximately balanced (within 20% of uniform distribution).

**Validates: Requirements 4.1**

### Property 14: Minority Class Oversampling

*For any* imbalanced dataset with minority class proportion < 0.3, the Task_Aware_Sampler SHALL oversample minority classes such that they appear more frequently than their natural proportion.

**Validates: Requirements 4.2**

### Property 15: Minimum Class Representation

*For any* batch with batch_size >= num_classes sampled with balanced sampling, each class SHALL appear at least once in the batch.

**Validates: Requirements 4.5**

### Property 16: Window Overlap Correctness

*For any* slide divided into windows with stride S and window size M, consecutive windows SHALL overlap by (M - S) patches.

**Validates: Requirements 5.2**

### Property 17: Mean Pooling Aggregation

*For any* set of K window predictions {ŷ_1, ..., ŷ_K}, the aggregated prediction SHALL equal (1/K) * Σ ŷ_k.

**Validates: Requirements 5.4**

### Property 18: Stride Configuration Range

*For any* configured stride S in the range [0.5*M, M] where M is window size, the system SHALL successfully perform sliding window inference.

**Validates: Requirements 5.5**

### Property 19: Inference Output Shape

*For any* slide processed during inference, the output logits SHALL have shape [num_classes].

**Validates: Requirements 5.7**

### Property 20: Uncertainty Output Shape

*For any* batch of size B processed with uncertainty estimation, the uncertainty output SHALL have shape [B, 2] containing [epistemic, aleatoric] uncertainties.

**Validates: Requirements 6.3**

### Property 21: Dropout Activation for Uncertainty

*For any* inference request with uncertainty estimation enabled, dropout layers SHALL remain active during forward passes.

**Validates: Requirements 6.1, 6.4**

### Property 22: Uncertainty Normalization

*For any* computed uncertainty values (epistemic, aleatoric), the values SHALL be in the range [0, 1].

**Validates: Requirements 6.5**

### Property 23: Combined Uncertainty Formula

*For any* epistemic uncertainty E and aleatoric uncertainty A, the combined uncertainty SHALL equal sqrt(E² + A²).

**Validates: Requirements 6.6**

### Property 24: Foundation Model Compatibility

*For any* feature tensor with dimensions matching UNI (1024), CONCH (512), Phikon (768), or ResNet50 (2048), the nnMIL model SHALL successfully process the features.

**Validates: Requirements 7.1**

### Property 25: Automatic Dimension Detection

*For any* input feature tensor, the model SHALL correctly detect the feature dimension from the tensor shape.

**Validates: Requirements 7.2**

### Property 26: Adaptive Projection

*For any* input with feature_dim != configured hidden_dim, the model SHALL apply a learned projection layer to match dimensions.

**Validates: Requirements 7.3**

### Property 27: Weight Freezing

*For any* foundation model with frozen weights, training SHALL not update those weights (gradient should be None or zero).

**Validates: Requirements 7.4**

### Property 28: Configuration Loading

*For any* valid YAML configuration file, the system SHALL successfully load and parse the configuration without errors.

**Validates: Requirements 9.1**

### Property 29: Configuration Validation

*For any* invalid configuration parameter, the system SHALL raise a descriptive ValueError indicating which parameter is invalid and why.

**Validates: Requirements 9.3**

### Property 30: Configuration Inheritance

*For any* task-specific configuration that overrides base configuration, the final configuration SHALL contain task-specific values for overridden parameters and base values for non-overridden parameters.

**Validates: Requirements 9.4**

### Property 31: Configuration Logging

*For any* training run, all active configuration parameters SHALL be logged at the start of training.

**Validates: Requirements 9.5**

### Property 32: Configuration Persistence

*For any* saved model checkpoint, the checkpoint file SHALL contain the complete configuration used during training.

**Validates: Requirements 9.6**

### Property 33: API Compatibility

*For any* input tensor, both TransMIL and nnMIL models SHALL accept the input and return outputs with compatible shapes.

**Validates: Requirements 10.2, 10.6**

### Property 34: Weight Transfer

*For any* TransMIL checkpoint, compatible layers (feature projection, classifier) SHALL transfer to nnMIL with matching weights.

**Validates: Requirements 10.4**

### Property 35: Metrics Logging

*For any* training epoch, the system SHALL log training loss, validation loss, validation AUC, learning rate, and gradient norm.

**Validates: Requirements 11.1, 11.2**

### Property 36: Checkpoint Saving

*For any* training run with checkpoint interval N, the system SHALL save checkpoints at epochs that are multiples of N.

**Validates: Requirements 12.1**

### Property 37: Best Model Tracking

*For any* training run, the system SHALL maintain and save the checkpoint with the highest validation AUC.

**Validates: Requirements 12.2**

### Property 38: Early Stopping

*For any* training run with patience P, training SHALL stop if validation AUC does not improve for P consecutive epochs.

**Validates: Requirements 12.3**

### Property 39: Multi-Scale Input Handling

*For any* list of feature tensors [scale1, scale2, ..., scaleN] where each has shape [B, M, D_i], the multi-scale model SHALL successfully process all scales.

**Validates: Requirements 13.1**

### Property 40: Early Fusion Concatenation

*For any* multi-scale input with early fusion, features from all scales SHALL be concatenated along the feature dimension before attention computation.

**Validates: Requirements 13.3**

## Error Handling

### Input Validation Errors

**Invalid Tensor Shapes**:
```python
if features.dim() != 3:
    raise ValueError(
        f"Expected 3D tensor [batch, patches, features], got {features.dim()}D tensor"
    )
```

**Invalid Configuration**:
```python
if hidden_dim % num_heads != 0:
    raise ValueError(
        f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
    )
```

**Invalid Bag Length**:
```python
if not (100 <= bag_length <= 10000):
    raise ValueError(
        f"bag_length must be in range [100, 10000], got {bag_length}"
    )
```

### Training Errors

**Insufficient GPU Memory**:
```python
try:
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.warning("GPU OOM detected, enabling gradient accumulation")
        # Activate gradient accumulation
        accumulation_steps = calculate_accumulation_steps(batch_size, available_memory)
    else:
        raise
```

**NaN Loss Detection**:
```python
if torch.isnan(loss):
    logger.error(f"NaN loss detected at epoch {epoch}, step {step}")
    logger.error(f"Batch statistics: {compute_batch_stats(batch)}")
    raise RuntimeError("Training diverged: NaN loss detected")
```

**Checkpoint Loading Errors**:
```python
try:
    checkpoint = torch.load(checkpoint_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
except RuntimeError as e:
    if "size mismatch" in str(e):
        logger.warning("Checkpoint architecture mismatch, attempting partial load")
        load_compatible_weights(model, checkpoint)
    else:
        raise
```

### Inference Errors

**Empty Slide**:
```python
if num_patches == 0:
    logger.warning(f"Empty slide {slide_id}, returning default prediction")
    return default_prediction(num_classes)
```

**Corrupted Features**:
```python
if torch.isnan(features).any() or torch.isinf(features).any():
    raise ValueError(
        f"Corrupted features detected in slide {slide_id}: "
        f"NaN={torch.isnan(features).sum()}, Inf={torch.isinf(features).sum()}"
    )
```

### Graceful Degradation

**Missing Foundation Model**:
```python
try:
    foundation_model = load_foundation_model(model_name)
except ImportError:
    logger.warning(f"Foundation model {model_name} not available, using ResNet50")
    foundation_model = load_foundation_model("resnet50")
```

**Uncertainty Estimation Failure**:
```python
try:
    uncertainty = estimate_uncertainty(predictions)
except Exception as e:
    logger.warning(f"Uncertainty estimation failed: {e}, returning zeros")
    uncertainty = torch.zeros(batch_size, 2)
```

## Testing Strategy

### Unit Testing

**Model Architecture Tests**:
- Test nnMIL initialization with various configurations
- Test forward pass with different input shapes
- Test multi-scale input handling (early/late fusion)
- Test attention mechanism computation
- Test classifier head output shapes

**Sampling Tests**:
- Test fixed-length bag sampling (padding, truncation, random sampling)
- Test task-aware batch samplers (balanced, regression, survival)
- Test sliding window generation with various strides
- Test attention mask creation

**Uncertainty Tests**:
- Test epistemic uncertainty computation (MC dropout)
- Test aleatoric uncertainty computation
- Test combined uncertainty formula
- Test uncertainty normalization

**Configuration Tests**:
- Test YAML configuration loading
- Test configuration validation (valid/invalid parameters)
- Test configuration inheritance
- Test rule-based configuration derivation from dataset fingerprints

### Property-Based Testing

**Property Test Configuration**:
- Minimum 100 iterations per property test
- Use Hypothesis library for Python
- Each test tagged with: `Feature: nnmil-architecture-upgrade, Property {number}: {property_text}`

**Example Property Test**:
```python
from hypothesis import given, strategies as st
import pytest

@given(
    batch_size=st.integers(min_value=1, max_value=64),
    num_patches=st.integers(min_value=10, max_value=1000),
    feature_dim=st.integers(min_value=256, max_value=2560),
    num_classes=st.integers(min_value=2, max_value=10)
)
def test_property_1_input_output_shape_invariance(
    batch_size, num_patches, feature_dim, num_classes
):
    """
    Feature: nnmil-architecture-upgrade, Property 1: For any valid input tensor 
    with dimensions [batch_size, num_patches, feature_dim], the nnMIL model SHALL 
    accept the input and produce output logits with dimensions [batch_size, num_classes].
    """
    model = nnMIL(feature_dim=feature_dim, num_classes=num_classes)
    features = torch.randn(batch_size, num_patches, feature_dim)
    
    logits = model(features)
    
    assert logits.shape == (batch_size, num_classes)
```

**Property Test Coverage**:
- Shape invariants (Properties 1, 19, 20, 39)
- Mathematical formulas (Properties 5, 17, 23)
- Sampling correctness (Properties 7, 8, 9, 12)
- Configuration handling (Properties 28, 29, 30)
- Uncertainty computation (Properties 21, 22, 23)

### Integration Testing

**End-to-End Training Tests**:
- Train nnMIL for 5 epochs on synthetic data
- Verify convergence (loss decreases)
- Verify checkpointing (best model saved)
- Verify early stopping (stops after patience epochs)
- Verify metrics logging (all metrics logged)

**Benchmark Tests** (Requirements 8.1-8.6):
- Test on PatchCamelyon dataset (target: ≥93.94% AUC)
- Test on disease subtyping tasks (target: ≥80.7% BACC)
- Test on biomarker detection tasks (target: ≥77.1% AUC)
- Test on prognosis tasks (target: ≥0.640 C-Index)
- Compare training time vs. TransMIL (target: ≤120%)
- Compare GPU memory vs. TransMIL (target: ≤120%)

**Cross-Model Generalization Tests**:
- Test with UNI features (1024-dim)
- Test with CONCH features (512-dim)
- Test with Phikon features (768-dim)
- Test with ResNet50 features (2048-dim)
- Verify performance within 5% AUC across models

### Backward Compatibility Tests

**TransMIL Compatibility**:
- Test unified training interface with both models
- Test checkpoint loading (TransMIL → nnMIL)
- Test weight transfer (compatible layers)
- Test migration script (checkpoint conversion)
- Test API compatibility (same input/output format)

### Test Coverage Target

- **Overall**: ≥85% coverage on all nnMIL components
- **Core Model**: ≥90% coverage on nnMIL class
- **Samplers**: ≥85% coverage on batch samplers
- **Uncertainty**: ≥85% coverage on uncertainty estimation
- **Configuration**: ≥80% coverage on config management

## Performance Considerations

### Training Performance

**Large-Batch Optimization Benefits**:
- **Faster Convergence**: Batch size 32 achieves convergence in ~20% fewer epochs than batch size 1
- **Better Gradient Estimates**: Larger batches provide more stable gradient estimates
- **GPU Utilization**: Better GPU utilization with larger batches (higher throughput)

**Memory Optimization**:
- **Gradient Accumulation**: Automatically activated when GPU memory insufficient
- **Feature Subspace Sampling**: Reduces memory footprint during attention computation (H=256 vs. D=1024-2560)
- **Mixed Precision Training**: Support for FP16 training to reduce memory usage

**Training Time Estimates** (based on research):
- Small datasets (500-1000 WSIs): 0.5-1 hour on 8×L40 GPUs
- Medium datasets (2000-5000 WSIs): 2-3 hours
- Large datasets (10000+ WSIs): 5-8 hours

### Inference Performance

**Sliding-Window Inference**:
- **Computational Cost**: K = (D - H) / stride + 1 forward passes per slide
- **Default Configuration**: D=1024, H=256, stride=64 → K=13 passes
- **Inference Time**: ~0.1-0.5 seconds per WSI (depends on patch count)

**Uncertainty Estimation Overhead**:
- **MC Dropout**: 10 forward passes → 10× inference time
- **Sliding Window**: Already provides ensemble → no additional cost
- **Trade-off**: Sliding window provides uncertainty "for free" compared to MC dropout

**Optimization Strategies**:
- **Batch Inference**: Process multiple slides in parallel
- **Feature Caching**: Cache foundation model features (avoid recomputation)
- **Stride Tuning**: Larger stride (less overlap) → faster inference, slightly lower accuracy

### Memory Requirements

**Training Memory** (per GPU):
- **Model Parameters**: ~50 MB (12M parameters × 4 bytes)
- **Optimizer State**: ~100 MB (AdamW maintains 2 states per parameter)
- **Batch Data**: Batch size 32 × bag length 500 × feature dim 1024 × 4 bytes = ~64 MB
- **Activations**: ~200-500 MB (depends on model depth)
- **Total**: ~500-700 MB per GPU (fits comfortably on 16GB+ GPUs)

**Inference Memory**:
- **Model**: ~50 MB
- **Single Slide**: Patches 1000 × feature dim 1024 × 4 bytes = ~4 MB
- **Sliding Window**: K=13 windows × 4 MB = ~52 MB
- **Total**: ~100-150 MB (very lightweight)

### Scalability

**Dataset Scaling**:
- **10K WSIs**: Trains in ~5 hours on 8×L40 GPUs
- **100K WSIs**: Estimated ~50 hours (linear scaling)
- **Bottleneck**: Feature extraction (foundation model), not MIL training

**Model Scaling**:
- **Hidden Dim**: 128-512 (256 is sweet spot for performance/efficiency)
- **Bag Length**: 100-10000 (median/2 rule works well across datasets)
- **Batch Size**: 16-64 (32 is default, adjust for GPU memory)

## Deployment Considerations

### Model Serving

**Inference API**:
```python
class nnMILInferenceService:
    """
    Production inference service for nnMIL.
    
    Features:
    - Batch inference for throughput
    - Feature caching for repeated slides
    - Uncertainty quantification
    - Attention map visualization
    """
    
    def predict(
        self,
        slide_id: str,
        features: torch.Tensor,
        return_uncertainty: bool = True,
        return_attention: bool = False
    ) -> InferenceOutput:
        ...
```

**Deployment Options**:
1. **REST API**: Flask/FastAPI service for HTTP requests
2. **gRPC**: High-performance RPC for low-latency inference
3. **Batch Processing**: Offline processing of large slide cohorts
4. **Edge Deployment**: Lightweight model for on-device inference

### Model Versioning

**Checkpoint Format**:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config.to_dict(),
    'epoch': epoch,
    'best_val_auc': best_val_auc,
    'training_history': history,
    'metadata': {
        'nnmil_version': '1.0.0',
        'pytorch_version': torch.__version__,
        'foundation_model': 'UNI',
        'training_date': datetime.now().isoformat()
    }
}
```

**Version Compatibility**:
- **Semantic Versioning**: Major.Minor.Patch (e.g., 1.0.0)
- **Backward Compatibility**: Load checkpoints from previous minor versions
- **Migration Scripts**: Automatic conversion for major version upgrades

### Monitoring and Logging

**Training Metrics**:
- Loss curves (training/validation)
- AUC curves (validation)
- Learning rate schedule
- Gradient norms
- GPU memory usage
- Training throughput (samples/second)

**Inference Metrics**:
- Prediction latency (p50, p95, p99)
- Throughput (slides/second)
- Uncertainty distribution
- Attention weight statistics
- Error rates

**Integration with MLOps**:
- **TensorBoard**: Real-time training visualization
- **Weights & Biases**: Experiment tracking and hyperparameter tuning
- **MLflow**: Model registry and deployment tracking
- **Prometheus**: Production monitoring and alerting

## Migration Guide

### From TransMIL to nnMIL

**Step 1: Update Model Initialization**:
```python
# Old (TransMIL)
model = TransMIL(
    feature_dim=1024,
    hidden_dim=256,
    num_classes=2,
    num_layers=2,
    num_heads=8
)

# New (nnMIL)
model = nnMIL(
    feature_dim=1024,
    hidden_dim=256,  # Same as TransMIL
    num_classes=2,
    dropout=0.25     # Higher dropout than TransMIL (0.1)
)
```

**Step 2: Update Training Loop**:
```python
# Old (TransMIL) - batch size 1
for features, labels in dataloader:  # batch_size=1
    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

# New (nnMIL) - batch size 32 with task-aware sampling
sampler = BalancedBatchSampler(labels, batch_size=32)
dataloader = DataLoader(dataset, batch_sampler=sampler)

for features, labels in dataloader:  # batch_size=32
    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

**Step 3: Update Inference**:
```python
# Old (TransMIL) - single forward pass
logits = model(features)

# New (nnMIL) - sliding window with uncertainty
inference = SlidingWindowInference(hidden_dim=256, stride=64)
output = inference(model, features)
logits = output['logits']
uncertainty = output['epistemic_uncertainty']
```

**Step 4: Migrate Checkpoints**:
```python
# Use migration script
from nnmil.migration import migrate_transmil_checkpoint

nnmil_checkpoint = migrate_transmil_checkpoint(
    transmil_checkpoint_path='transmil_best.pth',
    output_path='nnmil_migrated.pth'
)

# Load migrated checkpoint
model = nnMIL(feature_dim=1024, num_classes=2)
model.load_state_dict(torch.load('nnmil_migrated.pth'))
```

### Configuration Migration

**Old TransMIL Config**:
```yaml
model:
  type: transmil
  feature_dim: 1024
  hidden_dim: 256
  num_layers: 2
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 1
  learning_rate: 1e-4
  epochs: 100
```

**New nnMIL Config**:
```yaml
model:
  type: nnmil
  feature_dim: 1024
  hidden_dim: 256
  dropout: 0.25  # Increased from 0.1

training:
  batch_size: 32  # Increased from 1
  learning_rate: 3e-4  # Scaled: 1e-4 * sqrt(32) ≈ 3e-4
  epochs: 100
  batch_sampler: balanced  # New: task-aware sampling

inference:
  sliding_window: true  # New: uncertainty estimation
  stride: 64
```

## Future Enhancements

### Planned Features

1. **Multi-Modal Fusion**: Integrate clinical data, genomics, and imaging
2. **Attention Visualization**: Interactive heatmaps for interpretability
3. **Active Learning**: Uncertainty-guided sample selection for annotation
4. **Federated Learning**: Privacy-preserving multi-institutional training
5. **AutoML Integration**: Automated hyperparameter tuning with Optuna/Ray Tune

### Research Directions

1. **Adaptive Bag Sampling**: Learn optimal bag length per slide
2. **Hierarchical Attention**: Multi-level attention (patch → region → slide)
3. **Contrastive Learning**: Self-supervised pretraining for MIL
4. **Causal Inference**: Identify causal relationships in pathology features
5. **Explainable AI**: Generate natural language explanations for predictions

## References

1. Luo, X., Xiang, J., Ji, Y., & Li, R. (2024). nnMIL: A generalizable multiple instance learning framework for computational pathology. *arXiv preprint arXiv:2511.14907*. [https://arxiv.org/html/2511.14907](https://arxiv.org/html/2511.14907)

2. Shao, Z., et al. (2021). TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification. *NeurIPS 2021*.

3. Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. *ICML 2018*.

4. Lu, M. Y., et al. (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. *Nature Biomedical Engineering*.

5. Isensee, F., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Authors**: HistoCore Development Team  
**Status**: Design Complete - Ready for Implementation
