# Requirements Document: nnMIL Architecture Upgrade

## Introduction

This document specifies requirements for upgrading the Multiple Instance Learning (MIL) architecture from TransMIL (2021) to nnMIL (2024, Stanford/NIH). nnMIL represents state-of-the-art performance with significant training innovations including large batch optimization, fixed-length bag sampling, task-aware batch samplers, and sliding-window inference. The upgrade aims to maintain or improve the current 93.94% AUC performance while adopting modern training techniques and gaining uncertainty quantification capabilities.

## Glossary

- **MIL_System**: The Multiple Instance Learning system responsible for whole-slide image classification
- **nnMIL_Model**: The neural network MIL model architecture from Stanford/NIH (2024)
- **TransMIL_Model**: The existing Transformer-based MIL model (2021)
- **Bag**: A collection of patches extracted from a whole-slide image
- **Fixed_Length_Bag**: A bag with a predetermined number of patches (nnMIL approach)
- **Variable_Length_Bag**: A bag with varying number of patches (TransMIL approach)
- **Batch_Sampler**: Component that creates batches of bags for training
- **Task_Aware_Sampler**: Batch sampler that considers task-specific characteristics
- **Sliding_Window_Inference**: Inference technique that processes overlapping windows of patches
- **Epistemic_Uncertainty**: Model uncertainty due to lack of knowledge (reducible with more data)
- **Aleatoric_Uncertainty**: Data uncertainty due to inherent noise (irreducible)
- **Foundation_Model**: Pre-trained feature extractor (e.g., UNI, CONCH, Phikon)
- **BACC**: Balanced Accuracy metric
- **C_Index**: Concordance Index for survival analysis

## Requirements

### Requirement 1: nnMIL Model Architecture Implementation

**User Story:** As a computational pathologist, I want to use the nnMIL model architecture, so that I can leverage state-of-the-art MIL performance with modern training techniques.

#### Acceptance Criteria

1. THE MIL_System SHALL implement the nnMIL_Model architecture as specified in the Stanford/NIH 2024 paper
2. THE nnMIL_Model SHALL accept feature tensors from Foundation_Model extractors with dimensions [batch_size, num_patches, feature_dim]
3. THE nnMIL_Model SHALL produce slide-level classification logits with dimensions [batch_size, num_classes]
4. THE nnMIL_Model SHALL support configurable hidden dimensions, number of layers, and attention heads
5. WHEN initialized, THE nnMIL_Model SHALL validate that hidden_dim is divisible by num_heads
6. THE nnMIL_Model SHALL maintain parameter efficiency comparable to TransMIL_Model (within 20% of 12.2M parameters)

### Requirement 2: Large Batch Training Optimization

**User Story:** As a machine learning engineer, I want to train with large batches (batch=32), so that I can achieve faster convergence and better performance than single-bag training.

#### Acceptance Criteria

1. THE MIL_System SHALL support batch sizes from 1 to 64 bags per training step
2. WHEN batch_size exceeds 1, THE MIL_System SHALL aggregate gradients across all bags in the batch before updating weights
3. THE MIL_System SHALL implement gradient accumulation WHEN GPU memory is insufficient for the requested batch size
4. THE MIL_System SHALL scale learning rate proportionally to batch size using the formula: lr_scaled = lr_base * sqrt(batch_size)
5. WHEN training with batch_size >= 16, THE MIL_System SHALL achieve convergence within 20% fewer epochs than batch_size = 1
6. THE MIL_System SHALL log effective batch size (batch_size * accumulation_steps) to training metrics

### Requirement 3: Fixed-Length Bag Sampling

**User Story:** As a machine learning engineer, I want to sample fixed-length bags from whole-slide images, so that I can efficiently batch multiple slides together during training.

#### Acceptance Criteria

1. THE MIL_System SHALL implement patch sampling that produces Fixed_Length_Bag instances with exactly N patches per bag
2. WHEN a slide contains fewer than N patches, THE MIL_System SHALL pad the bag with zero vectors to reach length N
3. WHEN a slide contains more than N patches, THE MIL_System SHALL randomly sample N patches without replacement during training
4. WHEN a slide contains more than N patches, THE MIL_System SHALL use sliding window sampling during inference
5. THE MIL_System SHALL support configurable bag lengths from 100 to 10000 patches
6. THE MIL_System SHALL create attention masks that mark padded positions as invalid
7. FOR ALL Fixed_Length_Bag instances, THE bag length SHALL equal the configured N value

### Requirement 4: Task-Aware Batch Sampler

**User Story:** As a machine learning engineer, I want task-aware batch sampling, so that I can balance class distributions and improve training stability.

#### Acceptance Criteria

1. THE MIL_System SHALL implement a Task_Aware_Sampler that balances class distributions within each batch
2. WHEN training on imbalanced datasets, THE Task_Aware_Sampler SHALL oversample minority classes to achieve approximately equal representation per batch
3. THE Task_Aware_Sampler SHALL support stratified sampling by slide-level labels
4. THE Task_Aware_Sampler SHALL support weighted sampling based on slide difficulty scores
5. WHEN configured for balanced sampling, THE Task_Aware_Sampler SHALL ensure each class appears at least once per batch WHEN batch_size >= num_classes
6. THE Task_Aware_Sampler SHALL shuffle samples within each epoch while maintaining balance constraints

### Requirement 5: Sliding-Window Inference

**User Story:** As a computational pathologist, I want sliding-window inference for large slides, so that I can process slides with more patches than the training bag length without losing information.

#### Acceptance Criteria

1. WHEN a slide contains more than N patches at inference time, THE MIL_System SHALL apply sliding window processing
2. THE MIL_System SHALL divide the slide into overlapping windows of size N patches with configurable stride
3. THE MIL_System SHALL process each window through the nnMIL_Model independently
4. THE MIL_System SHALL aggregate predictions from all windows using mean pooling of logits
5. THE MIL_System SHALL support configurable window stride from 50% to 100% of window size
6. WHEN stride equals window size, THE MIL_System SHALL perform non-overlapping window inference
7. THE MIL_System SHALL return aggregated slide-level predictions with dimensions [num_classes]

### Requirement 6: Uncertainty Quantification

**User Story:** As a clinical researcher, I want uncertainty estimates for predictions, so that I can identify cases requiring expert review and assess model confidence.

#### Acceptance Criteria

1. THE MIL_System SHALL compute Epistemic_Uncertainty using Monte Carlo dropout with at least 10 forward passes
2. THE MIL_System SHALL compute Aleatoric_Uncertainty by modeling prediction variance through the network
3. THE MIL_System SHALL return uncertainty estimates alongside predictions with dimensions [batch_size, 2] for [epistemic, aleatoric]
4. WHEN uncertainty is requested, THE MIL_System SHALL enable dropout during inference for epistemic uncertainty estimation
5. THE MIL_System SHALL normalize uncertainty values to the range [0, 1] for interpretability
6. THE MIL_System SHALL provide a combined uncertainty score computed as: sqrt(epistemic^2 + aleatoric^2)

### Requirement 7: Foundation Model Compatibility

**User Story:** As a machine learning engineer, I want plug-and-play compatibility with multiple foundation models, so that I can experiment with different feature extractors without changing the MIL architecture.

#### Acceptance Criteria

1. THE nnMIL_Model SHALL accept features from UNI, CONCH, Phikon, and ResNet50 Foundation_Model extractors
2. THE nnMIL_Model SHALL automatically detect feature dimensions from input tensors
3. WHEN feature_dim does not match the configured value, THE nnMIL_Model SHALL apply a learned projection layer to match dimensions
4. THE MIL_System SHALL support freezing Foundation_Model weights during training
5. THE MIL_System SHALL support fine-tuning Foundation_Model weights with configurable learning rate multiplier
6. THE nnMIL_Model SHALL maintain consistent performance (within 5% AUC) across different Foundation_Model extractors on the same task

### Requirement 8: Performance Benchmarking

**User Story:** As a computational pathologist, I want to validate nnMIL performance against TransMIL, so that I can confirm the upgrade maintains or improves state-of-the-art results.

#### Acceptance Criteria

1. THE MIL_System SHALL achieve at least 93.94% AUC on the PatchCamelyon benchmark (matching current TransMIL_Model performance)
2. WHEN evaluated on disease subtyping tasks, THE nnMIL_Model SHALL achieve at least 80.7% BACC
3. WHEN evaluated on biomarker detection tasks, THE nnMIL_Model SHALL achieve at least 77.1% AUC
4. WHEN evaluated on prognosis tasks, THE nnMIL_Model SHALL achieve at least 0.640 C_Index
5. THE MIL_System SHALL complete training in no more than 120% of TransMIL_Model training time for equivalent epochs
6. THE MIL_System SHALL use no more than 120% of TransMIL_Model GPU memory during training

### Requirement 9: Configuration Management

**User Story:** As a machine learning engineer, I want rule-based configuration for nnMIL hyperparameters, so that I can easily switch between different training setups and reproduce experiments.

#### Acceptance Criteria

1. THE MIL_System SHALL load nnMIL configuration from YAML files with schema validation
2. THE MIL_System SHALL provide default configurations for: disease_subtyping, biomarker_detection, and prognosis tasks
3. THE MIL_System SHALL validate all configuration parameters at load time and raise descriptive errors for invalid values
4. THE MIL_System SHALL support configuration inheritance where task-specific configs override base configs
5. THE MIL_System SHALL log all active configuration parameters at training start
6. THE MIL_System SHALL save the complete configuration alongside trained model checkpoints

### Requirement 10: Backward Compatibility and Migration

**User Story:** As a machine learning engineer, I want to migrate from TransMIL to nnMIL without breaking existing workflows, so that I can adopt the new architecture incrementally.

#### Acceptance Criteria

1. THE MIL_System SHALL maintain the TransMIL_Model implementation as a selectable architecture option
2. THE MIL_System SHALL provide a unified training interface that works with both TransMIL_Model and nnMIL_Model
3. THE MIL_System SHALL support loading TransMIL_Model checkpoints and continuing training with nnMIL_Model architecture
4. WHEN switching from TransMIL_Model to nnMIL_Model, THE MIL_System SHALL transfer compatible weights (feature projection, classifier head)
5. THE MIL_System SHALL provide a migration script that converts TransMIL_Model checkpoints to nnMIL_Model format
6. THE MIL_System SHALL maintain API compatibility for inference: both models accept the same input format and return the same output format

### Requirement 11: Training Monitoring and Logging

**User Story:** As a machine learning engineer, I want comprehensive training metrics, so that I can monitor nnMIL training progress and diagnose issues.

#### Acceptance Criteria

1. THE MIL_System SHALL log training loss, validation loss, and validation AUC every epoch
2. THE MIL_System SHALL log batch-level metrics including: batch loss, learning rate, gradient norm, and effective batch size
3. THE MIL_System SHALL compute and log class-wise performance metrics: precision, recall, F1 score per class
4. THE MIL_System SHALL track and log GPU memory usage and training throughput (samples/second)
5. THE MIL_System SHALL save training curves as plots: loss curves, AUC curves, and learning rate schedule
6. THE MIL_System SHALL integrate with TensorBoard for real-time training visualization

### Requirement 12: Model Checkpointing and Early Stopping

**User Story:** As a machine learning engineer, I want automatic checkpointing and early stopping, so that I can save the best model and avoid overfitting.

#### Acceptance Criteria

1. THE MIL_System SHALL save model checkpoints every N epochs where N is configurable (default: 5)
2. THE MIL_System SHALL save the best model checkpoint based on validation AUC
3. THE MIL_System SHALL implement early stopping that halts training WHEN validation AUC does not improve for K consecutive epochs (configurable K, default: 10)
4. THE MIL_System SHALL save checkpoint files containing: model weights, optimizer state, epoch number, and best validation AUC
5. THE MIL_System SHALL support resuming training from any saved checkpoint
6. THE MIL_System SHALL automatically clean up old checkpoints, keeping only the best and the N most recent checkpoints

### Requirement 13: Multi-Scale Feature Support

**User Story:** As a computational pathologist, I want to process features from multiple magnification levels, so that I can capture both fine-grained and contextual information.

#### Acceptance Criteria

1. THE nnMIL_Model SHALL accept multi-scale features as a list of tensors with dimensions [[batch, N, feat_dim_scale1], [batch, N, feat_dim_scale2], ...]
2. THE nnMIL_Model SHALL support early fusion (concatenate features before processing) and late fusion (separate processing then combine)
3. WHEN using early fusion, THE nnMIL_Model SHALL concatenate features from all scales along the feature dimension
4. WHEN using late fusion, THE nnMIL_Model SHALL process each scale independently then concatenate representations
5. THE nnMIL_Model SHALL support 1 to 4 magnification scales
6. THE MIL_System SHALL maintain the existing multi-scale API from TransMIL_Model for backward compatibility

### Requirement 14: Testing and Validation

**User Story:** As a software engineer, I want comprehensive tests for nnMIL, so that I can ensure correctness and prevent regressions.

#### Acceptance Criteria

1. THE MIL_System SHALL include unit tests for nnMIL_Model forward pass with single-scale and multi-scale inputs
2. THE MIL_System SHALL include unit tests for Fixed_Length_Bag sampling with padding and truncation
3. THE MIL_System SHALL include unit tests for Task_Aware_Sampler class balancing
4. THE MIL_System SHALL include unit tests for Sliding_Window_Inference with various stride values
5. THE MIL_System SHALL include integration tests that train nnMIL_Model for 5 epochs and verify convergence
6. THE MIL_System SHALL include property-based tests that verify: model output shapes, attention mask correctness, and gradient flow
7. THE MIL_System SHALL achieve at least 85% test coverage on all new nnMIL components

### Requirement 15: Documentation

**User Story:** As a developer, I want comprehensive documentation for nnMIL, so that I can understand the architecture and use it effectively.

#### Acceptance Criteria

1. THE MIL_System SHALL provide API documentation for nnMIL_Model with parameter descriptions and usage examples
2. THE MIL_System SHALL provide a migration guide from TransMIL_Model to nnMIL_Model with code examples
3. THE MIL_System SHALL provide a configuration reference documenting all nnMIL hyperparameters
4. THE MIL_System SHALL provide a training tutorial demonstrating: basic training, multi-scale training, and uncertainty quantification
5. THE MIL_System SHALL document performance benchmarks comparing nnMIL_Model to TransMIL_Model
6. THE MIL_System SHALL include docstrings for all public classes and methods following NumPy docstring format
