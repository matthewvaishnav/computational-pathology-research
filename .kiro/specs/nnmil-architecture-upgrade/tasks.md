# Implementation Plan: nnMIL Architecture Upgrade

## Overview

This implementation plan upgrades the Multiple Instance Learning (MIL) system from TransMIL (2021) to nnMIL (2024, Stanford/NIH). The upgrade focuses on training-centric innovations: large-batch optimization (batch=32), fixed-length bag sampling, task-aware batch samplers, sliding-window inference, and uncertainty quantification. The implementation maintains backward compatibility with TransMIL while introducing state-of-the-art training techniques that achieve 80.7% BACC on disease subtyping, 77.1% AUC on biomarker detection, and 0.640 C-Index on prognosis tasks.

**Key Implementation Strategy:**
- Build new nnMIL components alongside existing TransMIL (no breaking changes)
- Implement core model architecture first, then training infrastructure, then inference
- Add property-based tests for correctness properties defined in design
- Validate performance against TransMIL baseline (≥93.94% AUC on PatchCamelyon)

## Tasks

- [x] 1. Create nnMIL model architecture and core components
  - [x] 1.1 Implement nnMIL model class with gated attention mechanism
    - Create `src/models/nnmil.py` with nnMIL class
    - Implement gated attention: α_i = softmax(w^T(tanh(Vx')⊙σ(Ux')))
    - Implement feature projection layer (optional, only if feature_dim != hidden_dim)
    - Implement classifier head: Linear(D, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, num_classes)
    - Support configurable hidden_dim (default: 256), dropout (default: 0.25), num_classes
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [x] 1.2 Write property tests for nnMIL model architecture
    - **Property 1: Input/Output Shape Invariance**
    - **Validates: Requirements 1.2, 1.3**
    - **Property 2: Configuration Validation**
    - **Validates: Requirements 1.5**
    - **Property 3: Parameter Efficiency**
    - **Validates: Requirements 1.6**
  
  - [x] 1.3 Implement multi-scale support for nnMIL
    - Add multi_scale parameter to nnMIL class
    - Implement early fusion: concatenate features before attention
    - Implement late fusion: separate attention per scale, concatenate representations
    - Support 1-4 magnification scales
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  
  - [x] 1.4 Write property tests for multi-scale input handling
    - **Property 39: Multi-Scale Input Handling**
    - **Validates: Requirements 13.1**
    - **Property 40: Early Fusion Concatenation**
    - **Validates: Requirements 13.3**

- [x] 2. Checkpoint - Ensure nnMIL model tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement fixed-length bag sampling
  - [x] 3.1 Create FixedLengthBagSampler class
    - Create `src/data/bag_samplers.py` with FixedLengthBagSampler
    - Implement padding: zero vectors if N < M
    - Implement random sampling without replacement if N > M (training mode)
    - Implement sliding window sampling if N > M (inference mode)
    - Support configurable bag_length (100-10000), mode ('train'/'inference'), stride
    - Create attention masks marking padded positions as False
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_
  
  - [x] 3.2 Write property tests for fixed-length bag sampling
    - **Property 7: Fixed-Length Bag Invariant**
    - **Validates: Requirements 3.1, 3.7**
    - **Property 8: Padding Correctness**
    - **Validates: Requirements 3.2**
    - **Property 9: Sampling Without Replacement**
    - **Validates: Requirements 3.3**
    - **Property 10: Sliding Window Activation**
    - **Validates: Requirements 3.4, 5.1**
    - **Property 11: Bag Length Configuration Range**
    - **Validates: Requirements 3.5**
    - **Property 12: Attention Mask Correctness**
    - **Validates: Requirements 3.6**

- [ ] 4. Implement task-aware batch samplers
  - [x] 4.1 Create BalancedBatchSampler for classification
    - Add BalancedBatchSampler class to `src/data/batch_samplers.py`
    - Ensure approximately equal class representation per batch
    - Oversample minority classes to achieve balance
    - Support stratified sampling by slide-level labels
    - Support configurable batch_size (default: 32), shuffle (default: True)
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 4.6_
  
  - [x] 4.2 Create RegressionBatchSampler for regression tasks
    - Add RegressionBatchSampler class to `src/data/batch_samplers.py`
    - Divide target range into bins (default: 10 bins)
    - Sample proportionally from each bin for uniform coverage
    - Support configurable batch_size, num_bins
    - _Requirements: 4.1_
  
  - [x] 4.3 Create SurvivalBatchSampler for survival analysis
    - Add SurvivalBatchSampler class to `src/data/batch_samplers.py`
    - Maintain balanced event rates across batches
    - Balance temporal distributions (early vs. late events)
    - Support configurable batch_size
    - _Requirements: 4.1_
  
  - [x] 4.4 Write property tests for task-aware batch samplers
    - **Property 13: Balanced Batch Composition**
    - **Validates: Requirements 4.1**
    - **Property 14: Minority Class Oversampling**
    - **Validates: Requirements 4.2**
    - **Property 15: Minimum Class Representation**
    - **Validates: Requirements 4.5**

- [x] 5. Checkpoint - Ensure sampling tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement sliding-window inference and uncertainty quantification
  - [x] 6.1 Create SlidingWindowInference class
    - Create `src/inference/sliding_window.py` with SlidingWindowInference
    - Divide D-dimensional space into overlapping H-dimensional chunks
    - Default stride = H/4 (75% overlap)
    - Process each window through model independently
    - Aggregate predictions via mean pooling
    - Compute variance across windows for epistemic uncertainty
    - Return dict with: logits, epistemic_uncertainty, aleatoric_uncertainty, all_predictions
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  
  - [x] 6.2 Create UncertaintyEstimator class
    - Create `src/inference/uncertainty.py` with UncertaintyEstimator
    - Implement epistemic uncertainty: Var(ŷ_k) across K predictions
    - Implement aleatoric uncertainty: mean entropy for classification
    - Implement combined uncertainty: sqrt(epistemic² + aleatoric²)
    - Normalize uncertainty values to [0, 1] range
    - Support classification, regression, and survival tasks
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [ ] 6.3 Write property tests for sliding-window inference
    - **Property 16: Window Overlap Correctness**
    - **Validates: Requirements 5.2**
    - **Property 17: Mean Pooling Aggregation**
    - **Validates: Requirements 5.4**
    - **Property 18: Stride Configuration Range**
    - **Validates: Requirements 5.5**
    - **Property 19: Inference Output Shape**
    - **Validates: Requirements 5.7**
  
  - [ ] 6.4 Write property tests for uncertainty quantification
    - **Property 20: Uncertainty Output Shape**
    - **Validates: Requirements 6.3**
    - **Property 21: Dropout Activation for Uncertainty**
    - **Validates: Requirements 6.1, 6.4**
    - **Property 22: Uncertainty Normalization**
    - **Validates: Requirements 6.5**
    - **Property 23: Combined Uncertainty Formula**
    - **Validates: Requirements 6.6**

- [ ] 7. Implement rule-based configuration system
  - [x] 7.1 Create nnMILConfig class with dataset fingerprinting
    - Create `src/config/nnmil_config.py` with nnMILConfig class
    - Implement dataset fingerprint extraction: median patches, IQR, class prevalence
    - Derive bag_length = median_patches / 2
    - Set fixed hyperparameters: hidden_dim=256, dropout=0.25
    - Set task-specific defaults: lr=3e-4 (classification), lr=1e-4 (survival)
    - Implement from_dataset() classmethod for automatic configuration
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_
  
  - [x] 7.2 Create YAML configuration schema and validation
    - Define YAML schema for nnMIL configurations
    - Implement configuration validation with descriptive errors
    - Support configuration inheritance (task-specific overrides base config)
    - Create default configs: disease_subtyping.yaml, biomarker_detection.yaml, prognosis.yaml
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [ ] 7.3 Write property tests for configuration management
    - **Property 28: Configuration Loading**
    - **Validates: Requirements 9.1**
    - **Property 29: Configuration Validation**
    - **Validates: Requirements 9.3**
    - **Property 30: Configuration Inheritance**
    - **Validates: Requirements 9.4**
    - **Property 31: Configuration Logging**
    - **Validates: Requirements 9.5**
    - **Property 32: Configuration Persistence**
    - **Validates: Requirements 9.6**

- [x] 8. Checkpoint - Ensure configuration tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement nnMIL training infrastructure
  - [x] 9.1 Create nnMILTrainer class with large-batch optimization
    - Create `src/training/nnmil_trainer.py` with nnMILTrainer class
    - Support batch sizes 1-64 (default: 32)
    - Implement gradient accumulation for memory-constrained GPUs
    - Implement learning rate scaling: lr_scaled = lr_base * sqrt(batch_size)
    - Integrate task-aware batch samplers (balanced, regression, survival)
    - Log effective batch size: batch_size * accumulation_steps
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  
  - [x] 9.2 Implement training monitoring and logging
    - Log per-epoch metrics: train loss, val loss, val AUC
    - Log per-batch metrics: batch loss, learning rate, gradient norm, effective batch size
    - Compute class-wise metrics: precision, recall, F1 per class
    - Track GPU memory usage and training throughput (samples/second)
    - Save training curves as plots: loss, AUC, learning rate schedule
    - Integrate with TensorBoard for real-time visualization
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_
  
  - [x] 9.3 Implement checkpointing and early stopping
    - Save checkpoints every N epochs (configurable, default: 5)
    - Save best model based on validation AUC
    - Implement early stopping: halt if no improvement for K epochs (default: 10)
    - Checkpoint format: model weights, optimizer state, epoch, best AUC, config
    - Support resuming training from any checkpoint
    - Auto-cleanup: keep best + N most recent checkpoints
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_
  
  - [ ] 9.4 Write property tests for training infrastructure
    - **Property 4: Batch Size Flexibility**
    - **Validates: Requirements 2.1**
    - **Property 5: Learning Rate Scaling**
    - **Validates: Requirements 2.4**
    - **Property 6: Effective Batch Size Logging**
    - **Validates: Requirements 2.6**
    - **Property 35: Metrics Logging**
    - **Validates: Requirements 11.1, 11.2**
    - **Property 36: Checkpoint Saving**
    - **Validates: Requirements 12.1**
    - **Property 37: Best Model Tracking**
    - **Validates: Requirements 12.2**
    - **Property 38: Early Stopping**
    - **Validates: Requirements 12.3**

- [ ] 10. Implement foundation model compatibility layer
  - [x] 10.1 Create foundation model adapter with automatic dimension detection
    - Create `src/models/foundation_adapter.py` with FoundationModelAdapter
    - Support UNI (1024-dim), CONCH (512-dim), Phikon (768-dim), ResNet50 (2048-dim)
    - Automatically detect feature_dim from input tensor shape
    - Apply learned projection if feature_dim != hidden_dim
    - Support weight freezing and fine-tuning with configurable lr multiplier
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ] 10.2 Write property tests for foundation model compatibility
    - **Property 24: Foundation Model Compatibility**
    - **Validates: Requirements 7.1**
    - **Property 25: Automatic Dimension Detection**
    - **Validates: Requirements 7.2**
    - **Property 26: Adaptive Projection**
    - **Validates: Requirements 7.3**
    - **Property 27: Weight Freezing**
    - **Validates: Requirements 7.4**

- [x] 11. Checkpoint - Ensure training infrastructure tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement backward compatibility and migration tools
  - [x] 12.1 Create unified training interface for TransMIL and nnMIL
    - Update training scripts to support both model types via config
    - Ensure same input format and output format for both models
    - Maintain TransMIL as selectable architecture option
    - _Requirements: 10.1, 10.2, 10.6_
  
  - [x] 12.2 Create checkpoint migration script
    - Create `scripts/migrate_transmil_to_nnmil.py`
    - Load TransMIL checkpoint and extract compatible weights
    - Transfer feature projection and classifier head weights
    - Save as nnMIL checkpoint format with metadata
    - Provide migration guide in docstring
    - _Requirements: 10.3, 10.4, 10.5_
  
  - [ ] 12.3 Write property tests for backward compatibility
    - **Property 33: API Compatibility**
    - **Validates: Requirements 10.2, 10.6**
    - **Property 34: Weight Transfer**
    - **Validates: Requirements 10.4**

- [x] 13. Create data models and interfaces
  - [x] 13.1 Implement Bag, TrainingBatch, and InferenceOutput dataclasses
    - Create `src/data/data_models.py` with dataclasses
    - Bag: features [N, D], label, num_patches, slide_id, metadata
    - TrainingBatch: features [B, M, D], labels [B], masks [B, M], num_patches [B], slide_ids
    - InferenceOutput: logits, probabilities, attention_weights, uncertainties, slide_ids
    - _Requirements: 1.2, 1.3, 5.7, 6.3_

- [ ] 14. Write integration tests
  - [ ] 14.1 Write end-to-end training test
    - Train nnMIL for 5 epochs on synthetic data
    - Verify loss decreases (convergence)
    - Verify checkpointing (best model saved)
    - Verify early stopping (stops after patience epochs)
    - Verify metrics logging (all metrics logged)
    - _Requirements: 14.5_
  
  - [ ] 14.2 Write cross-model generalization tests
    - Test with UNI features (1024-dim)
    - Test with CONCH features (512-dim)
    - Test with Phikon features (768-dim)
    - Test with ResNet50 features (2048-dim)
    - Verify model accepts all feature dimensions
    - _Requirements: 7.1, 7.6_
  
  - [ ] 14.3 Write backward compatibility integration tests
    - Test unified training interface with TransMIL and nnMIL
    - Test checkpoint loading (TransMIL → nnMIL)
    - Test weight transfer (compatible layers)
    - Test migration script (checkpoint conversion)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [x] 15. Checkpoint - Ensure all integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Create documentation
  - [ ] 16.1 Write API documentation for nnMIL components
    - Document nnMIL class with parameter descriptions and usage examples
    - Document FixedLengthBagSampler, task-aware batch samplers
    - Document SlidingWindowInference and UncertaintyEstimator
    - Document nnMILConfig and configuration schema
    - Document nnMILTrainer with training examples
    - Follow NumPy docstring format
    - _Requirements: 15.1, 15.6_
  
  - [ ] 16.2 Write migration guide from TransMIL to nnMIL
    - Document model initialization changes
    - Document training loop changes (batch size, samplers)
    - Document inference changes (sliding window, uncertainty)
    - Document checkpoint migration process
    - Provide complete code examples for each step
    - _Requirements: 15.2_
  
  - [ ] 16.3 Write configuration reference and training tutorial
    - Document all nnMIL hyperparameters with descriptions
    - Document default configurations for each task type
    - Provide training tutorial: basic training, multi-scale, uncertainty
    - Include configuration examples for common scenarios
    - _Requirements: 15.3, 15.4_

- [ ] 17. Performance benchmarking (optional validation)
  - [ ] 17.1 Benchmark on PatchCamelyon dataset
    - Train nnMIL on PatchCamelyon with default config
    - Evaluate and verify AUC ≥ 93.94% (matching TransMIL baseline)
    - Compare training time vs. TransMIL (target: ≤120%)
    - Compare GPU memory vs. TransMIL (target: ≤120%)
    - _Requirements: 8.1, 8.5, 8.6_
  
  - [ ] 17.2 Document performance benchmarks
    - Create benchmark comparison table: nnMIL vs. TransMIL
    - Document training time, GPU memory, convergence speed
    - Document performance across foundation models (UNI, CONCH, Phikon, ResNet50)
    - _Requirements: 15.5_

- [ ] 18. Final checkpoint - Verify all requirements met
  - Ensure all tests pass, ask the user if questions arise.
  - Verify test coverage ≥85% on all nnMIL components
  - Verify all 40 correctness properties have corresponding tests
  - Verify documentation is complete

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from design document
- Integration tests validate end-to-end workflows
- Checkpoints ensure incremental validation at key milestones
- Implementation uses Python with PyTorch framework (as specified in design)
- All new code should maintain backward compatibility with existing TransMIL implementation
- Focus on training-centric innovations: large batches, task-aware sampling, sliding-window inference
