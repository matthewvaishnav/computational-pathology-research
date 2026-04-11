# Requirements Document

## Introduction

This feature implements state-of-the-art attention-based Multiple Instance Learning (MIL) architectures for whole-slide image classification. The repository currently has slide-level training infrastructure with simple mean/max pooling aggregation. This feature adds attention mechanisms that learn to weight patch importance, providing both improved performance and interpretability through attention heatmaps.

The implementation includes three attention-based architectures: Attention MIL (basic attention pooling), CLAM (clustering-constrained attention), and TransMIL (transformer-based MIL). These models integrate with the existing CAMELYONSlideDataset and training pipeline while maintaining backward compatibility with existing checkpoints.

## Glossary

- **MIL_Model**: Multiple Instance Learning model that processes bags of patch features
- **Attention_Mechanism**: Neural network module that computes importance weights for patches
- **Attention_Weights**: Learned weights indicating patch importance for classification
- **Attention_Heatmap**: Spatial visualization of attention weights overlaid on slide
- **Patch_Features**: Pre-extracted feature vectors for image patches from HDF5 files
- **Slide_Bag**: Collection of all patch features for a single whole-slide image
- **Gated_Attention**: Attention mechanism with element-wise gating for feature selection
- **Instance_Level_Attention**: Attention computed independently for each patch
- **Bag_Level_Attention**: Attention computed over aggregated bag representation
- **Training_Pipeline**: The existing experiments/train_camelyon.py infrastructure
- **Baseline_Models**: Current mean/max pooling aggregation methods
- **CLAM**: Clustering-Constrained Attention Multiple Instance Learning architecture
- **TransMIL**: Transformer-based Multiple Instance Learning architecture
- **Attention_MIL**: Basic attention-weighted pooling MIL architecture

## Requirements

### Requirement 1: Attention MIL Architecture

**User Story:** As a researcher, I want a basic attention-based MIL model, so that I can learn which patches are important for slide classification.

#### Acceptance Criteria

1. THE Attention_MIL SHALL compute attention weights for each patch in a Slide_Bag
2. THE Attention_MIL SHALL aggregate Patch_Features using learned Attention_Weights
3. THE Attention_MIL SHALL support both instance-level and bag-level attention modes
4. THE Attention_MIL SHALL output attention weights along with classification logits
5. WHEN processing variable-length bags, THE Attention_MIL SHALL mask padded patches
6. THE Attention_MIL SHALL support gated attention with element-wise feature gating
7. FOR ALL valid Slide_Bags, computing attention weights then aggregating features SHALL produce a fixed-size slide representation (invariant property)

### Requirement 2: CLAM Architecture

**User Story:** As a researcher, I want the CLAM architecture with clustering constraints, so that I can improve attention quality through instance clustering.

#### Acceptance Criteria

1. THE CLAM SHALL implement instance-level clustering of Patch_Features
2. THE CLAM SHALL compute attention weights constrained by cluster assignments
3. THE CLAM SHALL support both single-branch and multi-branch attention modes
4. THE CLAM SHALL output instance-level predictions and bag-level predictions
5. THE CLAM SHALL compute instance clustering loss for training
6. WHEN processing a Slide_Bag, THE CLAM SHALL assign each patch to a cluster
7. THE CLAM SHALL maintain separate attention branches for positive and negative classes in multi-branch mode

### Requirement 3: TransMIL Architecture

**User Story:** As a researcher, I want a transformer-based MIL model, so that I can capture long-range dependencies between patches.

#### Acceptance Criteria

1. THE TransMIL SHALL use multi-head self-attention to process Patch_Features
2. THE TransMIL SHALL support configurable number of transformer layers
3. THE TransMIL SHALL include positional encoding for spatial patch relationships
4. THE TransMIL SHALL aggregate features using the [CLS] token representation
5. THE TransMIL SHALL support layer normalization and dropout for regularization
6. WHEN processing patches, THE TransMIL SHALL compute self-attention across all patches
7. FOR ALL Slide_Bags, applying TransMIL then permuting patch order then applying TransMIL SHALL produce different results (order-dependent property)

### Requirement 4: Model Integration

**User Story:** As a researcher, I want attention models integrated with the existing training pipeline, so that I can train them using the same infrastructure as baseline models.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL support attention-based models via configuration
2. THE Training_Pipeline SHALL load attention model architectures from model registry
3. THE Training_Pipeline SHALL pass Slide_Bags to attention models with proper masking
4. THE Training_Pipeline SHALL save attention weights during validation
5. THE Training_Pipeline SHALL maintain backward compatibility with Baseline_Models
6. WHEN training an attention model, THE Training_Pipeline SHALL log attention statistics
7. THE Training_Pipeline SHALL support mixed training with both attention and baseline models

### Requirement 5: Attention Visualization

**User Story:** As a researcher, I want to visualize attention weights as heatmaps, so that I can interpret which regions the model focuses on.

#### Acceptance Criteria

1. THE Attention_Heatmap generator SHALL map attention weights to spatial coordinates
2. THE Attention_Heatmap generator SHALL support multiple color maps for visualization
3. THE Attention_Heatmap generator SHALL overlay attention on slide thumbnails
4. THE Attention_Heatmap generator SHALL save heatmaps as image files
5. THE Attention_Heatmap generator SHALL support batch generation for multiple slides
6. WHEN generating a heatmap, THE generator SHALL normalize attention weights to [0, 1]
7. THE Attention_Heatmap generator SHALL handle variable numbers of patches per slide

### Requirement 6: Performance Comparison

**User Story:** As a researcher, I want to compare attention models with baseline pooling methods, so that I can quantify performance improvements.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL compute metrics for both attention and baseline models
2. THE Training_Pipeline SHALL generate comparison tables with accuracy, AUC, F1 scores
3. THE Training_Pipeline SHALL compute statistical significance tests between models
4. THE Training_Pipeline SHALL save comparison results to JSON files
5. THE Training_Pipeline SHALL generate ROC curves for all models on the same plot
6. WHEN comparing models, THE Training_Pipeline SHALL use the same test set
7. THE Training_Pipeline SHALL report attention model inference time compared to baselines

### Requirement 7: Configuration Management

**User Story:** As a researcher, I want YAML configuration for attention models, so that I can easily switch between architectures and hyperparameters.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL read attention model type from configuration
2. THE Training_Pipeline SHALL support configuration for attention dimensions and layers
3. THE Training_Pipeline SHALL validate attention model configurations before training
4. THE Training_Pipeline SHALL support separate configs for Attention_MIL, CLAM, and TransMIL
5. THE Training_Pipeline SHALL allow configuration of gated attention and dropout rates
6. WHEN loading a config, THE Training_Pipeline SHALL provide clear error messages for invalid settings
7. THE Training_Pipeline SHALL support configuration inheritance from base model configs

### Requirement 8: Attention Weight Extraction

**User Story:** As a researcher, I want to extract and save attention weights for analysis, so that I can study model behavior and create visualizations offline.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL save attention weights to HDF5 files during evaluation
2. THE Training_Pipeline SHALL store attention weights with corresponding patch coordinates
3. THE Training_Pipeline SHALL include slide metadata with saved attention weights
4. THE Training_Pipeline SHALL support extraction for specific slides or entire test set
5. THE Training_Pipeline SHALL compress attention weight files for storage efficiency
6. WHEN extracting weights, THE Training_Pipeline SHALL preserve slide_id and patient_id
7. FOR ALL saved attention files, loading then accessing weights SHALL preserve original values (round-trip property)

### Requirement 9: Multi-Scale Feature Support

**User Story:** As a researcher, I want to use features from multiple magnifications, so that I can capture both fine-grained and contextual information.

#### Acceptance Criteria

1. THE MIL_Model SHALL accept features from multiple magnification levels
2. THE MIL_Model SHALL concatenate or fuse multi-scale features before attention
3. THE MIL_Model SHALL support independent attention computation per scale
4. THE MIL_Model SHALL support late fusion of scale-specific attention outputs
5. THE MIL_Model SHALL handle missing features at certain scales gracefully
6. WHEN processing multi-scale features, THE MIL_Model SHALL align spatial coordinates
7. THE MIL_Model SHALL support configuration of fusion strategy (early/late/hierarchical)

### Requirement 10: Testing and Validation

**User Story:** As a developer, I want comprehensive tests for attention models, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE test suite SHALL verify attention weights sum to 1 for each slide
2. THE test suite SHALL verify attention models handle variable-length bags correctly
3. THE test suite SHALL verify gradient flow through attention mechanisms
4. THE test suite SHALL verify attention masking for padded patches
5. THE test suite SHALL verify attention heatmap generation produces valid images
6. THE test suite SHALL verify model checkpoint saving and loading preserves attention weights
7. THE test suite SHALL run successfully with synthetic slide data

### Requirement 11: Documentation

**User Story:** As a user, I want clear documentation for attention models, so that I understand how to use them and interpret results.

#### Acceptance Criteria

1. THE documentation SHALL describe each attention architecture and its use cases
2. THE documentation SHALL provide example commands for training attention models
3. THE documentation SHALL explain attention weight interpretation
4. THE documentation SHALL document configuration options for each model type
5. THE documentation SHALL include example attention heatmap visualizations
6. THE documentation SHALL compare attention models with baseline pooling methods
7. THE documentation SHALL provide troubleshooting guidance for common issues

