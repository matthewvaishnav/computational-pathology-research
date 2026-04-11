# Implementation Plan: Attention-Based MIL Models

## Overview

This implementation plan adds three state-of-the-art attention-based Multiple Instance Learning (MIL) architectures to the existing slide-level training infrastructure: Attention MIL (basic attention pooling), CLAM (clustering-constrained attention), and TransMIL (transformer-based MIL). The implementation integrates with the existing CAMELYONSlideDataset and training pipeline while maintaining backward compatibility with baseline pooling models. Each task builds incrementally, starting with core model architectures, then training integration, visualization, model comparison, and comprehensive testing.

## Tasks

- [x] 1. Implement core attention model infrastructure
  - [x] 1.1 Create AttentionMILBase abstract base class in src/models/attention_mil.py
    - Define abstract methods: compute_attention, aggregate_features, forward
    - Implement __init__ with common parameters (feature_dim, hidden_dim, num_classes, dropout)
    - Add comprehensive docstrings with parameter descriptions and return types
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1_

  - [ ]* 1.2 Write unit tests for AttentionMILBase interface
    - Test that abstract methods raise NotImplementedError when not overridden
    - Test initialization with various parameter combinations
    - _Requirements: 10.1, 10.7_

- [x] 2. Implement AttentionMIL architecture
  - [x] 2.1 Implement AttentionMIL class with gated attention in src/models/attention_mil.py
    - Implement feature projection layer (Linear -> ReLU -> Dropout)
    - Implement gated attention mechanism (attention_V, attention_U, attention_w)
    - Implement simple attention mechanism as fallback (attention_net)
    - Implement classifier head (Linear -> ReLU -> Dropout -> Linear)
    - Support both instance-level and bag-level attention modes
    - _Requirements: 1.1, 1.2, 1.3, 1.6_

  - [x] 2.2 Implement compute_attention method for AttentionMIL
    - Project features through feature_proj
    - Compute attention scores using gated or simple attention
    - Apply mask to set padded patches to -inf before softmax
    - Apply softmax to normalize attention weights to sum to 1
    - _Requirements: 1.1, 1.5_

  - [x] 2.3 Implement aggregate_features method for AttentionMIL
    - Project features through feature_proj
    - Perform weighted sum using batch matrix multiplication
    - Return fixed-size slide representation
    - _Requirements: 1.2, 1.7_

  - [x] 2.4 Implement forward method for AttentionMIL
    - Create mask from num_patches parameter
    - Call compute_attention with mask
    - Call aggregate_features with attention weights
    - Pass through classifier to get logits
    - Return logits and optionally attention weights
    - _Requirements: 1.4, 1.5_

  - [ ]* 2.5 Write unit tests for AttentionMIL
    - Test forward pass with synthetic data (batch_size=4, num_patches=100)
    - Test attention weights sum to 1 for each slide
    - Test attention masking for padded patches
    - Test gradient flow through attention mechanism
    - Test both gated and simple attention modes
    - Test both instance and bag attention modes
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 3. Checkpoint - Ensure AttentionMIL works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement CLAM architecture
  - [x] 4.1 Implement CLAM class with clustering in src/models/attention_mil.py
    - Implement feature projection layer
    - Implement instance-level classifier for clustering (num_clusters output)
    - Implement attention branches (single or multi-branch)
    - Implement bag-level classifier
    - Support both single-branch and multi-branch attention modes
    - _Requirements: 2.1, 2.2, 2.3, 2.7_

  - [x] 4.2 Implement compute_instance_predictions method for CLAM
    - Project features through feature_proj
    - Pass through instance_classifier to get cluster logits
    - Return instance-level predictions for all patches
    - _Requirements: 2.1, 2.6_

  - [x] 4.3 Implement compute_attention method for CLAM
    - Project features through feature_proj
    - Compute attention scores using specified branch (pos/neg)
    - Apply mask to set padded patches to -inf before softmax
    - Apply softmax to normalize attention weights
    - _Requirements: 2.2, 2.6_

  - [x] 4.4 Implement forward method for CLAM
    - Create mask from num_patches parameter
    - Compute instance predictions
    - Compute attention for both branches (if multi_branch=True)
    - Aggregate features for each branch
    - Concatenate branch features (if multi_branch=True)
    - Pass through bag_classifier to get logits
    - Return logits, attention weights, and optionally instance predictions
    - _Requirements: 2.4, 2.5, 2.7_

  - [ ]* 4.5 Write unit tests for CLAM
    - Test forward pass with multi-branch attention
    - Test forward pass with single-branch attention
    - Test instance predictions shape and values
    - Test attention weights sum to 1 for each slide
    - Test gradient flow through clustering and attention
    - _Requirements: 10.1, 10.2, 10.3_

- [x] 5. Implement TransMIL architecture
  - [x] 5.1 Implement TransMIL class with transformer layers in src/models/attention_mil.py
    - Implement feature projection layer
    - Implement learnable positional encoding parameter
    - Implement learnable CLS token parameter
    - Create TransformerEncoder with specified num_layers and num_heads
    - Implement layer normalization
    - Implement classifier head
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.2 Implement compute_attention method for TransMIL
    - Return uniform attention weights as placeholder (transformer attention is internal)
    - Apply mask if provided
    - Normalize to sum to 1
    - _Requirements: 3.6_

  - [x] 5.3 Implement forward method for TransMIL
    - Project features through feature_proj
    - Add positional encoding if enabled
    - Prepend CLS token to sequence
    - Create attention mask for transformer (True for padding positions)
    - Apply transformer encoder
    - Extract CLS token representation
    - Apply layer normalization
    - Pass through classifier to get logits
    - Return logits and optionally attention weights
    - _Requirements: 3.4, 3.5, 3.6, 3.7_

  - [ ]* 5.4 Write unit tests for TransMIL
    - Test forward pass with synthetic data
    - Test CLS token aggregation
    - Test positional encoding application
    - Test transformer masking for padded patches
    - Test gradient flow through transformer layers
    - Test with different num_layers and num_heads configurations
    - _Requirements: 10.1, 10.2, 10.3_

- [ ] 6. Checkpoint - Ensure all three models work
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Integrate attention models with training pipeline
  - [x] 7.1 Create model factory function in src/models/attention_mil.py
    - Implement create_attention_model function that reads config
    - Support model_type: attention_mil, clam, transmil, mean, max
    - Extract model-specific configs (attention_mil, clam, transmil)
    - Instantiate appropriate model class with config parameters
    - Add validation for invalid model_type
    - Log model creation with key parameters
    - _Requirements: 4.1, 4.2, 7.1, 7.2, 7.3_

  - [x] 7.2 Create configuration validation function in experiments/train_camelyon.py
    - Implement validate_model_config function
    - Validate model_type is one of: attention_mil, clam, transmil, mean, max
    - Validate attention_mode is 'instance' or 'bag' for AttentionMIL
    - Validate num_clusters >= 2 for CLAM
    - Validate hidden_dim is divisible by num_heads for TransMIL
    - Raise ValueError with clear messages for invalid configs
    - _Requirements: 7.1, 7.6_

  - [x] 7.3 Update train_epoch function in experiments/train_camelyon.py
    - Check if model supports return_attention parameter using inspect
    - Call model with return_attention=save_attention flag
    - Handle both attention models and baseline models
    - Save attention weights to HDF5 if save_attention=True
    - Log attention statistics (mean, std, max, min)
    - _Requirements: 4.3, 4.4, 4.6, 8.1, 8.2_

  - [x] 7.4 Update validate function in experiments/train_camelyon.py
    - Check if model supports return_attention parameter
    - Call model with return_attention=True during validation
    - Save attention weights to HDF5 for all validation slides
    - Include slide_id, patient_id, label, prediction in HDF5 metadata
    - Only save valid patches (use num_patches to slice)
    - _Requirements: 4.4, 8.1, 8.2, 8.3, 8.6_

  - [ ]* 7.5 Write integration tests for training pipeline
    - Create synthetic CAMELYON data (10 slides, 50 patches each)
    - Test training AttentionMIL for 2 epochs
    - Test training CLAM for 2 epochs
    - Test training TransMIL for 2 epochs
    - Verify checkpoints are saved with correct structure
    - Verify attention weights are saved to HDF5
    - Verify backward compatibility with baseline models
    - _Requirements: 4.5, 10.6, 10.7_

- [x] 8. Implement attention weight extraction and storage
  - [x] 8.1 Create save_attention_weights function in src/utils/attention_utils.py
    - Accept attention_weights tensor, coordinates tensor, slide_id, output_dir
    - Validate attention_weights and coordinates have same length
    - Create output directory if it doesn't exist
    - Save to HDF5 with datasets: attention_weights, coordinates
    - Add slide_id as HDF5 attribute
    - Log save location
    - _Requirements: 8.1, 8.2, 8.5, 8.6_

  - [x] 8.2 Create load_attention_weights function in src/utils/attention_utils.py
    - Accept slide_id and attention_dir
    - Check if HDF5 file exists, return None if not found
    - Load attention_weights and coordinates from HDF5
    - Return tuple of (attention_weights, coordinates)
    - Handle exceptions gracefully with error logging
    - _Requirements: 8.3, 8.7_

  - [ ]* 8.3 Write unit tests for attention weight storage
    - Test saving attention weights to HDF5
    - Test loading attention weights from HDF5
    - Test round-trip preservation of values
    - Test handling of missing files
    - Test dimension mismatch error handling
    - _Requirements: 8.7, 10.6_

- [x] 9. Implement attention heatmap visualization
  - [x] 9.1 Create AttentionHeatmapGenerator class in src/visualization/attention_heatmap.py
    - Implement __init__ with attention_dir, output_dir, colormap, thumbnail_size
    - Create output directory in __init__
    - Store colormap from matplotlib
    - _Requirements: 5.1, 5.2, 5.4_

  - [x] 9.2 Implement load_attention_weights method in AttentionHeatmapGenerator
    - Load attention weights and coordinates from HDF5
    - Return None if file not found (with warning log)
    - Handle exceptions gracefully with error logging
    - _Requirements: 5.7_

  - [x] 9.3 Implement create_heatmap_array method in AttentionHeatmapGenerator
    - Normalize attention weights to [0, 1] range
    - Create canvas array with specified size
    - Map patch coordinates to canvas space
    - Place attention values at patch locations
    - Average overlapping regions
    - _Requirements: 5.1, 5.6, 5.7_

  - [x] 9.4 Implement generate_heatmap method in AttentionHeatmapGenerator
    - Load attention weights using load_attention_weights
    - Create heatmap array using create_heatmap_array
    - Create matplotlib figure
    - Load and display thumbnail if provided
    - Overlay heatmap with specified alpha transparency
    - Add colorbar with label
    - Set title and turn off axis
    - Save figure to output_dir
    - Return path to generated heatmap
    - _Requirements: 5.2, 5.3, 5.4, 5.6_

  - [x] 9.5 Implement generate_batch method in AttentionHeatmapGenerator
    - Iterate over list of slide_ids
    - Find thumbnail path if thumbnail_dir provided
    - Call generate_heatmap for each slide
    - Collect paths to generated heatmaps
    - Log number of heatmaps generated
    - _Requirements: 5.5_

  - [ ]* 9.6 Write unit tests for heatmap generation
    - Test heatmap generation with synthetic attention data
    - Test batch generation for multiple slides
    - Test handling of missing attention weights
    - Test colormap application
    - Verify output files are created
    - _Requirements: 10.5_

- [ ] 10. Checkpoint - Ensure visualization works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement model comparison infrastructure
  - [x] 11.1 Create train_single_model function in experiments/compare_attention_models.py
    - Accept model_type, config, output_dir parameters
    - Update config with model_type and checkpoint_dir
    - Call training function from train_camelyon.py
    - Evaluate model on test set
    - Return dictionary with test metrics (accuracy, auc, f1, inference_time)
    - _Requirements: 6.1, 6.2, 6.7_

  - [x] 11.2 Create compare_models function in experiments/compare_attention_models.py
    - Accept list of model_types, base config, output_dir
    - Iterate over model_types and call train_single_model
    - Collect results into list of dictionaries
    - Create pandas DataFrame from results
    - Save DataFrame to CSV file
    - Return DataFrame
    - _Requirements: 6.1, 6.2, 6.6_

  - [x] 11.3 Create plot_roc_curves function in experiments/compare_attention_models.py
    - Accept model_types, predictions dict, output_dir
    - Create matplotlib figure
    - For each model, compute ROC curve and AUC
    - Plot ROC curve with label showing AUC
    - Add random baseline (diagonal line)
    - Add labels, title, legend, grid
    - Save figure to output_dir
    - _Requirements: 6.5, 6.6_

  - [x] 11.4 Create statistical_significance_test function in experiments/compare_attention_models.py
    - Accept results DataFrame and baseline model name
    - Extract baseline AUC
    - For each model, compute p-value vs baseline (placeholder using simple comparison)
    - Return DataFrame with model_type and p_value columns
    - _Requirements: 6.3_

  - [x] 11.5 Implement main function in experiments/compare_attention_models.py
    - Parse command-line arguments (config, models list)
    - Load and validate config
    - Set random seed
    - Create output directory
    - Call compare_models function
    - Print comparison results table
    - Call statistical_significance_test and print results
    - Log completion message
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 11.6 Write integration tests for model comparison
    - Create synthetic CAMELYON data
    - Run comparison for 3 models (mean, attention_mil, clam)
    - Verify CSV file is created with correct columns
    - Verify ROC curve plot is generated
    - Verify all models complete training
    - _Requirements: 6.1, 6.2, 6.4_

- [x] 12. Create YAML configuration files
  - [x] 12.1 Create experiments/configs/attention_mil.yaml
    - Set model.wsi.model_type to "attention_mil"
    - Configure attention_mil section (gated=true, attention_mode="instance")
    - Set training parameters (batch_size=8, num_epochs=50, learning_rate=1e-4)
    - Configure attention weight saving (save_attention_weights=true, frequency=5)
    - Configure visualization (generate_heatmaps=true, colormap="jet")
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [x] 12.2 Create experiments/configs/clam.yaml
    - Set model.wsi.model_type to "clam"
    - Configure clam section (num_clusters=10, multi_branch=true, instance_loss_weight=0.3)
    - Set training parameters matching attention_mil.yaml
    - Configure attention weight saving and visualization
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [x] 12.3 Create experiments/configs/transmil.yaml
    - Set model.wsi.model_type to "transmil"
    - Configure transmil section (num_layers=2, num_heads=8, use_pos_encoding=true)
    - Set training parameters matching attention_mil.yaml
    - Configure attention weight saving and visualization
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [x] 12.4 Create experiments/configs/comparison.yaml
    - Set base configuration for model comparison
    - Configure data paths and training parameters
    - Set models list to compare (mean, max, attention_mil, clam, transmil)
    - Configure output directories for comparison results
    - _Requirements: 7.1, 7.7_

- [ ] 13. Implement multi-scale feature support
  - [ ] 13.1 Extend AttentionMIL to support multi-scale features
    - Add multi_scale parameter to __init__
    - Add scale-specific feature projection layers
    - Implement early fusion (concatenate before attention)
    - Implement late fusion (separate attention per scale, then combine)
    - Update forward method to handle multi-scale input
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.7_

  - [ ] 13.2 Extend CLAM to support multi-scale features
    - Add multi_scale parameter and fusion strategy
    - Implement scale-specific attention branches
    - Handle missing features at certain scales gracefully
    - Update forward method for multi-scale input
    - _Requirements: 9.1, 9.3, 9.4, 9.5, 9.7_

  - [ ] 13.3 Extend TransMIL to support multi-scale features
    - Add multi_scale parameter and fusion strategy
    - Implement scale-specific positional encodings
    - Align spatial coordinates across scales
    - Update forward method for multi-scale input
    - _Requirements: 9.1, 9.3, 9.6, 9.7_

  - [ ]* 13.4 Write unit tests for multi-scale support
    - Test early fusion with 2 scales
    - Test late fusion with 3 scales
    - Test handling of missing features at certain scales
    - Test coordinate alignment across scales
    - _Requirements: 9.5, 9.6_

- [ ] 14. Checkpoint - Ensure multi-scale support works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Create comprehensive documentation
  - [x] 15.1 Add attention models section to README.md
    - Document overview of three architectures
    - Provide quick start examples for training each model
    - Document configuration options for each model type
    - Explain when to use each architecture
    - Include performance comparison table
    - Document attention visualization workflow
    - Add troubleshooting section for common issues
    - Document limitations and future enhancements
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [x] 15.2 Add API documentation to src/models/attention_mil.py
    - Add comprehensive module docstring
    - Document AttentionMILBase interface with examples
    - Document AttentionMIL class with usage examples
    - Document CLAM class with usage examples
    - Document TransMIL class with usage examples
    - Document create_attention_model factory function
    - Include parameter descriptions and return types for all methods
    - _Requirements: 11.1, 11.2, 11.4_

  - [x] 15.3 Create example notebook for attention models
    - Create notebooks/attention_mil_tutorial.ipynb
    - Show how to load pre-trained attention model
    - Demonstrate inference on test slides
    - Show how to extract and visualize attention weights
    - Compare attention patterns across different models
    - Interpret attention heatmaps for tumor detection
    - _Requirements: 11.2, 11.3, 11.5_

  - [x] 15.4 Add docstrings to visualization module
    - Document AttentionHeatmapGenerator class
    - Document all methods with parameter descriptions
    - Add usage examples in docstrings
    - Document colormap options and their effects
    - _Requirements: 11.4, 11.5_

  - [x] 15.5 Add docstrings to comparison script
    - Document compare_attention_models.py module
    - Document all functions with parameter descriptions
    - Add usage examples in module docstring
    - Document command-line arguments
    - _Requirements: 11.2, 11.6_

- [x] 16. Create synthetic data generator for testing
  - [x] 16.1 Create create_synthetic_camelyon_data function in tests/utils.py
    - Accept data_dir, num_slides, num_patches, feature_dim parameters
    - Generate slide metadata with alternating labels
    - Create slide_index.json with train/val/test splits
    - Generate random feature vectors for each slide
    - Generate random patch coordinates
    - Save features and coordinates to HDF5 files
    - _Requirements: 10.7_

  - [ ]* 16.2 Write tests for synthetic data generator
    - Test slide index creation
    - Test feature file creation
    - Test data loading with CAMELYONSlideDataset
    - Verify split distribution
    - _Requirements: 10.7_

- [ ] 17. Final checkpoint - Verify complete implementation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Implementation uses Python with PyTorch for deep learning
- No property-based tests are included (this is infrastructure/neural network code)
- Unit tests and integration tests validate correctness through examples
- Focus is on integrating with existing slide-level training infrastructure
- All models work with pre-extracted HDF5 features, not raw WSI files
- Attention mechanisms provide both performance improvements and interpretability
- Multi-scale support enables capturing both fine-grained and contextual information
