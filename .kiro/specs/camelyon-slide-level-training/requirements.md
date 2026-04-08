# Requirements Document

## Introduction

This feature addresses the inconsistency between CAMELYON training and evaluation paths. Currently, `train_camelyon.py` claims to be slide-level but actually trains on individual patches wrapped as length-1 sequences, while `evaluate_camelyon.py` evaluates on full slides by aggregating all patches from each HDF5 feature file. This creates a train/eval mismatch that undermines model validity.

The solution introduces a true slide-level training dataset that loads complete slide bags of patch features from HDF5 caches, ensuring training and evaluation operate at the same granularity. The implementation uses mean/max pooling baselines and does not require raw WSI/OpenSlide work, keeping the scope practical and focused on the feature-cache baseline.

## Glossary

- **Training_Script**: The `experiments/train_camelyon.py` module responsible for model training
- **Evaluation_Script**: The `experiments/evaluate_camelyon.py` module responsible for model evaluation
- **Slide_Dataset**: A PyTorch Dataset class that returns complete slides (all patches) per sample
- **Patch_Dataset**: The existing `CAMELYONPatchDataset` class that returns individual patches
- **HDF5_Cache**: Pre-extracted patch feature files stored in HDF5 format at `data/camelyon/features/`
- **Slide_Bag**: A collection of all patch features belonging to a single whole-slide image
- **Aggregation_Method**: The pooling strategy (mean or max) used to combine patch features into slide-level representations
- **Collate_Function**: PyTorch DataLoader function that batches variable-length slide bags
- **Test_Suite**: The pytest test collection for CAMELYON functionality

## Requirements

### Requirement 1: Slide-Level Dataset Implementation

**User Story:** As a researcher, I want a dataset class that returns complete slides with all their patches, so that training operates at the same granularity as evaluation.

#### Acceptance Criteria

1. THE Slide_Dataset SHALL load all patch features for a slide from the corresponding HDF5_Cache file
2. WHEN a sample is requested, THE Slide_Dataset SHALL return a dictionary containing the slide_id, all patch features, coordinates, and label
3. THE Slide_Dataset SHALL support train, val, and test splits via the slide index
4. THE Slide_Dataset SHALL handle variable numbers of patches per slide without truncation
5. FOR ALL slides in the dataset, loading then accessing all patches SHALL preserve the original feature values (round-trip property)

### Requirement 2: Training Script Slide-Level Path

**User Story:** As a researcher, I want the training script to use true slide-level batching, so that each training sample represents a complete slide rather than a single patch.

#### Acceptance Criteria

1. THE Training_Script SHALL use Slide_Dataset instead of Patch_Dataset for data loading
2. THE Training_Script SHALL implement a Collate_Function that batches variable-length slide bags
3. WHEN processing a batch, THE Training_Script SHALL pass complete slide bags to the model
4. THE Training_Script SHALL apply Aggregation_Method (mean or max pooling) to combine patch features
5. THE Training_Script SHALL maintain backward compatibility with existing checkpoint format

### Requirement 3: Evaluation Consistency

**User Story:** As a researcher, I want evaluation to remain compatible with slide-level trained models, so that I can validate model performance correctly.

#### Acceptance Criteria

1. THE Evaluation_Script SHALL load and evaluate slide-level checkpoints without modification
2. WHEN evaluating a slide, THE Evaluation_Script SHALL use the same Aggregation_Method as training
3. THE Evaluation_Script SHALL compute slide-level metrics (accuracy, AUC, precision, recall, F1)
4. THE Evaluation_Script SHALL generate confusion matrix and ROC curve visualizations

### Requirement 4: Configuration Management

**User Story:** As a researcher, I want configuration options for slide-level training, so that I can control aggregation methods and batch sizes.

#### Acceptance Criteria

1. THE Training_Script SHALL read aggregation method from the configuration file
2. THE Training_Script SHALL support both "mean" and "max" pooling options
3. THE Training_Script SHALL allow configurable batch size for slide-level batching
4. THE Training_Script SHALL validate that required data paths exist before training

### Requirement 5: Testing and Validation

**User Story:** As a developer, I want targeted tests for the slide-level training path, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE Test_Suite SHALL include a test that verifies Slide_Dataset returns complete slides
2. THE Test_Suite SHALL include a test that verifies Collate_Function handles variable-length bags
3. THE Test_Suite SHALL include a test that verifies training can process at least one batch
4. THE Test_Suite SHALL include a test that verifies evaluation loads slide-level checkpoints
5. THE Test_Suite SHALL run successfully with synthetic CAMELYON data

### Requirement 6: Documentation and Honesty

**User Story:** As a user, I want accurate documentation about what the implementation provides, so that I understand its capabilities and limitations.

#### Acceptance Criteria

1. THE Training_Script SHALL document that it uses pre-extracted HDF5 features, not raw WSIs
2. THE Training_Script SHALL document the aggregation method being used
3. THE Training_Script SHALL clearly state this is a feature-cache baseline, not a full WSI pipeline
4. THE Training_Script SHALL provide example commands for running slide-level training
5. THE Evaluation_Script SHALL document compatibility with slide-level checkpoints
