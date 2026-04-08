# Implementation Plan: CAMELYON Slide-Level Training

## Overview

This implementation plan converts the CAMELYON training pipeline from patch-level to true slide-level training. Each task builds incrementally, starting with the core dataset infrastructure, then updating the training script, adding tests, and finally updating documentation. The implementation uses pre-extracted HDF5 feature caches with mean/max pooling aggregation, avoiding raw WSI complexity while establishing correct slide-level semantics.

## Tasks

- [ ] 1. Implement core slide-level dataset infrastructure
  - [ ] 1.1 Create CAMELYONSlideDataset class in src/data/camelyon_dataset.py
    - Implement `__init__` to load slide index and filter by split
    - Implement `__len__` to return number of slides
    - Implement `__getitem__` to load all patches from HDF5 for a single slide
    - Add validation for missing feature files with warning logs
    - Return dictionary with slide_id, patient_id, label, features, coordinates, num_patches
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 1.2 Implement collate_slide_bags function in src/data/camelyon_dataset.py
    - Accept list of slide dictionaries from dataset
    - Compute max_patches across batch
    - Pad features and coordinates tensors to max_patches
    - Return batch dictionary with padded tensors and metadata lists
    - _Requirements: 1.4, 2.2_

  - [ ]* 1.3 Write unit tests for CAMELYONSlideDataset
    - Test dataset length matches number of slides in split
    - Test `__getitem__` returns correct dictionary structure
    - Test split filtering (train/val/test)
    - Test handling of missing feature files
    - Test transform application if provided
    - _Requirements: 5.1_

  - [ ]* 1.4 Write unit tests for collate_slide_bags
    - Test padding to max length in batch
    - Test batch structure correctness
    - Test single-slide batch handling
    - Test metadata preservation (slide_ids, patient_ids)
    - _Requirements: 5.2_

- [ ] 2. Update training script for slide-level batching
  - [ ] 2.1 Create create_slide_dataloaders function in experiments/train_camelyon.py
    - Load CAMELYONSlideIndex from JSON
    - Create CAMELYONSlideDataset instances for train and val splits
    - Create DataLoader instances with collate_slide_bags
    - Configure batch_size, num_workers, pin_memory from config
    - Log dataset sizes
    - _Requirements: 2.1, 2.2_

  - [ ] 2.2 Update train_epoch function to handle slide-level batches
    - Extract features, labels, num_patches from batch dictionary
    - Pass features and num_patches to model forward
    - Compute loss and backpropagate
    - Log batch-level metrics
    - _Requirements: 2.3, 2.4_

  - [ ] 2.3 Update validate function for slide-level validation
    - Process slide-level batches from val_loader
    - Aggregate predictions and labels across slides
    - Compute slide-level metrics (accuracy, AUC, F1)
    - Return validation metrics dictionary
    - _Requirements: 3.1, 3.3_

  - [ ] 2.4 Add configuration validation in train_camelyon.py
    - Validate required config fields exist (data.root_dir, training.batch_size, etc.)
    - Validate aggregation method is "mean" or "max"
    - Validate data paths exist before training
    - Raise clear error messages for invalid configs
    - _Requirements: 4.1, 4.2, 4.4_

- [ ] 3. Checkpoint - Ensure training script runs
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Update model for masked aggregation
  - [ ] 4.1 Update SimpleSlideClassifier.forward in src/models/wsi_models.py
    - Accept optional num_patches parameter for masking
    - Implement masked mean pooling (only average over actual patches)
    - Keep max pooling unchanged (naturally handles padding)
    - Ensure backward compatibility with existing checkpoints
    - _Requirements: 2.4, 2.5_

  - [ ]* 4.2 Write unit tests for masked aggregation
    - Test mean pooling with num_patches masking
    - Test max pooling ignores padding
    - Test forward pass with variable-length inputs
    - Test backward compatibility with old checkpoint format
    - _Requirements: 2.5_

- [ ] 5. Create YAML configuration for slide-level training
  - [ ] 5.1 Create experiments/configs/camelyon.yaml
    - Set data.root_dir to "data/camelyon"
    - Set model.wsi.aggregation to "mean" (default)
    - Set training.batch_size to 8 (slides per batch)
    - Set training.num_epochs, learning_rate, weight_decay
    - Set checkpoint.checkpoint_dir and save_frequency
    - _Requirements: 4.1, 4.2, 4.3_

- [ ] 6. Update evaluation script for slide-level checkpoints
  - [ ] 6.1 Verify evaluate_camelyon.py loads slide-level checkpoints
    - Ensure evaluation uses CAMELYONSlideDataset
    - Ensure aggregation method matches training config
    - Compute slide-level metrics (accuracy, AUC, precision, recall, F1)
    - Generate confusion matrix and ROC curve visualizations
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 6.2 Write integration test for evaluation script
    - Create synthetic checkpoint and slide data
    - Run evaluation script
    - Verify metrics JSON is generated
    - Verify plots are created (if matplotlib available)
    - _Requirements: 5.4_

- [ ] 7. Checkpoint - Ensure evaluation works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Add comprehensive testing
  - [ ]* 8.1 Create synthetic CAMELYON data generator for testing
    - Generate synthetic slide_index.json with train/val/test splits
    - Generate synthetic HDF5 feature files with random features
    - Include variable numbers of patches per slide
    - Save to temporary directory for test isolation
    - _Requirements: 5.5_

  - [ ]* 8.2 Write end-to-end training integration test
    - Use synthetic data generator
    - Run training for 2 epochs
    - Verify checkpoint is saved with correct structure
    - Verify metrics are logged
    - _Requirements: 5.3, 5.5_

  - [ ]* 8.3 Write end-to-end evaluation integration test
    - Load checkpoint from training test
    - Run evaluation on synthetic test split
    - Verify metrics JSON is generated
    - Verify slide-level predictions are correct
    - _Requirements: 5.4, 5.5_

- [ ] 9. Update documentation
  - [ ] 9.1 Update train_camelyon.py module docstring
    - Document slide-level training approach
    - Clarify this is a feature-cache baseline (not raw WSI)
    - Document aggregation methods (mean/max)
    - Provide usage example with config path
    - Document data requirements (slide_index.json, HDF5 features)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ] 9.2 Add CAMELYON slide-level training section to README
    - Document data requirements (slide index, feature cache)
    - Provide training command example
    - Provide evaluation command example
    - Document aggregation method configuration
    - Document limitations (feature-cache baseline, no on-the-fly extraction)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 9.3 Update evaluate_camelyon.py docstring
    - Document compatibility with slide-level checkpoints
    - Document aggregation method usage
    - Provide usage example
    - _Requirements: 6.5_

- [ ] 10. Final checkpoint - Verify complete implementation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Implementation uses Python with PyTorch for deep learning
- Focus is on feature-cache baseline with HDF5 files, not raw WSI processing
- Mean/max pooling provides simple but effective slide-level aggregation
- Variable-length batching handled via custom collate function with padding
