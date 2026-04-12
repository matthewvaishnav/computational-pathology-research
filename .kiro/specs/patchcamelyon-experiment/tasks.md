# Implementation Plan: PatchCamelyon Experiment

## Overview

This implementation plan breaks down the PatchCamelyon experiment into discrete coding tasks following the 7-phase implementation plan from the design document. The experiment will train a binary classification model on histopathology patches using the existing multimodal framework in single-modality mode.

## Tasks

- [ ] 1. Implement PCamDataset class for data loading
  - [x] 1.1 Create src/data/pcam_dataset.py with PCamDataset class
    - Implement __init__ method with parameters: root_dir, split, transform, download, feature_extractor
    - Implement __len__ method returning number of samples
    - Implement __getitem__ method returning dict with 'wsi_features', 'label', 'image_id'
    - Add proper type hints and docstrings
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7_
  
  - [x] 1.2 Implement dataset download functionality
    - Add download() method using tensorflow_datasets API
    - Handle ImportError with clear message if tensorflow_datasets not installed
    - Organize downloaded data into train/val/test splits
    - Save data as numpy arrays in root_dir structure
    - _Requirements: 1.1, 1.2, 1.8_
  
  - [x] 1.3 Implement data preprocessing and augmentation
    - Add _apply_transforms() method for normalization to [0, 1]
    - Support torchvision transforms for augmentation (horizontal flip, vertical flip, color jitter)
    - Convert images to tensors with shape [3, 96, 96]
    - Handle missing or corrupted images with try-except and logging
    - _Requirements: 1.3, 1.4, 1.6, 1.8_
  
  - [ ]* 1.4 Write unit tests for PCamDataset
    - Test dataset initialization with different parameters
    - Test download functionality with mocked tensorflow_datasets
    - Test __getitem__ returns correct shapes and types
    - Test augmentation is applied correctly
    - Test error handling for missing files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

- [ ] 2. Implement feature extraction module
  - [x] 2.1 Create src/models/feature_extractors.py with ResNetFeatureExtractor class
    - Implement __init__ with parameters: model_name, pretrained, feature_dim
    - Load ResNet-18 from torchvision.models
    - Remove final classification layer to extract features
    - Implement forward() method returning [batch, 512] features
    - Add proper type hints and docstrings
    - _Requirements: 2.3_
  
  - [x] 2.2 Integrate feature extraction into PCamDataset
    - Add optional feature_extractor parameter to PCamDataset.__init__
    - Add _extract_features() method that applies feature extractor if provided
    - Update __getitem__ to return extracted features or raw images based on configuration
    - Ensure output shape is [1, 512] for compatibility with WSI encoder
    - _Requirements: 2.3_
  
  - [ ]* 2.3 Write unit tests for feature extraction
    - Test ResNetFeatureExtractor initialization with pretrained and random weights
    - Test forward pass with various input sizes
    - Test output feature dimension is correct
    - Test integration with PCamDataset
    - _Requirements: 2.3_

- [ ] 3. Create training configuration and script
  - [x] 3.1 Create experiments/configs/pcam.yaml configuration file
    - Define experiment metadata (name, description, tags)
    - Configure data parameters (dataset, root_dir, download, num_workers, augmentation)
    - Configure feature extraction (extract_features, feature_extractor settings)
    - Configure model architecture (modalities=[wsi], embed_dim, wsi encoder settings)
    - Configure task (type=classification, num_classes=2, classification head settings)
    - Configure training hyperparameters (num_epochs, batch_size, learning_rate, optimizer, scheduler)
    - Configure validation, checkpointing, early stopping, and logging
    - Set seed=42 for reproducibility
    - _Requirements: 2.1, 2.2, 2.6, 7.2, 7.3_
  
  - [x] 3.2 Create experiments/train_pcam.py training script
    - Implement create_pcam_dataloaders() function to create train/val/test DataLoaders
    - Implement create_single_modality_model() function to create WSI encoder and classification head
    - Implement train_epoch() function for one epoch of training
    - Implement validate() function for validation
    - Implement main() function with training loop, checkpoint saving, and logging
    - Set random seeds for reproducibility (torch, numpy, random)
    - Log configuration, PyTorch version, CUDA version, device info at start
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 7.1, 7.6_

- [ ] 4. Implement training loop with error handling
  - [x] 4.1 Add checkpoint saving and loading
    - Save checkpoint dict with epoch, global_step, encoder_state_dict, head_state_dict, optimizer_state_dict, scheduler_state_dict, metrics, config
    - Implement save_checkpoint() function called every N epochs
    - Implement load_checkpoint() function for resuming training
    - Save best checkpoint based on validation metric
    - _Requirements: 2.7, 3.5, 7.4, 7.8_
  
  - [x] 4.2 Add comprehensive error handling
    - Catch GPU out of memory errors and suggest reducing batch_size
    - Implement automatic batch size reduction on OOM if auto_batch_size=true
    - Detect NaN loss and terminate with informative error message
    - Validate checkpoint structure when loading
    - Handle missing checkpoint files gracefully
    - _Requirements: 3.8, 8.2, 8.3, 8.7_
  
  - [x] 4.3 Add logging and progress tracking
    - Log training progress every N batches (loss, accuracy, learning rate)
    - Log validation metrics after each epoch
    - Use TensorBoard for logging training curves
    - Log warnings for any skipped or problematic batches
    - _Requirements: 2.8, 3.7_
  
  - [ ]* 4.4 Write integration test for training
    - Test training for 1 epoch on small subset (100 samples)
    - Verify loss decreases during training
    - Verify checkpoint is saved correctly
    - Verify logs are created
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 5. Checkpoint - Verify training works end-to-end
  - Run training script for 1 epoch on full dataset
  - Ensure all tests pass, ask the user if questions arise

- [ ] 6. Implement evaluation script
  - [ ] 6.1 Create experiments/evaluate_pcam.py evaluation script
    - Implement load_checkpoint() function to load trained model
    - Implement evaluate_model() function to run inference on test set
    - Collect predictions, probabilities, and labels for all test samples
    - Handle evaluation errors gracefully (missing checkpoint, empty test set)
    - _Requirements: 4.1, 4.2, 8.5_
  
  - [ ] 6.2 Implement metrics computation
    - Implement compute_metrics() function using sklearn.metrics
    - Compute accuracy, AUC, precision, recall, F1-score
    - Generate confusion matrix as numpy array
    - Compute per-class metrics (precision, recall, F1 for each class)
    - _Requirements: 4.3, 4.4, 4.5, 4.6_
  
  - [ ] 6.3 Add metrics saving and reporting
    - Implement save_metrics() function to save metrics as JSON
    - Include training time, inference time, model parameters, hardware info in output
    - Log all metrics to console with clear formatting
    - Verify baseline accuracy threshold of 60% is met
    - _Requirements: 4.7, 4.8, 6.2, 6.3, 6.8_
  
  - [ ]* 6.4 Write unit tests for evaluation
    - Test metrics computation with known predictions and labels
    - Test confusion matrix generation
    - Test checkpoint loading with valid and invalid paths
    - Test JSON output file creation
    - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 7. Create visualization notebook
  - [ ] 7.1 Create experiments/notebooks/pcam_visualization.ipynb
    - Set up notebook with imports (matplotlib, seaborn, torch, numpy)
    - Add markdown cells explaining each section
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  
  - [ ] 7.2 Implement dataset exploration visualizations
    - Display grid of sample patches (8x8) with labels as titles
    - Plot class distribution bar chart for train/val/test splits
    - Compute and display image statistics (mean, std per channel)
    - _Requirements: 5.1_
  
  - [ ] 7.3 Implement training curves visualization
    - Load training logs from TensorBoard or checkpoint metrics
    - Plot loss curves (train and val) over epochs
    - Plot accuracy curves (train and val) over epochs
    - Plot learning rate schedule over epochs
    - _Requirements: 5.2, 5.3_
  
  - [ ] 7.4 Implement model performance visualizations
    - Load evaluation metrics JSON file
    - Plot confusion matrix as heatmap with annotations
    - Plot ROC curve with AUC value in legend
    - Plot precision-recall curve
    - _Requirements: 5.4, 5.5_
  
  - [ ] 7.5 Implement prediction analysis visualizations
    - Load test set predictions from evaluation
    - Display grid of correct predictions with confidence scores
    - Display grid of incorrect predictions with confidence scores
    - Plot histogram of prediction confidence distribution
    - _Requirements: 5.6_
  
  - [ ] 7.6 Save all plots to results directory
    - Create results/pcam/ directory if not exists
    - Save each plot as high-resolution PNG file
    - Add cell to display saved file paths
    - _Requirements: 5.7_

- [ ] 8. Update documentation with experimental results
  - [ ] 8.1 Add Experimental Results section to README
    - Create new section titled "Experimental Results: PatchCamelyon"
    - Document the experiment objective and dataset description
    - _Requirements: 6.1_
  
  - [ ] 8.2 Document training and evaluation commands
    - Add command to run training: python experiments/train_pcam.py --config experiments/configs/pcam.yaml
    - Add command to run evaluation: python experiments/evaluate_pcam.py --checkpoint <path>
    - Add command to run visualization notebook
    - Document expected training time and hardware requirements
    - _Requirements: 6.4, 6.5, 6.8_
  
  - [ ] 8.3 Document experimental results
    - Report final test accuracy achieved
    - Report AUC score achieved
    - Reference Model_Checkpoint location
    - Include links to visualization plots in results/pcam/
    - Add table comparing results to baseline expectations
    - _Requirements: 6.2, 6.3, 6.6, 6.7_
  
  - [ ] 8.4 Add reproducibility documentation
    - Document random seed used (42)
    - Document exact package versions in requirements.txt
    - Document PyTorch, CUDA versions used
    - Add note about numerical precision in reproducibility
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.7_

- [ ] 9. Add dataset validation and error handling
  - [ ] 9.1 Implement dataset validation function
    - Create validate_dataset() function in train_pcam.py
    - Check dataset is not empty
    - Sample first 10 items and validate shapes
    - Validate labels are in [0, 1]
    - Log validation results
    - _Requirements: 8.6_
  
  - [ ] 9.2 Add comprehensive error messages
    - Add error handling for dataset download failures with troubleshooting steps
    - Add error handling for validation set empty with warning
    - Add error handling for corrupted images in dataset
    - Ensure all error messages are actionable and clear
    - _Requirements: 8.1, 8.4, 8.8_
  
  - [ ]* 9.3 Write integration test for error handling
    - Test dataset download failure handling
    - Test GPU OOM handling
    - Test corrupted checkpoint handling
    - Test NaN loss detection
    - _Requirements: 8.1, 8.2, 8.3, 8.7_

- [ ] 10. Final integration and testing
  - [ ] 10.1 Run complete end-to-end experiment
    - Download PCam dataset
    - Train model for at least 5 epochs
    - Verify training completes without errors
    - Verify checkpoints are saved
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 10.2 Run evaluation and generate visualizations
    - Run evaluation script on trained model
    - Verify metrics meet baseline (>60% accuracy)
    - Run visualization notebook
    - Verify all plots are generated
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  
  - [ ] 10.3 Verify reproducibility
    - Run training twice with same seed and config
    - Verify results are identical within numerical precision
    - Test resuming from checkpoint
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.8_
  
  - [ ]* 10.4 Run all unit and integration tests
    - Run pytest on all test files
    - Verify all tests pass
    - Check test coverage is reasonable
    - _Requirements: All testing requirements_

- [ ] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The implementation uses Python and PyTorch as specified in the design document
- Checkpoints ensure incremental validation at key milestones
- Focus on reusing existing framework components where possible
- All code should include proper error handling and logging
- Configuration-driven approach allows easy experimentation
