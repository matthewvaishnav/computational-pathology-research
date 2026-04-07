# Requirements Document: PatchCamelyon Experiment

## Introduction

This document specifies requirements for adding a complete end-to-end experiment using the PatchCamelyon (PCam) dataset to the computational pathology research repository. The experiment will demonstrate that the framework works on real pathology data and provide baseline results for the portfolio. PatchCamelyon is a binary classification dataset of 96x96 pixel histopathology patches from lymph node sections, containing approximately 327,000 training images, 33,000 validation images, and 33,000 test images. The goal is to train a model to distinguish metastatic tissue from normal tissue, achieving reasonable baseline accuracy and generating visualizations that demonstrate the framework's capabilities.

## Glossary

- **PCam_Dataset**: The PatchCamelyon dataset containing 96x96 pixel histopathology patches with binary labels (metastatic vs normal tissue)
- **Dataset_Loader**: PyTorch Dataset class that loads and preprocesses PCam images and labels
- **Training_Script**: Python script that trains a model on PCam using the framework's architecture
- **Evaluation_Script**: Python script that evaluates trained model performance and computes metrics
- **Visualization_Notebook**: Jupyter notebook that generates plots and visualizations of results
- **Model_Checkpoint**: Saved PyTorch model weights after training
- **Training_Curves**: Plots showing loss and accuracy over training epochs
- **Confusion_Matrix**: 2x2 matrix showing true positives, false positives, true negatives, false negatives
- **ROC_Curve**: Receiver Operating Characteristic curve plotting true positive rate vs false positive rate
- **Baseline_Accuracy**: Minimum acceptable classification accuracy threshold of 60%
- **Single_Modality_Mode**: Configuration where only image features are used (no genomics or clinical text)
- **Results_Documentation**: Updated README section with experimental results and commands

## Requirements

### Requirement 1: Dataset Download and Preprocessing

**User Story:** As a researcher, I want to download and preprocess the PatchCamelyon dataset, so that I can train models on real histopathology data.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL download PCam data from TensorFlow Datasets or direct download source
2. WHEN PCam data is downloaded, THE Dataset_Loader SHALL organize images into train, validation, and test splits
3. THE Dataset_Loader SHALL normalize pixel values to the range expected by the framework
4. THE Dataset_Loader SHALL convert images to PyTorch tensors with shape [3, 96, 96]
5. THE Dataset_Loader SHALL provide binary labels (0 for normal, 1 for metastatic)
6. THE Dataset_Loader SHALL support data augmentation including random horizontal flip, random vertical flip, and color jitter
7. WHEN a sample is requested, THE Dataset_Loader SHALL return a dictionary with 'image' and 'label' keys
8. THE Dataset_Loader SHALL handle missing or corrupted images by skipping them and logging warnings

### Requirement 2: Single-Modality Training Configuration

**User Story:** As a researcher, I want to train a model using only image features, so that I can evaluate the framework on single-modality classification tasks.

#### Acceptance Criteria

1. THE Training_Script SHALL support a single-modality configuration that uses only WSI encoder
2. WHEN single-modality mode is enabled, THE Training_Script SHALL not require genomic or clinical text inputs
3. THE Training_Script SHALL use the existing WSIEncoder to process 96x96 patches as single-patch sequences
4. THE Training_Script SHALL add a classification head with 2 output classes
5. THE Training_Script SHALL use binary cross-entropy loss for training
6. THE Training_Script SHALL support configurable learning rate, batch size, and number of epochs
7. THE Training_Script SHALL save model checkpoints every N epochs where N is configurable
8. THE Training_Script SHALL log training loss and accuracy to TensorBoard

### Requirement 3: Model Training Execution

**User Story:** As a researcher, I want to train a model for at least 1 epoch, so that I can verify the framework works end-to-end on real data.

#### Acceptance Criteria

1. WHEN the Training_Script is executed, THE Training_Script SHALL complete at least 1 training epoch without errors
2. THE Training_Script SHALL process all training samples in batches
3. THE Training_Script SHALL compute gradients and update model weights
4. THE Training_Script SHALL validate on the validation set after each epoch
5. WHEN training completes, THE Training_Script SHALL save the final Model_Checkpoint
6. THE Training_Script SHALL complete 1 epoch within 30 minutes on a consumer GPU
7. THE Training_Script SHALL log progress including current epoch, batch number, and loss values
8. IF GPU memory is insufficient, THEN THE Training_Script SHALL reduce batch size automatically and log a warning

### Requirement 4: Model Evaluation and Metrics

**User Story:** As a researcher, I want to evaluate the trained model on the test set, so that I can measure its performance with standard metrics.

#### Acceptance Criteria

1. THE Evaluation_Script SHALL load a trained Model_Checkpoint
2. THE Evaluation_Script SHALL run inference on all test set samples
3. THE Evaluation_Script SHALL compute overall classification accuracy
4. THE Evaluation_Script SHALL compute Area Under ROC Curve (AUC)
5. THE Evaluation_Script SHALL generate a Confusion_Matrix with counts for each cell
6. THE Evaluation_Script SHALL compute precision, recall, and F1-score for both classes
7. WHEN evaluation completes, THE Evaluation_Script SHALL save metrics to a JSON file
8. THE Evaluation_Script SHALL achieve Baseline_Accuracy of at least 60% on the test set

### Requirement 5: Visualization Generation

**User Story:** As a researcher, I want to generate visualizations of the experiment, so that I can demonstrate the framework works and analyze results.

#### Acceptance Criteria

1. THE Visualization_Notebook SHALL display a grid of sample patches from the dataset with labels
2. THE Visualization_Notebook SHALL plot Training_Curves showing loss over epochs for train and validation sets
3. THE Visualization_Notebook SHALL plot Training_Curves showing accuracy over epochs for train and validation sets
4. THE Visualization_Notebook SHALL display the Confusion_Matrix as a heatmap with annotations
5. THE Visualization_Notebook SHALL plot the ROC_Curve with AUC value in the legend
6. THE Visualization_Notebook SHALL show example predictions with ground truth and predicted labels
7. THE Visualization_Notebook SHALL save all plots as PNG files to a results directory
8. WHERE attention mechanisms are used, THE Visualization_Notebook SHALL visualize attention weights overlaid on patches

### Requirement 6: Results Documentation

**User Story:** As a portfolio viewer, I want to see documented experimental results in the README, so that I can verify the framework has been tested on real data.

#### Acceptance Criteria

1. THE Results_Documentation SHALL include a new section titled "Experimental Results"
2. THE Results_Documentation SHALL report the final test accuracy achieved
3. THE Results_Documentation SHALL report the AUC score achieved
4. THE Results_Documentation SHALL include the command to reproduce the training
5. THE Results_Documentation SHALL include the command to reproduce the evaluation
6. THE Results_Documentation SHALL reference the saved Model_Checkpoint location
7. THE Results_Documentation SHALL include links to generated visualization plots
8. THE Results_Documentation SHALL state the training time and hardware used

### Requirement 7: Reproducibility and Configuration

**User Story:** As a researcher, I want reproducible training with documented configuration, so that others can replicate the experiment.

#### Acceptance Criteria

1. THE Training_Script SHALL set random seeds for PyTorch, NumPy, and Python random module
2. THE Training_Script SHALL use a YAML configuration file for all hyperparameters
3. THE Training_Script SHALL log the configuration file contents at the start of training
4. THE Training_Script SHALL save the configuration alongside the Model_Checkpoint
5. WHEN the same configuration and seed are used, THE Training_Script SHALL produce identical results within numerical precision
6. THE Training_Script SHALL log the PyTorch version, CUDA version, and device information
7. THE Training_Script SHALL create a requirements.txt file listing exact package versions used
8. THE Training_Script SHALL support resuming training from a checkpoint

### Requirement 8: Error Handling and Robustness

**User Story:** As a researcher, I want robust error handling, so that I can diagnose issues when they occur.

#### Acceptance Criteria

1. IF the dataset download fails, THEN THE Dataset_Loader SHALL log a clear error message with troubleshooting steps
2. IF GPU memory is exhausted, THEN THE Training_Script SHALL catch the error and suggest reducing batch size
3. IF a checkpoint file is corrupted, THEN THE Training_Script SHALL log an error and exit gracefully
4. IF the validation set is empty, THEN THE Training_Script SHALL log a warning and skip validation
5. IF evaluation is run without a checkpoint, THEN THE Evaluation_Script SHALL log an error indicating the checkpoint path is required
6. THE Training_Script SHALL validate that input images have the correct shape before processing
7. IF NaN loss is detected, THEN THE Training_Script SHALL log an error and terminate training
8. THE Evaluation_Script SHALL handle missing predictions gracefully by logging warnings and excluding them from metrics
