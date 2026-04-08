# Requirements Document: Full-Scale PatchCamelyon Experiments

## Introduction

This feature implements full-scale training and evaluation on the complete PatchCamelyon (PCam) dataset to address the "No experiments on full-scale published datasets" limitation identified in the project README. The current framework has been validated on a 500-sample synthetic subset achieving 94% accuracy, but lacks validation on the full 262K training samples and 32K test samples. This feature will enable rigorous benchmarking against published baselines and provide statistically validated results on a real, published pathology dataset.

## Glossary

- **PCam_Dataset**: The PatchCamelyon dataset containing 262,144 training samples, 32,768 validation samples, and 32,768 test samples of 96x96 RGB histopathology patches for binary metastatic tissue classification
- **Training_Pipeline**: The end-to-end system that downloads data, trains models, evaluates performance, and generates reports
- **Baseline_Model**: A reference model architecture (ResNet-50, DenseNet-121, or EfficientNet-B0) used for performance comparison
- **Bootstrap_CI**: Bootstrap confidence intervals computed by resampling predictions to estimate statistical uncertainty
- **GPU_Config**: Configuration settings optimized for GPU training including batch size, mixed precision, and memory management
- **Benchmark_Report**: A markdown document containing training results, evaluation metrics, baseline comparisons, and statistical validation
- **Feature_Extractor**: A pretrained convolutional neural network (ResNet, DenseNet, or EfficientNet) that converts raw images to feature vectors
- **Download_Manager**: The component responsible for downloading and validating the full PCam dataset from TensorFlow Datasets or GitHub
- **Comparison_Runner**: The system that trains multiple model variants and generates comparative performance analysis

## Requirements

### Requirement 1: Download Full PCam Dataset

**User Story:** As a researcher, I want to download the complete PCam dataset automatically, so that I can train and evaluate on the full 262K training samples without manual data preparation.

#### Acceptance Criteria

1. WHEN the Training_Pipeline is executed with download enabled, THE Download_Manager SHALL download the complete PCam_Dataset from TensorFlow Datasets or GitHub
2. WHEN the download completes, THE Download_Manager SHALL validate dataset integrity by verifying 262,144 training samples, 32,768 validation samples, and 32,768 test samples exist
3. WHEN the download completes, THE Download_Manager SHALL verify each sample has shape [3, 96, 96] and label in {0, 1}
4. IF the dataset already exists at the specified path, THEN THE Download_Manager SHALL skip downloading and proceed with validation
5. WHEN downloading, THE Download_Manager SHALL display progress information including download speed and estimated time remaining
6. IF the download fails, THEN THE Download_Manager SHALL provide a descriptive error message and cleanup partial downloads

### Requirement 2: GPU-Optimized Training Configuration

**User Story:** As a researcher, I want GPU-optimized training settings, so that I can complete full-scale training in 4-8 hours on a single GPU.

#### Acceptance Criteria

1. THE GPU_Config SHALL specify batch size between 128 and 512 based on available GPU memory
2. THE GPU_Config SHALL enable mixed precision training using automatic mixed precision
3. THE GPU_Config SHALL set num_workers between 4 and 8 for parallel data loading
4. THE GPU_Config SHALL enable gradient accumulation when batch size exceeds GPU memory capacity
5. WHEN GPU memory is insufficient, THE Training_Pipeline SHALL reduce batch size by half and retry training
6. THE GPU_Config SHALL enable pin_memory for faster CPU-to-GPU data transfer
7. THE GPU_Config SHALL set appropriate learning rate scaled to effective batch size

### Requirement 3: Train on Full PCam Dataset

**User Story:** As a researcher, I want to train models on all 262K training samples, so that I can evaluate performance on the complete dataset rather than a synthetic subset.

#### Acceptance Criteria

1. WHEN training is initiated, THE Training_Pipeline SHALL load all 262,144 training samples from PCam_Dataset
2. WHEN training is initiated, THE Training_Pipeline SHALL load all 32,768 validation samples from PCam_Dataset
3. THE Training_Pipeline SHALL train for 20 epochs with early stopping patience of 5 epochs
4. WHEN each epoch completes, THE Training_Pipeline SHALL compute training loss, accuracy, F1 score, and AUC
5. WHEN validation is performed, THE Training_Pipeline SHALL compute validation loss, accuracy, F1 score, and AUC
6. WHEN validation AUC improves, THE Training_Pipeline SHALL save the model checkpoint as best_model.pth
7. WHEN training completes or early stopping triggers, THE Training_Pipeline SHALL save final training metrics to a JSON file
8. THE Training_Pipeline SHALL complete training within 8 hours on a single GPU with 16GB VRAM

### Requirement 4: Evaluate on Full Test Set

**User Story:** As a researcher, I want to evaluate trained models on all 32K test samples, so that I can report statistically robust performance metrics.

#### Acceptance Criteria

1. WHEN evaluation is initiated, THE Training_Pipeline SHALL load all 32,768 test samples from PCam_Dataset
2. WHEN evaluation runs, THE Training_Pipeline SHALL compute test accuracy, precision, recall, F1 score, and AUC
3. WHEN evaluation runs, THE Training_Pipeline SHALL generate a confusion matrix visualization
4. WHEN evaluation runs, THE Training_Pipeline SHALL generate an ROC curve visualization
5. WHEN evaluation completes, THE Training_Pipeline SHALL save all metrics to a JSON file
6. WHEN evaluation completes, THE Training_Pipeline SHALL save all visualizations as PNG files
7. THE Training_Pipeline SHALL compute per-class precision, recall, and F1 scores

### Requirement 5: Implement Baseline Model Comparisons

**User Story:** As a researcher, I want to compare my model against published baselines (ResNet-50, DenseNet-121, EfficientNet-B0), so that I can demonstrate relative performance.

#### Acceptance Criteria

1. THE Comparison_Runner SHALL support training ResNet-50 as a Baseline_Model
2. THE Comparison_Runner SHALL support training DenseNet-121 as a Baseline_Model
3. THE Comparison_Runner SHALL support training EfficientNet-B0 as a Baseline_Model
4. WHEN comparison is initiated, THE Comparison_Runner SHALL train each Baseline_Model on the full PCam_Dataset
5. WHEN all models complete training, THE Comparison_Runner SHALL evaluate each model on the test set
6. WHEN evaluation completes, THE Comparison_Runner SHALL generate a comparison table with accuracy, AUC, F1, precision, and recall for each model
7. THE Comparison_Runner SHALL save comparison results to a markdown table in the Benchmark_Report

### Requirement 6: Statistical Validation with Confidence Intervals

**User Story:** As a researcher, I want bootstrap confidence intervals for all metrics, so that I can report statistically validated results with uncertainty estimates.

#### Acceptance Criteria

1. WHEN evaluation completes, THE Training_Pipeline SHALL compute Bootstrap_CI for test accuracy using 1000 bootstrap samples
2. WHEN evaluation completes, THE Training_Pipeline SHALL compute Bootstrap_CI for test AUC using 1000 bootstrap samples
3. WHEN evaluation completes, THE Training_Pipeline SHALL compute Bootstrap_CI for test F1 score using 1000 bootstrap samples
4. THE Training_Pipeline SHALL report confidence intervals as [lower_bound, upper_bound] at 95% confidence level
5. WHEN comparison completes, THE Comparison_Runner SHALL compute Bootstrap_CI for each Baseline_Model
6. THE Benchmark_Report SHALL include confidence intervals for all reported metrics

### Requirement 7: Update Documentation with Real Results

**User Story:** As a researcher, I want comprehensive benchmark documentation, so that others can understand the experimental setup and reproduce results.

#### Acceptance Criteria

1. WHEN training and evaluation complete, THE Training_Pipeline SHALL generate a Benchmark_Report in markdown format
2. THE Benchmark_Report SHALL include dataset details (262K train, 32K test, image dimensions)
3. THE Benchmark_Report SHALL include model architecture details (parameters, layers, feature dimensions)
4. THE Benchmark_Report SHALL include training configuration (epochs, batch size, learning rate, optimizer)
5. THE Benchmark_Report SHALL include final test metrics with confidence intervals
6. THE Benchmark_Report SHALL include comparison table with all Baseline_Model results
7. THE Benchmark_Report SHALL include commands to reproduce all experiments
8. THE Benchmark_Report SHALL include hardware specifications (GPU model, memory, training time)
9. WHEN the Benchmark_Report is generated, THE Training_Pipeline SHALL save it as PCAM_BENCHMARK_RESULTS.md

### Requirement 8: Maintain Backward Compatibility

**User Story:** As a developer, I want the synthetic data path to remain functional, so that existing tests and CI/CD pipelines continue working.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL support a synthetic_mode flag to use synthetic data instead of full dataset
2. WHEN synthetic_mode is enabled, THE Training_Pipeline SHALL use the existing 500-sample synthetic dataset
3. WHEN synthetic_mode is disabled, THE Training_Pipeline SHALL use the full PCam_Dataset
4. THE Training_Pipeline SHALL maintain the same API for both synthetic and full dataset modes
5. THE Training_Pipeline SHALL maintain the same output format for both synthetic and full dataset modes
6. WHEN tests run in CI/CD, THE Training_Pipeline SHALL default to synthetic_mode for fast execution

### Requirement 9: Reproducibility with Fixed Seeds

**User Story:** As a researcher, I want deterministic training with fixed random seeds, so that experiments are reproducible across different runs and platforms.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL set random seed for Python random module before training
2. THE Training_Pipeline SHALL set random seed for NumPy before training
3. THE Training_Pipeline SHALL set random seed for PyTorch CPU operations before training
4. THE Training_Pipeline SHALL set random seed for PyTorch CUDA operations before training
5. THE Training_Pipeline SHALL disable CUDNN benchmark mode for deterministic behavior
6. THE Training_Pipeline SHALL enable CUDNN deterministic mode for reproducible results
7. WHEN the same seed is used, THE Training_Pipeline SHALL produce identical results within floating-point precision tolerance

### Requirement 10: Cross-Platform Compatibility

**User Story:** As a researcher, I want the training pipeline to work on Windows, macOS, and Linux, so that I can run experiments on any available hardware.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL execute successfully on Windows 10 and Windows 11
2. THE Training_Pipeline SHALL execute successfully on macOS 12 and later
3. THE Training_Pipeline SHALL execute successfully on Ubuntu 20.04 and later
4. THE Training_Pipeline SHALL handle file paths correctly on all platforms using pathlib
5. THE Training_Pipeline SHALL detect CUDA availability and fall back to CPU if unavailable
6. WHEN running on CPU, THE Training_Pipeline SHALL log a warning and adjust batch size for memory constraints
7. THE Training_Pipeline SHALL handle line endings correctly on all platforms

### Requirement 11: Training Time Constraints

**User Story:** As a researcher, I want training to complete in a reasonable timeframe, so that I can iterate on experiments efficiently.

#### Acceptance Criteria

1. WHEN training on a GPU with 16GB VRAM, THE Training_Pipeline SHALL complete 20 epochs within 8 hours
2. WHEN training on a GPU with 24GB VRAM, THE Training_Pipeline SHALL complete 20 epochs within 6 hours
3. WHEN training on a GPU with 40GB VRAM, THE Training_Pipeline SHALL complete 20 epochs within 4 hours
4. WHEN each epoch completes, THE Training_Pipeline SHALL log elapsed time and estimated time remaining
5. WHEN training completes, THE Training_Pipeline SHALL log total training time in the Benchmark_Report
6. IF training exceeds 12 hours, THEN THE Training_Pipeline SHALL log a warning about potential configuration issues

### Requirement 12: Memory Management and Error Recovery

**User Story:** As a researcher, I want graceful handling of GPU out-of-memory errors, so that training can recover automatically without manual intervention.

#### Acceptance Criteria

1. IF GPU out-of-memory error occurs during training, THEN THE Training_Pipeline SHALL catch the error and log a warning
2. WHEN GPU out-of-memory error is caught, THE Training_Pipeline SHALL reduce batch size by 50%
3. WHEN batch size is reduced, THE Training_Pipeline SHALL verify new batch size is at least 8
4. WHEN batch size is reduced, THE Training_Pipeline SHALL clear GPU cache and retry training
5. IF GPU out-of-memory error occurs after batch size reduction, THEN THE Training_Pipeline SHALL terminate with a descriptive error message
6. THE Training_Pipeline SHALL log current GPU memory usage after each epoch
7. WHEN training resumes after error, THE Training_Pipeline SHALL load the last saved checkpoint

