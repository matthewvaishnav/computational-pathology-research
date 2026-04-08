# Implementation Plan: Full-Scale PatchCamelyon Experiments

## Overview

This implementation plan converts the full-scale PCam experiments design into actionable coding tasks. The plan builds on existing infrastructure (PCamDataset, train_pcam.py, evaluate_pcam.py, compare_pcam_baselines.py) and adds GPU-optimized configurations, bootstrap confidence intervals, baseline model variants, and comprehensive benchmark reporting.

The implementation follows a phased approach:
1. Create GPU-optimized and baseline configuration files
2. Implement statistical utilities for bootstrap confidence intervals
3. Enhance evaluation script with CI computation
4. Create benchmark report generator
5. Add baseline model configurations
6. Update documentation with reproduction instructions

## Tasks

- [x] 1. Create GPU-optimized configuration files
  - Create `experiments/configs/pcam_fullscale/` directory
  - Create `gpu_16gb.yaml` with batch_size=128, num_workers=6, use_amp=true
  - Create `gpu_24gb.yaml` with batch_size=256, num_workers=6, use_amp=true
  - Create `gpu_40gb.yaml` with batch_size=512, num_workers=8, use_amp=true
  - All configs should enable download=true, pin_memory=true, num_epochs=20
  - All configs should use early_stopping with patience=5
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 11.1, 11.2, 11.3_

- [x] 2. Implement bootstrap confidence interval utilities
  - [x] 2.1 Create `src/utils/statistical.py` module
    - Implement `compute_bootstrap_ci()` function with resampling logic
    - Support accuracy, AUC, F1, precision, recall metrics
    - Use n_bootstrap=1000, confidence_level=0.95 as defaults
    - Set random_state for reproducibility
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 9.1, 9.2_

  - [x] 2.2 Implement `compute_all_metrics_with_ci()` function
    - Call `compute_bootstrap_ci()` for each metric
    - Return dictionary with value, ci_lower, ci_upper for each metric
    - Handle edge cases (single class, all correct/wrong predictions)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 2.3 Write unit tests for bootstrap CI functions
    - Test with known distributions
    - Test edge cases (all correct, all wrong, single class)
    - Verify CI bounds are reasonable
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 3. Enhance evaluation script with bootstrap CI
  - [x] 3.1 Add `--compute-bootstrap-ci` flag to evaluate_pcam.py
    - Add `--bootstrap-samples` argument (default: 1000)
    - Add `--confidence-level` argument (default: 0.95)
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 3.2 Integrate bootstrap CI computation in evaluate_pcam.py
    - Import `compute_all_metrics_with_ci()` from statistical.py
    - Call when `--compute-bootstrap-ci` flag is set
    - Add CI results to metrics dictionary
    - Save bootstrap_config to metrics JSON
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

  - [ ]* 3.3 Write integration test for evaluation with CI
    - Load checkpoint from synthetic training
    - Run evaluation with `--compute-bootstrap-ci`
    - Verify CI bounds in output JSON
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [-] 4. Create baseline model configuration files
  - [x] 4.1 Create ResNet-50 baseline config
    - Create `experiments/configs/pcam_fullscale/baseline_resnet50.yaml`
    - Set feature_extractor.model=resnet50, feature_dim=2048
    - Use gpu_16gb settings as base
    - _Requirements: 5.1, 5.4_

  - [x] 4.2 Create DenseNet-121 baseline config
    - Create `experiments/configs/pcam_fullscale/baseline_densenet121.yaml`
    - Set feature_extractor.model=densenet121, feature_dim=1024
    - Use gpu_16gb settings as base
    - _Requirements: 5.2, 5.4_

  - [ ] 4.3 Create EfficientNet-B0 baseline config
    - Create `experiments/configs/pcam_fullscale/baseline_efficientnet_b0.yaml`
    - Set feature_extractor.model=efficientnet_b0, feature_dim=1280
    - Use gpu_16gb settings as base
    - _Requirements: 5.3, 5.4_

- [ ] 5. Checkpoint - Verify configurations load correctly
  - Ensure all YAML configs parse without errors
  - Verify feature extractor models are supported
  - Test loading configs with train_pcam.py --config flag
  - Ask the user if questions arise

- [ ] 6. Implement benchmark report generator
  - [ ] 6.1 Create `src/utils/benchmark_report.py` module
    - Implement `generate_benchmark_report()` function
    - Accept experiment_name, dataset_info, model_info, training_config, test_metrics
    - Accept optional comparison_results and hardware_info
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.8_

  - [ ] 6.2 Generate markdown sections in benchmark report
    - Executive summary with key results
    - Dataset description (262K train, 32K test, image dimensions)
    - Model architecture (parameters, layers, feature dimensions)
    - Training configuration (epochs, batch size, learning rate)
    - Test metrics with confidence intervals
    - Baseline comparison table (if provided)
    - Reproduction commands
    - Hardware specifications
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_

  - [ ]* 6.3 Write unit tests for report generation
    - Test markdown formatting
    - Test table generation
    - Test with missing optional fields
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 7. Integrate report generation into comparison runner
  - [ ] 7.1 Import `generate_benchmark_report()` in compare_pcam_baselines.py
    - Add call to report generator after comparison completes
    - Pass aggregated comparison results
    - Save report as PCAM_BENCHMARK_RESULTS.md
    - _Requirements: 5.7, 7.1, 7.9_

  - [ ] 7.2 Add bootstrap CI to comparison runner
    - Call evaluate_pcam.py with `--compute-bootstrap-ci` flag
    - Include CI results in comparison table
    - _Requirements: 5.6, 6.5, 6.6_

- [ ] 8. Update PCamDataset for full-scale download validation
  - [ ] 8.1 Enhance dataset validation in PCamDataset
    - Verify 262,144 training samples after download
    - Verify 32,768 validation samples after download
    - Verify 32,768 test samples after download
    - Log validation results
    - _Requirements: 1.2, 1.3_

  - [ ] 8.2 Add progress reporting to download
    - Display download speed and estimated time
    - Log download source (TFDS or GitHub)
    - _Requirements: 1.1, 1.5_

  - [ ] 8.3 Improve error handling for download failures
    - Cleanup partial downloads on failure
    - Provide descriptive error messages
    - _Requirements: 1.6, 12.1, 12.2_

- [ ] 9. Checkpoint - Test full pipeline on synthetic data
  - Run training with gpu_16gb.yaml on synthetic subset (500 samples)
  - Run evaluation with bootstrap CI
  - Run comparison with 2 baseline configs
  - Verify report generation
  - Ensure all tests pass, ask the user if questions arise

- [ ] 10. Create comprehensive documentation
  - [ ] 10.1 Create PCAM_FULLSCALE_GUIDE.md
    - Hardware requirements (16GB/24GB/40GB GPU)
    - Software dependencies and installation
    - Configuration selection guide
    - Step-by-step training instructions
    - Evaluation and comparison instructions
    - Troubleshooting common issues
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.7, 10.1, 10.2, 10.3, 11.1, 11.2, 11.3_

  - [ ] 10.2 Update README.md with full-scale experiments section
    - Add link to PCAM_FULLSCALE_GUIDE.md
    - Update "Current Limitations" section
    - Add example commands for full-scale training
    - _Requirements: 7.1, 7.7, 8.1, 8.2, 8.3_

  - [ ] 10.3 Create REPRODUCTION.md with exact commands
    - Commands to download dataset
    - Commands to train each baseline
    - Commands to evaluate models
    - Commands to generate comparison report
    - Expected outputs and file locations
    - _Requirements: 7.7, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

- [ ] 11. Ensure backward compatibility
  - [ ] 11.1 Verify synthetic mode still works
    - Run existing tests with synthetic data
    - Verify CI/CD pipeline passes
    - Ensure no breaking API changes
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ] 11.2 Add synthetic_mode flag to configs
    - Document synthetic vs full-scale mode
    - Ensure both modes use same API
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 12. Cross-platform compatibility validation
  - [ ] 12.1 Test on Windows
    - Verify file path handling with pathlib
    - Test CUDA detection and CPU fallback
    - Verify HDF5 data loading works
    - _Requirements: 10.1, 10.4, 10.5, 10.6, 10.7_

  - [ ] 12.2 Test on Linux
    - Verify all scripts execute successfully
    - Test GPU training and evaluation
    - Verify report generation
    - _Requirements: 10.3, 10.4, 10.5, 10.6, 10.7_

  - [ ]* 12.3 Test on macOS (if available)
    - Verify CPU-only training works
    - Test data loading and evaluation
    - _Requirements: 10.2, 10.4, 10.5, 10.6, 10.7_

- [ ] 13. Final checkpoint - Verify all requirements met
  - Review all 12 requirements and acceptance criteria
  - Verify all configuration files created
  - Verify bootstrap CI implemented and tested
  - Verify baseline models configured
  - Verify documentation complete
  - Verify backward compatibility maintained
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional testing tasks that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- The implementation leverages existing infrastructure to minimize code changes
- Bootstrap CI computation is opt-in via command-line flag
- Full-scale training requires GPU with 16GB+ VRAM
- Synthetic mode remains functional for CI/CD and quick testing
- Cross-platform compatibility is ensured through pathlib and platform detection
