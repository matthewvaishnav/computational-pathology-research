# Implementation Plan: Model Interpretability

## Overview

This implementation plan breaks down the model interpretability feature into discrete coding tasks. The feature adds comprehensive interpretability tools including Grad-CAM visualization, attention weight analysis, failure case identification, feature importance computation, and an interactive dashboard. The implementation extends existing visualization capabilities while maintaining backward compatibility with current evaluation workflows.

The tasks are organized to build incrementally: core interpretability components first (Grad-CAM, attention extensions), then analysis tools (failure analysis, feature importance), configuration infrastructure, and finally the interactive dashboard. Each task includes property-based tests to validate universal correctness properties defined in the design document.

## Tasks

- [x] 1. Set up interpretability module structure and core utilities
  - Create `src/interpretability/` directory with `__init__.py`
  - Create `src/interpretability/utils.py` with shared utilities (device management, tensor operations)
  - Set up `tests/interpretability/` directory with test infrastructure
  - Configure Hypothesis for property-based testing with 100 minimum iterations
  - Create `configs/interpretability/` directory for configuration files
  - _Requirements: 9.1, 9.6_

- [x] 2. Implement Grad-CAM generator for CNN feature extractors
  - [x] 2.1 Create GradCAMGenerator class with hook registration
    - Implement `__init__` method with model, target_layers, and device parameters
    - Implement forward and backward hook registration for target layers
    - Implement activation and gradient capture logic
    - Support ResNet, DenseNet, and EfficientNet architectures
    - _Requirements: 1.1, 1.2_

  - [x] 2.2 Implement Grad-CAM heatmap generation
    - Implement `generate` method computing gradient-weighted activations
    - Implement weighted combination: CAM = ReLU(Σ(α_k * A_k))
    - Implement bilinear upsampling to input resolution
    - Implement normalization to [0, 1] range
    - _Requirements: 1.4, 1.6_

  - [x]* 2.3 Write property test for Grad-CAM heatmap normalization
    - **Property 1: Grad-CAM Heatmap Normalization**
    - **Validates: Requirements 1.6**
    - Test that all heatmap values are in [0, 1] for any input

  - [x]* 2.4 Write property test for Grad-CAM architecture support
    - **Property 2: Grad-CAM Architecture Support**
    - **Validates: Requirements 1.2**
    - Test that all supported architectures produce valid heatmaps

  - [x]* 2.5 Write property test for Grad-CAM multi-layer output cardinality
    - **Property 3: Grad-CAM Multi-Layer Output Cardinality**
    - **Validates: Requirements 1.5**
    - Test that number of heatmaps equals number of target layers

  - [x] 2.6 Implement heatmap overlay and visualization
    - Implement `overlay_heatmap` method with configurable transparency and colormap
    - Implement `save_visualization` method producing 300+ DPI publication-quality figures
    - Support matplotlib colormaps (jet, viridis, plasma)
    - _Requirements: 1.3, 1.7_

  - [x]* 2.7 Write property test for Grad-CAM overlay validity
    - **Property 4: Grad-CAM Overlay Validity**
    - **Validates: Requirements 1.3**
    - Test that overlay produces valid RGB image with correct shape

  - [x]* 2.8 Write property test for Grad-CAM visualization round-trip
    - **Property 5: Grad-CAM Visualization Round-Trip**
    - **Validates: Requirements 1.8**
    - Test that save/load preserves heatmap values within 1% error

  - [x]* 2.9 Write unit tests for Grad-CAM generator
    - Test each architecture (ResNet18, ResNet50, DenseNet121, EfficientNet-B0)
    - Test edge cases (single pixel heatmap, all-zero gradients)
    - Test GPU and CPU execution
    - _Requirements: 1.1, 1.2, 8.7_

- [x] 3. Extend attention visualizer for new MIL architectures
  - [x] 3.1 Extend AttentionHeatmapGenerator with multi-architecture support
    - Implement `extract_attention_weights` method for AttentionMIL, CLAM, TransMIL
    - Handle multi-branch attention (CLAM positive/negative branches)
    - Implement attention weight normalization with warning logging
    - _Requirements: 2.1, 2.2, 2.8_

  - [ ]* 3.2 Write property test for attention weight extraction completeness
    - **Property 6: Attention Weight Extraction Completeness**
    - **Validates: Requirements 2.1, 2.2**
    - Test that weights are returned for all patches

  - [ ]* 3.3 Write property test for attention weight normalization
    - **Property 7: Attention Weight Normalization**
    - **Validates: Requirements 2.8**
    - Test that normalized weights sum to 1.0 within 1e-6 tolerance

  - [x] 3.4 Implement multi-head attention visualization
    - Implement `visualize_multi_head_attention` method creating grid visualizations
    - Support configurable number of attention heads
    - Generate separate heatmap for each head
    - _Requirements: 2.5_

  - [ ]* 3.5 Write property test for attention multi-head output cardinality
    - **Property 9: Attention Multi-Head Output Cardinality**
    - **Validates: Requirements 2.5**
    - Test that number of visualizations equals number of heads

  - [x] 3.6 Implement architecture comparison visualization
    - Implement `compare_architectures` method for side-by-side comparison
    - Support up to 4 architectures in single visualization
    - Map attention weights to slide coordinates and thumbnails
    - _Requirements: 2.3, 2.4, 2.6_

  - [ ]* 3.7 Write property test for attention top-k selection correctness
    - **Property 8: Attention Top-K Selection Correctness**
    - **Validates: Requirements 2.7**
    - Test that top-k patches have k highest attention values

  - [ ]* 3.8 Write unit tests for attention visualizer extensions
    - Test each MIL architecture (AttentionMIL, CLAM, TransMIL)
    - Test multi-head visualization with 1, 2, 4, 8 heads
    - Test architecture comparison with 2, 3, 4 models
    - _Requirements: 2.1, 2.2, 2.5, 2.6_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement failure analyzer for systematic error analysis
  - [ ] 5.1 Create FailureAnalyzer class with clustering support
    - Implement `__init__` with clustering_method, n_clusters, embedding_dim parameters
    - Support k-means, DBSCAN, and hierarchical clustering algorithms
    - Implement `identify_failures` method extracting misclassified samples
    - _Requirements: 3.1, 3.2_

  - [ ]* 5.2 Write property test for failure identification with confidence
    - **Property 10: Failure Identification with Confidence**
    - **Validates: Requirements 3.1, 3.2**
    - Test that all failures have associated confidence scores

  - [ ] 5.3 Implement failure clustering and analysis
    - Implement `cluster_failures` method using feature embeddings
    - Implement `analyze_cluster_characteristics` computing cluster statistics
    - Implement `identify_systematic_biases` analyzing failure distribution across subgroups
    - _Requirements: 3.3, 3.4, 3.5, 3.7_

  - [ ]* 5.4 Write property test for failure clustering completeness
    - **Property 11: Failure Clustering Completeness**
    - **Validates: Requirements 3.3, 3.4, 3.5**
    - Test that each failure is assigned to exactly one cluster

  - [ ]* 5.5 Write property test for systematic bias analysis completeness
    - **Property 13: Systematic Bias Analysis Completeness**
    - **Validates: Requirements 3.7**
    - Test that bias metrics are computed for all subgroups

  - [ ] 5.6 Implement failure report export
    - Implement `export_failure_report` method generating CSV with all required columns
    - Include slide_id, prediction, ground_truth, confidence, cluster_assignment
    - Handle edge case of zero failures (empty report with informational log)
    - _Requirements: 3.6, 3.8_

  - [ ]* 5.7 Write property test for failure CSV export completeness
    - **Property 12: Failure CSV Export Completeness**
    - **Validates: Requirements 3.6**
    - Test that CSV contains all required columns

  - [ ]* 5.8 Write unit tests for failure analyzer
    - Test each clustering algorithm (k-means, DBSCAN, hierarchical)
    - Test edge case of zero failures
    - Test systematic bias identification across clinical subgroups
    - _Requirements: 3.1, 3.3, 3.7, 3.8_

- [ ] 6. Implement feature importance calculator for clinical data
  - [ ] 6.1 Create FeatureImportanceCalculator class with multiple methods
    - Implement `__init__` with model, method, and device parameters
    - Support permutation, SHAP, and gradient-based attribution methods
    - Implement `compute_permutation_importance` with configurable n_repeats
    - _Requirements: 4.1, 4.2_

  - [ ]* 6.2 Write property test for feature importance method support
    - **Property 15: Feature Importance Method Support**
    - **Validates: Requirements 4.1, 4.2**
    - Test that all methods produce scores for all features

  - [ ] 6.3 Implement SHAP and gradient-based importance
    - Implement `compute_shap_values` using SHAP library
    - Implement `compute_gradient_importance` using PyTorch autograd
    - Implement normalization to [0, 1] range with sum=1.0
    - _Requirements: 4.2, 4.3_

  - [ ]* 6.4 Write property test for feature importance score normalization
    - **Property 14: Feature Importance Score Normalization**
    - **Validates: Requirements 4.3, 4.8**
    - Test that scores are in [0, 1] and sum to 1.0 within 1e-6 tolerance

  - [ ] 6.5 Implement feature ranking and confidence intervals
    - Implement `rank_features` method sorting by importance
    - Implement `compute_confidence_intervals` using bootstrap sampling
    - Support configurable confidence level (default 0.95)
    - _Requirements: 4.4, 4.6_

  - [ ]* 6.6 Write property test for feature importance top-k ranking
    - **Property 16: Feature Importance Top-K Ranking**
    - **Validates: Requirements 4.4**
    - Test that top-k features have k highest scores in descending order

  - [ ] 6.7 Implement feature importance visualization and export
    - Implement `visualize_importance` creating bar plots with optional confidence intervals
    - Implement `export_importance_scores` generating CSV with feature names and scores
    - _Requirements: 4.5, 4.7_

  - [ ]* 6.8 Write property test for feature importance CSV export completeness
    - **Property 17: Feature Importance CSV Export Completeness**
    - **Validates: Requirements 4.7**
    - Test that CSV contains all features with scores

  - [ ]* 6.9 Write unit tests for feature importance calculator
    - Test each method (permutation, SHAP, gradient)
    - Test confidence interval computation with different confidence levels
    - Test visualization generation with and without confidence intervals
    - _Requirements: 4.1, 4.2, 4.6_

- [ ] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement configuration parsers and pretty printers
  - [ ] 8.1 Create configuration data models
    - Create `src/interpretability/config.py`
    - Implement GradCAMConfig dataclass with validation method
    - Implement AttentionConfig dataclass
    - Implement AttentionData dataclass for HDF5 serialization
    - _Requirements: 5.1, 6.1_

  - [ ] 8.2 Implement Grad-CAM configuration parser
    - Implement GradCAMParser.parse method converting dict to GradCAMConfig
    - Implement validation for target layers (check existence in model)
    - Implement validation for transparency values in [0, 1]
    - Return descriptive error messages specifying invalid fields
    - _Requirements: 5.1, 5.2, 5.5, 5.6_

  - [ ]* 8.3 Write property test for Grad-CAM config validation
    - **Property 19: Grad-CAM Config Validation**
    - **Validates: Requirements 5.2, 5.5, 5.6**
    - Test that invalid configs are rejected with descriptive errors

  - [ ] 8.4 Implement Grad-CAM configuration pretty printer
    - Implement GradCAMPrettyPrinter.format method converting GradCAMConfig to dict
    - Ensure consistent field ordering and indentation
    - _Requirements: 5.3, 5.7_

  - [ ]* 8.5 Write property test for Grad-CAM config round-trip
    - **Property 18: Grad-CAM Config Round-Trip**
    - **Validates: Requirements 5.4**
    - Test that parse(pretty_print(config)) produces equivalent object

  - [ ] 8.6 Implement attention data parser
    - Implement AttentionParser.parse method reading HDF5 files
    - Implement validation for non-negative attention weights
    - Implement validation for coordinates within slide dimensions
    - Return descriptive error messages for invalid data
    - _Requirements: 6.1, 6.2, 6.5, 6.6_

  - [ ]* 8.7 Write property test for attention data validation
    - **Property 21: Attention Data Validation**
    - **Validates: Requirements 6.2, 6.5, 6.6**
    - Test that invalid data is rejected with descriptive errors

  - [ ] 8.8 Implement attention data pretty printer
    - Implement AttentionPrettyPrinter.format method writing HDF5 files
    - Use gzip compression level 4
    - Store attention weights, coordinates, slide_id, architecture
    - _Requirements: 6.3, 6.7_

  - [ ]* 8.9 Write property test for attention data round-trip
    - **Property 20: Attention Data Round-Trip**
    - **Validates: Requirements 6.4**
    - Test that parse(pretty_print(data)) preserves weights within 1e-6 tolerance

  - [ ]* 8.10 Write unit tests for configuration parsers
    - Test valid and invalid Grad-CAM configurations
    - Test valid and invalid attention data
    - Test error message clarity and specificity
    - _Requirements: 5.2, 6.2_

- [ ] 9. Implement interactive visualization dashboard
  - [ ] 9.1 Create Flask application structure
    - Create `src/interpretability/dashboard.py`
    - Implement InterpretabilityDashboard class with Flask app
    - Set up routes: /, /api/samples, /api/sample/<id>, /api/filter, /api/compare, /api/export
    - Implement caching infrastructure (in-memory with optional Redis)
    - _Requirements: 7.1, 7.7_

  - [ ] 9.2 Implement sample loading and filtering
    - Implement `load_sample` method loading Grad-CAM, attention, prediction, confidence
    - Implement `filter_samples` method with confidence range, correctness, clinical filters
    - Support keyboard navigation for efficient browsing
    - _Requirements: 7.2, 7.3, 7.6_

  - [ ]* 9.3 Write property test for dashboard sample filtering
    - **Property 22: Dashboard Sample Filtering**
    - **Validates: Requirements 7.3**
    - Test that returned samples match all filter criteria

  - [ ] 9.4 Implement sample comparison and export
    - Implement `compare_samples` method for side-by-side comparison (max 4 samples)
    - Implement `export_visualization` method generating publication-quality figures
    - Support PNG, PDF, SVG formats with configurable DPI
    - _Requirements: 7.4, 7.5_

  - [ ]* 9.5 Write property test for dashboard sample comparison cardinality
    - **Property 23: Dashboard Sample Comparison Cardinality**
    - **Validates: Requirements 7.4**
    - Test that comparison displays all specified samples (1-4)

  - [ ]* 9.6 Write property test for dashboard visualization caching
    - **Property 24: Dashboard Visualization Caching**
    - **Validates: Requirements 7.7**
    - Test that second load time ≤ first load time

  - [ ] 9.7 Create dashboard frontend
    - Create HTML templates with Plotly.js for interactive visualizations
    - Implement CSS styling for responsive layout
    - Implement JavaScript for API calls and dynamic updates
    - _Requirements: 7.1_

  - [ ]* 9.8 Write integration tests for dashboard
    - Test all API endpoints return correct status codes
    - Test filtering with various criteria combinations
    - Test comparison with 1, 2, 3, 4 samples
    - Test export in PNG, PDF, SVG formats
    - Test dashboard loads within 3 seconds
    - _Requirements: 7.1, 7.3, 7.4, 7.5, 7.8_

- [ ] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Integrate with existing evaluation scripts
  - [ ] 11.1 Add interpretability CLI flags to evaluation scripts
    - Add `--enable-gradcam` flag with target layer specification
    - Add `--enable-attention-viz` flag with architecture specification
    - Add `--enable-failure-analysis` flag with clustering parameters
    - Add `--enable-feature-importance` flag with method specification
    - Add `--interpretability-output-dir` flag for output directory configuration
    - _Requirements: 9.1, 9.2, 9.3_

  - [ ]* 11.2 Write property test for output directory configuration
    - **Property 27: Output Directory Configuration**
    - **Validates: Requirements 9.3**
    - Test that all visualizations are saved to specified directory

  - [ ] 11.3 Implement automatic visualization generation during evaluation
    - Hook into evaluation loop to generate visualizations
    - Preserve existing evaluation metrics and outputs
    - Log interpretability operations to existing logging system
    - _Requirements: 9.2, 9.5, 9.7_

  - [ ] 11.4 Add support for PCam and Camelyon datasets
    - Implement patch-level interpretability for PCam
    - Implement slide-level interpretability for Camelyon
    - Handle different coordinate systems and resolutions
    - _Requirements: 9.4_

  - [ ]* 11.5 Write property test for dataset type support
    - **Property 28: Dataset Type Support**
    - **Validates: Requirements 9.4**
    - Test that both PCam and Camelyon datasets are processed correctly

  - [ ]* 11.6 Write integration tests for evaluation script integration
    - Test CLI flags are parsed correctly
    - Test visualizations are generated during evaluation
    - Test existing metrics are preserved
    - Test logging integration
    - _Requirements: 9.1, 9.2, 9.5, 9.7_

- [ ] 12. Implement batch processing and GPU acceleration
  - [ ] 12.1 Add batch processing support to all components
    - Implement batch processing in GradCAMGenerator
    - Implement batch processing in AttentionHeatmapGenerator
    - Implement batch processing in FailureAnalyzer
    - Implement batch processing in FeatureImportanceCalculator
    - _Requirements: 8.3_

  - [ ]* 12.2 Write property test for batch processing completeness
    - **Property 25: Batch Processing Completeness**
    - **Validates: Requirements 8.3**
    - Test that all samples in batch are processed successfully

  - [ ] 12.3 Implement GPU acceleration with automatic fallback
    - Implement GPU device detection and placement
    - Implement automatic CPU fallback when GPU unavailable
    - Log warnings for resource constraints
    - _Requirements: 8.7_

  - [ ]* 12.4 Write property test for GPU acceleration support
    - **Property 26: GPU Acceleration Support**
    - **Validates: Requirements 8.7**
    - Test that operations execute on GPU when available

  - [ ] 12.5 Add progress indicators for long operations
    - Implement progress bars using tqdm
    - Show progress for operations exceeding 1 second
    - Display estimated time remaining
    - _Requirements: 8.5_

  - [ ]* 12.6 Write performance tests
    - Test Grad-CAM < 200ms per patch on GPU
    - Test attention extraction < 100ms per slide on GPU
    - Test feature importance < 5 seconds per model on CPU
    - Test dashboard loads < 3 seconds
    - _Requirements: 8.1, 8.2, 8.4, 7.8_

- [ ] 13. Create configuration files and examples
  - [ ] 13.1 Create default configuration files
    - Create `configs/interpretability/gradcam_default.yaml`
    - Create `configs/interpretability/attention_default.yaml`
    - Create `configs/interpretability/dashboard_default.yaml`
    - Include common use case configurations
    - _Requirements: 9.6, 10.4_

  - [ ] 13.2 Create Jupyter notebook examples
    - Create `examples/interpretability/gradcam_example.ipynb` demonstrating Grad-CAM usage
    - Create `examples/interpretability/attention_example.ipynb` demonstrating attention visualization
    - Create `examples/interpretability/failure_analysis_example.ipynb` demonstrating failure analysis
    - Create `examples/interpretability/feature_importance_example.ipynb` demonstrating feature importance
    - Include example outputs showing expected visualization formats
    - _Requirements: 10.2, 10.7_

  - [ ] 13.3 Create documentation
    - Write API documentation for all public classes and methods
    - Write command-line usage examples in README
    - Document computational requirements and expected performance
    - Create troubleshooting guide for common issues
    - _Requirements: 10.1, 10.3, 10.5, 10.6_

- [ ] 14. Final checkpoint - Ensure all tests pass and documentation is complete
  - Ensure all tests pass, ask the user if questions arise.
  - Verify all property-based tests reference design document properties
  - Verify all unit tests cover edge cases
  - Verify all integration tests validate end-to-end workflows
  - Verify documentation is complete and accurate

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at reasonable breaks
- Property tests validate universal correctness properties from design document
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- Performance tests validate computational efficiency requirements
- All code examples use Python with PyTorch, NumPy, Flask, and Hypothesis
- GPU acceleration is supported with automatic CPU fallback
- Dashboard uses Flask backend with Plotly.js frontend
- Configuration uses YAML files with validation
- All visualizations are publication-quality (300+ DPI)
