# Requirements Document: Model Interpretability

## Introduction

This document specifies requirements for adding interpretability tools to the computational pathology research framework. The feature enables researchers and clinicians to understand what deep learning models learn from histopathology images and why they make specific predictions. This is critical for clinical trust, debugging model failures, research publications, and regulatory compliance.

The interpretability system will provide Grad-CAM visualizations, attention weight analysis, failure case identification, feature importance computation, and interactive visualization capabilities for existing model architectures (ResNet, DenseNet, EfficientNet) with attention-based MIL aggregation.

## Glossary

- **Interpretability_System**: The complete model interpretability feature including Grad-CAM, attention visualization, and failure analysis
- **Grad_CAM_Generator**: Component that generates Gradient-weighted Class Activation Mapping visualizations
- **Attention_Visualizer**: Component that visualizes attention weights from MIL models
- **Failure_Analyzer**: Component that identifies and analyzes misclassified samples
- **Feature_Importance_Calculator**: Component that computes importance scores for clinical features
- **Visualization_Dashboard**: Interactive interface for exploring model decisions
- **CNN_Feature_Extractor**: Convolutional neural network models (ResNet, DenseNet, EfficientNet) that extract patch features
- **MIL_Model**: Multiple Instance Learning model that aggregates patch features for slide-level predictions
- **Heatmap**: Spatial visualization showing which image regions influence predictions
- **Attention_Weights**: Learned importance scores assigned to patches by MIL models
- **Failure_Case**: Sample where model prediction does not match ground truth label
- **Clinical_Features**: Patient metadata including age, tumor size, and other clinical variables
- **Publication_Quality_Figure**: Visualization meeting academic journal standards (300+ DPI, vector graphics)
- **Inference_Time**: Time required to generate predictions for a single sample

## Requirements

### Requirement 1: Grad-CAM Visualization for CNN Feature Extractors

**User Story:** As a researcher, I want to see Grad-CAM heatmaps overlaid on histopathology patches, so that I can understand which tissue regions drive model predictions.

#### Acceptance Criteria

1. WHEN a trained CNN_Feature_Extractor and input patch are provided, THE Grad_CAM_Generator SHALL compute gradient-weighted class activation maps
2. THE Grad_CAM_Generator SHALL support ResNet, DenseNet, and EfficientNet architectures
3. WHEN generating Grad-CAM visualizations, THE Grad_CAM_Generator SHALL overlay heatmaps on original patches with configurable transparency
4. THE Grad_CAM_Generator SHALL produce heatmaps at the same spatial resolution as the target convolutional layer
5. WHEN multiple target layers are specified, THE Grad_CAM_Generator SHALL generate separate heatmaps for each layer
6. THE Grad_CAM_Generator SHALL normalize heatmap values to the range [0, 1]
7. WHEN saving visualizations, THE Grad_CAM_Generator SHALL produce Publication_Quality_Figures with resolution of at least 300 DPI
8. FOR ALL valid input patches and trained models, generating then saving then loading a Grad-CAM visualization SHALL preserve the heatmap values within 1% relative error (round-trip property)

### Requirement 2: Attention Weight Visualization for MIL Models

**User Story:** As a pathologist, I want to see attention weights on slide-level predictions, so that I know which patches the model focused on.

#### Acceptance Criteria

1. WHEN a trained MIL_Model generates predictions, THE Attention_Visualizer SHALL extract Attention_Weights for all patches
2. THE Attention_Visualizer SHALL support AttentionMIL, CLAM, and TransMIL architectures
3. WHEN visualizing attention weights, THE Attention_Visualizer SHALL create spatial heatmaps showing patch importance
4. THE Attention_Visualizer SHALL map attention weights to patch coordinates on slide thumbnails
5. WHEN multiple attention heads exist, THE Attention_Visualizer SHALL generate separate visualizations for each head
6. THE Attention_Visualizer SHALL provide side-by-side comparison of attention patterns across different model architectures
7. THE Attention_Visualizer SHALL highlight the top-k highest attention patches with configurable k values
8. WHEN attention weights sum to a value other than 1.0, THE Attention_Visualizer SHALL normalize weights and log a warning message

### Requirement 3: Failure Case Analysis and Clustering

**User Story:** As a developer, I want to analyze failure cases to identify model weaknesses and biases, so that I can improve model performance.

#### Acceptance Criteria

1. WHEN evaluation results are provided, THE Failure_Analyzer SHALL identify all Failure_Cases where predictions do not match ground truth
2. THE Failure_Analyzer SHALL compute confidence scores for each Failure_Case
3. WHEN clustering failure cases, THE Failure_Analyzer SHALL group similar failures using feature embeddings
4. THE Failure_Analyzer SHALL generate cluster visualizations showing failure patterns
5. THE Failure_Analyzer SHALL compute statistics for each cluster including count, average confidence, and common characteristics
6. WHEN exporting failure analysis, THE Failure_Analyzer SHALL produce CSV files with slide IDs, predictions, ground truth, confidence scores, and cluster assignments
7. THE Failure_Analyzer SHALL identify systematic biases by analyzing failure distribution across clinical subgroups
8. WHEN no Failure_Cases exist, THE Failure_Analyzer SHALL return an empty report and log an informational message

### Requirement 4: Feature Importance for Clinical Data

**User Story:** As a clinical researcher, I want to see which clinical features contribute most to survival predictions, so that I can understand prognostic factors.

#### Acceptance Criteria

1. WHEN a trained model uses Clinical_Features, THE Feature_Importance_Calculator SHALL compute importance scores for each feature
2. THE Feature_Importance_Calculator SHALL support permutation importance, SHAP values, and gradient-based attribution methods
3. WHEN computing feature importance, THE Feature_Importance_Calculator SHALL generate scores normalized to the range [0, 1]
4. THE Feature_Importance_Calculator SHALL rank features by importance and identify the top-k most important features
5. THE Feature_Importance_Calculator SHALL generate bar plots showing feature importance scores
6. WHEN confidence intervals are requested, THE Feature_Importance_Calculator SHALL compute bootstrap confidence intervals with configurable sample count
7. THE Feature_Importance_Calculator SHALL export feature importance scores to CSV format with feature names and scores
8. FOR ALL valid feature sets, computing importance scores SHALL produce values that sum to 1.0 within numerical precision tolerance of 1e-6 (invariant property)

### Requirement 5: Grad-CAM Parser and Pretty Printer

**User Story:** As a developer, I want to parse and serialize Grad-CAM configurations, so that I can save and load visualization settings.

#### Acceptance Criteria

1. WHEN a Grad-CAM configuration dictionary is provided, THE Grad_CAM_Parser SHALL parse it into a GradCAMConfig object
2. WHEN an invalid configuration is provided, THE Grad_CAM_Parser SHALL return a descriptive error message specifying which field is invalid
3. THE Grad_CAM_Pretty_Printer SHALL format GradCAMConfig objects into valid configuration dictionaries
4. FOR ALL valid GradCAMConfig objects, parsing then printing then parsing SHALL produce an equivalent object (round-trip property)
5. THE Grad_CAM_Parser SHALL validate that target layer names exist in the specified model architecture
6. THE Grad_CAM_Parser SHALL validate that transparency values are in the range [0, 1]
7. THE Grad_CAM_Pretty_Printer SHALL format configuration dictionaries with consistent indentation and field ordering

### Requirement 6: Attention Weight Parser and Pretty Printer

**User Story:** As a developer, I want to parse and serialize attention weight data, so that I can save and load attention visualizations.

#### Acceptance Criteria

1. WHEN attention weight HDF5 files are provided, THE Attention_Parser SHALL parse them into AttentionData objects
2. WHEN an invalid HDF5 file is provided, THE Attention_Parser SHALL return a descriptive error message
3. THE Attention_Pretty_Printer SHALL format AttentionData objects into valid HDF5 files
4. FOR ALL valid AttentionData objects, parsing then printing then parsing SHALL produce equivalent attention weights within numerical precision tolerance of 1e-6 (round-trip property)
5. THE Attention_Parser SHALL validate that attention weights are non-negative
6. THE Attention_Parser SHALL validate that patch coordinates are within valid slide dimensions
7. THE Attention_Pretty_Printer SHALL compress HDF5 files using gzip compression level 4

### Requirement 7: Interactive Visualization Dashboard

**User Story:** As a researcher, I want an interactive dashboard to explore model decisions, so that I can efficiently analyze multiple samples.

#### Acceptance Criteria

1. THE Visualization_Dashboard SHALL provide a web-based interface for exploring interpretability results
2. WHEN a slide is selected, THE Visualization_Dashboard SHALL display Grad-CAM heatmaps, attention weights, and prediction confidence
3. THE Visualization_Dashboard SHALL support filtering samples by prediction confidence, correctness, and clinical attributes
4. THE Visualization_Dashboard SHALL provide side-by-side comparison of up to 4 samples
5. WHEN exporting visualizations, THE Visualization_Dashboard SHALL generate Publication_Quality_Figures
6. THE Visualization_Dashboard SHALL support keyboard navigation for efficient browsing
7. THE Visualization_Dashboard SHALL cache visualizations to reduce loading time for previously viewed samples
8. WHEN the dashboard is accessed, THE Visualization_Dashboard SHALL load within 3 seconds on standard hardware

### Requirement 8: Computational Efficiency

**User Story:** As a researcher, I want interpretability tools to be computationally efficient, so that they do not significantly slow down my analysis workflow.

#### Acceptance Criteria

1. WHEN generating Grad-CAM visualizations, THE Grad_CAM_Generator SHALL complete processing within 200 milliseconds per patch on GPU hardware
2. WHEN extracting attention weights, THE Attention_Visualizer SHALL complete processing within 100 milliseconds per slide on GPU hardware
3. THE Interpretability_System SHALL support batch processing of multiple samples to amortize overhead costs
4. WHEN computing feature importance, THE Feature_Importance_Calculator SHALL complete processing within 5 seconds per model on CPU hardware
5. THE Interpretability_System SHALL provide progress indicators for operations exceeding 1 second
6. WHEN generating visualizations, THE Interpretability_System SHALL increase Inference_Time by no more than 20% compared to standard inference
7. THE Interpretability_System SHALL support GPU acceleration for all computationally intensive operations

### Requirement 9: Integration with Existing Evaluation Scripts

**User Story:** As a developer, I want interpretability tools to integrate with existing evaluation scripts, so that I can generate visualizations during standard evaluation workflows.

#### Acceptance Criteria

1. THE Interpretability_System SHALL provide command-line interfaces compatible with existing evaluation scripts
2. WHEN evaluation scripts are executed with interpretability flags, THE Interpretability_System SHALL generate visualizations automatically
3. THE Interpretability_System SHALL save visualizations to configurable output directories
4. THE Interpretability_System SHALL support both patch-level (PCam) and slide-level (Camelyon) datasets
5. WHEN integrated with evaluation scripts, THE Interpretability_System SHALL preserve existing evaluation metrics and outputs
6. THE Interpretability_System SHALL provide configuration files for common interpretability workflows
7. THE Interpretability_System SHALL log interpretability operations to the same logging system used by evaluation scripts

### Requirement 10: Documentation and Examples

**User Story:** As a new user, I want comprehensive documentation and examples, so that I can quickly learn how to use interpretability tools.

#### Acceptance Criteria

1. THE Interpretability_System SHALL provide API documentation for all public classes and methods
2. THE Interpretability_System SHALL include Jupyter notebook examples demonstrating Grad-CAM, attention visualization, and failure analysis
3. THE Interpretability_System SHALL provide command-line usage examples in README files
4. THE Interpretability_System SHALL include example configuration files for common use cases
5. THE Interpretability_System SHALL document computational requirements and expected performance
6. THE Interpretability_System SHALL provide troubleshooting guides for common issues
7. THE Interpretability_System SHALL include example outputs showing expected visualization formats
