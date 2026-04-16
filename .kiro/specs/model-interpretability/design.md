# Design Document: Model Interpretability

## Overview

This design document specifies the architecture and implementation approach for adding comprehensive interpretability tools to the computational pathology research framework. The interpretability system will enable researchers and clinicians to understand deep learning model decisions through Grad-CAM visualizations, attention weight analysis, failure case identification, and feature importance computation.

The system extends the existing attention heatmap visualization capabilities (already implemented in `src/visualization/attention_heatmap.py`) with new Grad-CAM support for CNN feature extractors, systematic failure analysis, clinical feature importance calculation, and an interactive dashboard for exploring model decisions.

### Key Design Principles

1. **Modularity**: Each interpretability component (Grad-CAM, attention, failure analysis, feature importance) is implemented as an independent module with clear interfaces
2. **Extensibility**: Support for multiple CNN architectures (ResNet, DenseNet, EfficientNet) and MIL models (AttentionMIL, CLAM, TransMIL) through polymorphic design
3. **Efficiency**: GPU acceleration for computationally intensive operations, batch processing support, and caching for interactive workflows
4. **Integration**: Seamless integration with existing evaluation scripts and model architectures without breaking changes
5. **Reproducibility**: Serializable configurations and deterministic outputs for research reproducibility

## Architecture

### System Components

The interpretability system consists of five main components:

```
┌─────────────────────────────────────────────────────────────┐
│                  Interpretability System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Grad-CAM        │  │  Attention       │                │
│  │  Generator       │  │  Visualizer      │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │                     │                            │
│           └──────────┬──────────┘                            │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │  Visualization      │                           │
│           │  Dashboard          │                           │
│           └──────────┬──────────┘                           │
│                      │                                       │
│  ┌──────────────────┴──────────────────┐                   │
│  │                                       │                   │
│  ▼                                       ▼                   │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Failure         │  │  Feature         │                │
│  │  Analyzer        │  │  Importance      │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

1. **Grad-CAM Generator** → Produces patch-level heatmaps for CNN feature extractors
2. **Attention Visualizer** → Extends existing `AttentionHeatmapGenerator` for MIL models
3. **Failure Analyzer** → Identifies and clusters misclassified samples using embeddings
4. **Feature Importance Calculator** → Computes importance scores for clinical features
5. **Visualization Dashboard** → Web-based interface integrating all components

### Data Flow

```
Input Models & Data
        │
        ├─→ CNN Feature Extractor ─→ Grad-CAM Generator ─→ Patch Heatmaps
        │
        ├─→ MIL Model ─→ Attention Visualizer ─→ Slide Heatmaps
        │
        ├─→ Evaluation Results ─→ Failure Analyzer ─→ Failure Clusters
        │
        └─→ Clinical Features ─→ Feature Importance ─→ Importance Scores
                                                              │
                                                              ▼
                                                   Visualization Dashboard
```

## Components and Interfaces

### 1. Grad-CAM Generator

**Purpose**: Generate gradient-weighted class activation maps for CNN feature extractors to visualize which spatial regions in patches influence predictions.

**Location**: `src/interpretability/gradcam.py`

**Key Classes**:

```python
class GradCAMGenerator:
    """Generate Grad-CAM visualizations for CNN models.
    
    Supports ResNet, DenseNet, and EfficientNet architectures.
    Computes gradient-weighted activations at specified convolutional layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        device: str = "cuda"
    ):
        """Initialize Grad-CAM generator.
        
        Args:
            model: CNN feature extractor (ResNet, DenseNet, EfficientNet)
            target_layers: List of layer names to generate CAMs for
            device: Device for computation ('cuda' or 'cpu')
        """
        
    def generate(
        self,
        images: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate Grad-CAM heatmaps for input images.
        
        Args:
            images: [batch, 3, H, W] input patches
            class_idx: Target class index (None for predicted class)
            
        Returns:
            Dictionary mapping layer names to heatmaps [batch, H, W]
        """
        
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> np.ndarray:
        """Overlay heatmap on original image.
        
        Args:
            image: Original image [H, W, 3]
            heatmap: Grad-CAM heatmap [H, W]
            alpha: Transparency (0=transparent, 1=opaque)
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image [H, W, 3]
        """
        
    def save_visualization(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        output_path: Path,
        dpi: int = 300
    ) -> Path:
        """Save Grad-CAM visualization to file.
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            output_path: Output file path
            dpi: Resolution (default 300 for publication quality)
            
        Returns:
            Path to saved visualization
        """
```

**Architecture-Specific Target Layers**:

- **ResNet**: `layer4` (final convolutional block)
- **DenseNet**: `features.denseblock4` (final dense block)
- **EfficientNet**: `features.8` (final MBConv block)

**Implementation Strategy**:

1. Register forward and backward hooks on target layers
2. During forward pass, capture activations
3. During backward pass, capture gradients with respect to target class
4. Compute weighted combination: `CAM = ReLU(Σ(α_k * A_k))` where `α_k = global_avg_pool(∂y/∂A_k)`
5. Upsample CAM to input resolution using bilinear interpolation
6. Normalize to [0, 1] range

### 2. Attention Visualizer (Extension)

**Purpose**: Extend existing `AttentionHeatmapGenerator` to support new MIL architectures and multi-head attention visualization.

**Location**: `src/visualization/attention_heatmap.py` (extend existing class)

**New Methods**:

```python
class AttentionHeatmapGenerator:
    # ... existing methods ...
    
    def extract_attention_weights(
        self,
        model: AttentionMILBase,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights from MIL model.
        
        Supports AttentionMIL, CLAM (multi-branch), and TransMIL.
        
        Args:
            model: MIL model instance
            features: Patch features [batch, num_patches, feature_dim]
            num_patches: Valid patch counts [batch]
            
        Returns:
            Dictionary with attention weights:
                - AttentionMIL: {'attention': [batch, num_patches]}
                - CLAM: {'positive': [batch, num_patches], 'negative': [batch, num_patches]}
                - TransMIL: {'attention': [batch, num_patches]} (uniform for API compatibility)
        """
        
    def visualize_multi_head_attention(
        self,
        slide_id: str,
        attention_heads: Dict[int, np.ndarray],
        coordinates: np.ndarray,
        thumbnail_path: Optional[Path] = None
    ) -> Path:
        """Visualize attention patterns from multiple attention heads.
        
        Creates a grid visualization showing each attention head separately.
        
        Args:
            slide_id: Slide identifier
            attention_heads: Dictionary mapping head index to attention weights
            coordinates: Patch coordinates [num_patches, 2]
            thumbnail_path: Optional slide thumbnail
            
        Returns:
            Path to saved multi-head visualization
        """
        
    def compare_architectures(
        self,
        slide_id: str,
        architecture_attentions: Dict[str, np.ndarray],
        coordinates: np.ndarray,
        thumbnail_path: Optional[Path] = None
    ) -> Path:
        """Compare attention patterns across different MIL architectures.
        
        Creates side-by-side visualization of attention from different models.
        
        Args:
            slide_id: Slide identifier
            architecture_attentions: Dict mapping architecture name to attention weights
            coordinates: Patch coordinates
            thumbnail_path: Optional slide thumbnail
            
        Returns:
            Path to saved comparison visualization
        """
```

### 3. Failure Analyzer

**Purpose**: Identify and analyze failure cases to reveal systematic model weaknesses and biases.

**Location**: `src/interpretability/failure_analysis.py`

**Key Classes**:

```python
class FailureAnalyzer:
    """Analyze model failure cases and identify patterns.
    
    Clusters failures using feature embeddings and identifies
    systematic biases across clinical subgroups.
    """
    
    def __init__(
        self,
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
        embedding_dim: int = 256
    ):
        """Initialize failure analyzer.
        
        Args:
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters for k-means
            embedding_dim: Dimension of feature embeddings
        """
        
    def identify_failures(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        confidence_scores: np.ndarray,
        slide_ids: List[str]
    ) -> pd.DataFrame:
        """Identify failure cases from evaluation results.
        
        Args:
            predictions: Predicted labels [num_samples]
            ground_truth: True labels [num_samples]
            confidence_scores: Prediction confidence [num_samples]
            slide_ids: Slide identifiers
            
        Returns:
            DataFrame with columns: slide_id, prediction, ground_truth,
            confidence, is_failure
        """
        
    def cluster_failures(
        self,
        failure_embeddings: np.ndarray,
        failure_metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """Cluster failure cases using embeddings.
        
        Args:
            failure_embeddings: Feature embeddings [num_failures, embedding_dim]
            failure_metadata: Metadata for failure cases
            
        Returns:
            DataFrame with added 'cluster_id' column
        """
        
    def analyze_cluster_characteristics(
        self,
        clustered_failures: pd.DataFrame,
        clinical_features: Optional[pd.DataFrame] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of each failure cluster.
        
        Args:
            clustered_failures: DataFrame with cluster assignments
            clinical_features: Optional clinical metadata
            
        Returns:
            Dictionary mapping cluster_id to statistics:
                - count: Number of failures in cluster
                - avg_confidence: Average prediction confidence
                - common_characteristics: Most common clinical features
                - representative_samples: Sample slide IDs
        """
        
    def identify_systematic_biases(
        self,
        failures: pd.DataFrame,
        clinical_subgroups: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Identify systematic biases across clinical subgroups.
        
        Args:
            failures: DataFrame with failure cases
            clinical_subgroups: Dictionary mapping subgroup name to slide IDs
            
        Returns:
            Dictionary mapping subgroup name to failure rate
        """
        
    def export_failure_report(
        self,
        failures: pd.DataFrame,
        cluster_stats: Dict[int, Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """Export comprehensive failure analysis report.
        
        Args:
            failures: DataFrame with failure cases and clusters
            cluster_stats: Cluster statistics
            output_path: Output CSV path
            
        Returns:
            Path to saved report
        """
```

### 4. Feature Importance Calculator

**Purpose**: Compute importance scores for clinical features in multimodal models.

**Location**: `src/interpretability/feature_importance.py`

**Key Classes**:

```python
class FeatureImportanceCalculator:
    """Calculate feature importance for clinical data.
    
    Supports permutation importance, SHAP values, and gradient-based attribution.
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "permutation",
        device: str = "cuda"
    ):
        """Initialize feature importance calculator.
        
        Args:
            model: Trained model with clinical features
            method: Importance method ('permutation', 'shap', 'gradient')
            device: Device for computation
        """
        
    def compute_permutation_importance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Compute permutation importance scores.
        
        Args:
            features: Clinical features [num_samples, num_features]
            labels: Ground truth labels [num_samples]
            feature_names: Names of features
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        
    def compute_shap_values(
        self,
        features: np.ndarray,
        feature_names: List[str],
        background_samples: int = 100
    ) -> Dict[str, float]:
        """Compute SHAP (SHapley Additive exPlanations) values.
        
        Args:
            features: Clinical features [num_samples, num_features]
            feature_names: Names of features
            background_samples: Number of background samples for SHAP
            
        Returns:
            Dictionary mapping feature names to SHAP importance scores
        """
        
    def compute_gradient_importance(
        self,
        features: torch.Tensor,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute gradient-based feature importance.
        
        Args:
            features: Clinical features [num_samples, num_features]
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to gradient-based importance
        """
        
    def rank_features(
        self,
        importance_scores: Dict[str, float],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Rank features by importance.
        
        Args:
            importance_scores: Feature importance scores
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, score) tuples sorted by importance
        """
        
    def compute_confidence_intervals(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float, float]]:
        """Compute bootstrap confidence intervals for importance scores.
        
        Args:
            features: Clinical features
            labels: Ground truth labels
            feature_names: Names of features
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Dictionary mapping feature names to (mean, lower_ci, upper_ci)
        """
        
    def visualize_importance(
        self,
        importance_scores: Dict[str, float],
        output_path: Path,
        confidence_intervals: Optional[Dict[str, Tuple[float, float, float]]] = None
    ) -> Path:
        """Create bar plot of feature importance.
        
        Args:
            importance_scores: Feature importance scores
            output_path: Output file path
            confidence_intervals: Optional confidence intervals
            
        Returns:
            Path to saved visualization
        """
        
    def export_importance_scores(
        self,
        importance_scores: Dict[str, float],
        output_path: Path,
        confidence_intervals: Optional[Dict[str, Tuple[float, float, float]]] = None
    ) -> Path:
        """Export importance scores to CSV.
        
        Args:
            importance_scores: Feature importance scores
            output_path: Output CSV path
            confidence_intervals: Optional confidence intervals
            
        Returns:
            Path to saved CSV
        """
```

### 5. Configuration Parsers and Pretty Printers

**Purpose**: Parse and serialize interpretability configurations for reproducibility.

**Location**: `src/interpretability/config.py`

**Key Classes**:

```python
@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM visualization."""
    model_name: str  # 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0'
    target_layers: List[str]
    transparency: float = 0.5
    colormap: str = "jet"
    dpi: int = 300
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        

@dataclass
class AttentionConfig:
    """Configuration for attention visualization."""
    architecture: str  # 'AttentionMIL', 'CLAM', 'TransMIL'
    visualize_multi_head: bool = False
    top_k_patches: int = 10
    colormap: str = "jet"
    

class GradCAMParser:
    """Parse Grad-CAM configurations from dictionaries."""
    
    @staticmethod
    def parse(config_dict: Dict[str, Any]) -> GradCAMConfig:
        """Parse configuration dictionary into GradCAMConfig object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            GradCAMConfig object
            
        Raises:
            ValueError: If configuration is invalid
        """
        
    @staticmethod
    def validate_target_layers(
        model_name: str,
        target_layers: List[str]
    ) -> bool:
        """Validate that target layers exist in model architecture.
        
        Args:
            model_name: Model architecture name
            target_layers: List of layer names
            
        Returns:
            True if all layers exist, False otherwise
        """


class GradCAMPrettyPrinter:
    """Format Grad-CAM configurations for serialization."""
    
    @staticmethod
    def format(config: GradCAMConfig) -> Dict[str, Any]:
        """Format GradCAMConfig object into dictionary.
        
        Args:
            config: GradCAMConfig object
            
        Returns:
            Configuration dictionary with consistent formatting
        """


class AttentionParser:
    """Parse attention weight data from HDF5 files."""
    
    @staticmethod
    def parse(hdf5_path: Path) -> AttentionData:
        """Parse HDF5 file into AttentionData object.
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            AttentionData object
            
        Raises:
            ValueError: If HDF5 file is invalid
        """
        
    @staticmethod
    def validate_attention_weights(weights: np.ndarray) -> bool:
        """Validate attention weights are non-negative.
        
        Args:
            weights: Attention weights array
            
        Returns:
            True if valid, False otherwise
        """


@dataclass
class AttentionData:
    """Container for attention weight data."""
    attention_weights: np.ndarray  # [num_patches]
    coordinates: np.ndarray  # [num_patches, 2]
    slide_id: str
    architecture: str
    

class AttentionPrettyPrinter:
    """Format attention data for HDF5 serialization."""
    
    @staticmethod
    def format(
        attention_data: AttentionData,
        output_path: Path,
        compression: str = "gzip",
        compression_level: int = 4
    ) -> Path:
        """Format AttentionData into HDF5 file.
        
        Args:
            attention_data: AttentionData object
            output_path: Output HDF5 path
            compression: Compression algorithm
            compression_level: Compression level (0-9)
            
        Returns:
            Path to saved HDF5 file
        """
```

### 6. Interactive Visualization Dashboard

**Purpose**: Web-based interface for exploring interpretability results.

**Location**: `src/interpretability/dashboard.py`

**Technology Stack**:
- **Backend**: Flask (lightweight Python web framework)
- **Frontend**: HTML/CSS/JavaScript with Plotly.js for interactive visualizations
- **Caching**: Redis or in-memory cache for previously viewed samples

**Key Components**:

```python
class InterpretabilityDashboard:
    """Web-based dashboard for model interpretability.
    
    Provides interactive interface for exploring Grad-CAM, attention weights,
    failure cases, and feature importance.
    """
    
    def __init__(
        self,
        gradcam_generator: GradCAMGenerator,
        attention_visualizer: AttentionHeatmapGenerator,
        failure_analyzer: FailureAnalyzer,
        feature_importance: FeatureImportanceCalculator,
        cache_dir: Optional[Path] = None,
        port: int = 5000
    ):
        """Initialize dashboard.
        
        Args:
            gradcam_generator: Grad-CAM generator instance
            attention_visualizer: Attention visualizer instance
            failure_analyzer: Failure analyzer instance
            feature_importance: Feature importance calculator instance
            cache_dir: Optional directory for caching visualizations
            port: Port for web server
        """
        
    def start(self, host: str = "0.0.0.0", debug: bool = False):
        """Start dashboard web server.
        
        Args:
            host: Host address
            debug: Enable debug mode
        """
        
    def load_sample(self, slide_id: str) -> Dict[str, Any]:
        """Load interpretability data for a sample.
        
        Args:
            slide_id: Slide identifier
            
        Returns:
            Dictionary containing:
                - gradcam_heatmaps: Grad-CAM visualizations
                - attention_weights: Attention heatmaps
                - prediction: Model prediction
                - confidence: Prediction confidence
                - clinical_features: Clinical metadata
        """
        
    def filter_samples(
        self,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        correctness: Optional[bool] = None,
        clinical_filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Filter samples by criteria.
        
        Args:
            min_confidence: Minimum prediction confidence
            max_confidence: Maximum prediction confidence
            correctness: Filter by correct/incorrect predictions
            clinical_filters: Dictionary of clinical attribute filters
            
        Returns:
            List of slide IDs matching filters
        """
        
    def compare_samples(
        self,
        slide_ids: List[str],
        comparison_type: str = "side_by_side"
    ) -> Dict[str, Any]:
        """Compare interpretability results across samples.
        
        Args:
            slide_ids: List of slide IDs to compare (max 4)
            comparison_type: 'side_by_side' or 'overlay'
            
        Returns:
            Dictionary with comparison visualizations
        """
        
    def export_visualization(
        self,
        slide_id: str,
        output_path: Path,
        format: str = "png",
        dpi: int = 300
    ) -> Path:
        """Export visualization to file.
        
        Args:
            slide_id: Slide identifier
            output_path: Output file path
            format: Output format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
            
        Returns:
            Path to saved file
        """
```

**Dashboard Routes**:

- `/` - Main dashboard interface
- `/api/samples` - List available samples with metadata
- `/api/sample/<slide_id>` - Load interpretability data for sample
- `/api/filter` - Filter samples by criteria
- `/api/compare` - Compare multiple samples
- `/api/export` - Export visualization

## Data Models

### Grad-CAM Output

```python
@dataclass
class GradCAMOutput:
    """Output from Grad-CAM generation."""
    slide_id: str
    patch_id: str
    layer_name: str
    heatmap: np.ndarray  # [H, W] normalized to [0, 1]
    original_image: np.ndarray  # [H, W, 3]
    overlaid_image: np.ndarray  # [H, W, 3]
    class_idx: int
    confidence: float
```

### Attention Output

```python
@dataclass
class AttentionOutput:
    """Output from attention visualization."""
    slide_id: str
    architecture: str
    attention_weights: Union[np.ndarray, Dict[str, np.ndarray]]  # Single or multi-branch
    coordinates: np.ndarray  # [num_patches, 2]
    heatmap_path: Path
    top_k_patches: List[int]
```

### Failure Analysis Output

```python
@dataclass
class FailureAnalysisOutput:
    """Output from failure analysis."""
    total_failures: int
    failure_rate: float
    clusters: Dict[int, Dict[str, Any]]
    systematic_biases: Dict[str, float]
    failure_report_path: Path
```

### Feature Importance Output

```python
@dataclass
class FeatureImportanceOutput:
    """Output from feature importance calculation."""
    method: str  # 'permutation', 'shap', 'gradient'
    importance_scores: Dict[str, float]
    ranked_features: List[Tuple[str, float]]
    confidence_intervals: Optional[Dict[str, Tuple[float, float, float]]]
    visualization_path: Path
    csv_path: Path
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Before defining correctness properties, I need to assess whether property-based testing is appropriate for this feature. The model interpretability system involves:

1. **Grad-CAM generation**: Pure computation on tensors with deterministic outputs
2. **Attention visualization**: Extends existing visualization with deterministic rendering
3. **Failure analysis**: Clustering and statistical analysis with deterministic algorithms
4. **Feature importance**: Statistical computations with deterministic outputs
5. **Configuration parsing**: String/data structure transformations

**Assessment**: Property-based testing IS appropriate for this feature because:
- Core computations are pure functions with clear input/output behavior
- Universal properties exist (round-trips, invariants, normalization)
- Input space is large (various model architectures, image sizes, configurations)
- Testing parsers, serializers, and data transformations is ideal for PBT

Now I'll perform the prework analysis to identify testable properties.



### Property 1: Grad-CAM Heatmap Normalization

*For any* generated Grad-CAM heatmap from any CNN architecture and input patch, all heatmap values SHALL be in the range [0, 1].

**Validates: Requirements 1.6**

### Property 2: Grad-CAM Architecture Support

*For any* CNN architecture in {ResNet18, ResNet50, DenseNet121, EfficientNet-B0} and valid input patch, Grad-CAM generation SHALL succeed and produce a valid heatmap.

**Validates: Requirements 1.2**

### Property 3: Grad-CAM Multi-Layer Output Cardinality

*For any* list of target layers, the number of generated Grad-CAM heatmaps SHALL equal the number of target layers specified.

**Validates: Requirements 1.5**

### Property 4: Grad-CAM Overlay Validity

*For any* image, heatmap, and transparency value in [0, 1], the overlay operation SHALL produce a valid RGB image with shape matching the input image.

**Validates: Requirements 1.3**

### Property 5: Grad-CAM Visualization Round-Trip

*For any* valid Grad-CAM heatmap, saving then loading the visualization SHALL preserve heatmap values within 1% relative error.

**Validates: Requirements 1.8**

### Property 6: Attention Weight Extraction Completeness

*For any* MIL model architecture in {AttentionMIL, CLAM, TransMIL} and patch features, attention weight extraction SHALL return weights for all patches.

**Validates: Requirements 2.1, 2.2**

### Property 7: Attention Weight Normalization

*For any* unnormalized attention weights, normalization SHALL produce weights that sum to 1.0 within numerical precision tolerance of 1e-6.

**Validates: Requirements 2.8**

### Property 8: Attention Top-K Selection Correctness

*For any* attention weights and integer k, the selected top-k patches SHALL have the k highest attention values in descending order.

**Validates: Requirements 2.7**

### Property 9: Attention Multi-Head Output Cardinality

*For any* number of attention heads, the number of generated visualizations SHALL equal the number of attention heads.

**Validates: Requirements 2.5**

### Property 10: Failure Identification with Confidence

*For any* predictions and ground truth labels, all identified failure cases (where prediction ≠ ground truth) SHALL have associated confidence scores.

**Validates: Requirements 3.1, 3.2**

### Property 11: Failure Clustering Completeness

*For any* failure embeddings, clustering SHALL assign each failure to exactly one cluster, and both visualizations and statistics SHALL be generated for all clusters.

**Validates: Requirements 3.3, 3.4, 3.5**

### Property 12: Failure CSV Export Completeness

*For any* failure analysis results, the exported CSV SHALL contain all required columns: slide_id, prediction, ground_truth, confidence, and cluster_assignment.

**Validates: Requirements 3.6**

### Property 13: Systematic Bias Analysis Completeness

*For any* failure cases and clinical subgroups, bias metrics SHALL be computed for all specified subgroups.

**Validates: Requirements 3.7**

### Property 14: Feature Importance Score Normalization

*For any* computed feature importance scores, all values SHALL be in the range [0, 1] and sum to 1.0 within numerical precision tolerance of 1e-6.

**Validates: Requirements 4.3, 4.8**

### Property 15: Feature Importance Method Support

*For any* importance method in {permutation, SHAP, gradient} and valid clinical features, importance calculation SHALL succeed and produce scores for all features.

**Validates: Requirements 4.1, 4.2**

### Property 16: Feature Importance Top-K Ranking

*For any* feature importance scores and integer k, the top-k ranked features SHALL have the k highest importance scores in descending order.

**Validates: Requirements 4.4**

### Property 17: Feature Importance CSV Export Completeness

*For any* feature importance results, the exported CSV SHALL contain feature names and corresponding importance scores for all features.

**Validates: Requirements 4.7**

### Property 18: Grad-CAM Config Round-Trip

*For any* valid GradCAMConfig object, the operation parse(pretty_print(config)) SHALL produce an equivalent GradCAMConfig object.

**Validates: Requirements 5.4**

### Property 19: Grad-CAM Config Validation

*For any* configuration dictionary with invalid target layers or transparency values outside [0, 1], the parser SHALL reject the configuration with a descriptive error message specifying the invalid field.

**Validates: Requirements 5.2, 5.5, 5.6**

### Property 20: Attention Data Round-Trip

*For any* valid AttentionData object, the operation parse(pretty_print(data)) SHALL produce attention weights equivalent to the original within numerical precision tolerance of 1e-6.

**Validates: Requirements 6.4**

### Property 21: Attention Data Validation

*For any* attention weights with negative values or coordinates outside valid slide dimensions, the parser SHALL reject the data with a descriptive error message.

**Validates: Requirements 6.2, 6.5, 6.6**

### Property 22: Dashboard Sample Filtering

*For any* filter criteria (confidence range, correctness, clinical attributes), the returned samples SHALL match all specified filter criteria.

**Validates: Requirements 7.3**

### Property 23: Dashboard Sample Comparison Cardinality

*For any* list of 1-4 sample IDs, the comparison visualization SHALL display all specified samples side-by-side.

**Validates: Requirements 7.4**

### Property 24: Dashboard Visualization Caching

*For any* previously viewed sample, the second load time SHALL be less than or equal to the first load time (caching improves or maintains performance).

**Validates: Requirements 7.7**

### Property 25: Batch Processing Completeness

*For any* batch of samples, batch processing SHALL successfully process all samples in the batch.

**Validates: Requirements 8.3**

### Property 26: GPU Acceleration Support

*For any* computationally intensive operation, when a GPU is available, the operation SHALL execute on the GPU device.

**Validates: Requirements 8.7**

### Property 27: Output Directory Configuration

*For any* specified output directory path, all generated visualizations SHALL be saved to that directory.

**Validates: Requirements 9.3**

### Property 28: Dataset Type Support

*For any* dataset type in {PCam (patch-level), Camelyon (slide-level)}, the interpretability system SHALL process the dataset correctly.

**Validates: Requirements 9.4**

## Error Handling

### Error Categories

The interpretability system handles four categories of errors:

1. **Input Validation Errors**: Invalid configurations, malformed data, out-of-range values
2. **Model Compatibility Errors**: Unsupported architectures, missing layers, incompatible model states
3. **Resource Errors**: Out of memory, GPU unavailable, disk space exhausted
4. **Runtime Errors**: Numerical instability, clustering failures, visualization rendering errors

### Error Handling Strategy

**Validation Errors**:
- Validate all inputs at API boundaries before processing
- Return descriptive error messages specifying which field/value is invalid
- Provide suggestions for valid values when possible
- Example: "Invalid transparency value 1.5. Must be in range [0, 1]."

**Model Compatibility Errors**:
- Check model architecture against supported list before processing
- Validate target layers exist in model before hooking
- Provide clear error messages with supported architectures
- Example: "Layer 'layer5' not found in ResNet18. Valid layers: ['layer1', 'layer2', 'layer3', 'layer4']"

**Resource Errors**:
- Implement graceful degradation (fall back to CPU if GPU unavailable)
- Provide memory-efficient batch processing options
- Log warnings for resource constraints
- Example: "GPU memory insufficient for batch size 64. Reducing to batch size 32."

**Runtime Errors**:
- Catch numerical errors (NaN, Inf) and provide informative messages
- Implement fallback strategies for clustering failures (reduce clusters, try different algorithm)
- Log detailed error context for debugging
- Example: "Clustering failed with k=10. Trying k=5."

### Error Recovery

**Automatic Recovery**:
- GPU unavailable → Fall back to CPU
- Clustering fails → Reduce number of clusters or try different algorithm
- Visualization rendering fails → Save raw data for manual inspection

**User Intervention Required**:
- Invalid configuration → User must fix configuration
- Unsupported architecture → User must use supported architecture
- Insufficient disk space → User must free disk space

### Logging Strategy

All errors are logged with:
- Timestamp
- Error category
- Component name
- Detailed error message
- Stack trace (for unexpected errors)
- Suggested resolution (when available)

Example log entry:
```
2024-01-15 10:30:45 ERROR [GradCAMGenerator] Input validation failed: 
  Invalid transparency value 1.5. Must be in range [0, 1].
  Suggestion: Use transparency between 0.0 (transparent) and 1.0 (opaque).
```

## Testing Strategy

### Dual Testing Approach

The interpretability system uses both unit tests and property-based tests for comprehensive coverage:

**Unit Tests**: Verify specific examples, edge cases, and integration points
- Example: Test Grad-CAM on a specific ResNet18 model with known input
- Example: Test failure analyzer with zero failures (edge case)
- Example: Test dashboard API endpoints return correct status codes

**Property Tests**: Verify universal properties across all inputs
- Example: For any heatmap, values are in [0, 1]
- Example: For any config, round-trip preserves data
- Example: For any attention weights, normalization produces sum=1.0

### Property-Based Testing Configuration

**Testing Library**: Hypothesis (Python property-based testing library)

**Test Configuration**:
- Minimum 100 iterations per property test
- Each property test references its design document property
- Tag format: `# Feature: model-interpretability, Property {number}: {property_text}`

**Example Property Test**:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    heatmap=npst.arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(min_value=1, max_value=512),
            st.integers(min_value=1, max_value=512)
        ),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False)
    )
)
@settings(max_examples=100)
def test_gradcam_normalization(heatmap):
    """
    Feature: model-interpretability, Property 1: Grad-CAM Heatmap Normalization
    
    For any generated Grad-CAM heatmap, all values SHALL be in [0, 1].
    """
    generator = GradCAMGenerator(model=mock_model, target_layers=['layer4'])
    normalized = generator._normalize_heatmap(heatmap)
    
    assert np.all(normalized >= 0.0), "Heatmap contains values < 0"
    assert np.all(normalized <= 1.0), "Heatmap contains values > 1"
```

### Unit Test Coverage

**Component-Level Tests**:
- Grad-CAM Generator: Test each architecture (ResNet, DenseNet, EfficientNet)
- Attention Visualizer: Test each MIL architecture (AttentionMIL, CLAM, TransMIL)
- Failure Analyzer: Test clustering algorithms (k-means, DBSCAN, hierarchical)
- Feature Importance: Test each method (permutation, SHAP, gradient)
- Parsers: Test valid and invalid configurations

**Integration Tests**:
- Dashboard: Test API endpoints, filtering, comparison
- Evaluation Scripts: Test CLI integration, automatic visualization generation
- End-to-End: Test complete workflow from model to visualization

**Performance Tests**:
- Grad-CAM: Verify < 200ms per patch on GPU
- Attention: Verify < 100ms per slide on GPU
- Feature Importance: Verify < 5 seconds per model on CPU
- Dashboard: Verify < 3 second load time

### Test Data

**Synthetic Data**:
- Generated using Hypothesis for property tests
- Controlled distributions for edge case testing
- Deterministic seeds for reproducibility

**Real Data**:
- Sample patches from PCam dataset
- Sample slides from Camelyon dataset
- Pre-trained model checkpoints
- Known-good visualizations for regression testing

### Continuous Integration

**Test Execution**:
- Run unit tests on every commit
- Run property tests on every pull request
- Run integration tests nightly
- Run performance tests weekly

**Test Environment**:
- CPU-only tests: Run on standard CI runners
- GPU tests: Run on GPU-enabled runners
- Dashboard tests: Run with headless browser

### Test Metrics

**Coverage Targets**:
- Line coverage: > 90%
- Branch coverage: > 85%
- Property coverage: 100% of defined properties

**Quality Gates**:
- All tests must pass before merge
- No decrease in coverage allowed
- Performance tests must meet SLAs

## Implementation Notes

### Technology Stack

**Core Libraries**:
- PyTorch: Deep learning framework for model operations
- NumPy: Numerical computations
- Matplotlib: Static visualizations
- Plotly: Interactive visualizations
- Flask: Web dashboard backend
- H5py: HDF5 file I/O
- Scikit-learn: Clustering and statistical methods
- SHAP: SHAP value computation
- Hypothesis: Property-based testing

**Optional Dependencies**:
- Redis: Caching for dashboard (optional, falls back to in-memory)
- Pillow: Image processing
- OpenSlide: Whole-slide image reading

### File Structure

```
src/interpretability/
├── __init__.py
├── gradcam.py              # Grad-CAM generator
├── failure_analysis.py     # Failure analyzer
├── feature_importance.py   # Feature importance calculator
├── config.py               # Configuration parsers and data models
├── dashboard.py            # Web dashboard
└── utils.py                # Shared utilities

src/visualization/
├── __init__.py
├── attention_heatmap.py    # Extended attention visualizer (existing)
└── timeline.py             # Timeline visualizer (existing)

tests/interpretability/
├── test_gradcam.py
├── test_failure_analysis.py
├── test_feature_importance.py
├── test_config.py
├── test_dashboard.py
└── test_properties.py      # Property-based tests

examples/interpretability/
├── gradcam_example.ipynb
├── attention_example.ipynb
├── failure_analysis_example.ipynb
└── feature_importance_example.ipynb

configs/interpretability/
├── gradcam_default.yaml
├── attention_default.yaml
└── dashboard_default.yaml
```

### Integration Points

**Existing Code**:
- `src/models/feature_extractors.py`: CNN models for Grad-CAM
- `src/models/attention_mil.py`: MIL models for attention extraction
- `src/visualization/attention_heatmap.py`: Existing attention visualization (extend)
- Evaluation scripts: Add interpretability flags

**New Code**:
- `src/interpretability/`: New module for interpretability components
- `tests/interpretability/`: New test suite
- `examples/interpretability/`: New example notebooks
- `configs/interpretability/`: New configuration files

### Performance Optimization

**GPU Acceleration**:
- Use PyTorch's automatic GPU placement
- Batch process multiple samples together
- Cache intermediate activations

**Memory Management**:
- Process large slides in chunks
- Clear GPU cache between batches
- Use memory-mapped HDF5 for large datasets

**Caching Strategy**:
- Cache generated visualizations in dashboard
- Cache attention weights after extraction
- Cache feature importance scores

### Backward Compatibility

**No Breaking Changes**:
- All new functionality is opt-in via flags
- Existing evaluation scripts work without modification
- New dependencies are optional where possible

**Deprecation Policy**:
- No existing APIs are deprecated
- New APIs follow existing naming conventions
- Documentation clearly marks new features

## Deployment Considerations

### Installation

**Required Dependencies**:
```bash
pip install torch torchvision numpy matplotlib h5py scikit-learn flask hypothesis
```

**Optional Dependencies**:
```bash
pip install shap redis pillow openslide-python plotly
```

### Configuration

**Environment Variables**:
- `INTERPRETABILITY_CACHE_DIR`: Cache directory for visualizations
- `INTERPRETABILITY_GPU`: Enable/disable GPU acceleration
- `DASHBOARD_PORT`: Port for web dashboard

**Configuration Files**:
- YAML files in `configs/interpretability/`
- Override defaults with command-line arguments

### Resource Requirements

**Minimum Requirements**:
- CPU: 4 cores
- RAM: 16 GB
- Disk: 10 GB for cache
- GPU: Optional (CUDA-capable for acceleration)

**Recommended Requirements**:
- CPU: 8+ cores
- RAM: 32 GB
- Disk: 50 GB for cache
- GPU: NVIDIA GPU with 8+ GB VRAM

### Monitoring

**Metrics to Track**:
- Grad-CAM generation time per patch
- Attention extraction time per slide
- Feature importance computation time
- Dashboard response time
- Cache hit rate

**Logging**:
- All operations logged to standard logging system
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Structured logging for easy parsing

## Future Enhancements

### Planned Features

1. **Additional Interpretability Methods**:
   - Integrated Gradients
   - Layer-wise Relevance Propagation (LRP)
   - Saliency maps

2. **Enhanced Visualizations**:
   - 3D visualizations for multi-scale analysis
   - Animated attention over time for temporal models
   - Interactive heatmap editing

3. **Advanced Analytics**:
   - Automated failure pattern detection using ML
   - Comparative analysis across model versions
   - Uncertainty quantification for interpretability

4. **Collaboration Features**:
   - Multi-user dashboard with annotations
   - Export to clinical reporting formats
   - Integration with PACS systems

### Research Directions

1. **Interpretability Metrics**:
   - Quantitative measures of explanation quality
   - Faithfulness metrics for attention weights
   - Stability analysis for Grad-CAM

2. **Clinical Validation**:
   - User studies with pathologists
   - Comparison with expert annotations
   - Clinical utility assessment

3. **Scalability**:
   - Distributed processing for large cohorts
   - Real-time interpretability for inference
   - Cloud deployment options

## References

### Academic Papers

1. Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)
3. Lu et al. "Data-efficient and weakly supervised computational pathology on whole-slide images" (Nature Biomedical Engineering 2021)
4. Lundberg & Lee. "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017) - SHAP
5. Ribeiro et al. "Why Should I Trust You?: Explaining the Predictions of Any Classifier" (KDD 2016) - LIME

### Technical Documentation

1. PyTorch Hooks: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
2. Hypothesis Documentation: https://hypothesis.readthedocs.io/
3. Flask Documentation: https://flask.palletsprojects.com/
4. SHAP Documentation: https://shap.readthedocs.io/

### Related Work

1. Captum (PyTorch interpretability library): https://captum.ai/
2. Alibi Explain: https://docs.seldon.io/projects/alibi/
3. InterpretML: https://interpret.ml/

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Authors**: Kiro AI Agent  
**Status**: Ready for Review
