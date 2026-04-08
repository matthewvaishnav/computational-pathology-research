# Design Document: Full-Scale PatchCamelyon Experiments

## Overview

This design implements full-scale training and evaluation on the complete PatchCamelyon (PCam) dataset to validate the multimodal pathology framework on a real, published dataset. The system extends existing PCamDataset, training, and evaluation infrastructure to handle 262K training samples with GPU-optimized configurations, baseline model comparisons, and statistical validation through bootstrap confidence intervals.

The design follows a minimal-change approach, leveraging existing components:
- PCamDataset class with TFDS download capability (already implemented)
- Training script with early stopping and checkpointing (already implemented)
- Evaluation script with comprehensive metrics (already implemented)
- Comparison runner infrastructure (already implemented)

New components focus on:
- GPU-optimized configuration files
- Bootstrap CI utility function
- Baseline model configuration variants
- Benchmark report generation
- Enhanced documentation

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Full-Scale PCam System                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────────────┐
                              │                                 │
                    ┌─────────▼────────┐            ┌──────────▼─────────┐
                    │  Data Pipeline   │            │  Training Pipeline │
                    └─────────┬────────┘            └──────────┬─────────┘
                              │                                 │
        ┌─────────────────────┼─────────────────────┬──────────┼──────────┬────────────┐
        │                     │                     │          │          │            │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────┐  ┌──▼──────┐  ┌▼────────┐  ┌▼──────────┐
│  PCamDataset   │  │ Download Manager│  │ GPU Configs │  │ Trainer │  │Evaluator│  │Comparator │
│  (existing)    │  │   (existing)    │  │    (new)    │  │(existing│  │(existing│  │ (existing)│
└───────┬────────┘  └────────┬────────┘  └────────┬────┘  └──┬──────┘  └┬────────┘  └┬──────────┘
        │                    │                     │          │          │            │
        │                    │                     │          │          │            │
        └────────────────────┴─────────────────────┴──────────┴──────────┴────────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │  Statistical Utils │
                                          │  - Bootstrap CI    │
                                          │  - Report Gen      │
                                          │     (new)          │
                                          └────────────────────┘
```

### Data Flow

```
1. Download Phase:
   PCamDataset.download() → TFDS/GitHub → HDF5 files
   ↓
   Validation: 262K train + 32K val + 32K test

2. Training Phase:
   DataLoader → Feature Extractor → WSI Encoder → Classification Head
   ↓
   Checkpoints + Metrics (JSON)

3. Evaluation Phase:
   Test DataLoader → Model → Predictions
   ↓
   Metrics + Visualizations + Bootstrap CI

4. Comparison Phase:
   Multiple Configs → Train → Evaluate → Aggregate
   ↓
   Comparison Table + Benchmark Report
```

## Components and Interfaces

### 1. GPU-Optimized Configuration Files

**Location:** `experiments/configs/pcam_fullscale/`

**Files:**
- `gpu_16gb.yaml` - For GPUs with 16GB VRAM (batch_size=128)
- `gpu_24gb.yaml` - For GPUs with 24GB VRAM (batch_size=256)
- `gpu_40gb.yaml` - For GPUs with 40GB VRAM (batch_size=512)
- `baseline_resnet50.yaml` - ResNet-50 baseline
- `baseline_densenet121.yaml` - DenseNet-121 baseline
- `baseline_efficientnet_b0.yaml` - EfficientNet-B0 baseline

**Configuration Structure:**
```yaml
experiment:
  name: pcam_fullscale_gpu16gb
  description: Full-scale PCam training on 262K samples
  tags: [pcam, fullscale, gpu-optimized]

data:
  dataset: pcam
  root_dir: ./data/pcam
  download: true  # Enable automatic download
  num_workers: 6  # Parallel data loading
  pin_memory: true  # Fast CPU-to-GPU transfer

model:
  modalities: [wsi]
  embed_dim: 256
  feature_extractor:
    model: resnet18  # or resnet50, densenet121, efficientnet_b0
    pretrained: true
    feature_dim: 512

training:
  num_epochs: 20
  batch_size: 128  # Adjusted per GPU memory
  learning_rate: 1e-3
  weight_decay: 1e-4
  use_amp: true  # Mixed precision
  max_grad_norm: 1.0
  gradient_accumulation_steps: 1  # Increase if batch_size too large

validation:
  interval: 1
  metric: val_auc
  maximize: true

early_stopping:
  enabled: true
  patience: 5
  min_delta: 0.001

checkpoint:
  save_interval: 5
  save_best: true
  checkpoint_dir: ./checkpoints/pcam_fullscale

logging:
  log_dir: ./logs/pcam_fullscale
  log_interval: 100
  use_tensorboard: true

evaluation:
  generate_plots: true
  output_dir: ./results/pcam_fullscale
  compute_bootstrap_ci: true  # New flag
  bootstrap_samples: 1000
  confidence_level: 0.95

seed: 42
device: cuda
```

### 2. Bootstrap Confidence Interval Utility

**Location:** `src/utils/statistical.py` (new file)

**Interface:**
```python
def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_prob: Predicted probabilities [N]
        metric_fn: Function that computes metric from (y_true, y_pred, y_prob)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
        
    Example:
        >>> accuracy_ci = compute_bootstrap_ci(
        ...     y_true, y_pred, y_prob,
        ...     lambda yt, yp, yprob: accuracy_score(yt, yp),
        ...     n_bootstrap=1000
        ... )
        >>> print(f"Accuracy: {accuracy_ci[0]:.4f} [{accuracy_ci[1]:.4f}, {accuracy_ci[2]:.4f}]")
    """
    pass

def compute_all_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute all classification metrics with bootstrap CIs.
    
    Returns:
        Dictionary with structure:
        {
            'accuracy': {'value': 0.95, 'ci_lower': 0.94, 'ci_upper': 0.96},
            'auc': {'value': 0.97, 'ci_lower': 0.96, 'ci_upper': 0.98},
            'f1': {'value': 0.94, 'ci_lower': 0.93, 'ci_upper': 0.95},
            'precision': {'value': 0.93, 'ci_lower': 0.92, 'ci_upper': 0.94},
            'recall': {'value': 0.95, 'ci_lower': 0.94, 'ci_upper': 0.96}
        }
    """
    pass
```

### 3. Enhanced Evaluation Script

**Modification:** `experiments/evaluate_pcam.py`

**Changes:**
- Add `--compute-bootstrap-ci` flag
- Add `--bootstrap-samples` argument (default: 1000)
- Integrate `compute_all_metrics_with_ci()` when flag is enabled
- Save CI results in metrics JSON

**Updated metrics.json structure:**
```json
{
  "accuracy": 0.9523,
  "accuracy_ci_lower": 0.9487,
  "accuracy_ci_upper": 0.9558,
  "auc": 0.9812,
  "auc_ci_lower": 0.9789,
  "auc_ci_upper": 0.9834,
  "f1": 0.9501,
  "f1_ci_lower": 0.9463,
  "f1_ci_upper": 0.9537,
  "precision": 0.9456,
  "precision_ci_lower": 0.9415,
  "precision_ci_upper": 0.9496,
  "recall": 0.9547,
  "recall_ci_lower": 0.9511,
  "recall_ci_upper": 0.9582,
  "confusion_matrix": [[15234, 543], [421, 16570]],
  "bootstrap_config": {
    "n_samples": 1000,
    "confidence_level": 0.95,
    "random_state": 42
  }
}
```

### 4. Benchmark Report Generator

**Location:** `src/utils/benchmark_report.py` (new file)

**Interface:**
```python
def generate_benchmark_report(
    experiment_name: str,
    dataset_info: Dict[str, Any],
    model_info: Dict[str, Any],
    training_config: Dict[str, Any],
    test_metrics: Dict[str, Any],
    comparison_results: Optional[Dict[str, Any]] = None,
    hardware_info: Optional[Dict[str, Any]] = None,
    output_path: str = "PCAM_BENCHMARK_RESULTS.md"
) -> None:
    """
    Generate comprehensive benchmark report in markdown format.
    
    Args:
        experiment_name: Name of the experiment
        dataset_info: Dataset statistics and details
        model_info: Model architecture and parameters
        training_config: Training hyperparameters
        test_metrics: Test set metrics with CIs
        comparison_results: Optional baseline comparison results
        hardware_info: Optional hardware specifications
        output_path: Path to save the report
        
    Generates:
        - Executive summary
        - Dataset description
        - Model architecture
        - Training configuration
        - Results with confidence intervals
        - Baseline comparison table
        - Reproduction commands
        - Hardware specifications
    """
    pass
```

### 5. Extended Comparison Runner

**Modification:** `experiments/compare_pcam_baselines.py`

**Changes:**
- Add support for full-scale configs in `experiments/configs/pcam_fullscale/`
- Integrate bootstrap CI computation for each baseline
- Generate comparison table with CIs
- Call `generate_benchmark_report()` after comparison completes

**No API changes** - existing interface remains compatible.

## Data Models

### Configuration Schema

```python
@dataclass
class FullScalePCamConfig:
    """Configuration for full-scale PCam experiments."""
    
    # Experiment metadata
    experiment_name: str
    description: str
    tags: List[str]
    
    # Data configuration
    data_root: str = "./data/pcam"
    download_enabled: bool = True
    num_workers: int = 6
    pin_memory: bool = True
    
    # Model configuration
    feature_extractor_model: str = "resnet18"  # resnet18, resnet50, densenet121, efficientnet_b0
    pretrained: bool = True
    feature_dim: int = 512
    embed_dim: int = 256
    
    # Training configuration
    num_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Validation configuration
    val_interval: int = 1
    val_metric: str = "val_auc"
    maximize_metric: bool = True
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints/pcam_fullscale"
    save_interval: int = 5
    save_best: bool = True
    
    # Logging
    log_dir: str = "./logs/pcam_fullscale"
    log_interval: int = 100
    use_tensorboard: bool = True
    
    # Evaluation
    output_dir: str = "./results/pcam_fullscale"
    generate_plots: bool = True
    compute_bootstrap_ci: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Reproducibility
    seed: int = 42
    device: str = "cuda"
```

### Metrics Schema

```python
@dataclass
class MetricsWithCI:
    """Metrics with bootstrap confidence intervals."""
    
    value: float
    ci_lower: float
    ci_upper: float
    
    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"

@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    
    accuracy: MetricsWithCI
    auc: MetricsWithCI
    f1: MetricsWithCI
    precision: MetricsWithCI
    recall: MetricsWithCI
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    bootstrap_config: Dict[str, Any]
    hardware_info: Dict[str, Any]
    inference_time_seconds: float
    samples_per_second: float
```

### Comparison Results Schema

```python
@dataclass
class BaselineResult:
    """Results for a single baseline model."""
    
    model_name: str
    config_path: str
    checkpoint_path: str
    training_time_seconds: float
    test_metrics: EvaluationResults
    model_parameters: int

@dataclass
class ComparisonResults:
    """Aggregated comparison results."""
    
    timestamp: str
    baselines: List[BaselineResult]
    best_model: str
    best_accuracy: float
    
    def to_markdown_table(self) -> str:
        """Generate markdown comparison table."""
        pass
```

## Error Handling

### GPU Out-of-Memory Recovery

**Strategy:** Automatic batch size reduction with retry

**Implementation in train_pcam.py (already exists):**
```python
def reduce_batch_size_on_oom(config, feature_extractor, encoder, head, optimizer, scheduler):
    """Reduce batch size and attempt recovery from GPU OOM."""
    current_batch_size = config["training"]["batch_size"]
    new_batch_size = max(8, current_batch_size // 2)
    
    logger.warning(f"GPU OOM. Reducing batch size: {current_batch_size} → {new_batch_size}")
    config["training"]["batch_size"] = new_batch_size
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    return config
```

**Error Flow:**
```
Training → GPU OOM Exception
    ↓
Catch RuntimeError with "out of memory"
    ↓
reduce_batch_size_on_oom()
    ↓
Recreate DataLoaders with new batch_size
    ↓
Resume training from last checkpoint
    ↓
If OOM again with batch_size < 8 → Fail with error message
```

### Download Failure Recovery

**Strategy:** Cleanup partial downloads and provide clear error messages

**Implementation in PCamDataset (already exists):**
```python
def download(self):
    """Download PCam dataset with error handling."""
    try:
        if TFDS_AVAILABLE:
            self._download_via_tfds()
        else:
            self._download_direct()
        self._process_downloaded_data()
        self._save_metadata()
    except Exception as e:
        # Cleanup partial downloads
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)
        raise RuntimeError(f"Download failed: {e}\nPlease check network connection and try again.")
```

### Dataset Validation Errors

**Strategy:** Validate dataset integrity before training

**Implementation in train_pcam.py (already exists):**
```python
def validate_dataset(dataset: PCamDataset):
    """Validate dataset integrity."""
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")
    
    # Check samples
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        if sample["image"].shape != torch.Size([3, 96, 96]):
            raise RuntimeError(f"Invalid image shape: {sample['image'].shape}")
        if sample["label"].item() not in (0, 1):
            raise RuntimeError(f"Invalid label: {sample['label'].item()}")
```

### Cross-Platform Path Handling

**Strategy:** Use pathlib for all file operations

**Example:**
```python
from pathlib import Path

# Instead of: config_path = "experiments/configs/pcam.yaml"
config_path = Path("experiments") / "configs" / "pcam.yaml"

# Instead of: os.path.join(root, "checkpoints", "best_model.pth")
checkpoint_path = Path(root) / "checkpoints" / "best_model.pth"
```

## Testing Strategy

### Unit Tests

**Test Coverage:**
1. Bootstrap CI computation
   - Test with known distributions
   - Verify CI bounds are reasonable
   - Test edge cases (all correct, all wrong)

2. Configuration loading
   - Test GPU config variants load correctly
   - Test baseline config variants load correctly
   - Test config validation

3. Report generation
   - Test markdown formatting
   - Test table generation
   - Test with missing optional fields

**Test Files:**
- `tests/unit/test_statistical_utils.py`
- `tests/unit/test_benchmark_report.py`
- `tests/unit/test_fullscale_configs.py`

### Integration Tests

**Test Scenarios:**
1. End-to-end training on synthetic subset (fast)
   - Use existing 500-sample synthetic data
   - Train for 2 epochs
   - Verify checkpoint saved
   - Verify metrics computed

2. Evaluation with bootstrap CI
   - Load checkpoint
   - Run evaluation with CI computation
   - Verify CI bounds in output

3. Comparison runner with multiple configs
   - Run 2 baseline configs on synthetic data
   - Verify comparison table generated
   - Verify report generated

**Test Files:**
- `tests/integration/test_fullscale_training.py`
- `tests/integration/test_bootstrap_evaluation.py`
- `tests/integration/test_baseline_comparison.py`

### Manual Testing Checklist

**Pre-release validation:**
- [ ] Download full PCam dataset (262K samples)
- [ ] Train ResNet-18 on full dataset with GPU config
- [ ] Verify training completes in < 8 hours on 16GB GPU
- [ ] Evaluate on full test set (32K samples)
- [ ] Verify bootstrap CI computed correctly
- [ ] Run baseline comparison (ResNet-50, DenseNet-121)
- [ ] Verify benchmark report generated
- [ ] Test on Windows, macOS, Linux
- [ ] Test GPU OOM recovery by reducing available memory
- [ ] Verify backward compatibility with synthetic mode

## Implementation Plan

### Phase 1: Core Infrastructure (Priority: High)

**Tasks:**
1. Create `src/utils/statistical.py` with bootstrap CI functions
2. Create GPU-optimized config files in `experiments/configs/pcam_fullscale/`
3. Create baseline config variants (ResNet-50, DenseNet-121, EfficientNet-B0)
4. Add `--compute-bootstrap-ci` flag to `evaluate_pcam.py`
5. Integrate bootstrap CI computation in evaluation

**Estimated Time:** 4-6 hours

### Phase 2: Reporting and Documentation (Priority: High)

**Tasks:**
1. Create `src/utils/benchmark_report.py` with report generation
2. Extend `compare_pcam_baselines.py` to call report generator
3. Create `PCAM_BENCHMARK_RESULTS.md` template
4. Update README with full-scale experiment instructions

**Estimated Time:** 3-4 hours

### Phase 3: Testing and Validation (Priority: Medium)

**Tasks:**
1. Write unit tests for bootstrap CI
2. Write unit tests for report generation
3. Write integration tests for full pipeline
4. Run manual testing checklist
5. Fix any discovered issues

**Estimated Time:** 4-6 hours

### Phase 4: Full-Scale Experiments (Priority: Medium)

**Tasks:**
1. Download full PCam dataset
2. Train ResNet-18 baseline on full dataset
3. Train ResNet-50, DenseNet-121, EfficientNet-B0 baselines
4. Generate benchmark report with all results
5. Update documentation with final results

**Estimated Time:** 12-24 hours (mostly GPU time)

## Deployment Considerations

### Hardware Requirements

**Minimum:**
- GPU: 16GB VRAM (NVIDIA RTX 4080, A4000, or equivalent)
- RAM: 32GB system memory
- Storage: 50GB free space (dataset + checkpoints + logs)
- OS: Windows 10+, macOS 12+, or Ubuntu 20.04+

**Recommended:**
- GPU: 24GB VRAM (NVIDIA RTX 4090, A5000, or equivalent)
- RAM: 64GB system memory
- Storage: 100GB free space
- OS: Ubuntu 22.04 LTS

**Optimal:**
- GPU: 40GB VRAM (NVIDIA A100, A6000, or equivalent)
- RAM: 128GB system memory
- Storage: 200GB NVMe SSD
- OS: Ubuntu 22.04 LTS

### Software Dependencies

**Required:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- TensorFlow Datasets (for PCam download)
- scikit-learn (for metrics)
- matplotlib, seaborn (for visualizations)
- tqdm (for progress bars)
- pyyaml (for config loading)

**Installation:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow-datasets scikit-learn matplotlib seaborn tqdm pyyaml h5py
```

### Configuration Selection Guide

**Choose config based on available GPU memory:**

| GPU Memory | Config File | Batch Size | Expected Time |
|------------|-------------|------------|---------------|
| 16GB | `gpu_16gb.yaml` | 128 | 8 hours |
| 24GB | `gpu_24gb.yaml` | 256 | 6 hours |
| 40GB+ | `gpu_40gb.yaml` | 512 | 4 hours |

**For CPU-only systems:**
- Use `synthetic_mode=true` for testing
- Full-scale training not recommended (would take days)

### Monitoring and Logging

**TensorBoard:**
```bash
tensorboard --logdir logs/pcam_fullscale
```

**Training Status:**
- Check `logs/pcam_fullscale/training_status.json` for real-time progress
- Monitor GPU usage: `nvidia-smi -l 1`
- Monitor disk space: `df -h`

**Checkpoints:**
- Best model: `checkpoints/pcam_fullscale/best_model.pth`
- Periodic checkpoints: `checkpoints/pcam_fullscale/checkpoint_epoch_N.pth`

## Backward Compatibility

### Synthetic Mode Preservation

**Existing behavior maintained:**
```python
# Synthetic mode (existing, unchanged)
config = {
    "data": {
        "root_dir": "./data/pcam_synthetic",
        "download": False,  # Synthetic data already exists
        "num_workers": 0
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 32
    }
}
```

**Full-scale mode (new):**
```python
# Full-scale mode (new configs)
config = {
    "data": {
        "root_dir": "./data/pcam",
        "download": True,  # Download full dataset
        "num_workers": 6
    },
    "training": {
        "num_epochs": 20,
        "batch_size": 128
    }
}
```

### API Compatibility

**No breaking changes:**
- All existing scripts work without modification
- New features are opt-in via config flags
- Existing tests continue to pass
- CI/CD pipelines unaffected

### Migration Path

**For users with existing experiments:**
1. Keep existing configs in `experiments/configs/`
2. New full-scale configs in `experiments/configs/pcam_fullscale/`
3. Use `--config` flag to select desired config
4. No code changes required

## Performance Optimization

### Data Loading

**Optimizations:**
- Parallel data loading with `num_workers=6`
- Pin memory for faster CPU-to-GPU transfer
- HDF5 lazy loading (already implemented in PCamDataset)
- Prefetching with `pin_memory=True`

**Expected throughput:**
- 16GB GPU: ~2000 samples/second
- 24GB GPU: ~3000 samples/second
- 40GB GPU: ~5000 samples/second

### Training Speed

**Optimizations:**
- Mixed precision training (AMP) for 2x speedup
- Gradient accumulation for large effective batch sizes
- Efficient feature extraction (batch-time, not per-sample)
- Early stopping to avoid unnecessary epochs

**Expected training time per epoch:**
- 16GB GPU: ~24 minutes/epoch
- 24GB GPU: ~16 minutes/epoch
- 40GB GPU: ~10 minutes/epoch

### Memory Management

**Strategies:**
- Automatic batch size reduction on OOM
- Gradient checkpointing (if needed)
- Clear GPU cache between epochs
- Monitor memory usage and log warnings

## Security Considerations

**Data Privacy:**
- PCam dataset is publicly available (no privacy concerns)
- No patient identifiers in dataset
- Safe for public benchmarking

**Code Security:**
- No external API calls (except TFDS download)
- No user input validation needed (config files only)
- No authentication required

**Dependency Security:**
- Use pinned versions in requirements.txt
- Regular dependency updates
- Scan for vulnerabilities with `pip-audit`

## Future Enhancements

**Potential improvements (out of scope for this feature):**

1. **Distributed Training:**
   - Multi-GPU support with DataParallel or DistributedDataParallel
   - Reduce training time to < 2 hours

2. **Advanced Baselines:**
   - Vision Transformers (ViT, Swin)
   - Self-supervised pretraining (SimCLR, MoCo)
   - Ensemble methods

3. **Hyperparameter Optimization:**
   - Automated hyperparameter search with Optuna
   - Learning rate finder
   - Architecture search

4. **Extended Analysis:**
   - Per-class error analysis
   - Failure case visualization
   - Attention map visualization
   - Grad-CAM heatmaps

5. **Cloud Deployment:**
   - AWS SageMaker integration
   - Google Cloud TPU support
   - Azure ML integration

## Conclusion

This design provides a comprehensive solution for full-scale PCam experiments while maintaining backward compatibility and minimizing code changes. The implementation leverages existing infrastructure, adds GPU-optimized configurations, implements statistical validation through bootstrap confidence intervals, and generates professional benchmark reports.

Key benefits:
- **Minimal code changes:** Extends existing components rather than rewriting
- **GPU-optimized:** Configurations for 16GB, 24GB, and 40GB GPUs
- **Statistically rigorous:** Bootstrap confidence intervals for all metrics
- **Reproducible:** Fixed seeds, deterministic training, comprehensive logging
- **Cross-platform:** Works on Windows, macOS, and Linux
- **Backward compatible:** Existing tests and workflows unaffected
- **Well-documented:** Comprehensive benchmark reports and reproduction instructions

The design addresses all 12 requirements and provides a solid foundation for validating the multimodal pathology framework on real, published datasets.

