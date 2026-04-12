# PCam Visualization Notebook

This directory contains Jupyter notebooks for visualizing PatchCamelyon experiment results.

## pcam_visualization.ipynb

A comprehensive visualization notebook that generates all required plots for the PCam experiment.

### Features

The notebook includes the following sections:

#### 1. Dataset Exploration
- **Sample Image Grid**: Displays an 8x8 grid of sample patches with labels
- **Class Distribution**: Bar chart showing the distribution of normal vs metastatic samples
- **Image Statistics**: Mean and standard deviation per RGB channel

#### 2. Training Curves
- **Loss Curves**: Training and validation loss over epochs
- **Accuracy Curves**: Training and validation accuracy over epochs
- **Learning Rate Schedule**: Learning rate changes over training (if available)

#### 3. Model Performance
- **Confusion Matrix**: Heatmap showing true positives, false positives, true negatives, and false negatives
- **ROC Curve**: Receiver Operating Characteristic curve with AUC score
- **Precision-Recall Curve**: Precision vs recall with average precision score

#### 4. Prediction Analysis
- **Correct Predictions**: Grid of correctly classified samples with confidence scores
- **Incorrect Predictions**: Grid of misclassified samples with confidence scores
- **Confidence Distribution**: Histograms showing prediction confidence for correct and incorrect predictions

#### 5. Output
All plots are automatically saved to `results/pcam/` directory as high-resolution PNG files.

### Usage

#### Prerequisites
```bash
pip install jupyter matplotlib seaborn scikit-learn torch numpy
```

#### Running the Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook experiments/notebooks/pcam_visualization.ipynb
   ```

2. **Run All Cells**: Execute all cells in order (Cell → Run All)

3. **View Results**: Plots are displayed inline and saved to `results/pcam/`

#### Running as Python Script

You can also convert and run the notebook as a Python script:

```bash
# Convert notebook to Python script
jupyter nbconvert --to script pcam_visualization.ipynb

# Run the script
python pcam_visualization.py
```

### Data Requirements

The notebook expects the following data files:

#### Required Files
- **Evaluation Metrics**: `results/pcam_eval_test/metrics.json` or `results/pcam/metrics.json`
  - Contains: predictions, probabilities, labels, confusion matrix, accuracy, AUC, etc.

#### Optional Files
- **Training Checkpoint**: `checkpoints/pcam/best_model.pth`
  - Contains: training history with loss and accuracy per epoch
- **TensorBoard Logs**: `logs/pcam/`
  - Contains: TensorBoard event files with training metrics

#### Fallback Behavior
If data files are not found, the notebook will:
- Use sample/synthetic data for demonstration
- Display warnings indicating which files are missing
- Still generate all plots with placeholder data

### Output Files

All plots are saved to `results/pcam/` with the following names:

| File | Description |
|------|-------------|
| `sample_grid.png` | 8x8 grid of sample images |
| `class_distribution.png` | Bar chart of class distribution |
| `image_statistics.png` | Mean and std per channel |
| `loss_curves.png` | Training and validation loss |
| `accuracy_curves.png` | Training and validation accuracy |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `precision_recall_curve.png` | Precision-recall curve |
| `correct_predictions.png` | Grid of correct predictions |
| `incorrect_predictions.png` | Grid of incorrect predictions |
| `confidence_histogram.png` | Confidence distribution histograms |

### Customization

You can customize the notebook by modifying the following parameters:

```python
# Configuration section (first code cell)
DATA_ROOT = Path('../../data/pcam')  # Dataset location
SAMPLE_SIZE = 64  # Number of samples to display (8x8 grid)
OUTPUT_DIR = Path('../../results/pcam')  # Output directory for plots

# Plot settings
plt.rcParams['figure.dpi'] = 100  # Display DPI
plt.rcParams['savefig.dpi'] = 300  # Save DPI
plt.rcParams['figure.figsize'] = (10, 8)  # Default figure size
```

### Troubleshooting

#### Issue: "Could not load dataset"
- **Solution**: Ensure the PCam dataset is downloaded to `data/pcam/`
- **Alternative**: The notebook will use sample data for demonstration

#### Issue: "No metrics file found"
- **Solution**: Run the evaluation script first: `python experiments/evaluate_pcam.py`
- **Alternative**: The notebook will generate sample metrics for demonstration

#### Issue: "Checkpoint not found"
- **Solution**: Run training first: `python experiments/train_pcam.py`
- **Alternative**: The notebook will use sample training curves

#### Issue: Import errors
- **Solution**: Install missing packages:
  ```bash
  pip install matplotlib seaborn scikit-learn torch numpy jupyter
  ```

### Integration with Experiment Workflow

This notebook is part of the PCam experiment workflow:

1. **Train Model**: `python experiments/train_pcam.py --config experiments/configs/pcam.yaml`
2. **Evaluate Model**: `python experiments/evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth`
3. **Visualize Results**: Run this notebook to generate all plots

### Requirements Validation

This notebook validates the following requirements from the spec:

- **Requirement 5.1**: Dataset exploration visualizations
- **Requirement 5.2**: Training loss curves
- **Requirement 5.3**: Training accuracy curves
- **Requirement 5.4**: Confusion matrix visualization
- **Requirement 5.5**: ROC curve with AUC
- **Requirement 5.6**: Prediction analysis with confidence scores
- **Requirement 5.7**: Save all plots to results directory

### Notes

- The notebook uses a non-interactive matplotlib backend when run as a script
- All plots are saved with high resolution (300 DPI) for publication quality
- The notebook is designed to be robust and will work even with missing data files
- Seaborn styling is applied for professional-looking plots
