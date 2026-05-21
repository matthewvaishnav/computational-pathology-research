# TransnnMIL Implementation Summary

## Overview

TransnnMIL is a novel dual-branch Multiple Instance Learning (MIL) architecture that fuses **TransMIL** (transformer-based) and **nnMIL** (gated attention) through a learnable scalar gate parameter. This implementation is fully integrated into the existing computational pathology research codebase and ready for immediate training.

## Architecture

```
Input: Bag of patch embeddings [B, K, D]
│
├─ Branch A (TransMIL)
│  ├─ Feature projection: Linear(D, H) → ReLU → Dropout
│  ├─ Positional encoding: DISABLED (for random sub-bags)
│  ├─ CLS token prepended
│  ├─ Transformer encoder (2 layers, 8 heads)
│  ├─ Extract CLS token representation
│  └─ Classifier: Linear(H, H) → ReLU → Dropout → Linear(H, C)
│     → logits_A [B, C]
│
├─ Branch B (nnMIL)
│  ├─ Optional feature projection (only if D ≠ H)
│  ├─ Gated attention in H-dimensional subspace
│  │  ├─ V branch: tanh(Linear(D, H))
│  │  ├─ U branch: sigmoid(Linear(D, H))
│  │  └─ Attention: softmax(Linear(V ⊙ U, 1))
│  ├─ Aggregate in FULL D-dimensional space
│  └─ Classifier: Linear(D, H) → ReLU → Dropout(0.25) → Linear(H, C)
│     → logits_B [B, C]
│
└─ Fusion
   ├─ Gate: σ(gate_param) ∈ (0, 1)
   └─ Output: gate * logits_A + (1 - gate) * logits_B
      → final_logits [B, C]
```

## Key Design Decisions

### 1. Positional Encoding Disabled
- **Rationale**: Random sub-bag sampling during training means absolute positions are not meaningful
- **Implementation**: `use_pos_encoding=False` by default in TransnnMIL
- **Impact**: Model focuses on patch content rather than spatial arrangement

### 2. Learnable Fusion Gate
- **Initialization**: `gate_param = 0.0` → `sigmoid(0) = 0.5` (equal weight)
- **Training**: Gate learns optimal balance between branches
- **Interpretation**: 
  - `gate → 1.0`: Model relies more on TransMIL (transformer attention)
  - `gate → 0.0`: Model relies more on nnMIL (gated attention)
  - `gate ≈ 0.5`: Balanced fusion

### 3. Differential Dropout Rates
- **TransMIL branch**: `dropout = 0.1` (standard for transformers)
- **nnMIL branch**: `dropout = 0.25` (higher, as per nnMIL paper)
- **Rationale**: nnMIL uses higher dropout for regularization

### 4. Attention Aggregation Strategy
- **TransMIL**: Attention computed via transformer self-attention, aggregated via CLS token
- **nnMIL**: Gated attention computed in H-dim subspace, aggregation in full D-dim space
- **Returned attention**: From TransMIL branch for interpretability

## Files Created/Modified

### New Files
1. **`src/models/transnnmil.py`** (283 lines)
   - Complete TransnnMIL class implementation
   - Dual-branch forward pass with fusion
   - Helper methods: `get_gate_value()`, `get_branch_outputs()`
   - Comprehensive docstrings and examples

2. **`test_transnnmil.py`** (250 lines)
   - 6 comprehensive test cases
   - Validates model instantiation, forward pass, attention, factory, branches, and learning
   - All tests pass successfully

### Modified Files
1. **`src/models/factory.py`**
   - Added `from .transnnmil import TransnnMIL` import
   - Added `elif model_type == "transnnmil":` block
   - Updated docstrings to mention TransnnMIL
   - Updated error message to include transnnmil

## Usage

### 1. Direct Instantiation
```python
from src.models.transnnmil import TransnnMIL

model = TransnnMIL(
    feature_dim=1024,      # Foundation model embedding dimension
    hidden_dim=256,        # Hidden dimension for both branches
    num_classes=2,         # Binary classification
    num_layers=2,          # Transformer layers in Branch A
    num_heads=8,           # Attention heads in Branch A
    dropout=0.1,           # Dropout for TransMIL branch
    use_pos_encoding=False # Disabled for random sub-bags
)

# Forward pass
features = torch.randn(4, 100, 1024)  # [batch, patches, features]
num_patches = torch.tensor([100, 80, 90, 100])
logits = model(features, num_patches)

# Get attention weights
logits, attention = model(features, num_patches, return_attention=True)

# Check gate value
gate = model.get_gate_value()
print(f"TransMIL weight: {gate:.3f}, nnMIL weight: {1-gate:.3f}")
```

### 2. Factory Pattern (Recommended)
```python
from src.models.factory import create_attention_model

config = {
    'model_type': 'transnnmil',
    'hidden_dim': 256,
    'num_classes': 2,
    'dropout': 0.1,
    'transnnmil': {
        'num_layers': 2,
        'num_heads': 8,
        'use_pos_encoding': False
    }
}

model = create_attention_model(config, feature_dim=1024)
```

### 3. Training Script
```bash
# Train TransnnMIL on PatchCamelyon
python train.py --model transnnmil --dataset pcam --epochs 20

# Train with custom config
python train.py --config configs/transnnmil_pcam.yaml
```

### 4. Analyzing Branch Contributions
```python
# Get individual branch outputs
logits_a, logits_b, logits_fused = model.get_branch_outputs(features, num_patches)

# Compute branch agreement
preds_a = logits_a.argmax(dim=1)
preds_b = logits_b.argmax(dim=1)
agreement = (preds_a == preds_b).float().mean()
print(f"Branch agreement: {agreement:.2%}")

# Analyze gate evolution during training
gate_values = []
for epoch in range(num_epochs):
    # ... training loop ...
    gate_values.append(model.get_gate_value())

# Plot gate evolution
import matplotlib.pyplot as plt
plt.plot(gate_values)
plt.xlabel('Epoch')
plt.ylabel('Gate Value (TransMIL weight)')
plt.title('Fusion Gate Evolution During Training')
plt.show()
```

## Compatibility

### ✅ Compatible With
- **nnMILTrainer**: No modifications needed
- **FixedLengthBagSampler**: Works with random sub-bags and sliding windows
- **Task-aware batch samplers**: Balanced, regression, survival
- **Existing training scripts**: Just change `--model transnnmil`
- **Inference pipelines**: Sliding window uncertainty estimation
- **Multi-GPU training**: DDP compatible

### ⚠️ Not Yet Implemented
- Multi-scale support (can be added if needed)
- Custom fusion strategies beyond scalar gate
- Branch-specific learning rates

## Testing

Run the comprehensive test suite:
```bash
python test_transnnmil.py
```

**Test Results:**
```
✓ Test 1: Direct Instantiation - PASSED
✓ Test 2: Forward Pass - PASSED
✓ Test 3: Attention Weights - PASSED
✓ Test 4: Factory Creation - PASSED
✓ Test 5: Branch Outputs - PASSED
✓ Test 6: Gate Parameter Learning - PASSED

ALL TESTS PASSED!
```

## Performance Expectations

### Computational Cost
- **Parameters**: ~2x single branch (both TransMIL and nnMIL)
- **FLOPs**: ~2x single branch (parallel processing)
- **Memory**: Similar to TransMIL (transformer dominates)
- **Training time**: ~10-20% slower than single branch

### Expected Benefits
1. **Robustness**: Fusion reduces reliance on single architecture
2. **Adaptability**: Gate learns task-specific balance
3. **Uncertainty**: Branch disagreement indicates uncertainty
4. **Interpretability**: Can analyze which branch contributes more

### Baseline Comparisons
- **vs TransMIL alone**: May improve on tasks where local attention helps
- **vs nnMIL alone**: May improve on tasks requiring global context
- **vs ensemble**: More efficient than separate models (shared computation)

## Uncertainty Estimation

TransnnMIL supports uncertainty estimation via:

### 1. Sliding Window Variance
```python
# Inference with sliding windows
windows = sampler.get_sliding_windows(slide_features)
predictions = []

for window in windows:
    logits = model(window)
    predictions.append(logits.softmax(dim=1))

# Slide-level prediction (mean)
slide_pred = torch.stack(predictions).mean(dim=0)

# Uncertainty (variance)
slide_uncertainty = torch.stack(predictions).var(dim=0)
```

### 2. Branch Disagreement
```python
# Get branch outputs
logits_a, logits_b, _ = model.get_branch_outputs(features, num_patches)

# Compute disagreement
probs_a = logits_a.softmax(dim=1)
probs_b = logits_b.softmax(dim=1)
disagreement = (probs_a - probs_b).abs().mean(dim=1)

# High disagreement → high uncertainty
uncertain_samples = disagreement > threshold
```

## Future Extensions

### Potential Improvements
1. **Adaptive gate per sample**: Learn sample-specific fusion weights
2. **Multi-scale support**: Extend to multi-magnification inputs
3. **Attention fusion**: Fuse attention weights instead of logits
4. **Branch-specific losses**: Train branches with different objectives
5. **Dynamic branching**: Skip branches based on confidence

### Research Questions
1. Does the gate converge to favor one branch?
2. How does gate value correlate with task difficulty?
3. Can branch disagreement predict model errors?
4. Does fusion improve calibration?

## References

- **TransMIL**: Shao et al., "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification", NeurIPS 2021
- **nnMIL**: Stanford/NIH, "No-New-Net Multiple Instance Learning for Histopathology", arXiv 2024
- **MIL Survey**: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018

## Support

For issues or questions:
1. Check test suite: `python test_transnnmil.py`
2. Review docstrings in `src/models/transnnmil.py`
3. Open GitHub issue with error details

## License

Same as parent repository (MIT License)

---

**Implementation Date**: 2026
**Status**: ✅ Complete and tested
**Ready for**: Immediate training and experimentation
