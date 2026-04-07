# Input Validation Module

Comprehensive input validation for the computational pathology framework. Provides validation functions for tensor operations, model inputs, and multimodal batch data with helpful error messages.

## Features

- ✅ **Core tensor validation**: Shape, range, NaN/Inf detection, batch size
- ✅ **Modality-specific validation**: WSI features, genomic data, clinical text
- ✅ **Batch validation**: Complete multimodal batch validation with missing modality support
- ✅ **Decorator support**: `@validate_inputs` for automatic validation
- ✅ **Production-ready**: Can be disabled via environment variable
- ✅ **Helpful error messages**: Clear explanations with suggestions for fixing issues
- ✅ **Comprehensive tests**: 54 unit tests with 88% coverage

## Quick Start

```python
import torch
from src.utils.validation import (
    validate_wsi_features,
    validate_genomic_features,
    validate_multimodal_batch,
    validate_inputs
)

# Validate WSI features
wsi_features = torch.randn(16, 100, 1024)
validate_wsi_features(wsi_features)

# Validate complete batch
batch = {
    'wsi_features': torch.randn(16, 100, 1024),
    'genomic': torch.randn(16, 2000),
    'clinical_text': torch.randint(0, 30000, (16, 128)),
    'labels': torch.randint(0, 2, (16,))
}
validate_multimodal_batch(batch)

# Use decorator for automatic validation
class MyModel(nn.Module):
    @validate_inputs
    def forward(self, batch):
        # Validation happens automatically
        return self.process(batch)
```

## Core Validation Functions

### `validate_tensor_shape(tensor, expected_shape, name)`

Validates tensor shape with support for variable dimensions.

```python
tensor = torch.randn(16, 100, 1024)

# Exact shape match
validate_tensor_shape(tensor, (16, 100, 1024), "wsi_features")

# Variable batch size (use None)
validate_tensor_shape(tensor, (None, 100, 1024), "wsi_features")
```

**Error Example:**
```
Shape mismatch for wsi_features at dimension 2:
  Expected: (16, 100, 1024)
  Received: (16, 100, 512)
  Dimension 2: expected 1024, got 512
Suggestion: Verify the input preprocessing and data loading pipeline
```

### `validate_tensor_range(tensor, min_val, max_val, name)`

Validates tensor values are within expected range.

```python
tensor = torch.randn(16, 1024) * 0.1

# Check normalized range
validate_tensor_range(tensor, -1.0, 1.0, "normalized_features")

# Check only minimum
validate_tensor_range(tensor, min_val=0.0, name="positive_features")
```

**Error Example:**
```
Value range error for normalized_features:
  Expected maximum: 1.0
  Actual maximum: 5.234567
  Actual range: [-0.123456, 5.234567]
Suggestion: Check normalization or scaling of input data
```

### `validate_no_nan_inf(tensor, name)`

Validates tensor contains no NaN or Inf values.

```python
tensor = torch.randn(16, 1024)
validate_no_nan_inf(tensor, "features")
```

**Error Example:**
```
Invalid values detected in features:
  NaN values: 10 / 16384 (0.06%)
  Inf values: 2 / 16384 (0.01%)
Suggestions:
  - Check for division by zero in preprocessing
  - Verify data loading pipeline for corrupted data
  - Add gradient clipping if this occurs during training
  - Check for numerical instability in model computations
```

### `validate_batch_size(tensor, expected_batch_size, name)`

Validates tensor has expected batch size (first dimension).

```python
tensor = torch.randn(16, 1024)
validate_batch_size(tensor, 16, "features")
```

## Modality-Specific Validation

### `validate_wsi_features(features, expected_feature_dim=1024, name)`

Validates WSI (Whole Slide Image) features.

**Expected shape:** `[batch_size, num_patches, feature_dim]`

```python
wsi_features = torch.randn(16, 100, 1024)
validate_wsi_features(wsi_features)

# Custom feature dimension
wsi_features = torch.randn(16, 100, 2048)
validate_wsi_features(wsi_features, expected_feature_dim=2048)
```

**Checks:**
- 3D tensor shape
- Feature dimension matches expected
- No NaN/Inf values
- At least 1 patch per sample

### `validate_genomic_features(features, expected_feature_dim=2000, name)`

Validates genomic features.

**Expected shape:** `[batch_size, num_genes]`

```python
genomic = torch.randn(16, 2000)
validate_genomic_features(genomic)
```

**Checks:**
- 2D tensor shape
- Gene count matches expected
- No NaN/Inf values

### `validate_clinical_text(tokens, max_seq_length=None, name)`

Validates clinical text token IDs.

**Expected shape:** `[batch_size, seq_length]`

```python
tokens = torch.randint(0, 30000, (16, 128))
validate_clinical_text(tokens, max_seq_length=512)
```

**Checks:**
- 2D tensor shape
- Integer dtype
- No negative token IDs
- Sequence length within limit (if specified)

### `validate_multimodal_batch(batch, expected_batch_size=None, require_all_modalities=False)`

Validates complete multimodal batch dictionary.

```python
batch = {
    'wsi_features': torch.randn(16, 100, 1024),
    'genomic': torch.randn(16, 2000),
    'clinical_text': torch.randint(0, 30000, (16, 128)),
    'labels': torch.randint(0, 2, (16,))
}

# Allow missing modalities
validate_multimodal_batch(batch, require_all_modalities=False)

# Require all modalities
validate_multimodal_batch(batch, require_all_modalities=True)
```

**Checks:**
- Batch is a dictionary
- Consistent batch size across modalities
- Each modality passes its specific validation
- All required modalities present (if specified)

**Supports:**
- Missing modalities (None values)
- Variable-length patch sequences (list of tensors)
- Optional modalities

## Decorator for Automatic Validation

### `@validate_inputs`

Decorator for automatic input validation on model forward methods.

```python
import torch.nn as nn
from src.utils.validation import validate_inputs

class MultimodalModel(nn.Module):
    @validate_inputs
    def forward(self, batch):
        # Batch is automatically validated before this code runs
        wsi = batch['wsi_features']
        genomic = batch['genomic']
        # ... process features
        return output
```

**Features:**
- Validates batch dictionary automatically
- Works with both positional and keyword arguments
- Preserves function name and docstring
- Can be disabled globally

## Disabling Validation

### For Production Use

Validation can be disabled for production to improve performance:

**Method 1: Environment Variable**
```bash
export DISABLE_VALIDATION=1
python train.py
```

**Method 2: Programmatically**
```python
from src.utils.validation import set_validation_enabled

# Disable validation
set_validation_enabled(False)

# Re-enable validation
set_validation_enabled(True)
```

**Method 3: Check Status**
```python
from src.utils.validation import is_validation_enabled

if is_validation_enabled():
    print("Validation is active")
```

## Utility Functions

### `get_validation_summary(batch)`

Generate a summary of batch contents for debugging.

```python
from src.utils.validation import get_validation_summary

batch = {
    'wsi_features': torch.randn(16, 100, 1024),
    'genomic': torch.randn(16, 2000),
    'clinical_text': None,
}

print(get_validation_summary(batch))
```

**Output:**
```
Batch Summary:
  wsi_features: torch.Tensor [16, 100, 1024] dtype=torch.float32
  genomic: torch.Tensor [16, 2000] dtype=torch.float32
  clinical_text: None
```

## Error Messages

All validation errors include:
- **What was expected**: Clear specification of requirements
- **What was received**: Actual values/shapes
- **Suggestions**: Actionable advice for fixing the issue

Example error message:
```
Feature dimension mismatch for wsi_features:
  Expected: [batch_size, num_patches, 1024]
  Received: (16, 100, 512)
  Feature dim: expected 1024, got 512
Suggestion: Verify feature extractor output dimension (e.g., ResNet, ViT)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all validation tests
pytest tests/test_validation.py -v

# Run with coverage
pytest tests/test_validation.py --cov=src.utils.validation --cov-report=html
```

**Test Coverage:**
- 54 unit tests
- 88% code coverage
- Tests for all validation functions
- Tests for error messages
- Tests for decorator functionality
- Tests for enable/disable functionality

## Examples

See `examples/validation_usage.py` for comprehensive usage examples:

```bash
python examples/validation_usage.py
```

Examples include:
1. Basic tensor validation
2. Modality-specific validation
3. Complete batch validation
4. Handling missing modalities
5. Using the decorator
6. Error message demonstrations
7. Disabling validation
8. Variable-length patch sequences

## Integration with Existing Code

### In Data Loaders

```python
from src.utils.validation import validate_multimodal_batch

def collate_fn(batch):
    # ... collate logic
    collated_batch = {
        'wsi_features': ...,
        'genomic': ...,
        'clinical_text': ...,
    }
    
    # Validate before returning
    validate_multimodal_batch(collated_batch)
    return collated_batch
```

### In Training Loop

```python
from src.utils.validation import validate_multimodal_batch, set_validation_enabled

# Enable validation during development
set_validation_enabled(True)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Validate batch
        validate_multimodal_batch(batch)
        
        # Train
        output = model(batch)
        loss = criterion(output, batch['labels'])
        loss.backward()
        optimizer.step()
```

### In Model Definition

```python
from src.utils.validation import validate_inputs

class MultimodalFusionModel(nn.Module):
    @validate_inputs
    def forward(self, batch):
        # Automatic validation
        wsi_emb = self.wsi_encoder(batch['wsi_features'])
        genomic_emb = self.genomic_encoder(batch['genomic'])
        fused = self.fusion_layer([wsi_emb, genomic_emb])
        return fused
```

## Best Practices

1. **Enable during development**: Keep validation enabled to catch issues early
2. **Disable in production**: Set `DISABLE_VALIDATION=1` for performance
3. **Use specific validators**: Use modality-specific validators for better error messages
4. **Validate at boundaries**: Validate data at input boundaries (data loaders, model inputs)
5. **Check error messages**: Read error messages carefully - they include suggestions
6. **Test with validation**: Run tests with validation enabled to catch issues

## Performance Considerations

- Validation adds minimal overhead (~1-5% in most cases)
- Can be completely disabled for production
- Most expensive checks: NaN/Inf detection on large tensors
- Shape checks are very fast
- Decorator adds negligible overhead

## API Reference

### Core Functions
- `validate_tensor_shape(tensor, expected_shape, name)`
- `validate_tensor_range(tensor, min_val, max_val, name)`
- `validate_no_nan_inf(tensor, name)`
- `validate_batch_size(tensor, expected_batch_size, name)`

### Modality Functions
- `validate_wsi_features(features, expected_feature_dim, name)`
- `validate_genomic_features(features, expected_feature_dim, name)`
- `validate_clinical_text(tokens, max_seq_length, name)`
- `validate_multimodal_batch(batch, expected_batch_size, require_all_modalities)`

### Utilities
- `@validate_inputs` - Decorator for automatic validation
- `is_validation_enabled()` - Check if validation is enabled
- `set_validation_enabled(enabled)` - Enable/disable validation
- `get_validation_summary(batch)` - Get batch summary for debugging

### Exceptions
- `ValidationError` - Raised when validation fails

## Contributing

When adding new validation functions:
1. Follow the existing pattern for error messages
2. Include helpful suggestions in error messages
3. Add comprehensive unit tests
4. Update this README with examples
5. Respect the `_VALIDATION_ENABLED` flag

## License

Part of the computational pathology research framework.
