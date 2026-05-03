# Example: Property-Based Test Strategy

## Problem Description

Design property-based test strategy for medical image preprocessing pipeline. Need to verify correctness properties (normalization preserves relative intensities, augmentation preserves labels, pipeline is deterministic with fixed seed) using Hypothesis framework.

## Classification

```typescript
const classifier = new ProblemClassifier();
const result = classifier.classifyProblem(
  'Design property-based test strategy for image preprocessing pipeline'
);

// Result:
// {
//   shouldDelegate: true,
//   classification: {
//     delegationType: 'test_strategy',
//     complexity: 'moderate',
//     requiredContext: [
//       'preprocessing_code',
//       'existing_tests',
//       'data_schemas'
//     ]
//   },
//   recommendation: 'Formal correctness properties with property-based testing. Opus delegation recommended.'
// }
```

## Context Bundle

```markdown
# Context Bundle: Property-Based Test Strategy

## Problem Summary
Design property-based test strategy for medical image preprocessing pipeline. Verify correctness properties: normalization preserves relative intensities, augmentation preserves labels, pipeline deterministic with fixed seed.

## Relevant Code

### src/preprocessing/normalize.py (lines 10-35)
```python
def normalize_image(image: np.ndarray, method: str = 'z_score') -> np.ndarray:
    """Normalize image intensities"""
    if method == 'z_score':
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return image - mean
        return (image - mean) / std
    elif method == 'min_max':
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val == min_val:
            return np.zeros_like(image)
        return (image - min_val) / (max_val - min_val)
    else:
        raise ValueError(f'Unknown normalization method: {method}')
```

### src/preprocessing/augment.py (lines 15-50)
```python
def augment_image(
    image: np.ndarray,
    label: int,
    rotation_range: float = 15.0,
    flip_horizontal: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Apply random augmentations to image"""
    if seed is not None:
        np.random.seed(seed)
    
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    image = rotate(image, angle, reshape=False)
    
    # Random horizontal flip
    if flip_horizontal and np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # Label should not change
    return image, label
```

### src/preprocessing/pipeline.py (lines 20-45)
```python
class PreprocessingPipeline:
    """Image preprocessing pipeline"""
    def __init__(
        self,
        normalize: bool = True,
        augment: bool = False,
        seed: Optional[int] = None
    ):
        self.normalize = normalize
        self.augment = augment
        self.seed = seed
    
    def process(
        self,
        image: np.ndarray,
        label: int
    ) -> Tuple[np.ndarray, int]:
        """Process image through pipeline"""
        if self.normalize:
            image = normalize_image(image, method='z_score')
        
        if self.augment:
            image, label = augment_image(
                image, label, seed=self.seed
            )
        
        return image, label
```

## Documentation Excerpts

**From:** docs/preprocessing.md
Preprocessing pipeline: normalize → augment → output. Normalization uses z-score (mean=0, std=1). Augmentation: random rotation (±15°), horizontal flip (50% chance). Pipeline must be deterministic when seed provided for reproducibility.

**From:** tests/test_preprocessing.py
Current tests: unit tests for each function with fixed inputs. No property-based tests. No tests for edge cases (zero std, constant images, extreme rotations).

## Constraints
- Images: 96x96 RGB, uint8 (0-255)
- Labels: binary (0=benign, 1=malignant)
- Normalization must preserve relative intensities
- Augmentation must preserve labels
- Pipeline must be deterministic with fixed seed
- Must handle edge cases (zero std, constant images)

## Context Manifest

| Source | Type | Size | Relevance |
|--------|------|------|-----------|
| src/preprocessing/normalize.py | Code | 1.2 KB | High |
| src/preprocessing/augment.py | Code | 1.5 KB | High |
| src/preprocessing/pipeline.py | Code | 1.3 KB | High |
| docs/preprocessing.md | Doc | 0.8 KB | Medium |
| tests/test_preprocessing.py | Code | 2.0 KB | Medium |
```

## Delegation Request

```markdown
# Delegation Request: Property-Based Test Strategy

## Objective
Design a comprehensive property-based test strategy for medical image preprocessing pipeline using Hypothesis framework. The strategy must verify key correctness properties: normalization preserves relative intensities, augmentation preserves labels, pipeline is deterministic with fixed seed, and edge cases are handled correctly.

## Background
Current testing uses unit tests with fixed inputs. This misses edge cases and doesn't verify general correctness properties. Need property-based tests to:
- Generate diverse test inputs automatically
- Verify properties hold for all valid inputs
- Find edge cases that break assumptions
- Provide confidence in correctness

## Expected Artifacts
Please generate the following:

1. **Test Strategy Document** — Markdown document including:
   - Correctness properties to verify
   - Test case categories (normalization, augmentation, pipeline, edge cases)
   - Hypothesis strategies for generating test data
   - Expected test coverage and success criteria

2. **Test Implementation Guide** — Detailed implementation steps:
   - Property 1: Normalization preserves relative intensities
   - Property 2: Augmentation preserves labels
   - Property 3: Pipeline determinism with fixed seed
   - Property 4: Edge case handling (zero std, constant images)
   - Hypothesis strategies for each property
   - Example test code with Hypothesis decorators

3. **Test Data Generators** — Python code for Hypothesis strategies:
   - Image generator (96x96 RGB, various distributions)
   - Label generator (binary)
   - Edge case generators (zero std, constant, extreme values)
   - Composite generators for pipeline testing

## Output Format Requirements
- Strategy: Use markdown with sections for each property
- Implementation guide: Use markdown numbered lists with code examples
- Generators: Use Python code blocks with Hypothesis strategies

## Context
[Context Bundle inserted here]

## Questions to Address
1. What correctness properties should be verified?
2. How to generate diverse test images with Hypothesis?
3. How to verify "normalization preserves relative intensities"?
4. How to test determinism with fixed seed?
5. What edge cases need explicit testing?
6. How to structure tests for maintainability?
7. What test coverage is sufficient?
8. How to handle flaky tests from randomness?

## Success Criteria
- All key correctness properties have property-based tests
- Tests generate diverse inputs (at least 100 examples per property)
- Tests find edge cases automatically
- Tests are deterministic and reproducible
- Test suite runs in <30 seconds
- Clear failure messages when properties violated
```

## Opus Response (Sample)

```markdown
# Property-Based Test Strategy for Image Preprocessing Pipeline

## Correctness Properties

### Property 1: Normalization Preserves Relative Intensities

**Formal Statement:** For any two pixels p1, p2 in image I, if p1 > p2 before normalization, then normalize(p1) > normalize(p2) after normalization (assuming non-constant image).

**Rationale:** Normalization should only rescale intensities, not reorder them. This ensures that relative brightness relationships are preserved.

**Test Approach:**
- Generate random images with known intensity relationships
- Apply normalization
- Verify ordering preserved for all pixel pairs

### Property 2: Normalization is Bounded

**Formal Statement:** For z-score normalization, output values should be approximately in range [-3, 3] for 99.7% of pixels (assuming normal distribution). For min-max normalization, output should be exactly in [0, 1].

**Rationale:** Normalized values should be in expected ranges to prevent downstream issues.

**Test Approach:**
- Generate random images
- Apply normalization
- Verify output ranges

### Property 3: Augmentation Preserves Labels

**Formal Statement:** For any image I with label L, augment(I, L) = (I', L) where I' is augmented image and label remains L.

**Rationale:** Geometric transformations (rotation, flip) should not change image label.

**Test Approach:**
- Generate random images with labels
- Apply augmentation
- Verify label unchanged

### Property 4: Pipeline Determinism with Fixed Seed

**Formal Statement:** For any image I, label L, and seed S, pipeline(I, L, seed=S) produces identical output across multiple runs.

**Rationale:** Reproducibility is critical for debugging and scientific validity.

**Test Approach:**
- Generate random images
- Process with fixed seed multiple times
- Verify outputs identical

### Property 5: Edge Case Handling

**Formal Statement:** Pipeline handles edge cases without errors:
- Constant images (zero std)
- Extreme values (all 0, all 255)
- Single-pixel differences

**Rationale:** Real-world data contains edge cases that must be handled gracefully.

**Test Approach:**
- Generate edge case images explicitly
- Verify no exceptions raised
- Verify reasonable outputs

## Test Implementation Guide

### Test 1: Normalization Preserves Relative Intensities

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
from src.preprocessing.normalize import normalize_image

@given(
    image=npst.arrays(
        dtype=np.float32,
        shape=(96, 96, 3),
        elements=st.floats(min_value=0, max_value=255, allow_nan=False)
    ).filter(lambda x: np.std(x) > 1e-6)  # Exclude constant images
)
def test_normalization_preserves_relative_intensities(image):
    """Property: Normalization preserves pixel ordering"""
    normalized = normalize_image(image, method='z_score')
    
    # Sample random pixel pairs
    for _ in range(100):
        i1, j1, k1 = np.random.randint(0, 96), np.random.randint(0, 96), np.random.randint(0, 3)
        i2, j2, k2 = np.random.randint(0, 96), np.random.randint(0, 96), np.random.randint(0, 3)
        
        original_order = image[i1, j1, k1] > image[i2, j2, k2]
        normalized_order = normalized[i1, j1, k1] > normalized[i2, j2, k2]
        
        assert original_order == normalized_order, \
            f"Ordering violated: ({i1},{j1},{k1}) vs ({i2},{j2},{k2})"
```

### Test 2: Normalization is Bounded

```python
@given(
    image=npst.arrays(
        dtype=np.float32,
        shape=(96, 96, 3),
        elements=st.floats(min_value=0, max_value=255, allow_nan=False)
    ).filter(lambda x: np.std(x) > 1e-6)
)
def test_z_score_normalization_bounded(image):
    """Property: Z-score normalized values in reasonable range"""
    normalized = normalize_image(image, method='z_score')
    
    # 99.7% of values should be in [-3, 3] for normal distribution
    # Use [-5, 5] as conservative bound
    assert np.all(np.abs(normalized) < 5), \
        f"Normalized values out of range: min={np.min(normalized)}, max={np.max(normalized)}"

@given(
    image=npst.arrays(
        dtype=np.float32,
        shape=(96, 96, 3),
        elements=st.floats(min_value=0, max_value=255, allow_nan=False)
    ).filter(lambda x: np.max(x) > np.min(x))  # Exclude constant images
)
def test_min_max_normalization_bounded(image):
    """Property: Min-max normalized values in [0, 1]"""
    normalized = normalize_image(image, method='min_max')
    
    assert np.all(normalized >= 0) and np.all(normalized <= 1), \
        f"Normalized values out of [0, 1]: min={np.min(normalized)}, max={np.max(normalized)}"
```

### Test 3: Augmentation Preserves Labels

```python
from src.preprocessing.augment import augment_image

@given(
    image=npst.arrays(
        dtype=np.float32,
        shape=(96, 96, 3),
        elements=st.floats(min_value=0, max_value=255, allow_nan=False)
    ),
    label=st.integers(min_value=0, max_value=1),
    seed=st.integers(min_value=0, max_value=2**31 - 1)
)
def test_augmentation_preserves_labels(image, label, seed):
    """Property: Augmentation does not change labels"""
    augmented_image, augmented_label = augment_image(
        image, label, seed=seed
    )
    
    assert augmented_label == label, \
        f"Label changed: {label} -> {augmented_label}"
```

### Test 4: Pipeline Determinism with Fixed Seed

```python
from src.preprocessing.pipeline import PreprocessingPipeline

@given(
    image=npst.arrays(
        dtype=np.float32,
        shape=(96, 96, 3),
        elements=st.floats(min_value=0, max_value=255, allow_nan=False)
    ),
    label=st.integers(min_value=0, max_value=1),
    seed=st.integers(min_value=0, max_value=2**31 - 1)
)
def test_pipeline_determinism(image, label, seed):
    """Property: Pipeline with fixed seed produces identical outputs"""
    pipeline = PreprocessingPipeline(
        normalize=True,
        augment=True,
        seed=seed
    )
    
    # Run pipeline twice with same seed
    output1, label1 = pipeline.process(image.copy(), label)
    output2, label2 = pipeline.process(image.copy(), label)
    
    assert np.allclose(output1, output2, rtol=1e-6), \
        "Pipeline outputs differ with same seed"
    assert label1 == label2, \
        "Labels differ with same seed"
```

### Test 5: Edge Case Handling

```python
@given(
    constant_value=st.floats(min_value=0, max_value=255, allow_nan=False)
)
def test_constant_image_normalization(constant_value):
    """Property: Constant images handled gracefully"""
    image = np.full((96, 96, 3), constant_value, dtype=np.float32)
    
    # Should not raise exception
    normalized = normalize_image(image, method='z_score')
    
    # Output should be zero (or constant - mean)
    assert np.all(np.abs(normalized) < 1e-6), \
        "Constant image normalization failed"

def test_extreme_value_images():
    """Property: Extreme value images handled correctly"""
    # All zeros
    image_zeros = np.zeros((96, 96, 3), dtype=np.float32)
    normalized_zeros = normalize_image(image_zeros, method='z_score')
    assert np.all(np.abs(normalized_zeros) < 1e-6)
    
    # All 255
    image_max = np.full((96, 96, 3), 255, dtype=np.float32)
    normalized_max = normalize_image(image_max, method='z_score')
    assert np.all(np.abs(normalized_max) < 1e-6)
    
    # Single pixel different
    image_single = np.zeros((96, 96, 3), dtype=np.float32)
    image_single[0, 0, 0] = 255
    normalized_single = normalize_image(image_single, method='z_score')
    # Should not raise exception
    assert normalized_single.shape == (96, 96, 3)
```

## Test Data Generators

```python
from hypothesis import strategies as st
import hypothesis.extra.numpy as npst
import numpy as np

# Basic image generator
image_strategy = npst.arrays(
    dtype=np.float32,
    shape=(96, 96, 3),
    elements=st.floats(min_value=0, max_value=255, allow_nan=False)
)

# Non-constant image generator (for normalization tests)
non_constant_image_strategy = image_strategy.filter(
    lambda x: np.std(x) > 1e-6
)

# Label generator
label_strategy = st.integers(min_value=0, max_value=1)

# Seed generator
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)

# Edge case generators
constant_image_strategy = st.builds(
    lambda val: np.full((96, 96, 3), val, dtype=np.float32),
    st.floats(min_value=0, max_value=255, allow_nan=False)
)

extreme_image_strategy = st.sampled_from([
    np.zeros((96, 96, 3), dtype=np.float32),  # All zeros
    np.full((96, 96, 3), 255, dtype=np.float32),  # All 255
    np.random.uniform(0, 1, (96, 96, 3)).astype(np.float32),  # Very small values
    np.random.uniform(254, 255, (96, 96, 3)).astype(np.float32),  # Very large values
])

# Composite generator for pipeline testing
@st.composite
def pipeline_input_strategy(draw):
    """Generate (image, label, seed) tuple for pipeline testing"""
    image = draw(image_strategy)
    label = draw(label_strategy)
    seed = draw(seed_strategy)
    return image, label, seed
```

## Test Coverage and Success Criteria

### Coverage Targets

| Component | Property Tests | Edge Case Tests | Total Examples |
|-----------|---------------|-----------------|----------------|
| Normalization | 3 | 3 | 300 |
| Augmentation | 1 | 0 | 100 |
| Pipeline | 1 | 0 | 100 |
| **Total** | **5** | **3** | **500** |

### Success Criteria

1. **All properties pass** — No property violations found in 100+ examples
2. **Edge cases handled** — No exceptions for constant images, extreme values
3. **Determinism verified** — Fixed seed produces identical outputs
4. **Fast execution** — Test suite completes in <30 seconds
5. **Clear failures** — Failure messages include counterexamples

### Running Tests

```bash
# Run all property-based tests
pytest tests/test_preprocessing_properties.py -v

# Run with more examples (slower, more thorough)
pytest tests/test_preprocessing_properties.py -v --hypothesis-examples=1000

# Run with specific seed for reproducibility
pytest tests/test_preprocessing_properties.py -v --hypothesis-seed=12345
```

## Maintenance and Extension

### Adding New Properties

1. **Identify property** — What invariant should hold?
2. **Write formal statement** — Precise mathematical description
3. **Create test** — Use `@given` decorator with appropriate strategies
4. **Verify with examples** — Run with 100+ examples
5. **Document** — Add to test strategy document

### Handling Flaky Tests

- **Use fixed seeds** — `--hypothesis-seed=X` for reproducibility
- **Increase tolerance** — Use `np.allclose()` with appropriate `rtol`/`atol`
- **Filter invalid inputs** — Use `.filter()` to exclude edge cases
- **Shrink counterexamples** — Hypothesis automatically finds minimal failing example

### Performance Optimization

- **Reduce examples** — Use `@settings(max_examples=50)` for faster tests
- **Profile slow tests** — Use `--hypothesis-profile=dev` for faster feedback
- **Parallelize** — Use `pytest-xdist` for parallel execution
