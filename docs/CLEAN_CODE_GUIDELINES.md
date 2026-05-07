# Clean Code Guidelines

## Overview
This document outlines clean code practices for the HistoCore project.

## Principles

### 1. No Magic Numbers
❌ **Bad:**
```python
if len(password) < 8:
    raise ValueError("Password too short")
```

✅ **Good:**
```python
from src.constants import MIN_PASSWORD_LENGTH

if len(password) < MIN_PASSWORD_LENGTH:
    raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
```

### 2. Use Type Hints
❌ **Bad:**
```python
def process_data(data, config):
    return result
```

✅ **Good:**
```python
def process_data(data: np.ndarray, config: Dict[str, Any]) -> ProcessingResult:
    return result
```

### 3. Return Objects, Not Tuples
❌ **Bad:**
```python
def validate_file(file):
    return True, "image/jpeg", "safe_file.jpg"
```

✅ **Good:**
```python
def validate_file(file) -> FileValidationResult:
    return FileValidationResult(
        is_valid=True,
        mime_type="image/jpeg",
        safe_filename="safe_file.jpg"
    )
```

### 4. Use Dataclasses for Configuration
❌ **Bad:**
```python
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}
```

✅ **Good:**
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

config = TrainingConfig()
```

### 5. Extract Complex Functions
❌ **Bad:**
```python
def train_epoch(model, data, optimizer):
    # 500 lines of code
    pass
```

✅ **Good:**
```python
def train_epoch(model, data, optimizer):
    metrics = TrainingMetrics()
    for batch in data:
        loss = process_batch(model, batch, optimizer)
        metrics.update(loss)
    return metrics.get_results()

def process_batch(model, batch, optimizer):
    # Focused function
    pass
```

### 6. Use Enums for Constants
❌ **Bad:**
```python
if status_code == 200:
    pass
elif status_code == 404:
    pass
```

✅ **Good:**
```python
from src.http_status import HTTPStatus

if status_code == HTTPStatus.OK:
    pass
elif status_code == HTTPStatus.NOT_FOUND:
    pass
```

### 7. Single Responsibility Principle
Each function should do one thing well.

❌ **Bad:**
```python
def process_and_save_and_validate(data):
    # Does too many things
    validated = validate(data)
    processed = process(validated)
    save(processed)
    return processed
```

✅ **Good:**
```python
def validate_data(data) -> ValidationResult:
    # Only validates
    pass

def process_data(data) -> ProcessedData:
    # Only processes
    pass

def save_data(data) -> None:
    # Only saves
    pass
```

### 8. Meaningful Names
❌ **Bad:**
```python
def f(x, y):
    return x + y

data = [1, 2, 3]
for i in data:
    print(i)
```

✅ **Good:**
```python
def calculate_total_cost(base_price: float, tax_rate: float) -> float:
    return base_price + (base_price * tax_rate)

patient_ids = [1, 2, 3]
for patient_id in patient_ids:
    print(patient_id)
```

### 9. Early Returns
❌ **Bad:**
```python
def process(data):
    if data is not None:
        if len(data) > 0:
            if validate(data):
                return process_valid_data(data)
    return None
```

✅ **Good:**
```python
def process(data):
    if data is None:
        return None
    if len(data) == 0:
        return None
    if not validate(data):
        return None
    return process_valid_data(data)
```

### 10. Avoid Deep Nesting
❌ **Bad:**
```python
if condition1:
    if condition2:
        if condition3:
            if condition4:
                do_something()
```

✅ **Good:**
```python
if not condition1:
    return
if not condition2:
    return
if not condition3:
    return
if not condition4:
    return
do_something()
```

## Function Length
- **Maximum:** 50 lines
- **Ideal:** 20 lines
- **If longer:** Extract helper functions

## Cyclomatic Complexity
- **Maximum:** 10
- **Ideal:** 5
- **If higher:** Simplify logic or extract functions

## Module Organization
```
src/
├── constants.py          # All constants
├── http_status.py        # HTTP status codes
├── config/
│   └── experiment_config.py  # Configuration dataclasses
├── models/
│   └── results.py        # Result objects
├── training/
│   └── training_utils.py # Training helpers
└── utils/
    ├── validators.py     # Validation functions
    └── helpers.py        # Helper functions
```

## Testing
Every refactored function should have tests:
```python
def test_validate_password():
    # Test with valid password
    result = validate_password("SecurePass123!")
    assert result.is_valid
    
    # Test with weak password
    result = validate_password("weak")
    assert not result.is_valid
```

## Documentation
Every public function needs a docstring:
```python
def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> MetricsResult:
    """Calculate classification metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        
    Returns:
        MetricsResult with accuracy, F1, AUC
        
    Raises:
        ValueError: If arrays have different lengths
    """
    pass
```

## Code Review Checklist
- [ ] No magic numbers
- [ ] Type hints on all functions
- [ ] Functions < 50 lines
- [ ] Cyclomatic complexity < 10
- [ ] Meaningful variable names
- [ ] Single responsibility per function
- [ ] Early returns used
- [ ] No deep nesting (max 3 levels)
- [ ] Docstrings on public functions
- [ ] Tests for new code

## Tools
```bash
# Check complexity
radon cc src/ -a -nb

# Check code quality
pylint src/

# Format code
black src/

# Type checking
mypy src/

# Run tests
pytest tests/
```

## References
- [Clean Code by Robert C. Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
