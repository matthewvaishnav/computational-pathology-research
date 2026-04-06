# Contributing to Computational Pathology Research Repository

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility for mistakes

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- CUDA-capable GPU (optional, for GPU training)

### Setting Up Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/computational-pathology.git
   cd computational-pathology
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Development dependencies
   pip install -r requirements-dev.txt
   
   # Or use make
   make install-dev
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   make test
   make lint
   ```

## Development Workflow

### Branch Strategy

- `main`: Stable production code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write code** following our [coding standards](#coding-standards)
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run tests** to ensure nothing breaks:
   ```bash
   make test
   ```
5. **Check code quality**:
   ```bash
   make lint
   make format
   make type-check
   ```

### Committing Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <subject>

# Examples:
git commit -m "feat(models): add attention mechanism to fusion module"
git commit -m "fix(data): handle missing clinical features"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(encoders): add unit tests for WSI encoder"
```

**Commit Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort

### Code Formatting

We use automated formatters:

```bash
# Format code
make format

# Or manually:
black src/ tests/ --line-length 100
isort src/ tests/ --profile black
```

### Type Hints

Use type hints for all function signatures:

```python
def process_features(
    features: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Process input features."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 100
) -> Dict[str, float]:
    """
    Train the model.
    
    Args:
        model: PyTorch model to train
        data_loader: DataLoader for training data
        epochs: Number of training epochs
    
    Returns:
        Dictionary containing training metrics
    
    Raises:
        ValueError: If epochs is negative
    
    Example:
        >>> model = MultimodalFusionModel()
        >>> loader = DataLoader(dataset)
        >>> metrics = train_model(model, loader, epochs=10)
    """
    pass
```

### Code Organization

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path

# 2. Third-party imports
import numpy as np
import torch
import torch.nn as nn

# 3. Local imports
from src.models.encoders import WSIEncoder
from src.data.loaders import MultimodalDataset
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

```python
def test_wsi_encoder_output_shape():
    """Test that WSI encoder produces correct output shape."""
    encoder = WSIEncoder(input_dim=224, embed_dim=256)
    input_tensor = torch.randn(32, 224)
    output = encoder(input_tensor)
    assert output.shape == (32, 256)
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_encoders.py

# Run specific test
pytest tests/test_encoders.py::test_wsi_encoder_output_shape

# Run in watch mode
make test-watch
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

```bash
# Generate coverage report
make test-cov

# View HTML report
open htmlcov/index.html
```

## Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Include type hints
- Provide usage examples for complex functions

### README Updates

Update README.md when:
- Adding new features
- Changing installation process
- Modifying usage instructions
- Adding dependencies

### API Documentation

We use Sphinx for API documentation:

```bash
# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout your-branch
   git rebase develop
   ```

2. **Run all checks**:
   ```bash
   make check-all
   ```

3. **Update documentation**:
   - Update README if needed
   - Add docstrings to new code
   - Update CHANGELOG.md

### Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** on GitHub:
   - Use a descriptive title
   - Reference related issues
   - Describe changes in detail
   - Add screenshots if applicable

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Updated documentation
   
   ## Related Issues
   Closes #123
   
   ## Screenshots (if applicable)
   ```

### Review Process

- At least one approval required
- All CI checks must pass
- Address reviewer feedback
- Keep PR focused and small

### After Approval

- Squash commits if requested
- Maintainer will merge PR
- Delete feature branch after merge

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10]
- PyTorch version: [e.g., 2.0.0]

**Additional context**
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of desired feature

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other relevant information
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: [maintainer@example.com]

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in publications (for significant contributions)

### Maintainers

Current maintainers:
- [Name] (@username) - Lead Maintainer
- [Name] (@username) - Core Contributor

## Development Tips

### Useful Commands

```bash
# Quick development check
make dev-check

# Run demos
make demo

# Profile performance
python scripts/profile.py --checkpoint checkpoints/best_model.pth

# Export model
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx
```

### Debugging

```bash
# Run with debugger
python -m pdb experiments/train.py

# Use ipdb for better debugging
pip install ipdb
# Add breakpoint in code:
import ipdb; ipdb.set_trace()
```

### Performance Optimization

- Profile before optimizing
- Use PyTorch profiler for bottlenecks
- Consider mixed precision training
- Optimize data loading pipeline

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Questions?

Don't hesitate to ask questions! We're here to help:
- Open an issue with the `question` label
- Start a discussion on GitHub Discussions
- Contact maintainers directly

Thank you for contributing! 🎉
