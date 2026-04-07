# Contributing to Computational Pathology Research Framework

Thank you for your interest in contributing to this computational pathology research framework! We welcome contributions from researchers, developers, and domain experts. This guide will help you get started.

## Welcome

We're building a research framework for multimodal fusion architectures in computational pathology. Whether you're fixing bugs, adding features, improving documentation, or proposing new research directions, your contributions are valued and appreciated.

This is a collaborative research project, and we believe in fostering an inclusive, respectful, and productive environment for all contributors.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and considerate in all interactions
- Welcome diverse perspectives and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the research community
- Show empathy towards other community members

Unacceptable behavior includes harassment, discriminatory comments, personal attacks, or any conduct that creates an intimidating or hostile environment. If you experience or witness unacceptable behavior, please report it by opening a confidential issue or contacting the maintainers directly.

## Getting Started

### Prerequisites

- Python 3.9 or higher (3.9, 3.10, 3.11 supported)
- Git for version control
- Basic understanding of PyTorch and deep learning
- Familiarity with computational pathology concepts (helpful but not required)

### Development Environment Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/your-username/computational-pathology-research.git
cd computational-pathology-research
```

2. **Create a virtual environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n pathology python=3.10
conda activate pathology
```

3. **Install dependencies**:
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development tools
pip install -e ".[dev]"

# Or install individually
pip install black>=23.0.0 flake8>=6.0.0 mypy>=1.0.0 isort>=5.12.0
```

4. **Verify installation**:
```bash
# Run tests to ensure everything works
pytest tests/ -v

# Check that imports work
python -c "import src; print('Installation successful!')"
```

5. **Set up pre-commit hooks** (optional but recommended):
```bash
pip install pre-commit
pre-commit install
```

### Understanding the Codebase

Before contributing, familiarize yourself with the repository structure:

- `src/`: Core source code
  - `data/`: Data loading and preprocessing
  - `models/`: Model architectures (encoders, fusion, temporal reasoning)
  - `pretraining/`: Self-supervised learning objectives
- `experiments/`: Training scripts and configurations
- `tests/`: Unit tests and integration tests
- `docs/`: Additional documentation

Key documentation to review:
- `README.md`: Project overview and quick start
- `ARCHITECTURE.md`: Detailed architecture documentation
- `docs/multimodal_architecture.md`: Multimodal fusion details

## Code Style Guidelines

We follow strict code style guidelines to maintain consistency and readability.

### Python Style

We use **Black** for code formatting with a line length of 100 characters:

```bash
# Format your code before committing
black src/ tests/ experiments/ --line-length 100

# Check formatting without making changes
black --check src/ tests/ experiments/
```

### Import Sorting

We use **isort** to organize imports, configured to be compatible with Black:

```bash
# Sort imports
isort src/ tests/ experiments/

# Check import sorting
isort --check-only src/ tests/ experiments/
```

Configuration is in `.isort.cfg`:
```ini
[settings]
profile = black
line_length = 100
```

### Linting

We use **flake8** for linting:

```bash
# Run flake8
flake8 src/ tests/ experiments/ --max-line-length=127 --max-complexity=10

# Critical errors only
flake8 src/ tests/ --select=E9,F63,F7,F82 --show-source
```

### Type Hints

We strongly encourage type hints for all functions and methods:

```python
from typing import Dict, List, Optional, Tuple
import torch

def process_batch(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a batch through the model.
    
    Args:
        batch: Dictionary containing input tensors
        model: PyTorch model for inference
        device: Device to run inference on
        
    Returns:
        Tuple of (embeddings, predictions)
    """
    # Implementation
    pass
```

Run **mypy** for type checking:

```bash
# Type check your code
mypy src/ --ignore-missing-imports --no-strict-optional

# Type check specific file
mypy src/models/multimodal.py
```

### Docstring Style

Use Google-style docstrings for all public functions, classes, and methods:

```python
def train_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float = 1e-4
) -> Dict[str, List[float]]:
    """Train a model on the provided dataset.
    
    This function implements the standard training loop with gradient
    descent optimization.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader providing training batches
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (default: 1e-4)
        
    Returns:
        Dictionary containing training metrics:
            - 'loss': List of loss values per epoch
            - 'accuracy': List of accuracy values per epoch
            
    Raises:
        ValueError: If epochs is less than 1
        RuntimeError: If CUDA is not available but model is on GPU
        
    Example:
        >>> model = MultimodalFusionModel(embed_dim=256)
        >>> dataloader = DataLoader(dataset, batch_size=16)
        >>> metrics = train_model(model, dataloader, epochs=10)
        >>> print(f"Final loss: {metrics['loss'][-1]:.4f}")
    """
    # Implementation
    pass
```

### Code Quality Checklist

Before submitting code, ensure:

- [ ] Code is formatted with Black (line length 100)
- [ ] Imports are sorted with isort
- [ ] No flake8 errors or warnings
- [ ] Type hints are provided for function signatures
- [ ] Docstrings are complete and follow Google style
- [ ] Code passes mypy type checking
- [ ] All tests pass
- [ ] New functionality has corresponding tests
- [ ] Code coverage is maintained or improved

## Testing Requirements

We maintain high testing standards to ensure code reliability.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_multimodal.py -v

# Run specific test function
pytest tests/test_multimodal.py::test_multimodal_fusion -v

# Run tests matching a pattern
pytest tests/ -k "fusion" -v
```

### Coverage Requirements

We aim for **80%+ code coverage** for all core modules:

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
start htmlcov/index.html  # On Windows
```

Coverage thresholds:
- Core models (`src/models/`): **90%+**
- Data loading (`src/data/`): **85%+**
- Utilities and helpers: **80%+**
- Experimental scripts: **70%+**

### Writing Tests

All new functionality must include tests. Use pytest for testing:

```python
# tests/test_new_feature.py
import pytest
import torch
from src.models import NewFeature

class TestNewFeature:
    """Test suite for NewFeature class."""
    
    def test_initialization(self):
        """Test that NewFeature initializes correctly."""
        feature = NewFeature(input_dim=256, output_dim=128)
        assert feature.input_dim == 256
        assert feature.output_dim == 128
    
    def test_forward_pass(self):
        """Test forward pass with valid input."""
        feature = NewFeature(input_dim=256, output_dim=128)
        x = torch.randn(8, 256)
        output = feature(x)
        
        assert output.shape == (8, 128)
        assert not torch.isnan(output).any()
    
    def test_invalid_input_shape(self):
        """Test that invalid input raises appropriate error."""
        feature = NewFeature(input_dim=256, output_dim=128)
        x = torch.randn(8, 512)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected input dimension 256"):
            feature(x)
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_different_batch_sizes(self, batch_size):
        """Test that model handles different batch sizes."""
        feature = NewFeature(input_dim=256, output_dim=128)
        x = torch.randn(batch_size, 256)
        output = feature(x)
        
        assert output.shape == (batch_size, 128)
```

### Test Organization

- Place tests in `tests/` directory
- Mirror the source code structure (e.g., `tests/test_models/test_fusion.py` for `src/models/fusion.py`)
- Use descriptive test names that explain what is being tested
- Group related tests in classes
- Use fixtures for common setup code
- Use parametrize for testing multiple inputs

### Continuous Integration

All tests run automatically on GitHub Actions for:
- Python versions: 3.9, 3.10, 3.11
- Operating systems: Ubuntu, Windows, macOS

Your pull request must pass all CI checks before merging.

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

2. **Make your changes**:
- Write clean, well-documented code
- Follow code style guidelines
- Add tests for new functionality
- Update documentation as needed

3. **Run the full test suite**:
```bash
# Format code
black src/ tests/ experiments/
isort src/ tests/ experiments/

# Run linting
flake8 src/ tests/ experiments/

# Run type checking
mypy src/ --ignore-missing-imports

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term

# Verify all checks pass
echo "All checks passed!"
```

4. **Commit your changes**:
```bash
git add .
git commit -m "feat: add cross-modal attention mechanism"
```

### Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

**Format**: `<type>(<scope>): <description>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature change or bug fix)
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks, dependency updates

**Examples**:
```bash
feat(fusion): add cross-modal attention mechanism
fix(data): handle missing modalities in dataloader
docs(readme): update installation instructions
test(models): add tests for temporal reasoning
refactor(encoders): simplify genomic encoder architecture
perf(inference): optimize batch processing for WSI features
```

**Commit message guidelines**:
- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor to..." not "moves cursor to...")
- Keep first line under 72 characters
- Add detailed description after blank line if needed
- Reference issues and PRs when relevant (`Fixes #123`, `Closes #456`)

### Submitting the Pull Request

1. **Push your branch**:
```bash
git push origin feature/your-feature-name
```

2. **Create pull request on GitHub**:
- Go to the repository on GitHub
- Click "New Pull Request"
- Select your branch
- Fill out the PR template

3. **PR Description Template**:
```markdown
## Description
Brief description of what this PR does.

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- List of specific changes made
- Another change
- And another

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed (describe what you tested)

## Documentation
- [ ] Code is well-documented with docstrings
- [ ] README updated (if needed)
- [ ] Architecture docs updated (if needed)

## Checklist
- [ ] Code follows style guidelines (Black, isort)
- [ ] No linting errors (flake8)
- [ ] Type hints added (mypy passes)
- [ ] Tests pass with good coverage
- [ ] Commit messages follow conventions
- [ ] PR title follows conventional commits format

## Related Issues
Fixes #123
Related to #456
```

### Review Process

1. **Automated checks**: CI will run all tests and checks automatically
2. **Code review**: Maintainers will review your code and provide feedback
3. **Address feedback**: Make requested changes and push updates
4. **Approval**: Once approved, a maintainer will merge your PR

**Review timeline**:
- Initial review: Within 3-5 business days
- Follow-up reviews: Within 1-2 business days
- Urgent fixes: Within 24 hours

**What reviewers look for**:
- Code correctness and logic
- Test coverage and quality
- Documentation completeness
- Performance implications
- Security considerations
- Compatibility with existing code

## Documentation Requirements

### Code Documentation

All public APIs must be documented:

```python
class MultimodalFusionModel(torch.nn.Module):
    """Multimodal fusion model with cross-modal attention.
    
    This model integrates whole-slide image features, genomic profiles,
    and clinical text through attention-based fusion mechanisms.
    
    Args:
        embed_dim: Embedding dimension for all modalities (default: 256)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        
    Attributes:
        wsi_encoder: Encoder for WSI features
        genomic_encoder: Encoder for genomic data
        text_encoder: Encoder for clinical text
        fusion_layer: Cross-modal attention fusion
        
    Example:
        >>> model = MultimodalFusionModel(embed_dim=256)
        >>> batch = {
        ...     'wsi_features': torch.randn(8, 100, 1024),
        ...     'genomic': torch.randn(8, 2000),
        ...     'clinical_text': torch.randint(0, 30000, (8, 128))
        ... }
        >>> embeddings = model(batch)
        >>> print(embeddings.shape)
        torch.Size([8, 256])
    """
    pass
```

### README Updates

If your changes affect usage, update the README:
- Installation instructions
- Quick start examples
- API changes
- New features or capabilities

### Architecture Documentation

For significant architectural changes, update `ARCHITECTURE.md`:
- New components or modules
- Changes to data flow
- Performance characteristics
- Design decisions and rationale

### Inline Comments

Use inline comments for complex logic:

```python
# Compute attention weights between all modality pairs
# Shape: [batch_size, num_modalities, num_modalities, embed_dim]
attention_weights = self.compute_cross_modal_attention(
    query=modality_embeddings,
    key=modality_embeddings,
    value=modality_embeddings,
    mask=modality_mask  # Mask for missing modalities
)
```

## Issue Reporting Guidelines

### Before Creating an Issue

1. **Search existing issues**: Check if the issue already exists
2. **Check documentation**: Ensure it's not a usage question answered in docs
3. **Reproduce the issue**: Verify you can consistently reproduce the problem
4. **Gather information**: Collect relevant details (error messages, environment, etc.)

### Bug Reports

Use the bug report template:

```markdown
## Bug Description
Clear and concise description of the bug.

## To Reproduce
Steps to reproduce the behavior:
1. Run command '...'
2. With configuration '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Error Messages
```
Paste full error message and stack trace here
```

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., RTX 3090]

## Additional Context
Any other relevant information.

## Possible Solution
(Optional) Suggest a fix if you have one.
```

### Feature Requests

Use the feature request template:

```markdown
## Feature Description
Clear description of the proposed feature.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
Describe how you envision this feature working.

## Alternatives Considered
Other approaches you've considered.

## Additional Context
Any other relevant information, references, or examples.

## Implementation Notes
(Optional) Technical details if you have implementation ideas.
```

### Questions and Discussions

For questions about usage, architecture, or research directions:
- Use GitHub Discussions (if enabled)
- Tag with appropriate labels (`question`, `discussion`, `research`)
- Provide context about what you're trying to achieve

## Development Workflow

### Typical Development Cycle

1. **Pick an issue**: Find an issue to work on or create one
2. **Discuss approach**: Comment on the issue to discuss your approach
3. **Create branch**: Create a feature branch from `main`
4. **Develop**: Write code, tests, and documentation
5. **Test locally**: Run all tests and checks
6. **Commit**: Make atomic commits with clear messages
7. **Push**: Push your branch to your fork
8. **Create PR**: Submit a pull request
9. **Address feedback**: Respond to review comments
10. **Merge**: Once approved, your PR will be merged

### Branch Naming

Use descriptive branch names:
- `feature/cross-modal-attention`
- `fix/missing-modality-handling`
- `docs/update-installation-guide`
- `test/add-fusion-tests`
- `refactor/simplify-encoder-architecture`

### Keeping Your Fork Updated

```bash
# Add upstream remote (one time)
git remote add upstream https://github.com/original-org/computational-pathology-research.git

# Fetch upstream changes
git fetch upstream

# Update your main branch
git checkout main
git merge upstream/main

# Update your feature branch
git checkout feature/your-feature
git rebase main
```

## Advanced Topics

### Adding New Models

When adding new model architectures:

1. Create module in `src/models/`
2. Inherit from `torch.nn.Module`
3. Implement `__init__` and `forward` methods
4. Add comprehensive docstrings
5. Create tests in `tests/test_models/`
6. Add configuration in `experiments/configs/`
7. Update architecture documentation

### Adding New Datasets

When adding dataset support:

1. Create dataset class in `src/data/loaders.py`
2. Inherit from `torch.utils.data.Dataset`
3. Implement `__len__` and `__getitem__`
4. Add preprocessing utilities if needed
5. Create tests with synthetic data
6. Document dataset format in `data/README.md`
7. Add example configuration

### Adding Experiments

When adding new experiments:

1. Create script in `experiments/`
2. Use Hydra for configuration management
3. Add config file in `experiments/configs/`
4. Implement logging with TensorBoard
5. Save checkpoints and results
6. Document experiment in docstring
7. Add to CI if appropriate

## Performance Considerations

### Memory Optimization

- Use gradient checkpointing for large models
- Implement efficient data loading with caching
- Profile memory usage with `torch.cuda.memory_summary()`
- Consider mixed-precision training (FP16/BF16)

### Computational Efficiency

- Vectorize operations when possible
- Avoid unnecessary data transfers between CPU/GPU
- Use efficient PyTorch operations
- Profile code with `torch.profiler`

### Benchmarking

When making performance claims:
- Provide benchmarking code
- Report hardware specifications
- Include multiple runs with standard deviation
- Compare against baselines

## Security Considerations

### Data Privacy

- Never commit real patient data
- Use synthetic data for tests
- Sanitize any example data
- Follow HIPAA guidelines for any clinical data

### Dependency Security

- Keep dependencies updated
- Review security advisories
- Use `pip-audit` to check for vulnerabilities
- Pin dependency versions in `requirements.txt`

### Code Security

- Avoid hardcoded credentials
- Use environment variables for sensitive config
- Validate all inputs
- Follow secure coding practices

## Getting Help

### Resources

- **Documentation**: Start with README.md and ARCHITECTURE.md
- **Examples**: Check `examples/` directory
- **Tests**: Look at `tests/` for usage examples
- **Issues**: Search existing issues for similar problems

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code review and technical discussion

### Response Times

- Issues: Acknowledged within 3-5 business days
- Pull requests: Initial review within 3-5 business days
- Questions: Response within 1 week

## Recognition

We value all contributions! Contributors will be:
- Listed in the repository contributors page
- Acknowledged in release notes for significant contributions
- Credited in academic citations when appropriate

## License

By contributing, you agree that your contributions will be licensed under the MIT License, the same license as the project.

## Thank You!

Thank you for contributing to computational pathology research! Your efforts help advance the field and improve tools for the research community.

If you have questions about contributing, feel free to open an issue or reach out to the maintainers.

---

**Happy Contributing! 🔬🧬💻**
