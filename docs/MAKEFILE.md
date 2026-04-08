# Makefile Documentation

This document describes all available Makefile targets for the Computational Pathology Research Framework.

## Quick Reference

```bash
make help              # Show all available targets
make install           # Install dependencies
make test              # Run tests
make demo              # Run quick demo
make docker-build      # Build Docker image
```

## Platform Support

### Linux/macOS
Use `make` directly:
```bash
make install
make test
```

### Windows
Use the batch file wrapper:
```bash
make.bat install
make.bat test
```

Or install `make` for Windows:
- Via Chocolatey: `choco install make`
- Via Scoop: `scoop install make`
- Via WSL: Use Linux commands in WSL

## Installation Targets

### `make install`
Install production dependencies.

**What it does**:
- Upgrades pip
- Installs requirements.txt
- Installs package in editable mode

**Usage**:
```bash
make install
```

### `make install-dev`
Install development dependencies including testing and linting tools.

**What it does**:
- Installs production dependencies
- Installs development dependencies (pytest, flake8, black, etc.)
- Installs pre-commit hooks

**Usage**:
```bash
make install-dev
```

### `make dev-setup`
Complete development environment setup.

**What it does**:
- Runs `install-dev`
- Configures pre-commit hooks
- Sets up development tools

**Usage**:
```bash
make dev-setup
```

## Testing Targets

### `make test`
Run all tests with verbose output.

**Usage**:
```bash
make test
```

**Output**: Test results in terminal

### `make test-cov`
Run tests with coverage reporting.

**Usage**:
```bash
make test-cov
```

**Output**: 
- Coverage report in terminal
- HTML report in `htmlcov/`

**View HTML report**:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### `make test-quick`
Run only quick tests (excludes slow tests).

**Usage**:
```bash
make test-quick
```

### `make test-watch`
Run tests in watch mode (re-runs on file changes).

**Usage**:
```bash
make test-watch
```

**Requirements**: `pytest-watch` (included in requirements-dev.txt)

## Code Quality Targets

### `make lint`
Run all linting checks.

**Checks**:
- flake8 (PEP 8 compliance)
- black (code formatting)
- isort (import sorting)

**Usage**:
```bash
make lint
```

### `make format`
Auto-format code with black and isort.

**Usage**:
```bash
make format
```

**What it does**:
- Formats all Python files with black
- Sorts imports with isort

### `make type-check`
Run static type checking with mypy.

**Usage**:
```bash
make type-check
```

### `make security`
Run security vulnerability scanning.

**Usage**:
```bash
make security
```

**Output**: `bandit-report.json`

### `make check-all`
Run all code quality checks (lint + type-check + security).

**Usage**:
```bash
make check-all
```

## Demo Targets

### `make demo`
Run quick demo (3 minutes).

**Usage**:
```bash
make demo
```

**Output**: `results/quick_demo/`

### `make demo-missing`
Run missing modality robustness demo.

**Usage**:
```bash
make demo-missing
```

**Output**: `results/missing_modality_demo/`

### `make demo-temporal`
Run temporal reasoning demo.

**Usage**:
```bash
make demo-temporal
```

**Output**: `results/temporal_demo/`

### `make demo-all`
Run all demos sequentially.

**Usage**:
```bash
make demo-all
```

**Time**: ~10 minutes total

## Training Targets

### `make train`
Train model with default configuration.

**Usage**:
```bash
make train
```

**Config**: `experiments/configs/default.yaml`

### `make train-quick`
Quick training for testing (5 epochs).

**Usage**:
```bash
make train-quick
```

**Config**: `experiments/configs/quick_demo.yaml`

### `make train-full`
Full production training (200 epochs).

**Usage**:
```bash
make train-full
```

**Config**: `experiments/configs/full_training.yaml`

**Custom training**:
```bash
python experiments/train.py --config-name <config>
```

## Evaluation Targets

### `make evaluate`
Evaluate trained model.

**Usage**:
```bash
make evaluate
```

**Requirements**: Trained model at `checkpoints/best_model.pth`

**Output**: `results/evaluation/`

### `make evaluate-ablation`
Run comprehensive ablation study.

**Usage**:
```bash
make evaluate-ablation
```

**What it does**:
- Tests each modality individually
- Tests missing modality robustness
- Generates detailed analysis

**Output**: `results/ablation/`

## Docker Targets

### `make docker-build`
Build Docker image.

**Usage**:
```bash
make docker-build
```

**Output**: Docker image `pathology-api:latest`

### `make docker-run`
Start Docker container with API.

**Usage**:
```bash
make docker-run
```

**Access**: http://localhost:8000

**API docs**: http://localhost:8000/docs

### `make docker-stop`
Stop Docker containers.

**Usage**:
```bash
make docker-stop
```

### `make docker-logs`
View container logs (follow mode).

**Usage**:
```bash
make docker-logs
```

**Exit**: Ctrl+C

### `make docker-shell`
Open bash shell in running container.

**Usage**:
```bash
make docker-shell
```

### `make docker-test`
Run Docker integration tests.

**Usage**:
```bash
make docker-test
```

**What it does**:
- Builds image
- Starts container
- Tests health endpoint
- Tests prediction endpoint
- Cleans up

## Jupyter Targets

### `make jupyter`
Start Jupyter notebook server.

**Usage**:
```bash
make jupyter
```

**Access**: http://localhost:8888

### `make jupyter-docker`
Start Jupyter in Docker container.

**Usage**:
```bash
make jupyter-docker
```

**Access**: http://localhost:8888

## Cleaning Targets

### `make clean`
Clean generated files and caches.

**Removes**:
- `__pycache__` directories
- `*.pyc` files
- `.pytest_cache`
- `.mypy_cache`
- `htmlcov/`
- `dist/` and `build/`

**Usage**:
```bash
make clean
```

### `make clean-results`
Clean result files from demos and evaluations.

**Usage**:
```bash
make clean-results
```

### `make clean-models`
Clean saved model checkpoints (with confirmation).

**Usage**:
```bash
make clean-models
```

**Warning**: This deletes all saved models!

### `make clean-all`
Clean everything except models.

**Usage**:
```bash
make clean-all
```

## Development Workflow Targets

### `make dev-check`
Run complete development check pipeline.

**What it does**:
1. Format code
2. Run linting
3. Run type checking
4. Run tests

**Usage**:
```bash
make dev-check
```

### `make ci`
Simulate CI pipeline locally.

**What it does**:
1. Linting
2. Type checking
3. Security scan
4. Tests with coverage
5. Docker build
6. Quick demo

**Usage**:
```bash
make ci
```

**Time**: ~15 minutes

## Utility Targets

### `make version`
Show version information.

**Usage**:
```bash
make version
```

**Output**:
- Python version
- PyTorch version
- Package version

### `make info`
Show project information.

**Usage**:
```bash
make info
```

**Output**:
- Project structure
- Key files
- Documentation links

### `make status`
Show project status.

**Usage**:
```bash
make status
```

**Output**:
- Git status
- Recent commits
- Docker containers
- Disk usage

### `make data-check`
Check data availability.

**Usage**:
```bash
make data-check
```

**Output**: Status of data directories

## Monitoring Targets

### `make monitor-training`
Start TensorBoard for monitoring training.

**Usage**:
```bash
make monitor-training
```

**Access**: http://localhost:6006

**Logs directory**: `logs/`

## Quick Workflows

### `make quick-start`
Quick start workflow for new users.

**What it does**:
1. Install dependencies
2. Run quick demo

**Usage**:
```bash
make quick-start
```

### `make full-workflow`
Complete workflow for development.

**What it does**:
1. Install dev dependencies
2. Run all checks
3. Run all demos

**Usage**:
```bash
make full-workflow
```

## Help Targets

### `make help`
Show all available targets with descriptions.

**Usage**:
```bash
make help
```

### `make help-docker`
Show Docker-specific help.

**Usage**:
```bash
make help-docker
```

### `make help-training`
Show training-specific help.

**Usage**:
```bash
make help-training
```

## Common Workflows

### First Time Setup
```bash
make install-dev
make test
make demo
```

### Daily Development
```bash
make format          # Format code
make dev-check       # Run all checks
make test            # Run tests
```

### Before Committing
```bash
make format          # Format code
make lint            # Check linting
make test-cov        # Run tests with coverage
```

### Before Pull Request
```bash
make ci              # Simulate full CI pipeline
```

### Training Workflow
```bash
make train-quick     # Quick test
make train           # Full training
make evaluate        # Evaluate results
```

### Docker Workflow
```bash
make docker-build    # Build image
make docker-run      # Start container
make docker-logs     # View logs
make docker-stop     # Stop container
```

### Release Workflow
```bash
make release-check   # Check release readiness
make clean-all       # Clean everything
make ci              # Run full CI
# Then create git tag
```

## Troubleshooting

### Issue: `make: command not found`
**Solution**: 
- On Windows: Use `make.bat` instead
- Or install make: `choco install make` (Windows)

### Issue: Tests fail
**Solution**:
```bash
make clean           # Clean caches
make install         # Reinstall dependencies
make test            # Run tests again
```

### Issue: Docker build fails
**Solution**:
```bash
make clean           # Clean build artifacts
docker system prune  # Clean Docker cache
make docker-build    # Rebuild
```

### Issue: Out of disk space
**Solution**:
```bash
make clean-all       # Clean generated files
docker system prune -a  # Clean Docker
make clean-models    # Clean models (if needed)
```

## Tips and Tricks

### Run multiple targets
```bash
make clean test demo
```

### Parallel test execution
```bash
pytest tests/ -n auto  # Use all CPU cores
```

### Watch for changes
```bash
make test-watch      # Auto-run tests on changes
```

### Custom configurations
```bash
# Override Makefile variables
make train CONFIG=my_config
```

### Quiet mode
```bash
make test > /dev/null  # Suppress output
```

### Dry run
```bash
make -n test         # Show commands without executing
```

## Environment Variables

Set these in your shell or `.env` file:

```bash
# Python
export PYTHONPATH=.

# CUDA
export CUDA_VISIBLE_DEVICES=0

# Logging
export LOG_LEVEL=INFO

# Weights & Biases
export WANDB_API_KEY=your_key
```

## Integration with IDEs

### VS Code
Add to `.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "make test",
      "group": "test"
    }
  ]
}
```

### PyCharm
1. Go to Run → Edit Configurations
2. Add new "Shell Script" configuration
3. Set script: `make test`

## Additional Resources

- Main README: `README.md`
- Docker guide: `DOCKER.md`
- Configuration guide: `experiments/configs/README.md`
- Workflow guide: `.github/workflows/README.md`
