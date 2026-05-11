# Development Environment Setup Guide

This guide covers setting up a secure development environment for the Computational Pathology Research platform.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Security Configuration](#security-configuration)
- [Development Tools](#development-tools)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or 3.11
- **RAM**: Minimum 8GB (16GB recommended for ML workloads)
- **Disk Space**: 10GB free space
- **GPU** (optional): CUDA-compatible GPU for accelerated training

### Required Software

```bash
# Python 3.10 or 3.11
python --version

# Git
git --version

# OpenSlide (for WSI processing)
# Ubuntu/Debian:
sudo apt-get install openslide-tools

# macOS:
brew install openslide

# Windows: Download from https://openslide.org/download/
```

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/computational-pathology-research.git
cd computational-pathology-research
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e ".[foundation]"

# Install PyTorch Geometric extensions
python -c "import torch; print(f'Installing PyG for torch {torch.__version__}')"
python -c "import torch, subprocess, sys; v=torch.__version__.split('+')[0]; subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch-scatter', 'torch-sparse', '-f', f'https://data.pyg.org/whl/torch-{v}+cpu.html'], check=True)"
```

### 4. Set Environment Variables

Create `.env` file in project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
```

**Development `.env` configuration:**

```bash
# Environment
ENVIRONMENT=development

# Security
SECRET_KEY=dev-secret-key-change-in-production
DEBUG=true

# Database
DATABASE_URL=sqlite:///./dev.db

# API
API_HOST=127.0.0.1
API_PORT=8000

# Model Storage
MODEL_CACHE_DIR=./models/cache
CHECKPOINT_DIR=./checkpoints

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=./logs/dev.log

# Feature Flags
ENABLE_PROFILING=true
ENABLE_DEBUG_TOOLBAR=true
```

## Security Configuration

### Development Security Settings

Development mode uses relaxed security policies for easier debugging:

```yaml
# config/security_config.yaml
development:
  environment_detection:
    default_environment: development
    require_explicit_production: true
  
  network_binding:
    default_host: "127.0.0.1"
    allow_0.0.0.0: false
    strict_binding: false
  
  model_downloads:
    require_pinned_revisions: false
    warn_on_unpinned: true
    allow_latest: true
  
  audit_trail:
    enabled: true
    log_level: DEBUG
    log_file: "logs/security_audit_dev.log"
```

### Security Best Practices for Development

1. **Never commit secrets**: Use `.env` files (gitignored) for sensitive data
2. **Use localhost binding**: Bind servers to `127.0.0.1`, not `0.0.0.0`
3. **Enable security warnings**: Pay attention to security warnings in logs
4. **Test security controls**: Run security tests before committing
5. **Keep dependencies updated**: Regularly update packages for security patches

### Running Security Scans Locally

```bash
# Run Bandit security scan
bandit -r src/ -f json -o bandit-report.json
python scripts/check_bandit_results.py bandit-report.json

# Check for vulnerable dependencies
pip-audit
safety check

# Run security tests
pytest tests/security/ -v
```

## Development Tools

### Code Quality Tools

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type check with mypy
mypy src/ --ignore-missing-imports
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### IDE Configuration

#### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- isort
- GitLens

`.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. Set Python interpreter to `.venv/bin/python`
2. Enable Black formatter: Settings → Tools → Black
3. Configure pytest: Settings → Tools → Python Integrated Tools → Testing

## Running the Application

### Start Development Server

```bash
# Start API server
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Or use the convenience script
python scripts/start_dev_server.py
```

### Access Services

- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health
- **Metrics**: http://127.0.0.1:8000/metrics

### Run Demo

```bash
# Quick demo (5 minutes)
python examples/run_quick_demo_simple.py

# Full demo with visualization
python scripts/run_demo.py
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow/property tests)
pytest tests/ -m "not slow and not property"

# Run security tests
pytest tests/security/ -v
```

### Test Data

```bash
# Generate synthetic test data
python scripts/generate_test_data.py

# Download sample datasets
python scripts/download_sample_data.py
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Reinstall package in editable mode
pip install -e .

# Verify installation
python -c "import src; print(src.__file__)"
```

#### OpenSlide Not Found

```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools python3-openslide

# macOS
brew install openslide

# Windows: Add OpenSlide bin directory to PATH
```

#### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CPU-only PyTorch for development
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Port Already in Use

```bash
# Find process using port 8000
# Linux/macOS:
lsof -i :8000

# Windows:
netstat -ano | findstr :8000

# Kill process or use different port
python -m uvicorn src.api.main:app --reload --port 8001
```

### Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Open GitHub issue with `question` label
- **Security**: Email security@example.com (do not open public issues)
- **Contributing**: See `CONTRIBUTING.md`

## Next Steps

1. **Read Security Documentation**: Review `docs/SECURITY.md`
2. **Review Contributing Guide**: See `docs/CONTRIBUTING_SECURITY.md`
3. **Run Tests**: Ensure all tests pass locally
4. **Start Coding**: Follow security best practices
5. **Submit PR**: Include tests and documentation

## Additional Resources

- [Production Deployment Guide](DEPLOY_PRODUCTION.md)
- [Security Testing Guide](SECURITY_TESTING.md)
- [Security Migration Guide](SECURITY_MIGRATION.md)
- [API Documentation](https://your-api-docs-url.com)
- [Architecture Overview](ARCHITECTURE.md)
