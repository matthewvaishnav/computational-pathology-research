# Dependency Management

This project uses a modular dependency structure for better maintainability and security.

## Installation Options

### Core Installation (Minimal)
```bash
pip install -r requirements-core.txt
```
Includes only essential dependencies for basic functionality.

### Development Installation
```bash
pip install -r requirements-dev.txt
```
Includes core + testing, linting, and development tools.

### Full Installation
```bash
pip install -r requirements.txt
```
Includes all dependencies (core + optional features).

### Using pyproject.toml (Recommended)
```bash
# Core only
pip install -e .

# With GUI support
pip install -e ".[gui]"

# With cloud integration
pip install -e ".[cloud]"

# Development setup
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## Dependency Files

- **requirements-core.txt**: Essential dependencies with version pins
- **requirements-dev.txt**: Development and testing tools
- **requirements-optional.txt**: Optional features (GUI, cloud, ML extensions)
- **requirements.txt**: Legacy file (use pyproject.toml instead)
- **pyproject.toml**: Modern Python packaging with optional dependency groups

## Version Pinning Strategy

All dependencies use upper bounds to prevent breaking changes:
- Format: `package>=X.Y.Z,<X+1.0.0`
- Critical packages (fastapi, pydantic) use exact pins: `package==X.Y.Z`

## Security Scanning

Run security checks before deployment:
```bash
./scripts/security_scan.sh
```

Or manually:
```bash
pip install safety
safety check --file requirements-core.txt
```

## Pre-commit Hooks

Install code quality checks:
```bash
pip install pre-commit
pre-commit install
```

Run manually:
```bash
pre-commit run --all-files
```

## Updating Dependencies

1. Update version in appropriate requirements file
2. Run security scan: `./scripts/security_scan.sh`
3. Run tests: `pytest tests/`
4. Update pyproject.toml if needed
5. Commit changes

## Troubleshooting

### Missing pip
```bash
# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

### Dependency Conflicts
```bash
# Check for conflicts
pip check

# Create fresh environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
```

### Platform-specific Issues
- **macOS**: OpenSlide requires `brew install openslide`
- **Linux**: OpenSlide requires `apt-get install openslide-tools`
- **Windows**: Use pre-built wheels or WSL2
