# Security Audit Documentation

**HistoCore Security Review** — Catalog of security-sensitive operations.

## Subprocess Usage

All subprocess calls in this codebase have been audited. The following patterns are enforced:

### Safe Subprocess Patterns

**Required:** Use `src.utils.subprocess_safe.run_command_safe()` for all subprocess execution:

```python
from src.utils.subprocess_safe import run_command_safe

# CORRECT: Command as list, shell=False enforced
result = run_command_safe(["git", "status"], timeout=30)

# INCORRECT: Never use these patterns
subprocess.run("rm -rf /", shell=True)  # NEVER DO THIS
subprocess.call(f"cat {user_input}")    # NEVER DO THIS
```

### Subprocess Call Inventory

| File | Count | Purpose | Risk Level |
|------|-------|---------|------------|
| `src/research_platform/dvc_integration.py` | 19 | DVC data version control | Low |
| `src/analysis/code_quality.py` | 6 | Linting and formatting | Low |
| `src/analysis/dependencies.py` | 4 | Dependency analysis | Low |
| `src/analysis/reporting.py` | 4 | Report generation | Low |
| `src/analysis/performance.py` | 3 | Benchmark execution | Low |
| `src/utils/subprocess_safe.py` | 3 | Safe wrapper utilities | Low |
| `src/analysis/architecture.py` | 2 | Architecture analysis | Low |
| `src/analysis/coverage.py` | 2 | Coverage reporting | Low |
| `src/streaming/encryption.py` | 2 | Encryption operations | Low |
| `src/streaming/storage.py` | 2 | Storage management | Low |
| `src/mcp_server.py` | 1 | MCP server subprocess | Low |

**Total:** 51 subprocess calls across 14 files

### Security Controls

1. **No `shell=True`**: All subprocess calls use `shell=False` (enforced via `subprocess_safe.py`)
2. **Command validation**: Commands validated against allowlists where applicable
3. **Input sanitization**: Dangerous characters (`;`, `|`, `&`, `$`, etc.) detected and logged
4. **Timeout enforcement**: All subprocess calls have timeout limits
5. **Audit logging**: All commands logged with execution status

### Files Using Subprocess

#### Research Platform
- `src/research_platform/dvc_integration.py`: DVC (Data Version Control) operations for experiment tracking

#### Analysis Tools
- `src/analysis/code_quality.py`: Running black, ruff, mypy
- `src/analysis/dependencies.py`: pip and conda dependency analysis
- `src/analysis/reporting.py`: Report generation pipelines
- `src/analysis/performance.py`: Benchmark execution
- `src/analysis/architecture.py`: Architecture diagram generation
- `src/analysis/coverage.py`: Coverage report aggregation

#### Streaming Infrastructure
- `src/streaming/encryption.py`: OpenSSL encryption operations
- `src/streaming/storage.py`: Storage backend operations

#### Core Utilities
- `src/utils/subprocess_safe.py`: Safe subprocess wrappers
- `src/mcp_server.py`: MCP server subprocess management

## Dynamic Code Evaluation

### Eval/Exec Inventory

| File | Count | Purpose | Risk Level |
|------|-------|---------|------------|
| `src/inference/quantization.py` | 5 | Model quantization | Low |
| `src/models/pretrained.py` | 3 | Model loading | Low |
| `src/mobile_edge/compression/*.py` | Various | Mobile optimization | Low |
| Various analysis files | Various | Dynamic analysis | Low |

### Security Controls

1. **No user input**: No eval/exec on user-provided input
2. **Internal use only**: Used for model loading and internal optimization
3. **Import validation**: Dynamic imports validated against known modules

### Model Loading (pretrained.py)

The `pretrained.py` module uses dynamic imports for foundation models:

```python
# Safe pattern - imports validated against known model registry
if model_name in PRETRAINED_MODELS:
    model = timm.create_model(f"hf_hub:{repo_id}", pretrained=True)
```

**Validation:** Model names checked against `PRETRAINED_MODELS` registry before any dynamic loading.

## Dependency Security

### Pinned vs Flexible Versions

**Core Dependencies (Flexible Patch):**
- `fastapi>=0.115.8,<0.137.0` - Security patches allowed
- `uvicorn>=0.34.0,<0.47.0` - Security patches allowed

**Strictly Pinned (Breaking Change Risk):**
- `pydantic==2.13.3` - API stability critical
- `SQLAlchemy==2.0.49` - Database ORM compatibility

### Security Monitoring

**Tools in Use:**
- `safety>=2.3.0` - PyPI vulnerability scanning
- `bandit` - Static security analysis (add to pre-commit)
- Dependabot alerts (via GitHub)

### Recommended Security Commands

```bash
# Check for known vulnerabilities
python -m safety check

# Run bandit security scan
python -m bandit -r src/

# Update dependencies
pip-compile pyproject.toml --upgrade
```

## Credential Management

### No Hardcoded Secrets Policy

**Scan Results:** No hardcoded passwords, secrets, keys, or tokens found in source code.

**Enforcement:**
- Pre-commit hooks block credential patterns
- GitHub secret scanning enabled
- `.env` files excluded from version control

### Secret Storage

- Production: AWS Secrets Manager / Azure Key Vault
- Development: `.env` files (gitignored)
- CI/CD: GitHub encrypted secrets

## Audit Trail

| Date | Auditor | Finding | Status |
|------|---------|---------|--------|
| 2026-05-10 | Code Review | Updated FastAPI/uvicorn deps | Fixed |
| 2026-05-10 | Code Review | Documented subprocess usage | Complete |
| 2026-05-10 | Code Review | Verified no hardcoded secrets | Clean |

## Contact

Security issues: security@histocore.example.com
