# Security Architecture

## Overview

HistoCore implements environment-aware security controls that adapt based on deployment context (Production, Development, Research). This document explains the security architecture, controls, and rationale.

## Security Controls

### 1. Jinja2 XSS Protection

**Risk**: Cross-site scripting (XSS) attacks through unescaped template variables.

**Control**: All Jinja2 environments enforce `autoescape=True` by default.

**Implementation**:
- `SecureJinja2Environment.create_environment()` creates environments with autoescape enabled
- Used in: `src/clinical/reporting.py`

**Verification**: Bandit scan verifies no HIGH severity Jinja2 autoescape issues.

### 2. Network Binding Security

**Risk**: Binding to `0.0.0.0` exposes services to all network interfaces, including public internet.

**Control**: Environment-based host binding policies.

**Implementation**:
- Production: Blocks `0.0.0.0` binding, requires explicit configuration
- Development: Defaults to `127.0.0.1` (localhost only)
- Research: Allows `0.0.0.0` with security warning for cluster computing

**Usage**:
```python
from src.security.network_binding import NetworkBindingManager

binding_manager = NetworkBindingManager()
safe_host = binding_manager.get_safe_host()
app.run(host=safe_host, port=5000)
```

**Verification**: Bandit scan verifies 0 MEDIUM severity `0.0.0.0` binding issues.

### 3. Model Download Security

**Risk**: Unpinned model downloads can introduce supply chain attacks or unexpected behavior changes.

**Control**: Revision pinning for all HuggingFace model downloads in production.

**Implementation**:
- `config/model_revisions.yaml` defines pinned revisions for all models
- `ModelDownloadManager` enforces revision pinning based on environment
- Production: Requires pinned revisions
- Development: Warns on unpinned models

**Usage**:
```python
from src.security.model_download import ModelDownloadManager

manager = ModelDownloadManager()
model = manager.download_model("owkin/phikon")  # Uses pinned revision from config
```

**Verification**: Bandit scan verifies 0 MEDIUM severity unpinned model download issues.

### 4. Temporary File Security

**Risk**: Hardcoded `/tmp` paths are insecure on multi-user systems and non-portable across platforms.

**Control**: Use `tempfile` module for secure temporary file creation with proper permissions.

**Implementation**:
- `TempFileManager.create_temp_file()` creates files with `0o600` permissions (owner read/write only)
- `TempFileManager.create_temp_directory()` creates directories with `0o700` permissions
- Automatic cleanup via context managers

**Usage**:
```python
from src.security.temp_file import TempFileManager

with TempFileManager.create_temp_file(suffix='.json') as tmp_path:
    # Use tmp_path
    pass  # Automatically cleaned up
```

**Verification**: Bandit scan verifies 0 MEDIUM severity hardcoded `/tmp` usage issues.

### 5. Pickle Deserialization Security

**Risk**: Unpickling untrusted data can execute arbitrary code.

**Control**: Source validation and restricted unpickler for untrusted sources.

**Implementation**:
- `PickleSecurityControl.safe_load()` validates pickle source before loading
- Trusted sources: `checkpoints/`, `models/`, `data/processed/`
- Untrusted sources: Use restricted unpickler or reject
- Production: Blocks untrusted pickle
- Development: Warns on untrusted pickle

**Usage**:
```python
from src.security.pickle_security import PickleSecurityControl

control = PickleSecurityControl()
data = control.safe_load(pickle_path)  # Validates source before loading
```

**Verification**: Bandit scan verifies 0 MEDIUM severity unsafe pickle deserialization issues.

### 6. URL Opening Security

**Risk**: Opening URLs without scheme validation can lead to SSRF attacks via `file://` URLs.

**Control**: URL scheme validation before opening.

**Implementation**:
- `URLFetcherControl.safe_urlopen()` validates URL scheme (only `http://` and `https://` allowed)
- Blocks `file://`, `ftp://`, and other unsafe schemes
- Inline validation in `src/clinical/pacs/notification_system.py`

**Usage**:
```python
from urllib.parse import urlparse

parsed = urlparse(url)
if parsed.scheme not in ('http', 'https'):
    raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310 - URL scheme validated above
    data = resp.read()
```

**Verification**: Bandit scan verifies 0 MEDIUM severity unsafe URL opening issues.

### 7. Credential Management

**Risk**: Hardcoded credentials in source code can be exposed in version control.

**Control**: All credentials loaded from environment variables or secure vaults.

**Implementation**:
- Production: Only accepts credentials from environment variables
- Development: Allows test credentials with explicit `# nosec` comments
- All database passwords, API keys, and tokens loaded via `os.getenv()`

**Usage**:
```python
import os

db_password = os.getenv("DB_PASSWORD")
if not db_password:
    raise ValueError("DB_PASSWORD environment variable not set")
```

**Verification**: Bandit scan verifies 0 MEDIUM severity hardcoded password issues.

## Environment-Based Policies

### Production Environment

- **Network Binding**: Blocks `0.0.0.0`, requires explicit configuration
- **Model Downloads**: Requires pinned revisions
- **Pickle**: Blocks untrusted sources
- **Credentials**: Only from environment variables
- **Audit Logging**: Tamper-evident logging enabled

### Development Environment

- **Network Binding**: Defaults to `127.0.0.1`
- **Model Downloads**: Warns on unpinned models
- **Pickle**: Warns on untrusted sources
- **Credentials**: Allows test credentials with `# nosec`
- **Audit Logging**: Standard logging

### Research Environment

- **Network Binding**: Allows `0.0.0.0` with warning (for cluster computing)
- **Model Downloads**: Allows unpinned models with warning
- **Pickle**: Allows untrusted sources with warning
- **Credentials**: Same as development
- **Audit Logging**: Standard logging

## Security Verification

### Bandit Scanning

All security controls are verified with Bandit static analysis:

```bash
# Scan entire codebase
python -m bandit -r src/ -f json -o bandit-report.json

# Check for HIGH/MEDIUM issues
python scripts/check_bandit_results.py bandit-report.json
```

### CI/CD Integration

- **Every PR**: Bandit scan runs and fails build on HIGH/MEDIUM issues
- **Weekly**: Comprehensive security audit with Safety and pip-audit
- **On main merge**: Full security report generated

### Manual Verification

```bash
# Run security verification script
python scripts/verify_security.py

# Generate security posture report
python scripts/security_posture_report.py
```

## Security Incident Response

See `docs/SECURITY_INCIDENT_RESPONSE.md` for incident response procedures.

## Security Testing

See `docs/SECURITY_TESTING.md` for security testing guide.

## Contributing Security Fixes

See `docs/CONTRIBUTING_SECURITY.md` for secure coding patterns and best practices.
