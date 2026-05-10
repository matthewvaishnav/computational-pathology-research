# Security Migration Guide

## Overview

This guide provides a migration path from insecure to secure patterns for existing code. All changes maintain backward compatibility in development mode while enforcing security in production.

## Migration Strategy

### Phased Rollout

1. **Development**: Relaxed policies with warnings
2. **Research**: Moderate policies with warnings
3. **Production**: Strict policies with enforcement

### Backward Compatibility

- Development mode maintains existing behavior
- Security warnings guide developers to fix issues
- No breaking changes to existing APIs
- Gradual adoption path

## Migration Paths

### 1. Jinja2 Template Rendering

**Before** (insecure):
```python
from jinja2 import Environment

env = Environment()
template = env.from_string("<h1>{{ title }}</h1>")
html = template.render(title=user_input)
```

**After** (secure):
```python
from src.security.jinja2_security import SecureJinja2Environment

env = SecureJinja2Environment.create_environment()
template = env.from_string("<h1>{{ title }}</h1>")
html = template.render(title=user_input)
```

**Migration Steps**:
1. Import `SecureJinja2Environment`
2. Replace `Environment()` with `SecureJinja2Environment.create_environment()`
3. Test in development (no breaking changes)
4. Deploy to production (autoescape enforced)

**Automated Migration**:
```bash
# Find all Jinja2 Environment() calls
grep -r "Environment()" src/

# Replace with secure version
sed -i 's/Environment()/SecureJinja2Environment.create_environment()/g' src/**/*.py
```

### 2. Server Binding

**Before** (insecure):
```python
app.run(host="0.0.0.0", port=5000)
```

**After** (secure):
```python
from src.security.network_binding import NetworkBindingManager

binding_manager = NetworkBindingManager()
safe_host = binding_manager.get_safe_host()
app.run(host=safe_host, port=5000)
```

**Migration Steps**:
1. Import `NetworkBindingManager`
2. Replace hardcoded `"0.0.0.0"` with `NetworkBindingManager().get_safe_host()`
3. Test in development (defaults to `127.0.0.1`)
4. Test in research (allows `0.0.0.0` with warning)
5. Deploy to production (blocks `0.0.0.0`)

**Automated Migration**:
```bash
# Find all 0.0.0.0 bindings
grep -r '0\.0\.0\.0' src/

# Manual review required - context-dependent
```

### 3. Model Downloads

**Before** (insecure):
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("owkin/phikon")
```

**After** (secure):
```python
from src.security.model_download import ModelDownloadManager

manager = ModelDownloadManager()
model = manager.download_model("owkin/phikon")
```

**Migration Steps**:
1. Add model revisions to `config/model_revisions.yaml`
2. Import `ModelDownloadManager`
3. Replace `from_pretrained()` with `manager.download_model()`
4. Test in development (warns on unpinned models)
5. Deploy to production (requires pinned revisions)

**Automated Migration**:
```bash
# Find all from_pretrained calls
grep -r "from_pretrained" src/

# Extract model names
grep -r "from_pretrained" src/ | grep -oP '"[^"]+"' | sort -u

# Add to config/model_revisions.yaml
```

### 4. Temporary Files

**Before** (insecure):
```python
tmp_file = "/tmp/data.json"
with open(tmp_file, 'w') as f:
    json.dump(data, f)
os.remove(tmp_file)
```

**After** (secure):
```python
from src.security.temp_file import TempFileManager

with TempFileManager.create_temp_file(suffix='.json') as tmp_path:
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
# Automatically cleaned up
```

**Migration Steps**:
1. Import `TempFileManager`
2. Replace hardcoded `/tmp/` paths with `TempFileManager.create_temp_file()`
3. Use context managers for automatic cleanup
4. Test in development (no breaking changes)
5. Deploy to production (secure permissions enforced)

**Automated Migration**:
```bash
# Find all /tmp usage
grep -r '/tmp' src/

# Manual review required - context-dependent
```

### 5. Pickle Deserialization

**Before** (insecure):
```python
import pickle

with open(file_path, 'rb') as f:
    data = pickle.load(f)
```

**After** (secure):
```python
from src.security.pickle_security import PickleSecurityControl

control = PickleSecurityControl()
data = control.safe_load(file_path)
```

**Migration Steps**:
1. Import `PickleSecurityControl`
2. Replace `pickle.load()` with `control.safe_load()`
3. Configure trusted sources in `config/security_config.yaml`
4. Test in development (warns on untrusted sources)
5. Deploy to production (blocks untrusted sources)

**Alternative**: Migrate to safer formats (JSON, SafeTensors)
```python
# Instead of pickle
import json

with open(file_path, 'w') as f:
    json.dump(data, f)

with open(file_path, 'r') as f:
    data = json.load(f)
```

**Automated Migration**:
```bash
# Find all pickle.load calls
grep -r "pickle\.load" src/

# Manual review required - assess trust level
```

### 6. URL Opening

**Before** (insecure):
```python
import urllib.request

with urllib.request.urlopen(url) as resp:
    data = resp.read()
```

**After** (secure):
```python
import urllib.request
from urllib.parse import urlparse

parsed = urlparse(url)
if parsed.scheme not in ('http', 'https'):
    raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310 - URL scheme validated above
    data = resp.read()
```

**Migration Steps**:
1. Add URL scheme validation before `urlopen()`
2. Block `file://`, `ftp://`, and other unsafe schemes
3. Add `# nosec B310` comment with justification
4. Test in development (no breaking changes)
5. Deploy to production (scheme validation enforced)

**Automated Migration**:
```bash
# Find all urlopen calls
grep -r "urlopen" src/

# Manual review required - add validation
```

### 7. Credentials

**Before** (insecure):
```python
DB_PASSWORD = "my_secret_password"
```

**After** (secure):
```python
import os

DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD environment variable not set")
```

**Migration Steps**:
1. Move credentials to environment variables
2. Update code to load from `os.getenv()`
3. Add validation for required credentials
4. Update deployment documentation
5. Rotate credentials after migration

**Automated Migration**:
```bash
# Find potential hardcoded credentials
grep -r "password.*=" src/ | grep -v "os.getenv"

# Manual review required - assess if real credentials
```

## Environment Configuration

### Development Setup

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Credentials (test values)
DB_PASSWORD=dev_password
SECRET_KEY=dev_secret_key
```

### Production Setup

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Credentials (from secure vault)
DB_PASSWORD=${VAULT_DB_PASSWORD}
SECRET_KEY=${VAULT_SECRET_KEY}
```

## Testing Migration

### Unit Tests

```python
import os
import pytest

def test_secure_jinja2_migration():
    """Test Jinja2 migration maintains functionality."""
    from src.security.jinja2_security import SecureJinja2Environment
    
    env = SecureJinja2Environment.create_environment()
    template = env.from_string("<h1>{{ title }}</h1>")
    html = template.render(title="Test")
    
    assert html == "<h1>Test</h1>"
    assert env.autoescape is True

def test_network_binding_migration():
    """Test network binding migration in different environments."""
    from src.security.network_binding import NetworkBindingManager
    
    # Development
    os.environ['ENVIRONMENT'] = 'development'
    manager = NetworkBindingManager()
    assert manager.get_safe_host() == '127.0.0.1'
    
    # Production
    os.environ['ENVIRONMENT'] = 'production'
    manager = NetworkBindingManager()
    assert manager.get_safe_host() != '0.0.0.0'
```

### Integration Tests

```bash
# Test in development environment
export ENVIRONMENT=development
python -m pytest tests/security/ -v

# Test in production environment
export ENVIRONMENT=production
python -m pytest tests/security/ -v
```

## Rollback Procedure

If issues arise during migration:

1. **Immediate**: Revert to previous version
2. **Assess**: Identify breaking changes
3. **Fix**: Update migration approach
4. **Test**: Verify fixes in staging
5. **Redeploy**: Apply corrected migration

## Migration Checklist

### Pre-Migration

- [ ] Review all security controls
- [ ] Identify affected code locations
- [ ] Create migration plan
- [ ] Set up test environments
- [ ] Document rollback procedure

### Migration

- [ ] Update Jinja2 template rendering
- [ ] Update server binding
- [ ] Update model downloads
- [ ] Update temporary file handling
- [ ] Update pickle deserialization
- [ ] Update URL opening
- [ ] Update credential management

### Post-Migration

- [ ] Run security verification: `python scripts/verify_security.py`
- [ ] Run Bandit scan: `python -m bandit -r src/`
- [ ] Run unit tests: `pytest tests/security/ -v`
- [ ] Run integration tests: `pytest tests/integration/ -v`
- [ ] Generate security report: `python scripts/generate_security_report.py`
- [ ] Update documentation
- [ ] Train team on new patterns

## Support

For migration questions or issues:
- Email: security-team@company.com
- Slack: #security-migration
- Documentation: docs/SECURITY.md
