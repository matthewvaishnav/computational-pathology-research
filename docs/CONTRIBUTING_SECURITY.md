# Security Contribution Guide

## Secure Coding Patterns

### Template Rendering (Jinja2)

**❌ Insecure**:
```python
from jinja2 import Environment

env = Environment()  # autoescape disabled by default
template = env.from_string("<h1>{{ title }}</h1>")
```

**✅ Secure**:
```python
from src.security.jinja2_security import SecureJinja2Environment

env = SecureJinja2Environment.create_environment()  # autoescape=True
template = env.from_string("<h1>{{ title }}</h1>")
```

### Server Binding

**❌ Insecure**:
```python
app.run(host="0.0.0.0", port=5000)  # Exposes to all interfaces
```

**✅ Secure**:
```python
from src.security.network_binding import NetworkBindingManager

binding_manager = NetworkBindingManager()
safe_host = binding_manager.get_safe_host()  # Environment-aware
app.run(host=safe_host, port=5000)
```

### Model Loading

**❌ Insecure**:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("owkin/phikon")  # Unpinned revision
```

**✅ Secure**:
```python
from src.security.model_download import ModelDownloadManager

manager = ModelDownloadManager()
model = manager.download_model("owkin/phikon")  # Uses pinned revision
```

### Temporary Files

**❌ Insecure**:
```python
tmp_file = "/tmp/data.json"  # Hardcoded path, insecure permissions
with open(tmp_file, 'w') as f:
    json.dump(data, f)
```

**✅ Secure**:
```python
from src.security.temp_file import TempFileManager

with TempFileManager.create_temp_file(suffix='.json') as tmp_path:
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
# Automatically cleaned up
```

### Pickle Deserialization

**❌ Insecure**:
```python
import pickle

with open(file_path, 'rb') as f:
    data = pickle.load(f)  # Unsafe for untrusted sources
```

**✅ Secure**:
```python
from src.security.pickle_security import PickleSecurityControl

control = PickleSecurityControl()
data = control.safe_load(file_path)  # Validates source
```

### URL Opening

**❌ Insecure**:
```python
import urllib.request

with urllib.request.urlopen(url) as resp:  # No scheme validation
    data = resp.read()
```

**✅ Secure**:
```python
import urllib.request
from urllib.parse import urlparse

parsed = urlparse(url)
if parsed.scheme not in ('http', 'https'):
    raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310 - URL scheme validated above
    data = resp.read()
```

### Credentials

**❌ Insecure**:
```python
DB_PASSWORD = "my_secret_password"  # Hardcoded credential
```

**✅ Secure**:
```python
import os

DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD environment variable not set")
```

## Bandit Suppression

Use `# nosec` comments only for false positives, with justification:

**✅ Good**:
```python
safe_host = NetworkBindingManager.get_safe_host()
click.echo(f"Access at: http://{safe_host if safe_host != '0.0.0.0' else 'localhost'}:5000")  # nosec B104 - String literal for display only
```

**❌ Bad**:
```python
app.run(host="0.0.0.0", port=5000)  # nosec - TODO: fix later
```

## Security Testing

### Run Bandit Locally

```bash
# Scan specific file
python -m bandit src/web/app.py

# Scan entire codebase
python -m bandit -r src/ -f json -o bandit-report.json

# Check for HIGH/MEDIUM issues
python scripts/check_bandit_results.py bandit-report.json
```

### Run Security Verification

```bash
python scripts/verify_security.py
```

## Pre-Commit Checklist

- [ ] No hardcoded credentials
- [ ] No `0.0.0.0` bindings without `NetworkBindingManager`
- [ ] No unpinned model downloads in production code
- [ ] No hardcoded `/tmp` paths
- [ ] No unsafe `pickle.load()` calls
- [ ] No `urllib.urlopen()` without scheme validation
- [ ] All Jinja2 environments have `autoescape=True`
- [ ] Bandit scan passes (0 HIGH/MEDIUM issues)

## Reporting Security Issues

**DO NOT** open public GitHub issues for security vulnerabilities.

Email security issues to: [security contact - configure this]

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

I will respond within 48 hours.
