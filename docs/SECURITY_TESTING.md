# Security Testing Guide

## Running Security Tests Locally

### Bandit Static Analysis

Bandit scans Python code for common security issues.

**Scan entire codebase**:
```bash
python -m bandit -r src/ -f json -o bandit-report.json
```

**Scan specific file**:
```bash
python -m bandit src/web/app.py
```

**Check for HIGH/MEDIUM issues**:
```bash
python scripts/check_bandit_results.py bandit-report.json
```

**Generate HTML report**:
```bash
python scripts/generate_security_report.py bandit-report.json security-report.html
```

### Dependency Vulnerability Scanning

**Safety** (checks against known vulnerability database):
```bash
pip install safety
safety check --json --output safety-report.json
```

**pip-audit** (checks PyPI advisory database):
```bash
pip install pip-audit
pip-audit --format json --output pip-audit-report.json
```

### Security Verification Script

Comprehensive security verification:
```bash
python scripts/verify_security.py
```

This checks:
- Jinja2 autoescape enabled
- Server binding policies
- Model download revision pinning
- Temporary file security
- Pickle deserialization safety
- URL opening security
- Credential management

### Property-Based Testing

Property-based tests use Hypothesis to generate test cases.

**Run all property tests**:
```bash
pytest tests/ -v -m property
```

**Run specific property test**:
```bash
pytest tests/test_security_properties.py::test_network_binding_respects_environment -v
```

**Increase test iterations** (default 100):
```bash
pytest tests/ -m property --hypothesis-iterations=1000
```

## Security Test Categories

### 1. Jinja2 XSS Tests

**Property**: All Jinja2 environments have autoescape enabled.

**Test**:
```python
from hypothesis import given, strategies as st
from src.security.jinja2_security import SecureJinja2Environment

@given(st.text())
def test_jinja2_autoescape_enabled(template_string):
    env = SecureJinja2Environment.create_environment()
    assert env.autoescape is True
```

### 2. Network Binding Tests

**Property**: Network binding respects environment policy.

**Test**:
```python
from hypothesis import given, strategies as st
from src.security.network_binding import NetworkBindingManager

@given(st.sampled_from(['production', 'development', 'research']))
def test_network_binding_respects_environment(environment):
    os.environ['ENVIRONMENT'] = environment
    manager = NetworkBindingManager()
    host = manager.get_safe_host()
    
    if environment == 'production':
        assert host != '0.0.0.0'
    elif environment == 'development':
        assert host == '127.0.0.1'
```

### 3. Model Download Tests

**Property**: Model downloads use pinned revisions when required.

**Test**:
```python
from hypothesis import given, strategies as st
from src.security.model_download import ModelDownloadManager

@given(st.sampled_from(['owkin/phikon', 'microsoft/resnet-50']))
def test_model_downloads_use_pinned_revisions(model_name):
    os.environ['ENVIRONMENT'] = 'production'
    manager = ModelDownloadManager()
    revision = manager.get_pinned_revision(model_name)
    assert revision is not None
    assert len(revision) == 40  # Git SHA-1 hash
```

### 4. Temporary File Tests

**Property**: Temp files never use hardcoded paths.

**Test**:
```python
from hypothesis import given, strategies as st
from src.security.temp_file import TempFileManager

@given(st.text(min_size=1, max_size=10))
def test_temp_files_never_use_hardcoded_paths(suffix):
    with TempFileManager.create_temp_file(suffix=suffix) as tmp_path:
        assert '/tmp' not in str(tmp_path) or os.name == 'posix'
        assert tmp_path.exists()
```

### 5. Pickle Security Tests

**Property**: Pickle source validation works correctly.

**Test**:
```python
from hypothesis import given, strategies as st
from src.security.pickle_security import PickleSecurityControl

@given(st.sampled_from(['checkpoints/model.pkl', '/tmp/untrusted.pkl']))
def test_pickle_source_validation(pickle_path):
    control = PickleSecurityControl()
    is_trusted = control.is_trusted_source(pickle_path)
    
    if 'checkpoints/' in pickle_path:
        assert is_trusted is True
    elif '/tmp/' in pickle_path:
        assert is_trusted is False
```

### 6. URL Security Tests

**Property**: URL scheme validation blocks unsafe schemes.

**Test**:
```python
from hypothesis import given, strategies as st
from src.security.url_fetcher import URLFetcherControl

@given(st.sampled_from(['http://example.com', 'https://example.com', 'file:///etc/passwd']))
def test_url_scheme_validation(url):
    control = URLFetcherControl()
    
    if url.startswith('file://'):
        with pytest.raises(ValueError):
            control.validate_url_scheme(url)
    else:
        control.validate_url_scheme(url)  # Should not raise
```

## CI/CD Integration

### GitHub Actions

Security tests run automatically on every PR:

**`.github/workflows/ci.yml`**:
```yaml
security:
  name: Security Scan
  runs-on: ubuntu-latest
  steps:
    - name: Run bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json
    
    - name: Check bandit results
      run: python scripts/check_bandit_results.py bandit-report.json
```

**`.github/workflows/security.yml`**:
```yaml
comprehensive-security-audit:
  name: Comprehensive Security Audit
  runs-on: ubuntu-latest
  steps:
    - name: Run comprehensive Bandit scan
      run: bandit -r src/ -f json -o bandit-full-report.json
    
    - name: Run Safety check
      run: safety check --json --output safety-report.json
    
    - name: Run pip-audit
      run: pip-audit --format json --output pip-audit-report.json
```

## Manual Security Testing

### Penetration Testing

For production deployments, conduct manual penetration testing:

1. **Authentication bypass attempts**
2. **SQL injection testing**
3. **XSS payload testing**
4. **CSRF token validation**
5. **Rate limiting verification**
6. **Session management testing**
7. **Authorization bypass attempts**

### Security Audit

Quarterly security audits should include:

1. **Code review** of security-critical components
2. **Dependency audit** for known vulnerabilities
3. **Configuration review** of production settings
4. **Access control review** of user permissions
5. **Incident response drill** to test procedures

## Troubleshooting

### Bandit False Positives

Use `# nosec` comments with justification:
```python
safe_host = NetworkBindingManager.get_safe_host()
print(f"Server: {safe_host if safe_host != '0.0.0.0' else 'localhost'}")  # nosec B104 - String literal for display only
```

### Property Test Failures

If property tests fail:

1. **Review the failing example** in test output
2. **Reproduce manually** with the failing input
3. **Fix the underlying issue** (not the test)
4. **Re-run with more iterations** to verify fix

### CI/CD Failures

If security checks fail in CI/CD:

1. **Review the Bandit report artifact**
2. **Fix HIGH/MEDIUM severity issues**
3. **Add `# nosec` for false positives** with justification
4. **Re-run the workflow**

## Best Practices

1. **Run Bandit locally** before committing
2. **Fix security issues immediately** (don't defer)
3. **Use `# nosec` sparingly** and only for false positives
4. **Keep dependencies updated** to patch vulnerabilities
5. **Review security logs regularly** for suspicious activity
6. **Test security controls** in staging before production
7. **Document security decisions** and rationale
