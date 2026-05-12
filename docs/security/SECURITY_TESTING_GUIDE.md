# Security Testing Guide

## Overview

Security testing validates that security controls work as intended and vulnerabilities are prevented. This guide covers security testing approaches for HistoCore.

## Testing Pyramid

```
         /\
        /  \  Property-Based Tests (Universal properties)
       /____\
      /      \  Integration Tests (End-to-end security)
     /________\
    /          \  Unit Tests (Individual controls)
   /____________\
```

## Unit Testing

### 1. Input Validation Tests

```python
import pytest
from src.security.validation import InputValidator

class TestInputValidation:
    """Test input validation security controls."""
    
    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32",
        ]
        
        for path in malicious_paths:
            with pytest.raises(ValueError):
                InputValidator.validate_path(
                    path,
                    allowed_dirs=[Path("/data")]
                )
    
    def test_sql_injection_blocked(self):
        """Test that SQL injection attempts are blocked."""
        sql_injections = [
            "admin' OR '1'='1",
            "'; DROP TABLE users--",
            "1' UNION SELECT * FROM passwords--",
        ]
        
        for injection in sql_injections:
            with pytest.raises(ValueError):
                InputValidator.validate_string(
                    injection,
                    pattern=r'^[a-zA-Z0-9_]+$'
                )
    
    def test_xss_blocked(self):
        """Test that XSS attempts are blocked."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
        ]
        
        for payload in xss_payloads:
            with pytest.raises(ValueError):
                InputValidator.validate_string(
                    payload,
                    pattern=r'^[a-zA-Z0-9\s]+$'
                )
```

### 2. Authentication Tests

```python
import pytest
from src.api.routers.auth import authenticate_user, create_access_token

class TestAuthentication:
    """Test authentication security controls."""
    
    def test_weak_password_rejected(self):
        """Test that weak passwords are rejected."""
        weak_passwords = [
            "123456",
            "password",
            "abc",
            "12345678",
        ]
        
        for password in weak_passwords:
            with pytest.raises(ValueError):
                validate_password_strength(password)
    
    def test_brute_force_protection(self):
        """Test that brute force attempts are blocked."""
        username = "test_user"
        
        # Attempt 5 failed logins
        for _ in range(5):
            with pytest.raises(HTTPException):
                authenticate_user(username, "wrong_password")
        
        # 6th attempt should be rate limited
        with pytest.raises(HTTPException) as exc:
            authenticate_user(username, "wrong_password")
        
        assert exc.value.status_code == 429  # Too Many Requests
    
    def test_token_expiration(self):
        """Test that expired tokens are rejected."""
        # Create token with 1 second expiry
        token = create_access_token(
            data={"sub": "user123"},
            expires_delta=timedelta(seconds=1)
        )
        
        # Wait for expiration
        time.sleep(2)
        
        # Token should be invalid
        with pytest.raises(HTTPException):
            verify_token(token)
```

### 3. Authorization Tests

```python
import pytest
from src.api.dependencies import check_permissions

class TestAuthorization:
    """Test authorization security controls."""
    
    def test_unauthorized_access_blocked(self):
        """Test that unauthorized access is blocked."""
        user = User(id=1, role="viewer")
        
        with pytest.raises(HTTPException) as exc:
            check_permissions(user, required_role="admin")
        
        assert exc.value.status_code == 403  # Forbidden
    
    def test_privilege_escalation_blocked(self):
        """Test that privilege escalation is blocked."""
        user = User(id=1, role="viewer")
        
        # Attempt to modify own role
        with pytest.raises(HTTPException):
            update_user_role(user, user.id, "admin")
    
    def test_horizontal_privilege_escalation_blocked(self):
        """Test that users can't access other users' data."""
        user1 = User(id=1, role="viewer")
        user2_data_id = 2
        
        with pytest.raises(HTTPException):
            get_user_data(user1, user2_data_id)
```

## Integration Testing

### 1. End-to-End Security Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)

class TestEndToEndSecurity:
    """Test end-to-end security flows."""
    
    def test_secure_file_upload_flow(self):
        """Test complete secure file upload flow."""
        # 1. Authenticate
        response = client.post("/auth/login", json={
            "username": "test_user",
            "password": "SecurePass123!"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        
        # 2. Upload file with malicious name
        files = {"file": ("../../etc/passwd", b"malicious content")}
        headers = {"Authorization": f"Bearer {token}"}
        
        response = client.post(
            "/upload",
            files=files,
            headers=headers
        )
        
        # Should be rejected
        assert response.status_code == 400
        assert "Invalid filename" in response.json()["detail"]
    
    def test_api_rate_limiting(self):
        """Test that API rate limiting works."""
        # Make 100 requests rapidly
        for i in range(100):
            response = client.get("/api/health")
            
            if i < 60:
                # First 60 should succeed
                assert response.status_code == 200
            else:
                # Remaining should be rate limited
                assert response.status_code == 429
```

### 2. Security Header Tests

```python
def test_security_headers_present():
    """Test that all security headers are present."""
    response = client.get("/")
    
    # CSP
    assert "Content-Security-Policy" in response.headers
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    
    # MIME sniffing protection
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    
    # Clickjacking protection
    assert response.headers["X-Frame-Options"] == "DENY"
    
    # HTTPS enforcement (production)
    if os.getenv('ENVIRONMENT') == 'production':
        assert "Strict-Transport-Security" in response.headers
    
    # XSS protection
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    
    # Referrer policy
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    
    # Information disclosure
    assert "Server" not in response.headers
    assert "X-Powered-By" not in response.headers
```

## Property-Based Testing

### 1. Hypothesis Tests

```python
from hypothesis import given, strategies as st
from src.security.validation import InputValidator

class TestSecurityProperties:
    """Test universal security properties."""
    
    @given(st.text())
    def test_no_path_traversal_possible(self, user_input):
        """Property: No input should allow path traversal."""
        try:
            validated = InputValidator.validate_path(
                user_input,
                allowed_dirs=[Path("/data")]
            )
            # If validation succeeds, path must be within allowed dir
            assert validated.is_relative_to(Path("/data"))
        except ValueError:
            # Validation rejection is acceptable
            pass
    
    @given(st.text(min_size=1, max_size=1000))
    def test_no_sql_injection_possible(self, user_input):
        """Property: No input should allow SQL injection."""
        try:
            validated = InputValidator.validate_string(
                user_input,
                pattern=r'^[a-zA-Z0-9_]+$'
            )
            # If validation succeeds, must be alphanumeric
            assert validated.isalnum() or '_' in validated
        except ValueError:
            # Validation rejection is acceptable
            pass
    
    @given(st.integers(min_value=-1000, max_value=1000))
    def test_batch_size_always_positive(self, value):
        """Property: Validated batch size is always positive."""
        try:
            batch_size = InputValidator.validate_batch_size(value)
            assert batch_size > 0
            assert batch_size <= 256
        except ValueError:
            # Rejection of invalid values is acceptable
            pass
```

## Penetration Testing

### 1. OWASP ZAP Integration

```bash
# Start ZAP in daemon mode
docker run -u zap -p 8080:8080 -d owasp/zap2docker-stable zap.sh -daemon -host 0.0.0.0 -port 8080 -config api.disablekey=true

# Run baseline scan
docker run -v $(pwd):/zap/wrk/:rw -t owasp/zap2docker-stable zap-baseline.py \
  -t http://localhost:8000 \
  -r zap-report.html

# Run full scan
docker run -v $(pwd):/zap/wrk/:rw -t owasp/zap2docker-stable zap-full-scan.py \
  -t http://localhost:8000 \
  -r zap-full-report.html
```

### 2. Burp Suite Tests

Manual testing checklist:

- [ ] SQL injection in all input fields
- [ ] XSS in all input fields
- [ ] CSRF token validation
- [ ] Session fixation
- [ ] Insecure direct object references
- [ ] Path traversal
- [ ] Command injection
- [ ] XXE (XML External Entity)
- [ ] SSRF (Server-Side Request Forgery)

## Static Analysis

### 1. Bandit Scan

```bash
# Run Bandit scan
bandit -r src -f json -o bandit-report.json

# Check results
python scripts/check_bandit_results.py bandit-report.json
```

### 2. Semgrep Scan

```bash
# Install Semgrep
pip install semgrep

# Run security rules
semgrep --config=p/security-audit src/

# Run OWASP Top 10 rules
semgrep --config=p/owasp-top-ten src/

# Custom rules
semgrep --config=.semgrep.yml src/
```

## Continuous Security Testing

### CI/CD Pipeline

```yaml
# .github/workflows/security-tests.yml
name: Security Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov hypothesis
      - name: Run security unit tests
        run: pytest tests/security/ -v --cov=src/security
  
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: pytest tests/integration/security/ -v
  
  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install hypothesis
      - name: Run property-based tests
        run: pytest tests/properties/ -v --hypothesis-show-statistics
  
  static-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r src -f json -o bandit-report.json
          python scripts/check_bandit_results.py bandit-report.json
      - name: Run Semgrep
        run: |
          pip install semgrep
          semgrep --config=p/security-audit src/
```

## Test Coverage

### Security Test Coverage Goals

- **Unit tests**: 100% coverage of security controls
- **Integration tests**: All critical security flows
- **Property tests**: All validation functions
- **Penetration tests**: Quarterly manual testing

### Measuring Coverage

```bash
# Run tests with coverage
pytest tests/security/ --cov=src/security --cov-report=html

# View coverage report
open htmlcov/index.html
```

## References

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [NIST SP 800-115 - Technical Guide to Information Security Testing](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-115.pdf)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [OWASP ZAP Documentation](https://www.zaproxy.org/docs/)
