# Security Headers Implementation

## Overview

HTTP security headers protect against common web vulnerabilities including XSS, clickjacking, MIME sniffing, and information leakage. This guide covers security header implementation for the platform web services.

## Required Headers

### 1. Content-Security-Policy (CSP)

Prevents XSS attacks by controlling resource loading.

```python
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        return response

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)
```

**Directives:**
- `default-src 'self'`: Only load resources from same origin
- `script-src`: Control JavaScript sources
- `style-src`: Control CSS sources
- `img-src`: Control image sources
- `frame-ancestors 'none'`: Prevent clickjacking
- `base-uri 'self'`: Prevent base tag injection

### 2. X-Content-Type-Options

Prevents MIME type sniffing.

```python
response.headers["X-Content-Type-Options"] = "nosniff"
```

**Purpose:** Forces browsers to respect declared Content-Type, preventing execution of misinterpreted files.

### 3. X-Frame-Options

Prevents clickjacking attacks.

```python
response.headers["X-Frame-Options"] = "DENY"
# Or for specific domains:
# response.headers["X-Frame-Options"] = "SAMEORIGIN"
```

**Options:**
- `DENY`: Never allow framing
- `SAMEORIGIN`: Allow framing from same origin
- `ALLOW-FROM uri`: Allow framing from specific URI (deprecated)

### 4. Strict-Transport-Security (HSTS)

Forces HTTPS connections.

```python
response.headers["Strict-Transport-Security"] = (
    "max-age=31536000; includeSubDomains; preload"
)
```

**Parameters:**
- `max-age=31536000`: Enforce HTTPS for 1 year
- `includeSubDomains`: Apply to all subdomains
- `preload`: Eligible for browser preload lists

### 5. X-XSS-Protection

Legacy XSS filter (deprecated but still useful for older browsers).

```python
response.headers["X-XSS-Protection"] = "1; mode=block"
```

**Modes:**
- `0`: Disable filter
- `1`: Enable filter
- `1; mode=block`: Enable and block page rendering on XSS detection

### 6. Referrer-Policy

Controls referrer information leakage.

```python
response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
```

**Policies:**
- `no-referrer`: Never send referrer
- `strict-origin`: Send origin only for HTTPS→HTTPS
- `strict-origin-when-cross-origin`: Full URL for same-origin, origin only for cross-origin

### 7. Permissions-Policy

Controls browser features and APIs.

```python
response.headers["Permissions-Policy"] = (
    "geolocation=(), "
    "microphone=(), "
    "camera=(), "
    "payment=(), "
    "usb=(), "
    "magnetometer=(), "
    "gyroscope=(), "
    "accelerometer=()"
)
```

## Complete Implementation

### FastAPI Middleware

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import os

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Comprehensive security headers middleware."""
    
    def __init__(self, app, environment: str = None):
        super().__init__(app)
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
    
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        
        # Content Security Policy
        if self.environment == 'production':
            csp = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        else:
            # Relaxed for development
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
        
        response.headers["Content-Security-Policy"] = csp
        
        # MIME type sniffing protection
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Clickjacking protection
        response.headers["X-Frame-Options"] = "DENY"
        
        # HTTPS enforcement (production only)
        if self.environment == 'production':
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # XSS protection (legacy)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
        )
        
        # Remove server header (information disclosure)
        response.headers.pop("Server", None)
        
        # Remove X-Powered-By (information disclosure)
        response.headers.pop("X-Powered-By", None)
        
        return response

# Usage
app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware, environment='production')
```

### Nginx Configuration

For production deployments behind Nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name api.the platform.example.com;
    
    # SSL configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=(), payment=(), usb=()" always;
    
    # Remove server tokens
    server_tokens off;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Testing

### Manual Testing

```bash
# Test security headers
curl -I https://api.the platform.example.com

# Expected output:
# Content-Security-Policy: default-src 'self'; ...
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
# X-XSS-Protection: 1; mode=block
# Referrer-Policy: strict-origin-when-cross-origin
# Permissions-Policy: geolocation=(), ...
```

### Automated Testing

```python
import pytest
from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)

def test_security_headers():
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
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
    
    # XSS protection
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    
    # Referrer policy
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    
    # Permissions policy
    assert "Permissions-Policy" in response.headers
    
    # Information disclosure
    assert "Server" not in response.headers
    assert "X-Powered-By" not in response.headers

def test_csp_blocks_inline_scripts():
    """Test that CSP blocks inline scripts in production."""
    if os.getenv('ENVIRONMENT') == 'production':
        response = client.get("/")
        csp = response.headers["Content-Security-Policy"]
        assert "'unsafe-inline'" not in csp or "script-src" not in csp
```

### Online Scanners

- [Mozilla Observatory](https://observatory.mozilla.org/)
- [Security Headers](https://securityheaders.com/)
- [SSL Labs](https://www.ssllabs.com/ssltest/)

## Environment-Specific Configuration

### Production

```python
# Strict CSP, no unsafe-inline/unsafe-eval
# HSTS enabled
# All security headers enforced
```

### Development

```python
# Relaxed CSP for hot reload
# HSTS disabled (localhost)
# Security headers present but permissive
```

### Testing

```python
# Minimal headers for test speed
# CSP report-only mode
```

## Common Issues

### Issue 1: CSP Blocks Legitimate Resources

**Symptom:** Console errors like "Refused to load script"

**Solution:** Add specific domains to CSP directives
```python
"script-src 'self' https://cdn.example.com"
```

### Issue 2: HSTS Breaks Local Development

**Symptom:** Browser forces HTTPS on localhost

**Solution:** Only enable HSTS in production
```python
if environment == 'production':
    response.headers["Strict-Transport-Security"] = "..."
```

### Issue 3: Frame-Ancestors Conflicts with X-Frame-Options

**Symptom:** Both headers present

**Solution:** Use CSP frame-ancestors (modern) or X-Frame-Options (legacy), not both
```python
# Modern approach
"frame-ancestors 'none'"

# Legacy approach
response.headers["X-Frame-Options"] = "DENY"
```

## References

- [OWASP Secure Headers Project](https://owasp.org/www-project-secure-headers/)
- [MDN Web Security](https://developer.mozilla.org/en-US/docs/Web/Security)
- [Content Security Policy Reference](https://content-security-policy.com/)
- [NIST SP 800-52 Rev. 2 - TLS Guidelines](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-52r2.pdf)
