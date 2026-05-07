# CSRF Protection Guide

## Overview
Cross-Site Request Forgery (CSRF) protection for the HistoCore API.

## Current Protection

### JWT Bearer Tokens (API)
- ✅ **Protected**: API endpoints using JWT Bearer tokens are NOT vulnerable to CSRF
- JWT tokens in Authorization header cannot be sent by malicious sites
- No additional CSRF protection needed for API-only endpoints

### Cookie-Based Authentication (Web UI)
- ⚠️ **Requires Protection**: Any cookie-based authentication MUST implement CSRF protection

## Implementation for Cookie-Based Auth

### Option 1: Double-Submit Cookie Pattern

```python
from fastapi import Cookie, Header, HTTPException
import secrets

def generate_csrf_token() -> str:
    """Generate CSRF token."""
    return secrets.token_urlsafe(32)

def verify_csrf_token(
    csrf_token_cookie: str = Cookie(None),
    csrf_token_header: str = Header(None, alias="X-CSRF-Token")
) -> bool:
    """Verify CSRF token matches."""
    if not csrf_token_cookie or not csrf_token_header:
        raise HTTPException(status_code=403, detail="CSRF token missing")
    
    if not secrets.compare_digest(csrf_token_cookie, csrf_token_header):
        raise HTTPException(status_code=403, detail="CSRF token mismatch")
    
    return True
```

### Option 2: SameSite Cookie Attribute

```python
from fastapi import Response

def set_session_cookie(response: Response, session_id: str):
    """Set session cookie with SameSite protection."""
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,  # Prevent JavaScript access
        secure=True,    # HTTPS only
        samesite="strict",  # CSRF protection
        max_age=3600
    )
```

### Option 3: fastapi-csrf-protect Library

```python
from fastapi_csrf_protect import CsrfProtect
from pydantic import BaseModel

class CsrfSettings(BaseModel):
    secret_key: str = "your-secret-key"
    cookie_samesite: str = "strict"

@CsrfProtect.load_config
def get_csrf_config():
    return CsrfSettings()

@app.post("/api/action")
async def protected_action(csrf_protect: CsrfProtect = Depends()):
    await csrf_protect.validate_csrf()
    # Process request
```

## Testing CSRF Protection

```python
import pytest
from fastapi.testclient import TestClient

def test_csrf_protection():
    """Test CSRF token validation."""
    client = TestClient(app)
    
    # Request without CSRF token should fail
    response = client.post("/api/action")
    assert response.status_code == 403
    
    # Request with valid CSRF token should succeed
    csrf_token = client.get("/api/csrf-token").json()["token"]
    response = client.post(
        "/api/action",
        headers={"X-CSRF-Token": csrf_token}
    )
    assert response.status_code == 200
```

## Deployment Checklist

- [ ] Identify all cookie-based authentication endpoints
- [ ] Implement CSRF protection (choose one method above)
- [ ] Set SameSite=Strict on all session cookies
- [ ] Add CSRF tests to test suite
- [ ] Document CSRF token usage for frontend developers
- [ ] Verify HTTPS is enforced (required for Secure cookies)

## References

- [OWASP CSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
