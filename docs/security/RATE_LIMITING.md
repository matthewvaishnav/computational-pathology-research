# Rate Limiting Configuration Guide

## Overview
Rate limiting protects the API from abuse, DoS attacks, and brute force attempts.

## Current Configuration

### Global Rate Limit
```python
# src/api/security.py
limiter = Limiter(key_func=get_client_ip, default_limits=["30/minute"])
```

### Endpoint-Specific Limits

#### Authentication Endpoints (Strict)
```python
@limiter.limit("5/minute")  # 5 attempts per minute
@app.post("/api/v1/auth/login")
async def login():
    pass

@limiter.limit("3/minute")  # 3 attempts per minute
@app.post("/api/v1/auth/register")
async def register():
    pass
```

#### Analysis Endpoints (Moderate)
```python
@limiter.limit("10/minute")  # 10 analyses per minute
@app.post("/api/v1/analyze/image")
async def analyze_image():
    pass
```

#### Read-Only Endpoints (Relaxed)
```python
@limiter.limit("100/minute")  # 100 requests per minute
@app.get("/api/v1/cases")
async def list_cases():
    pass
```

## Recommended Limits by Endpoint Type

| Endpoint Type | Rate Limit | Reason |
|--------------|------------|--------|
| Login | 5/minute | Prevent brute force |
| Registration | 3/minute | Prevent spam accounts |
| Password Reset | 3/minute | Prevent enumeration |
| File Upload | 10/minute | Prevent resource exhaustion |
| Analysis | 10/minute | Expensive computation |
| Read Operations | 100/minute | Low cost |
| Health Check | No limit | Monitoring |

## Implementation

### Per-User Rate Limiting
```python
from fastapi import Depends
from src.api.dependencies import get_current_user

def get_user_id(user = Depends(get_current_user)) -> str:
    """Get user ID for rate limiting."""
    return user.id

@limiter.limit("10/minute", key_func=get_user_id)
@app.post("/api/v1/analyze")
async def analyze(user = Depends(get_current_user)):
    pass
```

### Dynamic Rate Limiting
```python
def get_rate_limit(user = Depends(get_current_user)) -> str:
    """Dynamic rate limit based on user tier."""
    if user.tier == "premium":
        return "100/minute"
    elif user.tier == "standard":
        return "30/minute"
    else:
        return "10/minute"

@limiter.limit(get_rate_limit)
@app.post("/api/v1/analyze")
async def analyze(user = Depends(get_current_user)):
    pass
```

### Redis-Based Rate Limiting (Production)
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)
```

## Testing Rate Limits

```python
import pytest
from fastapi.testclient import TestClient

def test_rate_limit_login():
    """Test login rate limiting."""
    client = TestClient(app)
    
    # First 5 requests should succeed (or fail with 401)
    for i in range(5):
        response = client.post("/api/v1/auth/login", json={
            "username": "test",
            "password": "wrong"
        })
        assert response.status_code in [200, 401]
    
    # 6th request should be rate limited
    response = client.post("/api/v1/auth/login", json={
        "username": "test",
        "password": "wrong"
    })
    assert response.status_code == 429
```

## Monitoring

### Prometheus Metrics
```python
from prometheus_client import Counter

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit violations',
    ['endpoint', 'ip']
)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    rate_limit_exceeded.labels(
        endpoint=request.url.path,
        ip=get_client_ip(request)
    ).inc()
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )
```

## Deployment Checklist

- [ ] Configure Redis for distributed rate limiting
- [ ] Set appropriate limits per endpoint type
- [ ] Implement per-user rate limiting
- [ ] Add rate limit monitoring
- [ ] Document limits in API documentation
- [ ] Test rate limits in staging
- [ ] Set up alerts for excessive rate limit violations

## References

- [slowapi Documentation](https://slowapi.readthedocs.io/)
- [OWASP Rate Limiting](https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html)
