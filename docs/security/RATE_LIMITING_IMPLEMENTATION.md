# Rate Limiting Implementation

## Overview

HistoCore implements comprehensive rate limiting to protect against abuse, DoS attacks, and resource exhaustion. The rate limiting system is environment-aware and integrates with the security audit trail.

## Architecture

### Components

1. **RateLimitConfig**: Configuration dataclass defining rate limits
2. **RateLimiter**: Core rate limiting engine with sliding window algorithm
3. **rate_limit_middleware**: FastAPI middleware for automatic enforcement
4. **SecurityAuditTrail**: Logs all rate limit violations

### Rate Limit Tiers

```python
# Default configuration
RateLimitConfig(
    requests_per_minute=60,    # 1 request/second average
    requests_per_hour=1000,    # Burst allowance
    requests_per_day=10000     # Daily quota
)
```

## Usage

### FastAPI Integration

```python
from fastapi import FastAPI
from src.security.rate_limit import rate_limit_middleware, RateLimitConfig

app = FastAPI()

# Apply rate limiting middleware
config = RateLimitConfig(
    requests_per_minute=100,
    requests_per_hour=2000,
    requests_per_day=20000
)
app.middleware("http")(rate_limit_middleware(config))
```

### Manual Rate Limiting

```python
from src.security.rate_limit import RateLimiter, RateLimitConfig

# Create rate limiter
config = RateLimitConfig(requests_per_minute=30)
limiter = RateLimiter(config)

# Check rate limit
allowed, error_msg = limiter.check_rate_limit(client_id="user123")
if not allowed:
    raise HTTPException(status_code=429, detail=error_msg)

# Get client stats
stats = limiter.get_client_stats(client_id="user123")
print(f"Requests in last minute: {stats['requests_last_minute']}")
```

## Client Identification

Rate limits are enforced per client ID. The middleware uses:

1. **API Key** (if present in `X-API-Key` header)
2. **User ID** (if authenticated)
3. **IP Address** (fallback)

```python
# Priority order
client_id = (
    request.headers.get("X-API-Key") or
    getattr(request.state, "user_id", None) or
    request.client.host
)
```

## Response Headers

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640000000
```

## Error Responses

When rate limit exceeded:

```json
{
  "detail": "Rate limit exceeded: 60 requests per minute. Try again in 30 seconds.",
  "status_code": 429
}
```

## Environment-Specific Configuration

### Production
```yaml
rate_limits:
  api:
    requests_per_minute: 60
    requests_per_hour: 1000
    requests_per_day: 10000
  
  inference:
    requests_per_minute: 10
    requests_per_hour: 100
    requests_per_day: 500
```

### Development
```yaml
rate_limits:
  api:
    requests_per_minute: 1000  # Relaxed for testing
    requests_per_hour: 10000
    requests_per_day: 100000
```

## Monitoring

### Audit Trail

All rate limit violations are logged:

```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "event_type": "rate_limit_exceeded",
  "client_id": "192.168.1.100",
  "endpoint": "/api/v1/inference",
  "limit_type": "requests_per_minute",
  "limit_value": 60,
  "current_count": 61
}
```

### Metrics

Track rate limiting effectiveness:

```python
# Get stats for all clients
for client_id in limiter.request_history.keys():
    stats = limiter.get_client_stats(client_id)
    print(f"{client_id}: {stats}")
```

## Best Practices

1. **Set appropriate limits**: Balance security and usability
2. **Use tiered limits**: Different limits for different endpoints
3. **Monitor violations**: Track patterns to detect attacks
4. **Whitelist trusted clients**: Exempt internal services
5. **Implement backoff**: Suggest retry-after times

## Security Considerations

- **Distributed systems**: Use Redis for shared rate limit state
- **Clock skew**: Use monotonic time for window calculations
- **Memory limits**: Implement cleanup for old request history
- **Bypass protection**: Validate client IDs to prevent spoofing

## Testing

```python
import pytest
from src.security.rate_limit import RateLimiter, RateLimitConfig

def test_rate_limit_enforcement():
    config = RateLimitConfig(requests_per_minute=5)
    limiter = RateLimiter(config)
    
    # First 5 requests should succeed
    for i in range(5):
        allowed, _ = limiter.check_rate_limit("test_client")
        assert allowed
    
    # 6th request should fail
    allowed, error_msg = limiter.check_rate_limit("test_client")
    assert not allowed
    assert "Rate limit exceeded" in error_msg
```

## References

- [OWASP Rate Limiting Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html)
- [RFC 6585 - Additional HTTP Status Codes](https://tools.ietf.org/html/rfc6585#section-4)
- [NIST SP 800-53 SC-5 - Denial of Service Protection](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf)
