# API Security Best Practices

## Overview

This guide covers security best practices for HistoCore's REST API, including authentication, authorization, input validation, and rate limiting.

## Authentication

### JWT Token-Based Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    # Use strong secret key from environment
    secret_key = os.getenv("JWT_SECRET_KEY")
    if not secret_key:
        raise ValueError("JWT_SECRET_KEY must be set")
    
    encoded_jwt = jwt.encode(
        to_encode,
        secret_key,
        algorithm="HS256"
    )
    
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Validate JWT token and return current user."""
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET_KEY"),
            algorithms=["HS256"]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(user_id)
    if user is None:
        raise credentials_exception
    
    return user
```

### API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(
    api_key: str = Security(api_key_header)
) -> str:
    """Validate API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    
    # Validate against database
    if not is_valid_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
```

## Authorization

### Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import List

class Role(str, Enum):
    ADMIN = "admin"
    PATHOLOGIST = "pathologist"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class Permission(str, Enum):
    READ_PATIENTS = "read:patients"
    WRITE_PATIENTS = "write:patients"
    READ_MODELS = "read:models"
    WRITE_MODELS = "write:models"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"

ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.READ_PATIENTS,
        Permission.WRITE_PATIENTS,
        Permission.READ_MODELS,
        Permission.WRITE_MODELS,
        Permission.ADMIN_USERS,
        Permission.ADMIN_SYSTEM,
    ],
    Role.PATHOLOGIST: [
        Permission.READ_PATIENTS,
        Permission.WRITE_PATIENTS,
        Permission.READ_MODELS,
    ],
    Role.RESEARCHER: [
        Permission.READ_MODELS,
        Permission.WRITE_MODELS,
    ],
    Role.VIEWER: [
        Permission.READ_PATIENTS,
        Permission.READ_MODELS,
    ],
}

def check_permission(
    user: User,
    required_permission: Permission
) -> bool:
    """Check if user has required permission."""
    user_permissions = ROLE_PERMISSIONS.get(user.role, [])
    return required_permission in user_permissions

def require_permission(required_permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(
            *args,
            current_user: User = Depends(get_current_user),
            **kwargs
        ):
            if not check_permission(current_user, required_permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

# Usage
@app.post("/patients")
@require_permission(Permission.WRITE_PATIENTS)
async def create_patient(
    patient: PatientCreate,
    current_user: User = Depends(get_current_user)
):
    return create_patient_record(patient)
```

## Input Validation

### Request Body Validation

```python
from pydantic import BaseModel, Field, validator
from typing import Optional

class InferenceRequest(BaseModel):
    """Inference request with validation."""
    
    image_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Path to WSI image"
    )
    
    model_type: str = Field(
        ...,
        regex="^(resnet|vit|dino)$",
        description="Model type"
    )
    
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for inference"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold"
    )
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate image path."""
        from src.security.validation import InputValidator
        return InputValidator.validate_path(
            v,
            allowed_dirs=[Path("/data/images")],
            must_exist=True
        )
    
    class Config:
        schema_extra = {
            "example": {
                "image_path": "/data/images/slide001.svs",
                "model_type": "resnet",
                "batch_size": 32,
                "confidence_threshold": 0.5
            }
        }
```

### Query Parameter Validation

```python
from fastapi import Query

@app.get("/patients")
async def list_patients(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of results"
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Offset for pagination"
    ),
    search: Optional[str] = Query(
        default=None,
        max_length=100,
        regex="^[a-zA-Z0-9\\s]+$",
        description="Search query"
    ),
    current_user: User = Depends(get_current_user)
):
    """List patients with pagination."""
    return get_patients(limit=limit, offset=offset, search=search)
```

## Rate Limiting

### Endpoint-Specific Rate Limits

```python
from src.security.rate_limit import RateLimiter, RateLimitConfig

# Different limits for different endpoints
RATE_LIMITS = {
    "/api/v1/inference": RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=500
    ),
    "/api/v1/patients": RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000
    ),
    "/api/v1/auth/login": RateLimitConfig(
        requests_per_minute=5,
        requests_per_hour=20,
        requests_per_day=100
    ),
}

def get_rate_limiter(endpoint: str) -> RateLimiter:
    """Get rate limiter for endpoint."""
    config = RATE_LIMITS.get(endpoint, RateLimitConfig())
    return RateLimiter(config)

@app.post("/api/v1/inference")
async def run_inference(
    request: InferenceRequest,
    client_id: str = Depends(get_client_id)
):
    """Run inference with rate limiting."""
    limiter = get_rate_limiter("/api/v1/inference")
    
    allowed, error_msg = limiter.check_rate_limit(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=error_msg
        )
    
    return perform_inference(request)
```

## CORS Configuration

### Secure CORS Setup

```python
from fastapi.middleware.cors import CORSMiddleware

# Production CORS configuration
if os.getenv('ENVIRONMENT') == 'production':
    allowed_origins = [
        "https://app.histocore.example.com",
        "https://admin.histocore.example.com",
    ]
else:
    # Development - more permissive
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8080",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)
```

## Error Handling

### Secure Error Responses

```python
from src.utils.error_handling import SecureErrorResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions securely."""
    
    # Log full error internally
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host
        }
    )
    
    # Return generic error to client
    if os.getenv('ENVIRONMENT') == 'production':
        return JSONResponse(
            status_code=500,
            content=SecureErrorResponse.generic_error(500)
        )
    else:
        # Development - include details
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
```

## Request Size Limits

### Limit Request Body Size

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent DoS."""
    
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10 MB
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            
            if content_length:
                if int(content_length) > self.max_size:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"}
                    )
        
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)
```

## API Versioning

### Version-Specific Security

```python
from fastapi import APIRouter

# API v1 - legacy, more permissive
api_v1 = APIRouter(prefix="/api/v1")

@api_v1.get("/patients")
async def list_patients_v1():
    """Legacy endpoint - deprecated."""
    return {"warning": "This endpoint is deprecated. Use /api/v2/patients"}

# API v2 - current, strict security
api_v2 = APIRouter(
    prefix="/api/v2",
    dependencies=[Depends(get_current_user)]  # Require auth for all v2 endpoints
)

@api_v2.get("/patients")
async def list_patients_v2(
    current_user: User = Depends(get_current_user)
):
    """Current endpoint with strict security."""
    return get_patients()

app.include_router(api_v1)
app.include_router(api_v2)
```

## Audit Logging

### Log All API Requests

```python
from src.security.audit_trail import SecurityAuditTrail

audit = SecurityAuditTrail()

@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Log all API requests."""
    
    # Extract user info
    user_id = None
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
            user_id = payload.get("sub")
    except:
        pass
    
    # Log request
    audit.log_policy_applied(
        policy_name="api_request",
        decision="allowed",
        context={
            "method": request.method,
            "path": request.url.path,
            "user_id": user_id,
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent")
        }
    )
    
    response = await call_next(request)
    
    # Log response
    audit.log_policy_applied(
        policy_name="api_response",
        decision="completed",
        context={
            "status_code": response.status_code,
            "path": request.url.path,
            "user_id": user_id
        }
    )
    
    return response
```

## Testing

### Security Test Examples

```python
import pytest
from fastapi.testclient import TestClient

def test_authentication_required():
    """Test that endpoints require authentication."""
    response = client.get("/api/v2/patients")
    assert response.status_code == 401

def test_invalid_token_rejected():
    """Test that invalid tokens are rejected."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get("/api/v2/patients", headers=headers)
    assert response.status_code == 401

def test_authorization_enforced():
    """Test that authorization is enforced."""
    # Viewer role should not be able to create patients
    token = create_token_for_role(Role.VIEWER)
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.post(
        "/api/v2/patients",
        json={"name": "Test Patient"},
        headers=headers
    )
    assert response.status_code == 403

def test_rate_limiting():
    """Test that rate limiting works."""
    token = create_valid_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Make 100 requests
    for i in range(100):
        response = client.get("/api/v2/patients", headers=headers)
        
        if i < 60:
            assert response.status_code == 200
        else:
            assert response.status_code == 429

def test_input_validation():
    """Test that input validation works."""
    token = create_valid_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Invalid batch size
    response = client.post(
        "/api/v2/inference",
        json={
            "image_path": "/data/images/slide.svs",
            "model_type": "resnet",
            "batch_size": 1000  # Too large
        },
        headers=headers
    )
    assert response.status_code == 422
```

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)
- [NIST SP 800-63B - Digital Identity Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)
