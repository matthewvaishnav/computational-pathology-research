# Developer Guide: Adding Endpoints to API Routes

## Overview

This guide explains how to add new endpoints to the modular FastAPI application following the established patterns and best practices.

## Router Architecture

The API is organized into 5 domain-specific routers:

- **`auth.py`**: Authentication and authorization endpoints
- **`analysis.py`**: Image analysis, DICOM processing, and case management
- **`admin.py`**: Administrative operations requiring admin privileges
- **`mobile.py`**: Mobile device management and offline synchronization
- **`monitoring.py`**: Health checks, metrics, and system monitoring

## Adding a New Endpoint

### Step 1: Choose the Appropriate Router

Select the router that best matches your endpoint's domain:

```python
# For user authentication/authorization
src/api/routers/auth.py

# For image processing/analysis
src/api/routers/analysis.py

# For admin-only operations
src/api/routers/admin.py

# For mobile device features
src/api/routers/mobile.py

# For system monitoring/health
src/api/routers/monitoring.py
```

### Step 2: Follow the Standard Endpoint Pattern

All endpoints should follow this consistent pattern:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.api.dependencies import get_db_session, get_current_user
from src.api.validators import validate_email, validate_password
from src.database.models import User

router = APIRouter(prefix="/api/v1/domain", tags=["domain"])

# Request/Response models
class MyRequest(BaseModel):
    field1: str
    field2: int
    
class MyResponse(BaseModel):
    result: str
    success: bool

@router.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(
    # Request body
    request: MyRequest,
    # Path parameters
    resource_id: str,
    # Query parameters
    limit: int = 10,
    # Dependencies
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """
    Endpoint description for OpenAPI documentation.
    
    Args:
        request: Request body with required fields
        resource_id: Path parameter for resource identification
        limit: Optional query parameter for pagination
        db: Database session dependency
        current_user: Authenticated user dependency
        
    Returns:
        MyResponse: Success response with result data
        
    Raises:
        HTTPException: 400 for validation errors, 404 for not found, 500 for server errors
    """
    try:
        # Input validation
        if request.field1:
            validate_email(request.field1)  # Use shared validators
        
        # Business logic
        result = perform_operation(request, resource_id, db, current_user)
        
        # Return response
        return MyResponse(result=result, success=True)
        
    except HTTPException:
        # Re-raise HTTP exceptions (already have correct status code)
        raise
    except ValueError as e:
        # Client errors (400)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Server errors (500)
        logger.error(f"Operation failed: {e}")
        raise HTTPException(status_code=500, detail="Operation failed")
```

### Step 3: Use Shared Dependencies

Always use the shared dependency functions from `src/api/dependencies.py`:

```python
from src.api.dependencies import (
    get_db_session,        # Database session
    get_current_user,      # Authenticated user
    get_inference_engine,  # ML inference engine
    require_admin          # Admin-only endpoints
)

# Database access
@router.get("/data")
async def get_data(db: Session = Depends(get_db_session)):
    return db.query(MyModel).all()

# Authentication required
@router.get("/protected")
async def protected_endpoint(current_user: dict = Depends(get_current_user)):
    return {"user_id": current_user["id"]}

# Admin only
@router.get("/admin-only")
async def admin_endpoint(admin_user: dict = Depends(require_admin)):
    return {"admin": True}

# ML inference
@router.post("/predict")
async def predict(
    data: PredictionRequest,
    engine = Depends(get_inference_engine)
):
    return engine.predict(data.features)
```

### Step 4: Use Input Validators

Use the shared validation functions from `src/api/validators.py`:

```python
from src.api.validators import (
    validate_email,
    validate_password,
    validate_file_upload
)

@router.post("/register")
async def register_user(request: UserRegistration):
    # Validate email format
    validate_email(request.email)
    
    # Validate password strength
    validate_password(request.password)
    
    # Continue with registration logic
    ...

@router.post("/upload")
async def upload_file(file: UploadFile):
    file_content = await file.read()
    
    # Validate file upload (magic bytes, size, type)
    mime_type, safe_filename = validate_file_upload(file_content, file.filename)
    
    # Continue with file processing
    ...
```

### Step 5: Handle Errors Consistently

Follow the standard error handling pattern:

```python
import logging

logger = logging.getLogger(__name__)

@router.post("/my-endpoint")
async def my_endpoint(request: MyRequest):
    try:
        # Business logic here
        result = perform_operation(request)
        return {"result": result}
        
    except HTTPException:
        # Re-raise HTTP exceptions (already have correct status code)
        raise
    except ValueError as e:
        # Client errors (400) - invalid input, validation failures
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        # Authorization errors (403) - insufficient permissions
        raise HTTPException(status_code=403, detail="Access denied")
    except FileNotFoundError as e:
        # Not found errors (404) - resource doesn't exist
        raise HTTPException(status_code=404, detail="Resource not found")
    except Exception as e:
        # Server errors (500) - unexpected errors
        logger.error(f"Unexpected error in my_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Step 6: Add OpenAPI Documentation

Enhance your endpoint with comprehensive OpenAPI documentation:

```python
@router.post(
    "/my-endpoint",
    response_model=MyResponse,
    summary="Brief endpoint description",
    description="Detailed endpoint description with usage examples",
    responses={
        200: {"description": "Success response"},
        400: {"description": "Validation error"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"}
    }
)
async def my_endpoint(request: MyRequest):
    """
    Detailed endpoint documentation.
    
    This endpoint performs X operation and returns Y result.
    
    **Usage Example:**
    ```json
    {
        "field1": "example@email.com",
        "field2": 42
    }
    ```
    
    **Business Logic:**
    1. Validate input parameters
    2. Perform database operations
    3. Return formatted response
    """
    pass
```

### Step 7: Add Request/Response Examples

Include examples in your Pydantic models:

```python
class MyRequest(BaseModel):
    email: str
    age: int
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "age": 25
            }
        }

class MyResponse(BaseModel):
    user_id: int
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "message": "User created successfully"
            }
        }
```

## Testing Your Endpoint

### Step 1: Write Unit Tests

Create unit tests in the appropriate test file:

```python
# tests/api/test_my_router.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app

client = TestClient(app)

class TestMyEndpoint:
    @patch('src.api.routers.my_router.get_current_user')
    @patch('src.api.routers.my_router.get_db_session')
    def test_my_endpoint_success(self, mock_db, mock_user):
        # Mock dependencies
        mock_user.return_value = {"id": 1, "email": "test@example.com"}
        mock_db.return_value = Mock()
        
        # Test request
        response = client.post(
            "/api/v1/domain/my-endpoint",
            json={"field1": "test@example.com", "field2": 42},
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_my_endpoint_validation_error(self):
        # Test invalid input
        response = client.post(
            "/api/v1/domain/my-endpoint",
            json={"field1": "invalid-email", "field2": "not-a-number"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_my_endpoint_authentication_required(self):
        # Test without authentication
        response = client.post(
            "/api/v1/domain/my-endpoint",
            json={"field1": "test@example.com", "field2": 42}
        )
        
        assert response.status_code == 401  # Unauthorized
```

### Step 2: Write Integration Tests

Add integration tests for end-to-end workflows:

```python
# tests/api/test_integration_my_workflow.py
def test_complete_workflow():
    # Step 1: Register user
    register_response = client.post("/api/v1/auth/register", json={...})
    assert register_response.status_code == 200
    
    # Step 2: Login
    login_response = client.post("/api/v1/auth/login", json={...})
    token = login_response.json()["access_token"]
    
    # Step 3: Use your endpoint
    response = client.post(
        "/api/v1/domain/my-endpoint",
        json={...},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

### Step 3: Test Manually

Use the automatic OpenAPI documentation to test your endpoint:

1. Start the server: `python -m src.api.main`
2. Open http://localhost:8000/docs
3. Find your endpoint in the documentation
4. Click "Try it out" and test with sample data

## Security Considerations

### Authentication and Authorization

```python
# Public endpoint (no authentication required)
@router.get("/public")
async def public_endpoint():
    return {"message": "Public data"}

# Protected endpoint (authentication required)
@router.get("/protected")
async def protected_endpoint(current_user: dict = Depends(get_current_user)):
    return {"user_id": current_user["id"]}

# Admin-only endpoint
@router.get("/admin")
async def admin_endpoint(admin_user: dict = Depends(require_admin)):
    return {"admin": True}
```

### Input Validation

```python
# Always validate user input
@router.post("/create-user")
async def create_user(request: UserRequest):
    # Validate email
    validate_email(request.email)
    
    # Validate password
    validate_password(request.password)
    
    # Validate file uploads
    if request.avatar:
        validate_file_upload(request.avatar, request.avatar_filename)
```

### IDOR Protection

```python
# Ensure users can only access their own resources
@router.get("/cases/{case_id}")
async def get_case(
    case_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    # Query with user_id filter to prevent IDOR
    case = db.query(Case).filter(
        Case.id == case_id,
        Case.user_id == current_user["id"]  # IDOR protection
    ).first()
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    return case
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Apply rate limiting to sensitive endpoints
@router.post("/login")
@limiter.limit("5/minute")  # 5 requests per minute
async def login(request: Request, credentials: LoginRequest):
    # Login logic here
    pass
```

## Performance Considerations

### Database Queries

```python
# Use efficient database queries
@router.get("/users")
async def list_users(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db_session)
):
    # Use pagination to avoid large result sets
    users = db.query(User).offset(offset).limit(limit).all()
    return users

# Use joins instead of N+1 queries
@router.get("/cases-with-users")
async def get_cases_with_users(db: Session = Depends(get_db_session)):
    # Efficient join query
    cases = db.query(Case).join(User).all()
    return cases
```

### Async Operations

```python
# Use async for I/O operations
@router.post("/process-image")
async def process_image(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    # Read file asynchronously
    content = await file.read()
    
    # Queue background processing
    background_tasks.add_task(process_image_async, content)
    
    return {"message": "Processing started"}
```

## Deployment Considerations

### Environment Configuration

```python
# Use environment variables for configuration
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
```

### Health Checks

```python
# Add health checks for your endpoints
@router.get("/health")
async def health_check(db: Session = Depends(get_db_session)):
    try:
        # Test database connection
        db.execute("SELECT 1")
        
        # Test external dependencies
        # ... other health checks
        
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service unavailable")
```

## Best Practices Summary

1. **Follow the Pattern**: Use the established endpoint pattern consistently
2. **Use Shared Dependencies**: Leverage `dependencies.py` for common functionality
3. **Validate Input**: Always use `validators.py` for input validation
4. **Handle Errors**: Follow the standard error handling pattern
5. **Document Thoroughly**: Add comprehensive OpenAPI documentation
6. **Test Comprehensively**: Write unit and integration tests
7. **Secure by Default**: Apply authentication, authorization, and input validation
8. **Optimize Performance**: Use efficient database queries and async operations
9. **Monitor Health**: Add health checks and monitoring
10. **Configure Properly**: Use environment variables for configuration

## Common Patterns

### Pagination

```python
@router.get("/items")
async def list_items(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db_session)
):
    offset = (page - 1) * size
    items = db.query(Item).offset(offset).limit(size).all()
    total = db.query(Item).count()
    
    return {
        "items": items,
        "page": page,
        "size": size,
        "total": total,
        "pages": (total + size - 1) // size
    }
```

### File Upload

```python
@router.post("/upload")
async def upload_file(
    file: UploadFile,
    current_user: dict = Depends(get_current_user)
):
    # Read and validate file
    content = await file.read()
    mime_type, safe_filename = validate_file_upload(content, file.filename)
    
    # Save file
    file_path = save_file(content, safe_filename, current_user["id"])
    
    return {"filename": safe_filename, "path": file_path}
```

### Background Tasks

```python
@router.post("/process")
async def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    # Start background processing
    background_tasks.add_task(
        process_data_async,
        request.data,
        current_user["id"]
    )
    
    return {"message": "Processing started", "status": "queued"}
```

This guide provides a comprehensive framework for adding new endpoints while maintaining consistency, security, and performance across the API.