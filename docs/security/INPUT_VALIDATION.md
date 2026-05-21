# Input Validation Security Guide

## Overview

Input validation is the first line of defense against injection attacks, data corruption, and application crashes. This guide covers secure input validation patterns for the platform.

## Validation Principles

### 1. Whitelist Over Blacklist

**❌ Bad: Blacklist approach**
```python
def validate_username(username: str) -> bool:
    # Trying to block bad characters
    forbidden = ['<', '>', '&', '"', "'", ';', '--']
    return not any(char in username for char in forbidden)
```

**✅ Good: Whitelist approach**
```python
import re

def validate_username(username: str) -> bool:
    # Only allow alphanumeric and underscore
    return bool(re.match(r'^[a-zA-Z0-9_]{3,32}$', username))
```

### 2. Validate Early, Validate Often

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    age: int = Field(..., ge=0, le=150)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum() and '_' not in v:
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('email')
    def email_lowercase(cls, v):
        return v.lower()

app = FastAPI()

@app.post("/users")
async def create_user(user: UserCreate):
    # Input already validated by Pydantic
    return {"username": user.username, "email": user.email}
```

### 3. Fail Securely

```python
from src.security.validation import InputValidator

def process_file(filename: str):
    try:
        # Validate filename
        safe_filename = InputValidator.validate_image_file(filename)
    except ValueError as e:
        # Log the error
        logger.warning(f"Invalid filename rejected: {filename}")
        # Return generic error to user
        raise HTTPException(
            status_code=400,
            detail="Invalid file format"  # Don't leak validation details
        )
    
    # Process file
    return process_image(safe_filename)
```

## the platform Validation Utilities

### InputValidator Class

Located in `src/security/validation.py`:

```python
from src.security.validation import InputValidator
from pathlib import Path

# Path validation
safe_path = InputValidator.validate_path(
    path="/data/images/slide.svs",
    allowed_dirs=[Path("/data/images")],
    must_exist=True
)

# File extension validation
safe_file = InputValidator.validate_file_extension(
    filename="model.pth",
    allowed_extensions={'.pth', '.pt', '.ckpt'}
)

# String validation
safe_string = InputValidator.validate_string(
    value=user_input,
    min_length=1,
    max_length=100,
    pattern=r'^[a-zA-Z0-9\s]+$'
)

# Integer validation
safe_int = InputValidator.validate_integer(
    value=batch_size,
    min_value=1,
    max_value=256
)

# Float validation
safe_float = InputValidator.validate_float(
    value=threshold,
    min_value=0.0,
    max_value=1.0
)

# Batch size validation
batch_size = InputValidator.validate_batch_size(user_batch_size)

# Confidence threshold validation
threshold = InputValidator.validate_confidence_threshold(user_threshold)

# Filename sanitization
safe_filename = InputValidator.sanitize_filename(user_filename)
```

## Common Validation Patterns

### 1. File Upload Validation

```python
from fastapi import UploadFile, HTTPException
from pathlib import Path
import magic  # python-magic for MIME type detection

ALLOWED_EXTENSIONS = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs'}
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB

async def validate_wsi_upload(file: UploadFile) -> Path:
    """Validate whole slide image upload."""
    
    # 1. Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # 2. Sanitize filename
    safe_filename = InputValidator.sanitize_filename(file.filename)
    
    # 3. Check extension
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"
        )
    
    # 4. Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE} bytes"
        )
    
    # 5. Verify MIME type (optional but recommended)
    file_content = await file.read(2048)  # Read first 2KB
    await file.seek(0)  # Reset
    
    mime = magic.from_buffer(file_content, mime=True)
    if mime not in ['image/tiff', 'application/octet-stream']:
        raise HTTPException(
            status_code=400,
            detail="Invalid file content"
        )
    
    return Path(safe_filename)
```

### 2. SQL Injection Prevention

**❌ Bad: String concatenation**
```python
# NEVER DO THIS
def get_user(username: str):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)
```

**✅ Good: Parameterized queries**
```python
from sqlalchemy import text

def get_user(username: str):
    # Validate input first
    safe_username = InputValidator.validate_string(
        username,
        min_length=3,
        max_length=32,
        pattern=r'^[a-zA-Z0-9_]+$'
    )
    
    # Use parameterized query
    query = text("SELECT * FROM users WHERE username = :username")
    return db.execute(query, {"username": safe_username})
```

### 3. Path Traversal Prevention

**❌ Bad: Direct path construction**
```python
# NEVER DO THIS
def read_file(filename: str):
    path = f"/data/files/{filename}"
    with open(path, 'r') as f:
        return f.read()
# Risk: filename="../../../etc/passwd"
```

**✅ Good: Path validation**
```python
from pathlib import Path

def read_file(filename: str):
    # Sanitize filename
    safe_filename = InputValidator.sanitize_filename(filename)
    
    # Validate path
    base_dir = Path("/data/files")
    file_path = base_dir / safe_filename
    
    # Ensure path is within base directory
    try:
        file_path = file_path.resolve()
        file_path.relative_to(base_dir.resolve())
    except ValueError:
        raise ValueError("Invalid file path")
    
    # Validate with InputValidator
    validated_path = InputValidator.validate_path(
        path=file_path,
        allowed_dirs=[base_dir],
        must_exist=True
    )
    
    with open(validated_path, 'r') as f:
        return f.read()
```

### 4. Command Injection Prevention

**❌ Bad: Shell=True with user input**
```python
# NEVER DO THIS
import subprocess

def convert_image(filename: str):
    cmd = f"convert {filename} output.png"
    subprocess.run(cmd, shell=True)
# Risk: filename="input.jpg; rm -rf /"
```

**✅ Good: Argument list + validation**
```python
import subprocess
from src.utils.subprocess_safe import run_command_safe

def convert_image(filename: str):
    # Validate filename
    safe_filename = InputValidator.validate_image_file(filename)
    
    # Use argument list (no shell)
    result = run_command_safe(
        ['convert', str(safe_filename), 'output.png'],
        timeout=30
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")
    
    return result.stdout
```

### 5. XSS Prevention

**❌ Bad: Unescaped user input in templates**
```jinja2
<!-- NEVER DO THIS -->
<div>Welcome, {{ username }}!</div>
<!-- Risk: username="<script>alert('XSS')</script>" -->
```

**✅ Good: Auto-escaping + validation**
```python
from jinja2 import Environment
from src.security.jinja2_security import SecureJinja2Environment

# Create secure environment with auto-escape
env = SecureJinja2Environment.create_environment()

# Validate input
safe_username = InputValidator.validate_string(
    username,
    min_length=3,
    max_length=32,
    pattern=r'^[a-zA-Z0-9_]+$'
)

# Render template (auto-escaped)
template = env.from_string("<div>Welcome, {{ username }}!</div>")
html = template.render(username=safe_username)
```

## API Input Validation

### FastAPI with Pydantic

```python
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class ModelType(str, Enum):
    RESNET = "resnet"
    VIT = "vit"
    DINO = "dino"

class InferenceRequest(BaseModel):
    image_path: str = Field(..., min_length=1, max_length=500)
    model_type: ModelType
    batch_size: int = Field(default=32, ge=1, le=256)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @validator('image_path')
    def validate_image_path(cls, v):
        # Additional validation beyond Pydantic
        return InputValidator.validate_path(
            v,
            allowed_dirs=[Path("/data/images")],
            must_exist=True
        )

app = FastAPI()

@app.post("/inference")
async def run_inference(
    request: InferenceRequest,
    user_id: int = Query(..., ge=1),
    api_key: str = Query(..., min_length=32, max_length=64)
):
    # All inputs validated by Pydantic
    return {
        "status": "success",
        "model": request.model_type,
        "batch_size": request.batch_size
    }
```

## Testing

```python
import pytest
from src.security.validation import InputValidator

def test_path_traversal_blocked():
    """Test that path traversal attempts are blocked."""
    with pytest.raises(ValueError):
        InputValidator.validate_path(
            "../../../etc/passwd",
            allowed_dirs=[Path("/data")]
        )

def test_sql_injection_blocked():
    """Test that SQL injection attempts are blocked."""
    with pytest.raises(ValueError):
        InputValidator.validate_string(
            "admin' OR '1'='1",
            pattern=r'^[a-zA-Z0-9_]+$'
        )

def test_xss_blocked():
    """Test that XSS attempts are blocked."""
    with pytest.raises(ValueError):
        InputValidator.validate_string(
            "<script>alert('XSS')</script>",
            pattern=r'^[a-zA-Z0-9\s]+$'
        )

def test_command_injection_blocked():
    """Test that command injection attempts are blocked."""
    with pytest.raises(ValueError):
        InputValidator.sanitize_filename(
            "file.txt; rm -rf /"
        )
```

## References

- [OWASP Input Validation Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)
- [CWE-20: Improper Input Validation](https://cwe.mitre.org/data/definitions/20.html)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)
- [CWE-79: Cross-site Scripting (XSS)](https://cwe.mitre.org/data/definitions/79.html)
- [CWE-78: OS Command Injection](https://cwe.mitre.org/data/definitions/78.html)
- [CWE-22: Path Traversal](https://cwe.mitre.org/data/definitions/22.html)
