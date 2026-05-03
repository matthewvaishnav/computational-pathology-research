"""
API Input Validators

Validation functions for API inputs.
"""

import re
from pathlib import Path

from fastapi import HTTPException, UploadFile


def validate_email(email: str) -> None:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise HTTPException(status_code=400, detail="Invalid email format")


def validate_password(password: str) -> None:
    """Validate password strength."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    if not any(c.isupper() for c in password):
        raise HTTPException(status_code=400, detail="Password must contain uppercase letter")
    
    if not any(c.isdigit() for c in password):
        raise HTTPException(status_code=400, detail="Password must contain digit")


def validate_file_upload(file: UploadFile, max_size_mb: int = 100) -> None:
    """Validate uploaded file type and size."""
    allowed_extensions = {".svs", ".tif", ".tiff", ".ndpi", ".dcm", ".png", ".jpg", ".jpeg"}
    
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not allowed. Allowed: {', '.join(allowed_extensions)}"
        )
    
    if file.size and file.size > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds {max_size_mb}MB limit"
        )


def validate_username(username: str) -> None:
    """Validate username format."""
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    
    if len(username) > 50:
        raise HTTPException(status_code=400, detail="Username must be at most 50 characters")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise HTTPException(
            status_code=400,
            detail="Username can only contain letters, numbers, hyphens, and underscores"
        )


def validate_priority(priority: str) -> None:
    """Validate priority value."""
    allowed_priorities = {"low", "normal", "high", "urgent"}
    if priority not in allowed_priorities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority. Allowed: {', '.join(allowed_priorities)}"
        )


def validate_case_type(case_type: str) -> None:
    """Validate case type value."""
    allowed_types = {
        "breast_cancer_screening",
        "lymph_node_metastasis",
        "prostate_cancer",
        "lung_cancer",
        "colon_cancer"
    }
    if case_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid case type. Allowed: {', '.join(allowed_types)}"
        )
