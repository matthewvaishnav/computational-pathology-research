"""
Authentication Router

Handles user registration, login, and OAuth flows.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database import UserOperations, get_db_session
from src.api.dependencies import get_current_user
from src.api.security import (
    check_account_lockout,
    clear_failed_login,
    create_access_token,
    hash_password,
    limiter,
    log_security_event,
    record_failed_login,
    verify_password,
)
from src.api.validators import validate_email, validate_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

# Temporary in-memory user storage (will be replaced with database)
users_db: Dict[str, Dict] = {}


class UserRegistration(BaseModel):
    """User registration request model.

    Validates new user registration data with security controls.
    Role assignment is server-side only to prevent privilege escalation.
    """

    username: str
    email: str
    password: str

    class Config:
        # Prevent mass assignment of sensitive fields
        # role is set server-side, not from user input
        extra = "forbid"


class UserLogin(BaseModel):
    """User login request model.

    Validates user authentication credentials.
    """

    username: str
    password: str


@router.post("/register")
async def register_user(user_data: UserRegistration, request: Request):
    """Register new user with secure password hashing."""
    try:
        # Sanitize inputs
        username = user_data.username.strip().lower()
        email = user_data.email.strip().lower()

        # Validate email and password using centralized validators
        validate_email(email)
        validate_password(user_data.password)

        # Validate username format (alphanumeric, underscore, hyphen only)
        import re

        if not re.match(r"^[a-z0-9_-]{3,32}$", username):
            raise HTTPException(
                status_code=400,
                detail="Username must be 3-32 characters (lowercase letters, numbers, underscore, hyphen only)",
            )

        if username in users_db:
            log_security_event(
                "registration_failed",
                username=username,
                ip_address=request.client.host,
                details="User already exists",
                success=False,
            )
            raise HTTPException(status_code=409, detail="User already exists")

        hashed_password = hash_password(user_data.password)

        user_id = str(uuid.uuid4())
        # Role is always set to 'pathologist' for new registrations
        # Admin roles must be assigned by existing admins
        users_db[username] = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "password_hash": hashed_password,
            "role": "pathologist",  # Default role, cannot be overridden by user
            "created_at": datetime.now().isoformat(),
        }

        log_security_event(
            "user_registered",
            username=user_data.username,
            ip_address=request.client.host,
            details="Role: pathologist (default)",
            success=True,
        )

        return {"message": "User registered successfully", "user_id": user_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login")
@limiter.limit("5/minute")
async def login_user(login_data: UserLogin, request: Request):
    """User login with rate limiting and brute force protection."""
    username = login_data.username
    ip_address = request.client.host

    try:
        check_account_lockout(username)

        start_time = time.time()

        user_exists = username in users_db
        if user_exists:
            user = users_db[username]
            password_valid = verify_password(login_data.password, user["password_hash"])
        else:
            verify_password(login_data.password, hash_password("dummy_password_for_timing"))
            password_valid = False

        elapsed = time.time() - start_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)

        if not user_exists or not password_valid:
            record_failed_login(username)
            log_security_event(
                "login_failed",
                username=username,
                ip_address=ip_address,
                details="Invalid credentials",
                success=False,
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")

        clear_failed_login(username)

        access_token = create_access_token(
            data={"sub": user["user_id"], "username": username, "role": user["role"]}
        )

        log_security_event("login_success", username=username, ip_address=ip_address, success=True)

        return {
            "access_token": access_token,
            "token_type": "bearer",
        }  # nosec B105 - OAuth2 token type, not password  # nosec B105 - OAuth2 token type, not password

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        log_security_event(
            "login_error", username=username, ip_address=ip_address, details=str(e), success=False
        )
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me")
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
    }


@router.get("/oauth/login")
async def oauth_login(provider: str = "azure"):
    """Initiate OAuth 2.0 login flow."""
    try:
        from src.api.oauth import create_oauth_client

        oauth_client = create_oauth_client(provider=provider)
        auth_url, state = oauth_client.get_authorization_url()

        log_security_event("oauth_login_initiated", details=f"Provider: {provider}", success=True)

        return {"authorization_url": auth_url, "state": state, "provider": provider}

    except Exception as e:
        logger.error(f"OAuth login failed: {e}")
        log_security_event(
            "oauth_login_failed",
            details=f"Provider: {provider}, Error: {str(e)}",
            success=False,
        )
        raise HTTPException(status_code=500, detail="Failed to initiate OAuth login")


@router.get("/oauth/callback")
async def oauth_callback(request: Request, provider: str = "azure"):
    """Handle OAuth 2.0 callback."""
    try:
        from src.api.oauth import create_oauth_client, oauth_callback_handler

        oauth_client = create_oauth_client(provider=provider)
        result = await oauth_callback_handler(request, oauth_client)

        userinfo = result["userinfo"]
        access_token = result["access_token"]

        db = next(get_db_session())
        user_ops = UserOperations(db)

        email = userinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by OAuth provider")

        user = user_ops.get_user_by_email(email)

        if not user:
            username = userinfo.get("preferred_username") or email.split("@")[0]
            user = user_ops.create_user(
                username=username,
                email=email,
                password_hash="",  # nosec B106 - OAuth users have no password
                role="pathologist",
            )
            db.commit()
            logger.info(f"Created new OAuth user: {email}")

        jwt_token = create_access_token(
            data={
                "sub": str(user.id),
                "username": user.username,
                "role": user.role,
                "oauth_provider": provider,
            }
        )

        log_security_event(
            "oauth_login_success",
            username=user.username,
            ip_address=request.client.host,
            details=f"Provider: {provider}",
            success=True,
        )

        return {
            "access_token": jwt_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role,
            },
            "oauth_provider": provider,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        log_security_event(
            "oauth_callback_failed",
            ip_address=request.client.host,
            details=f"Provider: {provider}, Error: {str(e)}",
            success=False,
        )
        raise HTTPException(status_code=500, detail="OAuth authentication failed")
