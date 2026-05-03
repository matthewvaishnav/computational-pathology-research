"""
API Dependencies

Shared dependency functions for FastAPI endpoints.
"""

import logging
import uuid
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.database import UserOperations, get_db_session
from src.inference import InferenceEngine
from src.api.security import decode_access_token, log_security_event

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global inference engine
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get global inference engine instance."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
        try:
            _inference_engine.warm_up_model("breast_cancer")
        except Exception as e:
            logger.warning(f"Failed to warm up model: {e}")
    return _inference_engine


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session),
):
    """Get current authenticated user from JWT token."""
    if not credentials:
        log_security_event("authentication_failed", details="No credentials provided", success=False)
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        if not user_id:
            log_security_event("authentication_failed", details="Invalid token payload", success=False)
            raise HTTPException(status_code=401, detail="Invalid token")

        user_ops = UserOperations(db)
        user = user_ops.get_user_by_id(uuid.UUID(user_id))

        if not user:
            log_security_event("authentication_failed", username=user_id, details="User not found", success=False)
            raise HTTPException(status_code=401, detail="User not found")

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        log_security_event("authentication_error", details=str(e), success=False)
        raise HTTPException(status_code=401, detail="Authentication failed")
