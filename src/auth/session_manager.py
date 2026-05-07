"""
Secure Session Management

Provides secure session handling with proper expiration and validation.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class SecureSession:
    """Secure session with automatic expiration."""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        data: Dict[str, Any],
        timeout_minutes: int = 30,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.data = data
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.timeout_minutes = timeout_minutes
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        expiry = self.last_accessed + timedelta(minutes=self.timeout_minutes)
        return datetime.utcnow() > expiry
    
    def refresh(self):
        """Refresh session activity."""
        self.last_accessed = datetime.utcnow()
    
    def invalidate(self):
        """Invalidate session."""
        self.data.clear()


class SessionManager:
    """Manage secure sessions."""
    
    def __init__(self, timeout_minutes: int = 30):
        self.sessions: Dict[str, SecureSession] = {}
        self.timeout_minutes = timeout_minutes
    
    def create_session(self, user_id: str, data: Optional[Dict] = None) -> str:
        """Create new session.
        
        Args:
            user_id: User identifier
            data: Session data
            
        Returns:
            Session ID
        """
        # Generate cryptographically secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Create session
        session = SecureSession(
            session_id=session_id,
            user_id=user_id,
            data=data or {},
            timeout_minutes=self.timeout_minutes,
        )
        
        self.sessions[session_id] = session
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SecureSession]:
        """Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if valid, None otherwise
        """
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        # Check expiration
        if session.is_expired():
            self.invalidate_session(session_id)
            return None
        
        # Refresh activity
        session.refresh()
        
        return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate session.
        
        Args:
            session_id: Session to invalidate
        """
        if session_id in self.sessions:
            self.sessions[session_id].invalidate()
            del self.sessions[session_id]
    
    def cleanup_expired(self):
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired()
        ]
        
        for sid in expired:
            self.invalidate_session(sid)
