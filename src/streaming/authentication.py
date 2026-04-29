"""
Authentication & authorization for HistoCore Real-Time WSI Streaming.

OAuth 2.0, JWT tokens, RBAC, hospital identity integration.
"""

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# JWT imports
try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not available. Install: pip install pyjwt")


# ============================================================================
# User Roles & Permissions
# ============================================================================


class Role(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    PATHOLOGIST = "pathologist"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    TECHNICIAN = "technician"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions."""

    # Slide operations
    VIEW_SLIDES = "view_slides"
    PROCESS_SLIDES = "process_slides"
    DELETE_SLIDES = "delete_slides"
    EXPORT_SLIDES = "export_slides"

    # Results operations
    VIEW_RESULTS = "view_results"
    MODIFY_RESULTS = "modify_results"
    APPROVE_RESULTS = "approve_results"

    # System operations
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"

    # Data operations
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"


# Role-Permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.VIEW_SLIDES,
        Permission.PROCESS_SLIDES,
        Permission.DELETE_SLIDES,
        Permission.EXPORT_SLIDES,
        Permission.VIEW_RESULTS,
        Permission.MODIFY_RESULTS,
        Permission.APPROVE_RESULTS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SYSTEM,
        Permission.VIEW_AUDIT_LOGS,
        Permission.EXPORT_DATA,
        Permission.DELETE_DATA,
    },
    Role.PATHOLOGIST: {
        Permission.VIEW_SLIDES,
        Permission.PROCESS_SLIDES,
        Permission.EXPORT_SLIDES,
        Permission.VIEW_RESULTS,
        Permission.MODIFY_RESULTS,
        Permission.APPROVE_RESULTS,
        Permission.EXPORT_DATA,
    },
    Role.CLINICIAN: {
        Permission.VIEW_SLIDES,
        Permission.PROCESS_SLIDES,
        Permission.VIEW_RESULTS,
        Permission.EXPORT_DATA,
    },
    Role.RESEARCHER: {
        Permission.VIEW_SLIDES,
        Permission.PROCESS_SLIDES,
        Permission.VIEW_RESULTS,
        Permission.EXPORT_DATA,
    },
    Role.TECHNICIAN: {Permission.VIEW_SLIDES, Permission.PROCESS_SLIDES, Permission.VIEW_RESULTS},
    Role.VIEWER: {Permission.VIEW_SLIDES, Permission.VIEW_RESULTS},
}


# ============================================================================
# User Model
# ============================================================================


@dataclass
class User:
    """User model."""

    user_id: str
    username: str
    email: str
    roles: List[Role]
    organization: str
    department: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has permission."""
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        return False

    def has_role(self, role: Role) -> bool:
        """Check if user has role."""
        return role in self.roles

    def get_permissions(self) -> Set[Permission]:
        """Get all user permissions."""
        permissions = set()
        for role in self.roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))
        return permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": [r.value for r in self.roles],
            "organization": self.organization,
            "department": self.department,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


# ============================================================================
# JWT Token Manager
# ============================================================================


@dataclass
class TokenConfig:
    """JWT token configuration."""

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "histocore"
    audience: str = "histocore-api"


class JWTManager:
    """Manages JWT token creation and validation."""

    def __init__(self, config: TokenConfig):
        """Initialize JWT manager."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT required for JWT tokens")

        self.config = config
        logger.info("JWT manager initialized: algorithm=%s", config.algorithm)

    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)

        expire = datetime.utcnow() + expires_delta

        payload = {
            "sub": user.user_id,
            "username": user.username,
            "email": user.email,
            "roles": [r.value for r in user.roles],
            "organization": user.organization,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "access",
        }

        token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

        logger.debug("Created access token for user: %s", user.username)

        return token

    def create_refresh_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token."""
        if expires_delta is None:
            expires_delta = timedelta(days=self.config.refresh_token_expire_days)

        expire = datetime.utcnow() + expires_delta

        payload = {
            "sub": user.user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "refresh",
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }

        token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

        logger.debug("Created refresh token for user: %s", user.username)

        return token

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience,
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")

    def refresh_access_token(self, refresh_token: str, user: User) -> str:
        """Refresh access token using refresh token."""
        payload = self.verify_token(refresh_token)

        if payload.get("type") != "refresh":
            raise ValueError("Invalid token type")

        if payload.get("sub") != user.user_id:
            raise ValueError("Token user mismatch")

        return self.create_access_token(user)


# ============================================================================
# OAuth 2.0 Provider
# ============================================================================


@dataclass
class OAuth2Config:
    """OAuth 2.0 configuration."""

    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    redirect_uri: str
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])


class OAuth2Provider:
    """OAuth 2.0 provider integration."""

    def __init__(self, config: OAuth2Config):
        """Initialize OAuth 2.0 provider."""
        self.config = config
        logger.info("OAuth 2.0 provider initialized: client_id=%s", config.client_id)

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Get authorization URL for OAuth flow."""
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.config.authorization_endpoint}?{query_string}"

        return url

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        # This would make HTTP request to token endpoint
        # Simplified for example
        logger.info("Exchanging authorization code for token")

        # In production, use requests library:
        # response = requests.post(
        #     self.config.token_endpoint,
        #     data={
        #         'grant_type': 'authorization_code',
        #         'code': code,
        #         'redirect_uri': self.config.redirect_uri,
        #         'client_id': self.config.client_id,
        #         'client_secret': self.config.client_secret
        #     }
        # )
        # return response.json()

        return {
            "access_token": "mock_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "mock_refresh_token",
        }

    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from OAuth provider."""
        # This would make HTTP request to userinfo endpoint
        logger.info("Fetching user info from OAuth provider")

        # In production:
        # response = requests.get(
        #     self.config.userinfo_endpoint,
        #     headers={'Authorization': f'Bearer {access_token}'}
        # )
        # return response.json()

        return {
            "sub": "user123",
            "email": "user@hospital.org",
            "name": "Dr. Smith",
            "organization": "General Hospital",
        }


# ============================================================================
# Session Manager
# ============================================================================


@dataclass
class Session:
    """User session."""

    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at

    def is_idle(self, idle_timeout_minutes: int = 30) -> bool:
        """Check if session is idle."""
        idle_time = datetime.utcnow() - self.last_activity
        return idle_time > timedelta(minutes=idle_timeout_minutes)

    def refresh(self):
        """Refresh session activity."""
        self.last_activity = datetime.utcnow()


class SessionManager:
    """Manages user sessions."""

    def __init__(self, session_timeout_minutes: int = 30):
        """Initialize session manager."""
        self.session_timeout_minutes = session_timeout_minutes
        self.sessions: Dict[str, Session] = {}
        logger.info("Session manager initialized: timeout=%d min", session_timeout_minutes)

    def create_session(
        self, user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None
    ) -> Session:
        """Create new session."""
        session_id = secrets.token_urlsafe(32)
        now = datetime.utcnow()

        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(minutes=self.session_timeout_minutes),
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.sessions[session_id] = session

        logger.info("Created session: %s for user: %s", session_id, user_id)

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self.sessions.get(session_id)

        if session and session.is_expired():
            self.delete_session(session_id)
            return None

        return session

    def refresh_session(self, session_id: str) -> bool:
        """Refresh session activity."""
        session = self.get_session(session_id)

        if session:
            session.refresh()
            return True

        return False

    def delete_session(self, session_id: str):
        """Delete session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("Deleted session: %s", session_id)

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        expired = [sid for sid, session in self.sessions.items() if session.is_expired()]

        for session_id in expired:
            self.delete_session(session_id)

        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))


# ============================================================================
# RBAC Manager
# ============================================================================


class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(self):
        """Initialize RBAC manager."""
        self.role_permissions = ROLE_PERMISSIONS
        logger.info("RBAC manager initialized")

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission."""
        return user.has_permission(permission)

    def require_permission(self, user: User, permission: Permission):
        """Require permission or raise exception."""
        if not self.check_permission(user, permission):
            raise PermissionError(f"User {user.username} lacks permission: {permission.value}")

    def require_role(self, user: User, role: Role):
        """Require role or raise exception."""
        if not user.has_role(role):
            raise PermissionError(f"User {user.username} lacks role: {role.value}")

    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all user permissions."""
        return user.get_permissions()

    def add_role_to_user(self, user: User, role: Role):
        """Add role to user."""
        if role not in user.roles:
            user.roles.append(role)
            logger.info("Added role %s to user %s", role.value, user.username)

    def remove_role_from_user(self, user: User, role: Role):
        """Remove role from user."""
        if role in user.roles:
            user.roles.remove(role)
            logger.info("Removed role %s from user %s", role.value, user.username)


# ============================================================================
# Authentication Manager (Main Interface)
# ============================================================================


class AuthenticationManager:
    """Main authentication manager."""

    def __init__(
        self,
        jwt_config: TokenConfig,
        oauth_config: Optional[OAuth2Config] = None,
        session_timeout_minutes: int = 30,
    ):
        """Initialize authentication manager."""
        self.jwt_manager = JWTManager(jwt_config)
        self.oauth_provider = OAuth2Provider(oauth_config) if oauth_config else None
        self.session_manager = SessionManager(session_timeout_minutes)
        self.rbac_manager = RBACManager()

        # User storage (in production, use database)
        self.users: Dict[str, User] = {}

        logger.info("Authentication manager initialized")

    def register_user(
        self, username: str, email: str, roles: List[Role], organization: str, **kwargs
    ) -> User:
        """Register new user."""
        user_id = hashlib.sha256(f"{username}{email}".encode()).hexdigest()[:16]

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            organization=organization,
            **kwargs,
        )

        self.users[user_id] = user

        logger.info("Registered user: %s (roles=%s)", username, [r.value for r in roles])

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def authenticate_user(
        self, username: str, password: str, ip_address: Optional[str] = None
    ) -> Tuple[User, str, str]:
        """Authenticate user and return tokens."""
        # In production, verify password hash
        user = next((u for u in self.users.values() if u.username == username), None)

        if not user or not user.is_active:
            raise ValueError("Invalid credentials")

        # Update last login
        user.last_login = datetime.utcnow()

        # Create tokens
        access_token = self.jwt_manager.create_access_token(user)
        refresh_token = self.jwt_manager.create_refresh_token(user)

        # Create session
        session = self.session_manager.create_session(user.user_id, ip_address)

        logger.info("Authenticated user: %s", username)

        return user, access_token, refresh_token

    def verify_token(self, token: str) -> User:
        """Verify token and return user."""
        payload = self.jwt_manager.verify_token(token)

        user_id = payload.get("sub")
        user = self.get_user(user_id)

        if not user or not user.is_active:
            raise ValueError("User not found or inactive")

        return user

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check user permission."""
        return self.rbac_manager.check_permission(user, permission)

    def require_permission(self, user: User, permission: Permission):
        """Require permission or raise exception."""
        self.rbac_manager.require_permission(user, permission)


# ============================================================================
# Decorators
# ============================================================================


def require_auth(auth_manager: AuthenticationManager):
    """Decorator to require authentication."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            token = kwargs.get("token")
            if not token:
                raise ValueError("Authentication required")

            user = auth_manager.verify_token(token)
            kwargs["user"] = user

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_permission_decorator(auth_manager: AuthenticationManager, permission: Permission):
    """Decorator to require specific permission."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user:
                raise ValueError("User not found in context")

            auth_manager.require_permission(user, permission)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Convenience Functions
# ============================================================================


def create_auth_manager(
    secret_key: str, session_timeout_minutes: int = 30
) -> AuthenticationManager:
    """Create authentication manager with default config."""
    jwt_config = TokenConfig(secret_key=secret_key)

    return AuthenticationManager(
        jwt_config=jwt_config, session_timeout_minutes=session_timeout_minutes
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create auth manager
    auth = create_auth_manager(secret_key="your-secret-key-here")

    # Register users
    admin = auth.register_user(
        username="admin",
        email="admin@hospital.org",
        roles=[Role.ADMIN],
        organization="General Hospital",
    )

    pathologist = auth.register_user(
        username="dr_smith",
        email="smith@hospital.org",
        roles=[Role.PATHOLOGIST],
        organization="General Hospital",
        department="Pathology",
    )

    # Authenticate
    user, access_token, refresh_token = auth.authenticate_user("admin", "password")
    print(f"Access token: {access_token[:50]}...")

    # Verify token
    verified_user = auth.verify_token(access_token)
    print(f"Verified user: {verified_user.username}")

    # Check permissions
    can_manage = auth.check_permission(verified_user, Permission.MANAGE_SYSTEM)
    print(f"Can manage system: {can_manage}")
