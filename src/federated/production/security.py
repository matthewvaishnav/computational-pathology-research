"""Production security and authentication."""

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import jwt
from passlib.context import CryptContext
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .config import get_config
from .database import get_db_manager, AuditLog

logger = logging.getLogger(__name__)
config = get_config()
db_manager = get_db_manager()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Production security manager."""
    
    def __init__(self):
        self.secret_key = config.security.secret_key
        self.jwt_algorithm = config.security.jwt_algorithm
        self.jwt_expiration_hours = config.security.jwt_expiration_hours
    
    # Password management
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_policy(self, password: str) -> List[str]:
        """Validate password against policy."""
        errors = []
        
        if len(password) < config.security.min_password_length:
            errors.append(f"Password must be at least {config.security.min_password_length} characters")
        
        if config.security.require_special_chars:
            if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                errors.append("Password must contain at least one special character")
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        return errors
    
    # JWT token management
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.jwt_algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    # API key management
    def generate_api_key(self) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash."""
        return hmac.compare_digest(self.hash_api_key(api_key), hashed_key)
    
    # Certificate management
    def load_certificate(self, cert_path: str) -> x509.Certificate:
        """Load X.509 certificate."""
        with open(cert_path, "rb") as f:
            cert_data = f.read()
        return x509.load_pem_x509_certificate(cert_data)
    
    def verify_certificate_chain(self, cert_path: str, ca_cert_path: str) -> bool:
        """Verify certificate against CA."""
        try:
            cert = self.load_certificate(cert_path)
            ca_cert = self.load_certificate(ca_cert_path)
            
            # Check if certificate is signed by CA
            ca_public_key = ca_cert.public_key()
            ca_public_key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                cert.signature_algorithm_oid._name
            )
            
            # Check validity period
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Certificate verification failed: {e}")
            return False
    
    def get_certificate_info(self, cert_path: str) -> Dict[str, Any]:
        """Get certificate information."""
        cert = self.load_certificate(cert_path)
        
        return {
            "subject": cert.subject.rfc4514_string(),
            "issuer": cert.issuer.rfc4514_string(),
            "serial_number": str(cert.serial_number),
            "not_valid_before": cert.not_valid_before,
            "not_valid_after": cert.not_valid_after,
            "signature_algorithm": cert.signature_algorithm_oid._name,
        }


class AuditLogger:
    """HIPAA-compliant audit logger."""
    
    def __init__(self):
        self.enabled = config.security.audit_log_enabled
        self.log_path = config.security.audit_log_path
    
    def log_event(self, event_type: str, event_category: str, description: str,
                  user_id: str = None, client_id: str = None, resource_type: str = None,
                  resource_id: str = None, ip_address: str = None, user_agent: str = None,
                  success: bool = True, error_message: str = None, additional_data: Dict[str, Any] = None):
        """Log audit event."""
        if not self.enabled:
            return
        
        try:
            # Log to database
            db_manager.log_audit_event(
                event_type=event_type,
                event_category=event_category,
                description=description,
                user_id=user_id,
                client_id=client_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource_type=resource_type,
                resource_id=resource_id,
                success=success,
                error_message=error_message,
                additional_data=additional_data
            )
            
            # Also log to file for redundancy
            self._log_to_file(
                event_type, event_category, description, user_id, client_id,
                success, error_message
            )
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    def _log_to_file(self, event_type: str, event_category: str, description: str,
                     user_id: str, client_id: str, success: bool, error_message: str):
        """Log audit event to file."""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "event_category": event_category,
                "description": description,
                "user_id": user_id,
                "client_id": client_id,
                "success": success,
                "error_message": error_message
            }
            
            # Ensure log directory exists
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_path, "a") as f:
                f.write(f"{log_entry}\n")
                
        except Exception as e:
            logger.error(f"Failed to write audit log to file: {e}")
    
    # Predefined audit events
    def log_client_registration(self, client_id: str, user_id: str = None, ip_address: str = None):
        """Log client registration."""
        self.log_event(
            event_type="client_registration",
            event_category="access",
            description=f"Client {client_id} registered",
            user_id=user_id,
            client_id=client_id,
            ip_address=ip_address,
            resource_type="client",
            resource_id=client_id
        )
    
    def log_training_round_start(self, round_id: int, participants: List[str], user_id: str = None):
        """Log training round start."""
        self.log_event(
            event_type="training_round_start",
            event_category="system",
            description=f"Training round {round_id} started with {len(participants)} participants",
            user_id=user_id,
            resource_type="training_round",
            resource_id=str(round_id),
            additional_data={"participants": participants}
        )
    
    def log_model_update_submission(self, client_id: str, round_id: int, dataset_size: int):
        """Log model update submission."""
        self.log_event(
            event_type="model_update_submission",
            event_category="modification",
            description=f"Client {client_id} submitted model update for round {round_id}",
            client_id=client_id,
            resource_type="model_update",
            resource_id=f"{round_id}_{client_id}",
            additional_data={"dataset_size": dataset_size}
        )
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str = None, error_message: str = None):
        """Log authentication attempt."""
        self.log_event(
            event_type="authentication",
            event_category="access",
            description=f"Authentication attempt for user {user_id}",
            user_id=user_id,
            ip_address=ip_address,
            success=success,
            error_message=error_message
        )
    
    def log_data_access(self, user_id: str, resource_type: str, resource_id: str, action: str, ip_address: str = None):
        """Log data access."""
        self.log_event(
            event_type="data_access",
            event_category="access",
            description=f"User {user_id} {action} {resource_type} {resource_id}",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            additional_data={"action": action}
        )


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check if request is allowed under rate limit."""
        try:
            current = self.redis.get(key)
            if current is None:
                # First request in window
                self.redis.setex(key, window_seconds, 1)
                return True
            
            current_count = int(current)
            if current_count >= limit:
                return False
            
            # Increment counter
            self.redis.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if rate limiter is down
            return True
    
    def get_remaining(self, key: str, limit: int) -> int:
        """Get remaining requests in current window."""
        try:
            current = self.redis.get(key)
            if current is None:
                return limit
            return max(0, limit - int(current))
        except Exception:
            return limit


# Global instances
security_manager = SecurityManager()
audit_logger = AuditLogger()


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    return security_manager


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return audit_logger


def validate_security_config():
    """Validate security configuration."""
    errors = []
    
    # Check certificate files
    cert_files = [
        config.security.tls_cert_path,
        config.security.tls_key_path,
        config.security.ca_cert_path
    ]
    
    for cert_file in cert_files:
        if not Path(cert_file).exists():
            errors.append(f"Certificate file not found: {cert_file}")
    
    # Verify certificate chain
    if Path(config.security.tls_cert_path).exists() and Path(config.security.ca_cert_path).exists():
        if not security_manager.verify_certificate_chain(
            config.security.tls_cert_path,
            config.security.ca_cert_path
        ):
            errors.append("TLS certificate verification failed")
    
    # Check secret key strength
    if len(config.security.secret_key) < 32:
        errors.append("SECRET_KEY must be at least 32 characters")
    
    if errors:
        raise ValueError(f"Security validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True