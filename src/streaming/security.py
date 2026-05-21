"""
Security module for HistoCore Real-Time WSI Streaming.

Implements TLS 1.3 encryption, at-rest encryption, key management, and secure communications.
"""

import logging
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .authentication import (
    TLSManager,
    TokenGenerator,
    create_secure_headers,
    validate_password_strength,
)
from .authorization import AuthorizationManager
from .encryption import EncryptionManager, HSMManager, KeyManager

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "SecurityConfig",
    "TLSManager",
    "HSMManager",
    "EncryptionManager",
    "KeyManager",
    "TokenGenerator",
    "AuthorizationManager",
    "SecurityManager",
    "create_security_manager",
    "create_secure_headers",
    "validate_password_strength",
]


@dataclass
class SecurityConfig:
    """Security configuration."""

    # TLS configuration
    enable_tls: bool = True
    tls_version: str = "TLSv1.3"  # TLSv1.2, TLSv1.3
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_cert_path: Optional[str] = None
    verify_client_cert: bool = True

    # Encryption configuration
    enable_at_rest_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"  # AES-256-GCM, ChaCha20-Poly1305
    key_derivation_iterations: int = 100_000

    # Key management
    key_rotation_days: int = 90
    key_storage_path: str = "./keys"
    enable_key_rotation: bool = True

    # HSM configuration
    enable_hsm: bool = False
    hsm_library_path: Optional[str] = (
        None  # Path to PKCS#11 library (e.g., /usr/lib/softhsm/libsofthsm2.so)
    )
    hsm_slot_id: Optional[int] = None  # HSM slot ID
    hsm_pin: Optional[str] = None  # HSM PIN (should be from env var)
    hsm_key_label: str = "histocore_master_key"  # Key label in HSM

    # Security policies
    min_password_length: int = 12
    require_strong_passwords: bool = True
    session_timeout_minutes: int = 30
    max_failed_login_attempts: int = 5

    def __post_init__(self):
        """Validate security configuration."""
        if self.enable_tls and not (self.cert_path and self.key_path):
            logger.warning("TLS enabled but cert/key paths not provided")

        if self.tls_version not in ["TLSv1.2", "TLSv1.3"]:
            raise ValueError(f"Invalid TLS version: {self.tls_version}")

        if self.encryption_algorithm not in ["AES-256-GCM", "ChaCha20-Poly1305"]:
            raise ValueError(f"Invalid encryption algorithm: {self.encryption_algorithm}")

        if self.enable_hsm:
            if not self.hsm_library_path:
                raise ValueError("HSM enabled but hsm_library_path not provided")
            if self.hsm_slot_id is None:
                raise ValueError("HSM enabled but hsm_slot_id not provided")
            if not self.hsm_pin:
                logger.warning("HSM enabled but hsm_pin not provided - will fail at runtime")

        # Create key storage directory
        Path(self.key_storage_path).mkdir(parents=True, exist_ok=True)


class SecurityManager:
    """Main security manager coordinating all security components."""

    def __init__(self, config: SecurityConfig):
        """Initialize security manager."""
        self.config = config

        # Initialize components
        self.tls_manager = TLSManager(config) if config.enable_tls else None
        self.encryption_manager = (
            EncryptionManager(config) if config.enable_at_rest_encryption else None
        )
        self.key_manager = KeyManager(config)
        self.token_generator = TokenGenerator()
        self.authorization_manager = AuthorizationManager()
        self.hsm_manager = None
        if config.enable_hsm:
            self.hsm_manager = HSMManager(config)

        logger.info(
            "Security manager initialized: tls=%s encryption=%s hsm=%s",
            config.enable_tls,
            config.enable_at_rest_encryption,
            config.enable_hsm,
        )

    def initialize(self, master_password: Optional[str] = None):
        """Initialize security components."""
        if self.encryption_manager:
            self.encryption_manager.initialize_master_key(master_password)

        logger.info("Security manager initialized")

    def cleanup(self):
        """Cleanup resources (disconnect from HSM)."""
        if self.hsm_manager:
            self.hsm_manager.disconnect()
        if self.encryption_manager and self.encryption_manager.hsm_manager:
            self.encryption_manager.hsm_manager.disconnect()

    def get_ssl_context(self, server_side: bool = True) -> Optional[ssl.SSLContext]:
        """Get SSL context for secure connections."""
        if not self.tls_manager:
            return None

        return self.tls_manager.create_ssl_context(server_side=server_side)

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data."""
        if not self.encryption_manager:
            raise RuntimeError("Encryption not enabled")

        return self.encryption_manager.encrypt_data(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self.encryption_manager:
            raise RuntimeError("Encryption not enabled")

        return self.encryption_manager.decrypt_data(encrypted_data)

    def generate_secure_token(self) -> str:
        """Generate secure token."""
        return self.token_generator.generate_token()

    def generate_api_key(self) -> str:
        """Generate API key."""
        return self.token_generator.generate_api_key()


def create_security_manager(
    enable_tls: bool = True,
    enable_encryption: bool = True,
    key_storage_path: str = "./keys",
    enable_hsm: bool = False,
    hsm_library_path: Optional[str] = None,
    hsm_slot_id: Optional[int] = None,
    hsm_pin: Optional[str] = None,
) -> SecurityManager:
    """Create security manager with default configuration."""
    config = SecurityConfig(
        enable_tls=enable_tls,
        enable_at_rest_encryption=enable_encryption,
        key_storage_path=key_storage_path,
        enable_hsm=enable_hsm,
        hsm_library_path=hsm_library_path,
        hsm_slot_id=hsm_slot_id,
        hsm_pin=hsm_pin,
    )

    manager = SecurityManager(config)
    manager.initialize()

    return manager


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create security manager
    security = create_security_manager()

    # Generate self-signed cert for testing
    if security.tls_manager:
        cert_path, key_path = security.tls_manager.generate_self_signed_cert(
            "./test_cert.pem", "./test_key.pem"
        )
        print(f"Generated certificate: {cert_path}")

    # Test encryption
    if security.encryption_manager:
        data = b"Sensitive patient data"
        encrypted = security.encrypt_data(data)
        decrypted = security.decrypt_data(encrypted)
        print(f"Encryption test: {data == decrypted}")

    # Generate tokens
    token = security.generate_secure_token()
    api_key = security.generate_api_key()
    print(f"Token: {token[:20]}...")
    print(f"API Key: {api_key[:20]}...")
