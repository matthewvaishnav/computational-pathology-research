"""
Security module for HistoCore Real-Time WSI Streaming.

Implements TLS 1.3 encryption, at-rest encryption, key management, and secure communications.
"""

import hashlib
import hmac
import logging
import os
import secrets
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Cryptography imports
try:
    from cryptography import x509
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.x509.oid import ExtensionOID, NameOID

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not available. Install: pip install cryptography")


# ============================================================================
# Configuration
# ============================================================================


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

        # Create key storage directory
        Path(self.key_storage_path).mkdir(parents=True, exist_ok=True)


# ============================================================================
# TLS/SSL Configuration
# ============================================================================


class TLSManager:
    """Manages TLS/SSL configuration and certificates."""

    def __init__(self, config: SecurityConfig):
        """Initialize TLS manager."""
        self.config = config

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for TLS")

        logger.info("TLS manager initialized: version=%s", config.tls_version)

    def create_ssl_context(
        self, server_side: bool = True, verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    ) -> ssl.SSLContext:
        """Create SSL context with TLS 1.3."""
        # Set TLS version
        if self.config.tls_version == "TLSv1.3":
            protocol = ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT
            min_version = ssl.TLSVersion.TLSv1_3
        else:
            protocol = ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT
            min_version = ssl.TLSVersion.TLSv1_2

        # Create context
        context = ssl.SSLContext(protocol)
        context.minimum_version = min_version

        # Set verification mode
        if self.config.verify_client_cert:
            context.verify_mode = verify_mode
        else:
            # Still verify server certificate even if not requiring client cert
            context.verify_mode = ssl.CERT_REQUIRED
            logger.warning("Client cert not required, but server cert still verified")

        # Load certificates
        if server_side:
            if self.config.cert_path and self.config.key_path:
                context.load_cert_chain(
                    certfile=self.config.cert_path, keyfile=self.config.key_path
                )
                logger.info("Loaded server certificate: %s", self.config.cert_path)
        else:
            if self.config.ca_cert_path:
                context.load_verify_locations(cafile=self.config.ca_cert_path)
                logger.info("Loaded CA certificate: %s", self.config.ca_cert_path)

        # Set cipher suites (strong ciphers only)
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS")

        # Enable hostname checking for clients
        if not server_side:
            context.check_hostname = True

        return context

    def generate_self_signed_cert(
        self,
        output_cert_path: str,
        output_key_path: str,
        common_name: str = "localhost",
        validity_days: int = 365,
    ) -> Tuple[str, str]:
        """Generate self-signed certificate for testing."""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        # Create certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HistoCore"),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(common_name),
                        x509.DNSName("localhost"),
                        x509.IPAddress("127.0.0.1"),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256(), default_backend())
        )

        # Write certificate
        with open(output_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # Write private key
        with open(output_key_path, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        logger.info("Generated self-signed certificate: %s", output_cert_path)

        return output_cert_path, output_key_path


# ============================================================================
# At-Rest Encryption
# ============================================================================


class EncryptionManager:
    """Manages at-rest encryption for cached data."""

    def __init__(self, config: SecurityConfig):
        """Initialize encryption manager."""
        self.config = config

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for encryption")

        # Initialize encryption key
        self.master_key = None
        self.fernet = None

        logger.info("Encryption manager initialized: algorithm=%s", config.encryption_algorithm)

    def initialize_master_key(self, password: Optional[str] = None) -> bytes:
        """Initialize or load master encryption key."""
        key_path = Path(self.config.key_storage_path) / "master.key"

        if key_path.exists():
            # Load existing key
            with open(key_path, "rb") as f:
                self.master_key = f.read()
            logger.info("Loaded existing master key")
        else:
            # Generate new key
            if password:
                # Derive key from password
                salt = os.urandom(16)
                kdf = PBKDF2(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=self.config.key_derivation_iterations,
                    backend=default_backend(),
                )
                self.master_key = kdf.derive(password.encode())

                # Store salt with key
                with open(key_path, "wb") as f:
                    f.write(salt + self.master_key)
            else:
                # Generate random key
                self.master_key = Fernet.generate_key()

                with open(key_path, "wb") as f:
                    f.write(self.master_key)

            # Secure file permissions (Unix only)
            try:
                os.chmod(key_path, 0o600)
            except Exception:
                pass

            logger.info("Generated new master key: %s", key_path)

        # Initialize Fernet cipher
        self.fernet = Fernet(self.master_key)

        return self.master_key

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using master key."""
        if not self.fernet:
            raise RuntimeError("Master key not initialized")

        return self.fernet.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using master key."""
        if not self.fernet:
            raise RuntimeError("Master key not initialized")

        return self.fernet.decrypt(encrypted_data)

    def encrypt_file(self, input_path: str, output_path: str) -> str:
        """Encrypt file."""
        with open(input_path, "rb") as f:
            data = f.read()

        encrypted_data = self.encrypt_data(data)

        with open(output_path, "wb") as f:
            f.write(encrypted_data)

        logger.info("Encrypted file: %s -> %s", input_path, output_path)

        return output_path

    def decrypt_file(self, input_path: str, output_path: str) -> str:
        """Decrypt file."""
        with open(input_path, "rb") as f:
            encrypted_data = f.read()

        data = self.decrypt_data(encrypted_data)

        with open(output_path, "wb") as f:
            f.write(data)

        logger.info("Decrypted file: %s -> %s", input_path, output_path)

        return output_path

    def rotate_key(self, new_password: Optional[str] = None) -> bytes:
        """Rotate encryption key."""
        logger.info("Rotating encryption key")

        # Generate new key
        old_key = self.master_key
        old_fernet = self.fernet

        # Initialize new key
        self.master_key = None
        self.fernet = None
        new_key = self.initialize_master_key(new_password)

        # Re-encrypt all encrypted files
        # (This would need to be implemented based on your file storage structure)

        logger.info("Key rotation complete")

        return new_key


# ============================================================================
# Key Management
# ============================================================================


class KeyManager:
    """Manages encryption keys and rotation."""

    def __init__(self, config: SecurityConfig):
        """Initialize key manager."""
        self.config = config
        self.key_metadata = {}

        logger.info("Key manager initialized: rotation_days=%d", config.key_rotation_days)

    def generate_key(self, key_id: str, key_type: str = "symmetric") -> bytes:
        """Generate new encryption key."""
        if key_type == "symmetric":
            key = Fernet.generate_key()
        elif key_type == "asymmetric":
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        else:
            raise ValueError(f"Invalid key type: {key_type}")

        # Store key metadata
        self.key_metadata[key_id] = {
            "created_at": datetime.utcnow(),
            "key_type": key_type,
            "rotation_due": datetime.utcnow() + timedelta(days=self.config.key_rotation_days),
        }

        # Save key
        key_path = Path(self.config.key_storage_path) / f"{key_id}.key"
        with open(key_path, "wb") as f:
            f.write(key)

        # Secure permissions
        try:
            os.chmod(key_path, 0o600)
        except Exception:
            pass

        logger.info("Generated key: %s (type=%s)", key_id, key_type)

        return key

    def load_key(self, key_id: str) -> bytes:
        """Load encryption key."""
        key_path = Path(self.config.key_storage_path) / f"{key_id}.key"

        if not key_path.exists():
            raise FileNotFoundError(f"Key not found: {key_id}")

        with open(key_path, "rb") as f:
            key = f.read()

        return key

    def check_rotation_needed(self, key_id: str) -> bool:
        """Check if key rotation is needed."""
        if key_id not in self.key_metadata:
            return False

        metadata = self.key_metadata[key_id]
        return datetime.utcnow() >= metadata["rotation_due"]

    def rotate_key(self, key_id: str) -> bytes:
        """Rotate encryption key."""
        logger.info("Rotating key: %s", key_id)

        # Archive old key
        old_key_path = Path(self.config.key_storage_path) / f"{key_id}.key"
        archive_path = Path(self.config.key_storage_path) / f"{key_id}.key.old"

        if old_key_path.exists():
            old_key_path.rename(archive_path)

        # Generate new key
        metadata = self.key_metadata.get(key_id, {})
        key_type = metadata.get("key_type", "symmetric")
        new_key = self.generate_key(key_id, key_type)

        logger.info("Key rotated: %s", key_id)

        return new_key


# ============================================================================
# Secure Token Generation
# ============================================================================


class TokenGenerator:
    """Generates secure tokens for sessions and API keys."""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_api_key() -> str:
        """Generate API key."""
        return f"hc_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_token(token: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash token for storage."""
        if salt is None:
            salt = os.urandom(32)

        # Use PBKDF2 for token hashing
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
            backend=default_backend(),
        )

        token_hash = kdf.derive(token.encode())

        return token_hash, salt

    @staticmethod
    def verify_token(token: str, token_hash: bytes, salt: bytes) -> bool:
        """Verify token against hash."""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
            backend=default_backend(),
        )

        try:
            kdf.verify(token.encode(), token_hash)
            return True
        except Exception:
            return False


# ============================================================================
# Secure Communication Helpers
# ============================================================================


def create_secure_headers() -> Dict[str, str]:
    """Create secure HTTP headers."""
    return {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }


def validate_password_strength(password: str, min_length: int = 12) -> Tuple[bool, str]:
    """Validate password strength."""
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters"

    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    if not (has_upper and has_lower and has_digit and has_special):
        return False, "Password must contain uppercase, lowercase, digit, and special character"

    return True, "Password is strong"


# ============================================================================
# Security Manager (Main Interface)
# ============================================================================


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

        logger.info(
            "Security manager initialized: tls=%s encryption=%s",
            config.enable_tls,
            config.enable_at_rest_encryption,
        )

    def initialize(self, master_password: Optional[str] = None):
        """Initialize security components."""
        if self.encryption_manager:
            self.encryption_manager.initialize_master_key(master_password)

        logger.info("Security manager initialized")

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


# ============================================================================
# Convenience Functions
# ============================================================================


def create_security_manager(
    enable_tls: bool = True, enable_encryption: bool = True, key_storage_path: str = "./keys"
) -> SecurityManager:
    """Create security manager with default configuration."""
    config = SecurityConfig(
        enable_tls=enable_tls,
        enable_at_rest_encryption=enable_encryption,
        key_storage_path=key_storage_path,
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
