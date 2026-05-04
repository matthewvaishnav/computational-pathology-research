"""
Authentication Module

Implements TLS/SSL management, secure token generation, and password validation
for HistoCore Real-Time WSI Streaming.

Requirements: TLS 1.3 encryption, secure token generation, password policies
"""

import logging
import os
import secrets
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Cryptography imports
try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.x509.oid import NameOID

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not available. Install: pip install cryptography")


class TLSManager:
    """Manages TLS/SSL configuration and certificates."""

    def __init__(self, config):
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
        # Validate and sanitize common_name to prevent injection
        common_name = self._sanitize_common_name(common_name)
        
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
    
    def _sanitize_common_name(self, common_name: str) -> str:
        """Sanitize common_name to prevent injection attacks."""
        import re
        
        # Remove any control characters or special chars that could cause issues
        common_name = re.sub(r'[^\w\.\-]', '', common_name)
        
        # Limit length
        if len(common_name) > 64:
            common_name = common_name[:64]
        
        # Ensure not empty
        if not common_name:
            raise ValueError("Invalid common_name: empty after sanitization")
        
        return common_name


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


def create_secure_headers() -> dict:
    """Create secure HTTP headers."""
    return {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }
