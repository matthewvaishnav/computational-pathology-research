"""Cryptographic signing for audit records."""

import base64
import hashlib
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from .audit_models import AuditEvent


class CryptographicSigner:
    """Handles cryptographic signing of audit records."""

    def __init__(self, private_key: Optional[rsa.RSAPrivateKey] = None):
        """Initialize with RSA private key."""
        if private_key is None:
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.public_key_fingerprint = self._compute_key_fingerprint()

    def _compute_key_fingerprint(self) -> str:
        """Compute fingerprint of public key."""
        public_key_bytes = self.public_key.public_bytes(
            encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_key_bytes).hexdigest()[:16]

    def sign_event(self, event: AuditEvent) -> str:
        """Sign audit event and return base64-encoded signature."""
        content_hash = event.get_content_hash()
        signature = self.private_key.sign(
            content_hash.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def verify_signature(self, event: AuditEvent, signature: str) -> bool:
        """Verify signature of audit event."""
        try:
            signature_bytes = base64.b64decode(signature)
            content_hash = event.get_content_hash()

            self.public_key.verify(
                signature_bytes,
                content_hash.encode(),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return True
        except (InvalidSignature, ValueError):
            return False

    def export_public_key(self) -> str:
        """Export public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
        ).decode()

    def export_private_key(self, password: Optional[str] = None) -> str:
        """Export private key in PEM format."""
        encryption_algorithm = NoEncryption()
        if password:
            from cryptography.hazmat.primitives.serialization import BestAvailableEncryption

            encryption_algorithm = BestAvailableEncryption(password.encode())

        return self.private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm,
        ).decode()
