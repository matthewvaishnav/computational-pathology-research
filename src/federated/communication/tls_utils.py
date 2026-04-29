"""
TLS utilities for secure federated learning communication.

Provides certificate generation, validation, and gRPC channel setup
for mutual TLS authentication between coordinator and clients.
"""

import ipaddress
import logging
import os
import ssl
from datetime import datetime, timedelta
from typing import Optional, Tuple

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


class TLSManager:
    """Manages TLS certificates and secure gRPC channels."""

    def __init__(self, cert_dir: str = "./certs"):
        """
        Initialize TLS manager.

        Args:
            cert_dir: Directory to store certificates
        """
        self.cert_dir = cert_dir
        os.makedirs(cert_dir, exist_ok=True)

    def generate_ca_certificate(self) -> Tuple[bytes, bytes]:
        """
        Generate Certificate Authority (CA) certificate and private key.

        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Create certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HistoCore FL"),
                x509.NameAttribute(NameOID.COMMON_NAME, "HistoCore FL CA"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    key_cert_sign=True,
                    crl_sign=True,
                    digital_signature=False,
                    key_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Serialize to PEM
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        return cert_pem, key_pem

    def generate_server_certificate(
        self, ca_cert_pem: bytes, ca_key_pem: bytes, hostname: str = "localhost"
    ) -> Tuple[bytes, bytes]:
        """
        Generate server certificate signed by CA.

        Args:
            ca_cert_pem: CA certificate in PEM format
            ca_key_pem: CA private key in PEM format
            hostname: Server hostname

        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        # Load CA certificate and key
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
        ca_key = serialization.load_pem_private_key(ca_key_pem, password=None)

        # Generate server private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Create server certificate
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HistoCore FL"),
                x509.NameAttribute(NameOID.COMMON_NAME, hostname),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(hostname),
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.KeyUsage(
                    key_cert_sign=False,
                    crl_sign=False,
                    digital_signature=True,
                    key_encipherment=True,
                    key_agreement=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )

        # Serialize to PEM
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        return cert_pem, key_pem

    def generate_client_certificate(
        self, ca_cert_pem: bytes, ca_key_pem: bytes, client_id: str
    ) -> Tuple[bytes, bytes]:
        """
        Generate client certificate signed by CA.

        Args:
            ca_cert_pem: CA certificate in PEM format
            ca_key_pem: CA private key in PEM format
            client_id: Client identifier

        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        # Load CA certificate and key
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
        ca_key = serialization.load_pem_private_key(ca_key_pem, password=None)

        # Generate client private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Create client certificate
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HistoCore FL"),
                x509.NameAttribute(NameOID.COMMON_NAME, client_id),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.KeyUsage(
                    key_cert_sign=False,
                    crl_sign=False,
                    digital_signature=True,
                    key_encipherment=True,
                    key_agreement=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )

        # Serialize to PEM
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        return cert_pem, key_pem

    def setup_certificates(self, force_regenerate: bool = False):
        """
        Set up all certificates (CA, server, sample client).

        Args:
            force_regenerate: Force regeneration even if certs exist
        """
        ca_cert_path = os.path.join(self.cert_dir, "ca-cert.pem")
        ca_key_path = os.path.join(self.cert_dir, "ca-key.pem")

        # Generate CA if doesn't exist or force regenerate
        if force_regenerate or not (os.path.exists(ca_cert_path) and os.path.exists(ca_key_path)):
            logger.info("Generating CA certificate...")
            ca_cert_pem, ca_key_pem = self.generate_ca_certificate()

            with open(ca_cert_path, "wb") as f:
                f.write(ca_cert_pem)
            with open(ca_key_path, "wb") as f:
                f.write(ca_key_pem)
        else:
            # Load existing CA
            with open(ca_cert_path, "rb") as f:
                ca_cert_pem = f.read()
            with open(ca_key_path, "rb") as f:
                ca_key_pem = f.read()

        # Generate server certificate
        server_cert_path = os.path.join(self.cert_dir, "server-cert.pem")
        server_key_path = os.path.join(self.cert_dir, "server-key.pem")

        if force_regenerate or not (
            os.path.exists(server_cert_path) and os.path.exists(server_key_path)
        ):
            logger.info("Generating server certificate...")
            server_cert_pem, server_key_pem = self.generate_server_certificate(
                ca_cert_pem, ca_key_pem, "localhost"
            )

            with open(server_cert_path, "wb") as f:
                f.write(server_cert_pem)
            with open(server_key_path, "wb") as f:
                f.write(server_key_pem)

        # Generate sample client certificate
        client_cert_path = os.path.join(self.cert_dir, "client-cert.pem")
        client_key_path = os.path.join(self.cert_dir, "client-key.pem")

        if force_regenerate or not (
            os.path.exists(client_cert_path) and os.path.exists(client_key_path)
        ):
            logger.info("Generating sample client certificate...")
            client_cert_pem, client_key_pem = self.generate_client_certificate(
                ca_cert_pem, ca_key_pem, "sample_client"
            )

            with open(client_cert_path, "wb") as f:
                f.write(client_cert_pem)
            with open(client_key_path, "wb") as f:
                f.write(client_key_pem)

        logger.info(f"Certificates ready in {self.cert_dir}/")

    def create_server_credentials(self) -> grpc.ServerCredentials:
        """
        Create gRPC server credentials with mutual TLS.

        Returns:
            gRPC server credentials
        """
        # Load certificates
        ca_cert_path = os.path.join(self.cert_dir, "ca-cert.pem")
        server_cert_path = os.path.join(self.cert_dir, "server-cert.pem")
        server_key_path = os.path.join(self.cert_dir, "server-key.pem")

        with open(ca_cert_path, "rb") as f:
            ca_cert = f.read()
        with open(server_cert_path, "rb") as f:
            server_cert = f.read()
        with open(server_key_path, "rb") as f:
            server_key = f.read()

        # Create server credentials with mutual TLS
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)], root_certificates=ca_cert, require_client_auth=True
        )

        return credentials

    def create_client_credentials(
        self, client_cert_path: Optional[str] = None
    ) -> grpc.ChannelCredentials:
        """
        Create gRPC client credentials with mutual TLS.

        Args:
            client_cert_path: Path to client certificate (optional, uses sample if None)

        Returns:
            gRPC channel credentials
        """
        # Load CA certificate
        ca_cert_path = os.path.join(self.cert_dir, "ca-cert.pem")
        with open(ca_cert_path, "rb") as f:
            ca_cert = f.read()

        # Load client certificate
        if client_cert_path is None:
            client_cert_path = os.path.join(self.cert_dir, "client-cert.pem")
            client_key_path = os.path.join(self.cert_dir, "client-key.pem")
        else:
            client_key_path = client_cert_path.replace("-cert.pem", "-key.pem")

        with open(client_cert_path, "rb") as f:
            client_cert = f.read()
        with open(client_key_path, "rb") as f:
            client_key = f.read()

        # Create client credentials with mutual TLS
        credentials = grpc.ssl_channel_credentials(
            root_certificates=ca_cert, private_key=client_key, certificate_chain=client_cert
        )

        return credentials


def validate_certificate(cert_pem: bytes, ca_cert_pem: bytes) -> bool:
    """
    Validate certificate against CA.

    Args:
        cert_pem: Certificate to validate
        ca_cert_pem: CA certificate

    Returns:
        True if valid, False otherwise
    """
    try:
        cert = x509.load_pem_x509_certificate(cert_pem)
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)

        # Check if certificate is signed by CA
        ca_public_key = ca_cert.public_key()

        # Get the signature hash algorithm from the certificate
        from cryptography.hazmat.primitives.asymmetric import padding

        try:
            ca_public_key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )
        except Exception:
            return False

        # Check validity period
        now = datetime.utcnow()
        if now < cert.not_valid_before or now > cert.not_valid_after:
            return False

        return True
    except Exception as e:
        logger.error(f"Certificate validation failed: {e}")
        return False


if __name__ == "__main__":
    # Demo: Set up certificates
    tls_manager = TLSManager()
    tls_manager.setup_certificates(force_regenerate=True)
    print("TLS certificates generated successfully!")
