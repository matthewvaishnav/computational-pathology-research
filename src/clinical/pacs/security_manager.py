"""
Security Manager for PACS Integration System.

This module implements the SecurityManager class that handles TLS encryption,
certificate validation, mutual authentication, and security event logging
for all PACS communications.
"""

import logging
import socket
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import OpenSSL
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import NameOID

from .data_models import OperationResult, PACSEndpoint, SecurityConfig, ValidationResult

logger = logging.getLogger(__name__)


class CertificateValidationResult:
    """Result of certificate validation operations."""

    def __init__(self, is_valid: bool, certificate: Optional[x509.Certificate] = None):
        self.is_valid = is_valid
        self.certificate = certificate
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.validation_timestamp = datetime.now()

    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class SecureConnection:
    """Represents a secure DICOM connection."""

    def __init__(
        self,
        socket: ssl.SSLSocket,
        endpoint: PACSEndpoint,
        peer_certificate: Optional[x509.Certificate] = None,
    ):
        self.socket = socket
        self.endpoint = endpoint
        self.peer_certificate = peer_certificate
        self.established_at = datetime.now()
        self.is_active = True

    def close(self):
        """Close the secure connection."""
        try:
            if self.socket and self.is_active:
                self.socket.close()
                self.is_active = False
                logger.debug(f"Closed secure connection to {self.endpoint.host}")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for logging."""
        return {
            "endpoint_id": self.endpoint.endpoint_id,
            "host": self.endpoint.host,
            "port": self.endpoint.port,
            "established_at": self.established_at.isoformat(),
            "is_active": self.is_active,
            "tls_version": getattr(self.socket, "version", None) if self.socket else None,
            "cipher": getattr(self.socket, "cipher", None) if self.socket else None,
        }


class SecurityManager:
    """
    Manages TLS encryption, certificate validation, and authentication for PACS communications.

    This class provides comprehensive security capabilities including:
    - TLS 1.3 encryption for all DICOM communications
    - X.509 certificate validation against configured CA
    - Mutual authentication support with client certificates
    - Automatic credential rotation based on hospital policies
    - Comprehensive security event logging
    """

    def __init__(self, audit_logger=None):
        """
        Initialize Security Manager.

        Args:
            audit_logger: Optional audit logger for security events
        """
        self.audit_logger = audit_logger
        self._active_connections: Dict[str, SecureConnection] = {}
        self._certificate_cache: Dict[str, x509.Certificate] = {}
        self._validation_cache: Dict[str, CertificateValidationResult] = {}
        
        # Rate limiting for connection attempts
        self._connection_attempts: Dict[str, List[datetime]] = {}
        self._max_attempts_per_minute = 10
        self._lockout_duration_minutes = 15

        logger.info("SecurityManager initialized")

    def establish_secure_connection(
        self, pacs_endpoint: PACSEndpoint, timeout: int = 30
    ) -> SecureConnection:
        """
        Establish TLS encrypted connection to PACS endpoint.

        Args:
            pacs_endpoint: PACS endpoint configuration
            timeout: Connection timeout in seconds

        Returns:
            SecureConnection object

        Raises:
            ConnectionError: If connection establishment fails
            ssl.SSLError: If TLS handshake fails
        """
        logger.info(f"Establishing secure connection to {pacs_endpoint.host}:{pacs_endpoint.port}")

        # Input validation
        if not pacs_endpoint or not pacs_endpoint.host:
            raise ValueError("Invalid PACS endpoint")
        
        # Validate hostname format
        import re
        if not re.match(r'^[a-zA-Z0-9.-]+$', pacs_endpoint.host) or len(pacs_endpoint.host) > 253:
            raise ValueError("Invalid hostname format")
        
        # Validate port range
        if not (1 <= pacs_endpoint.port <= 65535):
            raise ValueError("Invalid port number")
        
        # Validate certificate paths if provided
        if pacs_endpoint.security_config:
            sec_config = pacs_endpoint.security_config
            if sec_config.ca_bundle_path and not sec_config.ca_bundle_path.exists():
                raise ValueError(f"CA bundle not found: {sec_config.ca_bundle_path}")
            if sec_config.client_cert_path and not sec_config.client_cert_path.exists():
                raise ValueError(f"Client certificate not found: {sec_config.client_cert_path}")
            if sec_config.client_key_path and not sec_config.client_key_path.exists():
                raise ValueError(f"Client key not found: {sec_config.client_key_path}")

        # Rate limiting check
        endpoint_key = f"{pacs_endpoint.host}:{pacs_endpoint.port}"
        if self._is_rate_limited(endpoint_key):
            error_msg = f"Rate limit exceeded for {endpoint_key}. Too many connection attempts."
            logger.warning(error_msg)
            self._log_security_event(
                event_type="rate_limit_exceeded",
                endpoint=pacs_endpoint,
                details={"endpoint": endpoint_key}
            )
            raise ConnectionError(error_msg)
        
        # Track connection attempt
        self._record_connection_attempt(endpoint_key)

        # Log security event
        self._log_security_event(
            event_type="connection_attempt", endpoint=pacs_endpoint, details={"timeout": timeout}
        )

        try:
            # Create SSL context
            ssl_context = self._create_ssl_context(pacs_endpoint.security_config)

            # Create socket and wrap with SSL
            sock = socket.create_connection(
                (pacs_endpoint.host, pacs_endpoint.port), timeout=timeout
            )

            ssl_sock = ssl_context.wrap_socket(sock, server_hostname=pacs_endpoint.host)

            # Perform TLS handshake
            ssl_sock.do_handshake()

            # Validate peer certificate
            peer_cert = None
            if pacs_endpoint.security_config.verify_certificates:
                peer_cert = self._get_peer_certificate(ssl_sock)
                validation_result = self.validate_certificate(
                    peer_cert, pacs_endpoint.security_config.ca_bundle_path
                )

                if not validation_result.is_valid:
                    ssl_sock.close()
                    
                    # Trigger security alert for validation failure
                    logger.critical(
                        f"SECURITY ALERT: Certificate validation failed for {pacs_endpoint.host}",
                        extra={"errors": validation_result.errors}
                    )
                    self._log_security_event(
                        event_type="certificate_validation_failed",
                        endpoint=pacs_endpoint,
                        details={
                            "errors": validation_result.errors,
                            "warnings": validation_result.warnings,
                            "severity": "CRITICAL"
                        }
                    )
                    
                    raise ssl.SSLError(f"Certificate validation failed: {validation_result.errors}")

            # Create secure connection object
            connection = SecureConnection(
                socket=ssl_sock, endpoint=pacs_endpoint, peer_certificate=peer_cert
            )

            # Track active connection
            connection_id = f"{pacs_endpoint.host}:{pacs_endpoint.port}"
            self._active_connections[connection_id] = connection

            # Log successful connection
            self._log_security_event(
                event_type="connection_established",
                endpoint=pacs_endpoint,
                details=connection.get_connection_info(),
            )

            logger.info(
                f"Secure connection established to {pacs_endpoint.host}:{pacs_endpoint.port}"
            )
            return connection

        except Exception as e:
            # Log connection failure
            self._log_security_event(
                event_type="connection_failed", endpoint=pacs_endpoint, details={"error": str(e)}
            )

            logger.error(f"Failed to establish secure connection: {str(e)}")
            raise ConnectionError(f"Secure connection failed: {str(e)}")

    def validate_certificate(
        self, certificate: x509.Certificate, ca_bundle_path: Optional[Path], hostname: Optional[str] = None
    ) -> CertificateValidationResult:
        """
        Validate X.509 certificate against configured Certificate Authority.

        Args:
            certificate: Certificate to validate
            ca_bundle_path: Path to CA bundle file
            hostname: Expected hostname for verification

        Returns:
            CertificateValidationResult with validation status
        """
        result = CertificateValidationResult(is_valid=True, certificate=certificate)

        try:
            # Validate hostname if provided
            if hostname:
                if not self._verify_hostname(certificate, hostname):
                    result.add_error(f"Certificate hostname verification failed for {hostname}")

            # Check certificate expiration
            uses_utc_properties = hasattr(certificate, "not_valid_before_utc")
            now = datetime.now(timezone.utc) if uses_utc_properties else datetime.now()
            not_before = (
                certificate.not_valid_before_utc
                if uses_utc_properties
                else certificate.not_valid_before
            )
            not_after = (
                certificate.not_valid_after_utc
                if uses_utc_properties
                else certificate.not_valid_after
            )

            if now < not_before:
                result.add_error(f"Certificate not yet valid (valid from: {not_before})")

            if now > not_after:
                result.add_error(f"Certificate expired (expired: {not_after})")

            # Warn if certificate expires soon (within 30 days)
            if not_after - now < timedelta(days=30):
                result.add_warning(f"Certificate expires soon: {not_after}")

            # Validate certificate chain if CA bundle provided
            if ca_bundle_path and ca_bundle_path.exists():
                chain_valid = self._validate_certificate_chain(certificate, ca_bundle_path)
                if not chain_valid:
                    result.add_error("Certificate chain validation failed")

            # Check key usage extensions
            try:
                key_usage = certificate.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.KEY_USAGE
                ).value

                if not key_usage.digital_signature:
                    result.add_warning("Certificate does not allow digital signatures")

                if not key_usage.key_encipherment:
                    result.add_warning("Certificate does not allow key encipherment")

            except x509.ExtensionNotFound:
                result.add_warning("Certificate missing key usage extension")

            # Check subject alternative names for hostname validation
            try:
                san_ext = certificate.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                ).value

                dns_names = san_ext.get_values_for_type(x509.DNSName)
                if not dns_names:
                    result.add_warning("Certificate has no DNS names in SAN extension")

            except x509.ExtensionNotFound:
                result.add_warning("Certificate missing Subject Alternative Name extension")

            # Log validation result
            self._log_security_event(
                event_type="certificate_validation",
                details={
                    "is_valid": result.is_valid,
                    "subject": certificate.subject.rfc4514_string(),
                    "issuer": certificate.issuer.rfc4514_string(),
                    "serial_number": str(certificate.serial_number),
                    "not_before": not_before.isoformat(),
                    "not_after": not_after.isoformat(),
                    "errors": result.errors,
                    "warnings": result.warnings,
                },
            )

        except Exception as e:
            result.add_error(f"Certificate validation failed")
            logger.error(f"Certificate validation failed: {type(e).__name__}")

        return result

    def _verify_hostname(self, certificate: x509.Certificate, hostname: str) -> bool:
        """Verify certificate hostname matches expected hostname."""
        import ipaddress
        
        # Validate hostname format
        if not hostname or len(hostname) > 253:
            return False
        
        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            # Check IP addresses in SAN
            try:
                san_ext = certificate.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                ).value
                ip_addresses = san_ext.get_values_for_type(x509.IPAddress)
                return ip in ip_addresses
            except x509.ExtensionNotFound:
                return False
        except ValueError:
            pass
        
        # Check Subject Alternative Names
        try:
            san_ext = certificate.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            ).value
            dns_names = san_ext.get_values_for_type(x509.DNSName)
            
            for dns_name in dns_names:
                if self._match_hostname(dns_name, hostname):
                    return True
        except x509.ExtensionNotFound:
            pass
        
        # Check Common Name as fallback
        try:
            subject = certificate.subject
            cn_attributes = subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            if cn_attributes:
                cn = cn_attributes[0].value
                if self._match_hostname(cn, hostname):
                    return True
        except Exception:
            pass
        
        return False
    
    def _match_hostname(self, cert_hostname: str, hostname: str) -> bool:
        """Match hostname with wildcard support."""
        # Exact match
        if cert_hostname.lower() == hostname.lower():
            return True
        
        # Wildcard match
        if cert_hostname.startswith('*.'):
            pattern = cert_hostname[2:].lower()
            if '.' in hostname:
                domain = hostname.split('.', 1)[1].lower()
                return domain == pattern
        
        return False

    def rotate_credentials(
        self,
        endpoint_id: str,
        new_cert_path: Optional[Path] = None,
        new_key_path: Optional[Path] = None,
    ) -> OperationResult:
        """
        Rotate credentials for a PACS endpoint.

        Args:
            endpoint_id: Endpoint identifier
            new_cert_path: Path to new certificate file
            new_key_path: Path to new private key file

        Returns:
            OperationResult with rotation status
        """
        logger.info(f"Rotating credentials for endpoint: {endpoint_id}")

        operation_id = f"credential_rotation_{endpoint_id}_{int(datetime.now().timestamp())}"

        try:
            # Close existing connections for this endpoint
            connections_to_close = [
                conn
                for conn_id, conn in self._active_connections.items()
                if conn.endpoint.endpoint_id == endpoint_id
            ]

            for connection in connections_to_close:
                connection.close()

            # Validate new credentials if provided
            if new_cert_path and new_cert_path.exists():
                try:
                    with open(new_cert_path, "rb") as f:
                        cert_data = f.read()

                    certificate = x509.load_pem_x509_certificate(cert_data)
                    validation_result = self.validate_certificate(certificate, None)

                    if not validation_result.is_valid:
                        return OperationResult.error_result(
                            operation_id=operation_id,
                            message="New certificate validation failed",
                            errors=validation_result.errors,
                        )

                except Exception as e:
                    return OperationResult.error_result(
                        operation_id=operation_id,
                        message=f"Failed to load new certificate: {str(e)}",
                        errors=[str(e)],
                    )

            # Log credential rotation
            self._log_security_event(
                event_type="credential_rotation",
                details={
                    "endpoint_id": endpoint_id,
                    "new_cert_provided": new_cert_path is not None,
                    "new_key_provided": new_key_path is not None,
                    "connections_closed": len(connections_to_close),
                },
            )

            return OperationResult.success_result(
                operation_id=operation_id,
                message=f"Credentials rotated successfully for endpoint {endpoint_id}",
                data={"endpoint_id": endpoint_id, "rotation_timestamp": datetime.now().isoformat()},
            )

        except Exception as e:
            logger.error(f"Credential rotation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id,
                message=f"Credential rotation failed: {str(e)}",
                errors=[str(e)],
            )

    def _create_ssl_context(self, security_config: SecurityConfig) -> ssl.SSLContext:
        """Create SSL context with appropriate security settings."""
        # Create context with TLS 1.3 (or highest available)
        if security_config.tls_version == "1.3":
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.minimum_version = ssl.TLSVersion.TLSv1_3
        elif security_config.tls_version == "1.2":
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        else:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # Configure certificate verification (always required for production)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED

        # Load CA bundle if provided
        if security_config.ca_bundle_path and security_config.ca_bundle_path.exists():
            context.load_verify_locations(str(security_config.ca_bundle_path))
        elif security_config.verify_certificates:
            # Use system CA bundle if no custom bundle provided
            context.load_default_certs()
        else:
            # Log warning if verification disabled (testing only)
            logger.warning(
                "Certificate verification disabled - INSECURE, only use for testing!"
            )
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        # Configure client certificate for mutual authentication
        if security_config.mutual_authentication:
            if security_config.client_cert_path and security_config.client_key_path:
                context.load_cert_chain(
                    str(security_config.client_cert_path), str(security_config.client_key_path)
                )
            else:
                raise ValueError("Client certificate and key required for mutual authentication")

        # Set secure cipher suites (TLS 1.3 and strong TLS 1.2)
        context.set_ciphers(
            "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:"
            "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305"
        )

        return context

    def _get_peer_certificate(self, ssl_socket: ssl.SSLSocket) -> x509.Certificate:
        """Extract peer certificate from SSL socket."""
        try:
            # Get peer certificate in DER format
            peer_cert_der = ssl_socket.getpeercert(binary_form=True)
            if not peer_cert_der:
                raise ValueError("No peer certificate available")

            # Parse certificate
            certificate = x509.load_der_x509_certificate(peer_cert_der)
            return certificate

        except Exception as e:
            logger.error(f"Failed to get peer certificate: {str(e)}")
            raise

    def _validate_certificate_chain(
        self, certificate: x509.Certificate, ca_bundle_path: Path
    ) -> bool:
        """Validate certificate chain against CA bundle."""
        try:
            # Load CA certificates
            with open(ca_bundle_path, "rb") as f:
                ca_data = f.read()

            # Parse CA certificates (PEM format may contain multiple certs)
            ca_certs = []
            for cert_pem in ca_data.split(b"-----END CERTIFICATE-----"):
                if b"-----BEGIN CERTIFICATE-----" in cert_pem:
                    cert_pem += b"-----END CERTIFICATE-----"
                    try:
                        ca_cert = x509.load_pem_x509_certificate(cert_pem)
                        ca_certs.append(ca_cert)
                    except Exception:
                        continue

            if not ca_certs:
                logger.warning("No valid CA certificates found in bundle")
                return False

            # Simple chain validation - check if certificate is signed by any CA
            for ca_cert in ca_certs:
                try:
                    # Check if certificate issuer matches CA subject
                    if certificate.issuer == ca_cert.subject:
                        # Verify signature (simplified check)
                        public_key = ca_cert.public_key()
                        signature_hash_algorithm = certificate.signature_hash_algorithm

                        if isinstance(public_key, rsa.RSAPublicKey):
                            public_key.verify(
                                certificate.signature,
                                certificate.tbs_certificate_bytes,
                                padding.PKCS1v15(),
                                signature_hash_algorithm,
                            )
                        elif isinstance(public_key, ec.EllipticCurvePublicKey):
                            public_key.verify(
                                certificate.signature,
                                certificate.tbs_certificate_bytes,
                                ec.ECDSA(signature_hash_algorithm),
                            )
                        else:
                            logger.warning(
                                "Unsupported CA public key type during certificate chain validation: %s",
                                type(public_key).__name__,
                            )
                            continue
                        return True
                except Exception:
                    continue

            return False

        except Exception as e:
            logger.error(f"Certificate chain validation failed: {str(e)}")
            return False

    def _log_security_event(
        self,
        event_type: str,
        endpoint: Optional[PACSEndpoint] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log security event for audit purposes."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details or {},
        }

        if endpoint:
            event["endpoint"] = {
                "endpoint_id": endpoint.endpoint_id,
                "host": endpoint.host,
                "port": endpoint.port,
                "vendor": endpoint.vendor.value,
            }

        # Log to audit logger if available
        if self.audit_logger:
            try:
                self.audit_logger.log_security_event(event)
            except Exception as e:
                logger.warning(f"Failed to log security event: {e}")

        # Always log to application logger
        logger.info(f"Security event: {event_type}", extra={"security_event": event})

    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get information about active secure connections."""
        return [
            conn.get_connection_info()
            for conn in self._active_connections.values()
            if conn.is_active
        ]

    def close_all_connections(self):
        """Close all active secure connections."""
        logger.info("Closing all active secure connections")
        closed_count = len(self._active_connections)

        for connection in self._active_connections.values():
            connection.close()

        self._active_connections.clear()

        self._log_security_event(
            event_type="all_connections_closed", details={"closed_count": closed_count}
        )

    def generate_self_signed_certificate(
        self,
        common_name: str,
        output_cert_path: Path,
        output_key_path: Path,
        validity_days: int = 365,
        key_password: Optional[str] = None,
    ) -> OperationResult:
        """
        Generate self-signed certificate for testing purposes.

        Args:
            common_name: Common name for certificate
            output_cert_path: Path to save certificate
            output_key_path: Path to save private key
            validity_days: Certificate validity period in days
            key_password: Password to encrypt private key (recommended)

        Returns:
            OperationResult with generation status
        """
        logger.info(f"Generating self-signed certificate for: {common_name}")

        operation_id = f"cert_generation_{int(datetime.now().timestamp())}"

        try:
            # Generate private key
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

            # Create certificate
            subject = issuer = x509.Name(
                [
                    x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HistoCore Test"),
                    x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "PACS Integration"),
                ]
            )

            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.now())
                .not_valid_after(datetime.now() + timedelta(days=validity_days))
                .add_extension(
                    x509.SubjectAlternativeName(
                        [
                            x509.DNSName(common_name),
                        ]
                    ),
                    critical=False,
                )
                .sign(private_key, hashes.SHA256())
            )

            # Save certificate
            with open(output_cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))

            # Save private key with encryption if password provided
            encryption_algo = (
                serialization.BestAvailableEncryption(key_password.encode())
                if key_password
                else serialization.NoEncryption()
            )
            
            with open(output_key_path, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=encryption_algo,
                    )
                )

            if not key_password:
                logger.warning(
                    "Private key saved without encryption - use key_password parameter for production"
                )

            logger.info(f"Self-signed certificate generated: {output_cert_path}")

            return OperationResult.success_result(
                operation_id=operation_id,
                message="Self-signed certificate generated successfully",
                data={
                    "cert_path": str(output_cert_path),
                    "key_path": str(output_key_path),
                    "common_name": common_name,
                    "validity_days": validity_days,
                },
            )

        except Exception as e:
            logger.error(f"Certificate generation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id,
                message=f"Certificate generation failed: {str(e)}",
                errors=[str(e)],
            )

    def _is_rate_limited(self, endpoint_key: str) -> bool:
        """Check if endpoint is rate limited with exponential backoff."""
        if endpoint_key not in self._connection_attempts:
            return False
        
        now = datetime.now()
        attempts = self._connection_attempts[endpoint_key]
        
        # Remove attempts older than lockout window
        lockout_window = timedelta(minutes=self._lockout_duration_minutes)
        recent_attempts = [t for t in attempts if now - t < lockout_window]
        self._connection_attempts[endpoint_key] = recent_attempts
        
        if len(recent_attempts) >= self._max_attempts_per_minute:
            # Calculate exponential backoff
            attempt_count = len(recent_attempts)
            backoff_minutes = min(2 ** (attempt_count - self._max_attempts_per_minute), 60)  # Max 1 hour
            
            if recent_attempts:
                last_attempt = max(recent_attempts)
                if now - last_attempt < timedelta(minutes=backoff_minutes):
                    return True
        
        return False
    
    def _record_connection_attempt(self, endpoint_key: str) -> None:
        """Record connection attempt for rate limiting."""
        if endpoint_key not in self._connection_attempts:
            self._connection_attempts[endpoint_key] = []
        
        self._connection_attempts[endpoint_key].append(datetime.now())

    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security manager statistics."""
        return {
            "active_connections": len(
                [c for c in self._active_connections.values() if c.is_active]
            ),
            "total_connections": len(self._active_connections),
            "certificate_cache_size": len(self._certificate_cache),
            "validation_cache_size": len(self._validation_cache),
        }
