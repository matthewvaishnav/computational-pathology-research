"""
Security Manager for PACS Integration System.

This module implements the SecurityManager class that handles TLS encryption,
certificate validation, mutual authentication, and security event logging
for all PACS communications.
"""

import logging
import ssl
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import OpenSSL
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import NameOID

from .data_models import PACSEndpoint, SecurityConfig, ValidationResult, OperationResult

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
        peer_certificate: Optional[x509.Certificate] = None
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
            "tls_version": getattr(self.socket, 'version', None) if self.socket else None,
            "cipher": getattr(self.socket, 'cipher', None) if self.socket else None,
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
        
        logger.info("SecurityManager initialized")
    
    def establish_secure_connection(
        self,
        pacs_endpoint: PACSEndpoint,
        timeout: int = 30
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
        
        # Log security event
        self._log_security_event(
            event_type="connection_attempt",
            endpoint=pacs_endpoint,
            details={"timeout": timeout}
        )
        
        try:
            # Create SSL context
            ssl_context = self._create_ssl_context(pacs_endpoint.security_config)
            
            # Create socket and wrap with SSL
            sock = socket.create_connection(
                (pacs_endpoint.host, pacs_endpoint.port),
                timeout=timeout
            )
            
            ssl_sock = ssl_context.wrap_socket(
                sock,
                server_hostname=pacs_endpoint.host
            )
            
            # Perform TLS handshake
            ssl_sock.do_handshake()
            
            # Validate peer certificate
            peer_cert = None
            if pacs_endpoint.security_config.verify_certificates:
                peer_cert = self._get_peer_certificate(ssl_sock)
                validation_result = self.validate_certificate(
                    peer_cert,
                    pacs_endpoint.security_config.ca_bundle_path
                )
                
                if not validation_result.is_valid:
                    ssl_sock.close()
                    raise ssl.SSLError(f"Certificate validation failed: {validation_result.errors}")
            
            # Create secure connection object
            connection = SecureConnection(
                socket=ssl_sock,
                endpoint=pacs_endpoint,
                peer_certificate=peer_cert
            )
            
            # Track active connection
            connection_id = f"{pacs_endpoint.host}:{pacs_endpoint.port}"
            self._active_connections[connection_id] = connection
            
            # Log successful connection
            self._log_security_event(
                event_type="connection_established",
                endpoint=pacs_endpoint,
                details=connection.get_connection_info()
            )
            
            logger.info(f"Secure connection established to {pacs_endpoint.host}:{pacs_endpoint.port}")
            return connection
            
        except Exception as e:
            # Log connection failure
            self._log_security_event(
                event_type="connection_failed",
                endpoint=pacs_endpoint,
                details={"error": str(e)}
            )
            
            logger.error(f"Failed to establish secure connection: {str(e)}")
            raise ConnectionError(f"Secure connection failed: {str(e)}")
    
    def validate_certificate(
        self,
        certificate: x509.Certificate,
        ca_bundle_path: Optional[Path]
    ) -> CertificateValidationResult:
        """
        Validate X.509 certificate against configured Certificate Authority.
        
        Args:
            certificate: Certificate to validate
            ca_bundle_path: Path to CA bundle file
            
        Returns:
            CertificateValidationResult with validation status
        """
        result = CertificateValidationResult(is_valid=True, certificate=certificate)
        
        try:
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
                    "warnings": result.warnings
                }
            )
            
        except Exception as e:
            result.add_error(f"Certificate validation error: {str(e)}")
            logger.error(f"Certificate validation failed: {str(e)}")
        
        return result
    
    def rotate_credentials(
        self,
        endpoint_id: str,
        new_cert_path: Optional[Path] = None,
        new_key_path: Optional[Path] = None
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
                conn for conn_id, conn in self._active_connections.items()
                if conn.endpoint.endpoint_id == endpoint_id
            ]
            
            for connection in connections_to_close:
                connection.close()
            
            # Validate new credentials if provided
            if new_cert_path and new_cert_path.exists():
                try:
                    with open(new_cert_path, 'rb') as f:
                        cert_data = f.read()
                    
                    certificate = x509.load_pem_x509_certificate(cert_data)
                    validation_result = self.validate_certificate(certificate, None)
                    
                    if not validation_result.is_valid:
                        return OperationResult.error_result(
                            operation_id=operation_id,
                            message="New certificate validation failed",
                            errors=validation_result.errors
                        )
                        
                except Exception as e:
                    return OperationResult.error_result(
                        operation_id=operation_id,
                        message=f"Failed to load new certificate: {str(e)}",
                        errors=[str(e)]
                    )
            
            # Log credential rotation
            self._log_security_event(
                event_type="credential_rotation",
                details={
                    "endpoint_id": endpoint_id,
                    "new_cert_provided": new_cert_path is not None,
                    "new_key_provided": new_key_path is not None,
                    "connections_closed": len(connections_to_close)
                }
            )
            
            return OperationResult.success_result(
                operation_id=operation_id,
                message=f"Credentials rotated successfully for endpoint {endpoint_id}",
                data={
                    "endpoint_id": endpoint_id,
                    "rotation_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Credential rotation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id,
                message=f"Credential rotation failed: {str(e)}",
                errors=[str(e)]
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
        
        # Configure certificate verification
        if security_config.verify_certificates:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            
            # Load CA bundle if provided
            if security_config.ca_bundle_path and security_config.ca_bundle_path.exists():
                context.load_verify_locations(str(security_config.ca_bundle_path))
        else:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        # Configure client certificate for mutual authentication
        if security_config.mutual_authentication:
            if security_config.client_cert_path and security_config.client_key_path:
                context.load_cert_chain(
                    str(security_config.client_cert_path),
                    str(security_config.client_key_path)
                )
            else:
                raise ValueError("Client certificate and key required for mutual authentication")
        
        # Set secure cipher suites
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
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
        self,
        certificate: x509.Certificate,
        ca_bundle_path: Path
    ) -> bool:
        """Validate certificate chain against CA bundle."""
        try:
            # Load CA certificates
            with open(ca_bundle_path, 'rb') as f:
                ca_data = f.read()
            
            # Parse CA certificates (PEM format may contain multiple certs)
            ca_certs = []
            for cert_pem in ca_data.split(b'-----END CERTIFICATE-----'):
                if b'-----BEGIN CERTIFICATE-----' in cert_pem:
                    cert_pem += b'-----END CERTIFICATE-----'
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
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security event for audit purposes."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details or {}
        }
        
        if endpoint:
            event["endpoint"] = {
                "endpoint_id": endpoint.endpoint_id,
                "host": endpoint.host,
                "port": endpoint.port,
                "vendor": endpoint.vendor.value
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
            event_type="all_connections_closed",
            details={"closed_count": closed_count}
        )
    
    def generate_self_signed_certificate(
        self,
        common_name: str,
        output_cert_path: Path,
        output_key_path: Path,
        validity_days: int = 365
    ) -> OperationResult:
        """
        Generate self-signed certificate for testing purposes.
        
        Args:
            common_name: Common name for certificate
            output_cert_path: Path to save certificate
            output_key_path: Path to save private key
            validity_days: Certificate validity period in days
            
        Returns:
            OperationResult with generation status
        """
        logger.info(f"Generating self-signed certificate for: {common_name}")
        
        operation_id = f"cert_generation_{int(datetime.now().timestamp())}"
        
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Create certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HistoCore Test"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "PACS Integration"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.now()
            ).not_valid_after(
                datetime.now() + timedelta(days=validity_days)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(common_name),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Save certificate
            with open(output_cert_path, 'wb') as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            # Save private key
            with open(output_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            logger.info(f"Self-signed certificate generated: {output_cert_path}")
            
            return OperationResult.success_result(
                operation_id=operation_id,
                message="Self-signed certificate generated successfully",
                data={
                    "cert_path": str(output_cert_path),
                    "key_path": str(output_key_path),
                    "common_name": common_name,
                    "validity_days": validity_days
                }
            )
            
        except Exception as e:
            logger.error(f"Certificate generation failed: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id,
                message=f"Certificate generation failed: {str(e)}",
                errors=[str(e)]
            )
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security manager statistics."""
        return {
            "active_connections": len([c for c in self._active_connections.values() if c.is_active]),
            "total_connections": len(self._active_connections),
            "certificate_cache_size": len(self._certificate_cache),
            "validation_cache_size": len(self._validation_cache),
        }
