"""
Mutual authentication for federated learning.

Implements certificate-based mutual authentication between
coordinator and clients with role-based access control.
"""

import hashlib
import logging
import time
from typing import Dict, Optional, Set

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication-related errors."""

    pass


class CertificateValidator:
    """Validates X.509 certificates for mutual authentication."""

    def __init__(self, ca_cert_pem: bytes):
        """
        Initialize certificate validator.

        Args:
            ca_cert_pem: CA certificate in PEM format
        """
        self.ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
        self.ca_public_key = self.ca_cert.public_key()

    def validate_certificate(self, cert_pem: bytes) -> Dict:
        """
        Validate certificate against CA and extract identity.

        Args:
            cert_pem: Certificate to validate in PEM format

        Returns:
            Dict with validation result and identity info

        Raises:
            AuthenticationError: If validation fails
        """
        import secrets
        
        # Constant-time validation to prevent timing attacks
        validation_start = time.time()
        is_valid = True
        error_msg = "Certificate validation failed"
        
        try:
            cert = x509.load_pem_x509_certificate(cert_pem)

            # Check certificate signature
            from cryptography.hazmat.primitives.asymmetric import padding

            try:
                self.ca_public_key.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    cert.signature_hash_algorithm,
                )
            except Exception:
                is_valid = False
                error_msg = "Certificate signature invalid"

            # Check validity period (with 1 second tolerance for clock skew)
            now = time.time()
            not_before = cert.not_valid_before.timestamp() - 1.0
            not_after = cert.not_valid_after.timestamp()

            if now < not_before or now > not_after:
                is_valid = False
                error_msg = "Certificate expired or not yet valid"

            # Extract identity information
            subject = cert.subject
            common_name = None
            organization = None

            for attribute in subject:
                if attribute.oid == x509.NameOID.COMMON_NAME:
                    common_name = attribute.value
                elif attribute.oid == x509.NameOID.ORGANIZATION_NAME:
                    organization = attribute.value

            # Validate common name format
            if common_name and not self._is_valid_common_name(common_name):
                is_valid = False
                error_msg = "Invalid certificate common name"

            # Compute certificate fingerprint
            cert_der = cert.public_bytes(serialization.Encoding.DER)
            fingerprint = hashlib.sha256(cert_der).hexdigest()

            # Constant-time delay to prevent timing attacks
            elapsed = time.time() - validation_start
            if elapsed < 0.1:  # Minimum 100ms validation time
                time.sleep(0.1 - elapsed + secrets.randbelow(50) / 1000.0)

            if not is_valid:
                raise AuthenticationError(error_msg)

            return {
                "valid": True,
                "common_name": common_name,
                "organization": organization,
                "serial_number": str(cert.serial_number),
                "fingerprint": fingerprint,
                "not_before": cert.not_valid_before,
                "not_after": cert.not_valid_after,
            }

        except AuthenticationError:
            raise
        except Exception:
            # Constant-time delay even for exceptions
            elapsed = time.time() - validation_start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed + secrets.randbelow(50) / 1000.0)
            raise AuthenticationError("Certificate validation failed")

    def _is_valid_common_name(self, common_name: str) -> bool:
        """Validate common name format to prevent injection attacks."""
        import re
        # Allow alphanumeric, hyphens, underscores, dots
        pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, common_name)) and len(common_name) <= 64


class RoleBasedAccessControl:
    """Role-based access control for FL operations."""

    def __init__(self):
        """Initialize RBAC system."""
        self.roles: Dict[str, Set[str]] = {
            "coordinator": {
                "start_round",
                "aggregate_updates",
                "broadcast_model",
                "manage_clients",
                "view_metrics",
            },
            "client": {"get_model", "submit_update", "get_round_status", "register"},
            "observer": {"view_metrics", "get_round_status"},
        }

        # Client ID to role mapping
        self.client_roles: Dict[str, str] = {}

        # Coordinator identities
        self.coordinators: Set[str] = set()

    def assign_role(self, client_id: str, role: str):
        """
        Assign role to client.

        Args:
            client_id: Client identifier
            role: Role name
        """
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")

        self.client_roles[client_id] = role

        if role == "coordinator":
            self.coordinators.add(client_id)

        logger.info(f"Assigned role '{role}' to client '{client_id}'")

    def check_permission(self, client_id: str, operation: str) -> bool:
        """
        Check if client has permission for operation.

        Args:
            client_id: Client identifier
            operation: Operation name

        Returns:
            True if permitted
        """
        role = self.client_roles.get(client_id)
        if not role:
            return False

        permissions = self.roles.get(role, set())
        return operation in permissions

    def is_coordinator(self, client_id: str) -> bool:
        """Check if client is a coordinator."""
        return client_id in self.coordinators


class MutualAuthInterceptor(grpc.ServerInterceptor):
    """gRPC server interceptor for mutual authentication."""

    def __init__(self, ca_cert_pem: bytes, rbac: RoleBasedAccessControl):
        """
        Initialize auth interceptor.

        Args:
            ca_cert_pem: CA certificate for validation
            rbac: Role-based access control system
        """
        self.validator = CertificateValidator(ca_cert_pem)
        self.rbac = rbac
        
        # Rate limiting for authentication attempts
        self._auth_attempts = {}  # client_ip -> (count, last_attempt)
        self._max_attempts = 5
        self._lockout_duration = 300  # 5 minutes

        # Method to operation mapping
        self.method_operations = {
            "/federated_learning.FederatedLearningService/RegisterClient": "register",
            "/federated_learning.FederatedLearningService/GetGlobalModel": "get_model",
            "/federated_learning.FederatedLearningService/SubmitUpdate": "submit_update",
            "/federated_learning.FederatedLearningService/GetRoundStatus": "get_round_status",
            "/federated_learning.FederatedLearningService/StartRound": "start_round",
        }

    def _validate_client_id(self, client_id: str) -> bool:
        """Validate client ID format to prevent injection attacks."""
        import re
        if not client_id or len(client_id) > 64:
            return False
        # Allow alphanumeric, hyphens, underscores
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, client_id))

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        
        if client_ip in self._auth_attempts:
            count, last_attempt = self._auth_attempts[client_ip]
            
            # Reset counter if lockout period expired
            if now - last_attempt > self._lockout_duration:
                self._auth_attempts[client_ip] = (1, now)
                return True
            
            # Check if exceeded max attempts
            if count >= self._max_attempts:
                return False
            
            # Increment counter
            self._auth_attempts[client_ip] = (count + 1, now)
        else:
            self._auth_attempts[client_ip] = (1, now)
        
        return True

    def intercept_service(self, continuation, handler_call_details):
        """Intercept gRPC service calls for authentication."""

        def auth_wrapper(request, context):
            try:
                # Get client IP for rate limiting
                peer = context.peer()
                client_ip = peer.split(':')[1] if ':' in peer else 'unknown'
                
                # Check rate limiting
                if not self._check_rate_limit(client_ip):
                    context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                    context.set_details("Too many authentication attempts")
                    return

                # Extract client certificate from TLS context
                auth_context = context.auth_context()
                peer_identity = None

                # Get peer certificate
                for key, value in auth_context.items():
                    if key == "x509_common_name":
                        peer_identity = value[0].decode() if value else None
                        break

                if not peer_identity:
                    context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                    context.set_details("Authentication required")
                    return

                # Validate peer identity format
                if not self._validate_client_id(peer_identity):
                    context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                    context.set_details("Invalid client identifier")
                    return

                # Get operation from method name
                method = handler_call_details.method
                operation = self.method_operations.get(method, "unknown")

                # Special handling for registration
                if operation == "register":
                    # Extract and validate client_id from request
                    client_id = getattr(request, "client_id", None)
                    if not client_id or not self._validate_client_id(client_id):
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("Invalid client ID format")
                        return
                    
                    # Verify client_id matches certificate identity
                    if client_id != peer_identity:
                        context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                        context.set_details("Client ID mismatch")
                        return

                    # Auto-assign client role for new registrations
                    if client_id not in self.rbac.client_roles:
                        # Determine role based on client_id pattern
                        if client_id.startswith("coordinator"):
                            self.rbac.assign_role(client_id, "coordinator")
                        else:
                            self.rbac.assign_role(client_id, "client")

                # Check permissions
                if not self.rbac.check_permission(peer_identity, operation):
                    context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                    context.set_details("Insufficient permissions")
                    return

                # Add client identity to context for use in handlers
                context.set_trailing_metadata([("client_id", peer_identity)])

                # Call original handler
                return continuation(request, context)

            except Exception as e:
                logger.error(f"Authentication error for {client_ip}: {type(e).__name__}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Authentication error")
                return

        return grpc.unary_unary_rpc_method_handler(auth_wrapper)


class AuthenticatedFLServer:
    """FL server with mutual authentication."""

    def __init__(
        self, servicer, port: int = 50051, cert_dir: str = "./certs", max_workers: int = 10
    ):
        """
        Initialize authenticated FL server.

        Args:
            servicer: FL service implementation
            port: Server port
            cert_dir: Certificate directory
            max_workers: Max worker threads
        """
        from concurrent import futures

        from .tls_utils import TLSManager

        self.servicer = servicer
        self.port = port

        # Initialize TLS manager
        self.tls_manager = TLSManager(cert_dir)
        self.tls_manager.setup_certificates()

        # Initialize RBAC
        self.rbac = RoleBasedAccessControl()

        # Load CA certificate for validation
        ca_cert_path = f"{cert_dir}/ca-cert.pem"
        with open(ca_cert_path, "rb") as f:
            ca_cert_pem = f.read()

        # Create auth interceptor
        auth_interceptor = MutualAuthInterceptor(ca_cert_pem, self.rbac)

        # Initialize server with auth interceptor
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers), interceptors=[auth_interceptor]
        )

        # Add servicer
        from .federated_learning_pb2_grpc import add_FederatedLearningServiceServicer_to_server

        add_FederatedLearningServiceServicer_to_server(servicer, self.server)

        # Add secure port
        credentials = self.tls_manager.create_server_credentials()
        self.server.add_secure_port(f"[::]:{port}", credentials)

        logger.info(f"Authenticated FL server initialized on port {port}")

    def add_coordinator(self, coordinator_id: str):
        """Add coordinator identity."""
        self.rbac.assign_role(coordinator_id, "coordinator")

    def add_client(self, client_id: str):
        """Add client identity."""
        self.rbac.assign_role(client_id, "client")

    def add_observer(self, observer_id: str):
        """Add observer identity."""
        self.rbac.assign_role(observer_id, "observer")

    def start(self):
        """Start the server."""
        self.server.start()
        logger.info(f"Authenticated FL server started on port {self.port}")

    def stop(self, grace_period: int = 5):
        """Stop the server."""
        self.server.stop(grace_period)

    def wait_for_termination(self):
        """Wait for server termination."""
        self.server.wait_for_termination()


class AuthenticatedFLClient:
    """FL client with mutual authentication."""

    def __init__(
        self,
        client_id: str,
        coordinator_host: str = "localhost",
        coordinator_port: int = 50051,
        cert_dir: str = "./certs",
    ):
        """
        Initialize authenticated FL client.

        Args:
            client_id: Client identifier
            coordinator_host: Coordinator hostname
            coordinator_port: Coordinator port
            cert_dir: Certificate directory
        """
        from .grpc_client import SecureFLClient

        self.client_id = client_id
        self.secure_client = SecureFLClient(client_id, coordinator_host, coordinator_port, cert_dir)

        # Load CA certificate for validation
        ca_cert_path = f"{cert_dir}/ca-cert.pem"
        with open(ca_cert_path, "rb") as f:
            ca_cert_pem = f.read()

        self.validator = CertificateValidator(ca_cert_pem)

        logger.info(f"Authenticated FL client {client_id} initialized")

    def validate_coordinator_certificate(self, cert_pem: bytes) -> bool:
        """
        Validate coordinator certificate.

        Args:
            cert_pem: Coordinator certificate

        Returns:
            True if valid
        """
        try:
            cert_info = self.validator.validate_certificate(cert_pem)

            # Check if it's a coordinator certificate
            if cert_info["organization"] != "HistoCore FL":
                return False

            # Additional coordinator-specific checks can be added here
            logger.info(f"Coordinator certificate validated: {cert_info['common_name']}")
            return True

        except AuthenticationError as e:
            logger.error(f"Coordinator certificate validation failed: {e}")
            return False

    def connect_and_authenticate(self) -> bool:
        """
        Connect to coordinator with mutual authentication.

        Returns:
            True if connection and authentication successful
        """
        try:
            # Connect using secure client
            if not self.secure_client.connect():
                return False

            # Register (this will trigger mutual auth)
            success = self.secure_client.register()

            if success:
                logger.info(f"Client {self.client_id} authenticated successfully")

            return success

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def __getattr__(self, name):
        """Delegate to secure client."""
        return getattr(self.secure_client, name)


def generate_client_certificates(
    ca_cert_pem: bytes, ca_key_pem: bytes, client_ids: list, cert_dir: str = "./certs"
) -> Dict[str, str]:
    """
    Generate certificates for multiple clients.

    Args:
        ca_cert_pem: CA certificate
        ca_key_pem: CA private key
        client_ids: List of client identifiers
        cert_dir: Certificate directory

    Returns:
        Dict mapping client_id to certificate path
    """
    import os
    import stat

    from .tls_utils import TLSManager

    tls_manager = TLSManager(cert_dir)
    cert_paths = {}

    # Validate all client IDs first
    for client_id in client_ids:
        if not client_id or len(client_id) > 64:
            raise ValueError(f"Invalid client ID: {client_id}")
        import re
        if not re.match(r'^[a-zA-Z0-9._-]+$', client_id):
            raise ValueError(f"Invalid client ID format: {client_id}")

    for client_id in client_ids:
        # Generate client certificate
        client_cert_pem, client_key_pem = tls_manager.generate_client_certificate(
            ca_cert_pem, ca_key_pem, client_id
        )

        # Save certificate with secure permissions
        cert_path = os.path.join(cert_dir, f"{client_id}-cert.pem")
        key_path = os.path.join(cert_dir, f"{client_id}-key.pem")

        # Write certificate (readable by owner and group)
        with open(cert_path, "wb") as f:
            f.write(client_cert_pem)
        os.chmod(cert_path, stat.S_IRUSR | stat.S_IRGRP)  # 0o440

        # Write private key (readable by owner only)
        with open(key_path, "wb") as f:
            f.write(client_key_pem)
        os.chmod(key_path, stat.S_IRUSR)  # 0o400

        cert_paths[client_id] = cert_path
        logger.info(f"Generated certificate for {client_id}")

    return cert_paths


if __name__ == "__main__":
    # Demo: Generate certificates for multiple clients
    from .tls_utils import TLSManager

    # Setup TLS
    tls_manager = TLSManager("./demo_certs")
    tls_manager.setup_certificates(force_regenerate=True)

    # Load CA
    with open("./demo_certs/ca-cert.pem", "rb") as f:
        ca_cert_pem = f.read()
    with open("./demo_certs/ca-key.pem", "rb") as f:
        ca_key_pem = f.read()

    # Generate client certificates
    client_ids = ["hospital_A", "hospital_B", "hospital_C", "coordinator_main"]
    cert_paths = generate_client_certificates(ca_cert_pem, ca_key_pem, client_ids, "./demo_certs")

    print("Generated certificates:")
    for client_id, path in cert_paths.items():
        print(f"  {client_id}: {path}")
