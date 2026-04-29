"""
Model Security Module for Real-Time WSI Streaming

This module implements comprehensive model security including integrity verification,
secure storage and distribution, and protection against adversarial attacks.

**Validates: Requirements 9.1.3**
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config_manager import StreamingConfig


class SecurityThreatType(Enum):
    """Types of security threats to models."""

    MODEL_TAMPERING = "model_tampering"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class SecurityLevel(Enum):
    """Security levels for model operations."""

    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelSignature:
    """Digital signature for model integrity verification."""

    model_hash: str
    signature: str
    timestamp: datetime
    signer_id: str
    algorithm: str
    key_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary."""
        return {
            "model_hash": self.model_hash,
            "signature": self.signature,
            "timestamp": self.timestamp.isoformat(),
            "signer_id": self.signer_id,
            "algorithm": self.algorithm,
            "key_fingerprint": self.key_fingerprint,
        }


@dataclass
class SecurityAuditLog:
    """Security audit log entry."""

    event_id: str
    event_type: str
    threat_type: Optional[SecurityThreatType]
    severity: str
    description: str
    user_id: Optional[str]
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "threat_type": self.threat_type.value if self.threat_type else None,
            "severity": self.severity,
            "description": self.description,
            "user_id": self.user_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AdversarialDetectionResult:
    """Result from adversarial attack detection."""

    is_adversarial: bool
    confidence_score: float
    detection_method: str
    threat_indicators: List[str]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ModelSecurityManager:
    """
    Comprehensive model security management system.

    This class implements security measures including:
    - Model integrity verification and digital signing
    - Secure model storage and distribution
    - Adversarial attack detection and mitigation
    - Security audit logging and monitoring
    """

    def __init__(self, config: StreamingConfig, security_config: Optional[Dict] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Security configuration
        self.security_config = security_config or {
            "security_level": SecurityLevel.HIGH,
            "enable_model_signing": True,
            "enable_adversarial_detection": True,
            "key_rotation_days": 90,
            "audit_retention_days": 365,
            "encryption_algorithm": "AES-256-GCM",
        }

        # Cryptographic keys
        self.signing_key: Optional[rsa.RSAPrivateKey] = None
        self.verification_key: Optional[rsa.RSAPublicKey] = None
        self.encryption_key: Optional[bytes] = None

        # Security state
        self.model_signatures: Dict[str, ModelSignature] = {}
        self.audit_logs: List[SecurityAuditLog] = []
        self.threat_detection_enabled = True

        # Initialize security infrastructure
        self._initialize_security()

    def _initialize_security(self) -> None:
        """Initialize security infrastructure."""
        try:
            # Generate or load cryptographic keys
            self._setup_cryptographic_keys()

            # Initialize audit logging
            self._setup_audit_logging()

            self.logger.info("Model security manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize security manager: {e}")
            raise

    def _setup_cryptographic_keys(self) -> None:
        """Setup cryptographic keys for signing and encryption."""
        # Generate RSA key pair for model signing
        self.signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.verification_key = self.signing_key.public_key()

        # Generate encryption key for secure storage
        self.encryption_key = Fernet.generate_key()

        self.logger.info("Cryptographic keys generated successfully")

    def _setup_audit_logging(self) -> None:
        """Setup security audit logging."""
        # Create audit log directory
        audit_dir = Path("security_logs")
        audit_dir.mkdir(exist_ok=True)

        # Log security initialization
        self._log_security_event(
            event_type="security_initialization",
            description="Model security manager initialized",
            severity="INFO",
        )

    def sign_model(self, model_path: str, signer_id: str) -> ModelSignature:
        """
        Create digital signature for model integrity verification.

        **Validates: Requirements 9.1.3.1**

        Args:
            model_path: Path to model file to sign
            signer_id: ID of the entity signing the model

        Returns:
            ModelSignature containing signature and metadata
        """
        try:
            # Calculate model hash
            model_hash = self._calculate_model_hash(model_path)

            # Create signature
            signature = self._create_digital_signature(model_hash)

            # Generate key fingerprint
            key_fingerprint = self._get_key_fingerprint(self.verification_key)

            # Create signature object
            model_signature = ModelSignature(
                model_hash=model_hash,
                signature=signature,
                timestamp=datetime.now(),
                signer_id=signer_id,
                algorithm="RSA-PSS-SHA256",
                key_fingerprint=key_fingerprint,
            )

            # Store signature
            self.model_signatures[model_path] = model_signature

            # Log signing event
            self._log_security_event(
                event_type="model_signing",
                description=f"Model signed: {model_path}",
                severity="INFO",
                metadata={"model_path": model_path, "signer_id": signer_id},
            )

            self.logger.info(f"Model signed successfully: {model_path}")
            return model_signature

        except Exception as e:
            self.logger.error(f"Failed to sign model {model_path}: {e}")
            self._log_security_event(
                event_type="model_signing_failure",
                description=f"Failed to sign model: {model_path}",
                severity="ERROR",
                metadata={"model_path": model_path, "error": str(e)},
            )
            raise

    def verify_model_integrity(self, model_path: str) -> bool:
        """
        Verify model integrity using digital signature.

        Args:
            model_path: Path to model file to verify

        Returns:
            True if model integrity is verified, False otherwise
        """
        try:
            # Check if signature exists
            if model_path not in self.model_signatures:
                self.logger.warning(f"No signature found for model: {model_path}")
                self._log_security_event(
                    event_type="signature_missing",
                    threat_type=SecurityThreatType.MODEL_TAMPERING,
                    description=f"No signature found for model: {model_path}",
                    severity="WARNING",
                    metadata={"model_path": model_path},
                )
                return False

            signature_obj = self.model_signatures[model_path]

            # Calculate current model hash
            current_hash = self._calculate_model_hash(model_path)

            # Verify hash matches signature
            if current_hash != signature_obj.model_hash:
                self.logger.error(f"Model hash mismatch for {model_path}")
                self._log_security_event(
                    event_type="model_tampering_detected",
                    threat_type=SecurityThreatType.MODEL_TAMPERING,
                    description=f"Model hash mismatch detected: {model_path}",
                    severity="CRITICAL",
                    metadata={
                        "model_path": model_path,
                        "expected_hash": signature_obj.model_hash,
                        "actual_hash": current_hash,
                    },
                )
                return False

            # Verify digital signature
            is_valid = self._verify_digital_signature(current_hash, signature_obj.signature)

            if is_valid:
                self.logger.info(f"Model integrity verified: {model_path}")
                self._log_security_event(
                    event_type="model_verification_success",
                    description=f"Model integrity verified: {model_path}",
                    severity="INFO",
                    metadata={"model_path": model_path},
                )
            else:
                self.logger.error(f"Invalid signature for model: {model_path}")
                self._log_security_event(
                    event_type="invalid_signature",
                    threat_type=SecurityThreatType.MODEL_TAMPERING,
                    description=f"Invalid signature detected: {model_path}",
                    severity="CRITICAL",
                    metadata={"model_path": model_path},
                )

            return is_valid

        except Exception as e:
            self.logger.error(f"Failed to verify model integrity {model_path}: {e}")
            self._log_security_event(
                event_type="verification_failure",
                description=f"Failed to verify model: {model_path}",
                severity="ERROR",
                metadata={"model_path": model_path, "error": str(e)},
            )
            return False

    def encrypt_model(self, model_path: str, output_path: str) -> bool:
        """
        Encrypt model for secure storage and distribution.

        **Validates: Requirements 9.1.3.2**

        Args:
            model_path: Path to model file to encrypt
            output_path: Path for encrypted model output

        Returns:
            True if encryption successful, False otherwise
        """
        try:
            # Create Fernet cipher
            cipher = Fernet(self.encryption_key)

            # Read model file
            with open(model_path, "rb") as f:
                model_data = f.read()

            # Encrypt model data
            encrypted_data = cipher.encrypt(model_data)

            # Write encrypted model
            with open(output_path, "wb") as f:
                f.write(encrypted_data)

            # Log encryption event
            self._log_security_event(
                event_type="model_encryption",
                description=f"Model encrypted: {model_path} -> {output_path}",
                severity="INFO",
                metadata={"source_path": model_path, "encrypted_path": output_path},
            )

            self.logger.info(f"Model encrypted successfully: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to encrypt model {model_path}: {e}")
            self._log_security_event(
                event_type="encryption_failure",
                description=f"Failed to encrypt model: {model_path}",
                severity="ERROR",
                metadata={"model_path": model_path, "error": str(e)},
            )
            return False

    def decrypt_model(self, encrypted_path: str, output_path: str) -> bool:
        """
        Decrypt model for loading and use.

        Args:
            encrypted_path: Path to encrypted model file
            output_path: Path for decrypted model output

        Returns:
            True if decryption successful, False otherwise
        """
        try:
            # Create Fernet cipher
            cipher = Fernet(self.encryption_key)

            # Read encrypted model
            with open(encrypted_path, "rb") as f:
                encrypted_data = f.read()

            # Decrypt model data
            decrypted_data = cipher.decrypt(encrypted_data)

            # Write decrypted model
            with open(output_path, "wb") as f:
                f.write(decrypted_data)

            # Log decryption event
            self._log_security_event(
                event_type="model_decryption",
                description=f"Model decrypted: {encrypted_path} -> {output_path}",
                severity="INFO",
                metadata={"encrypted_path": encrypted_path, "decrypted_path": output_path},
            )

            self.logger.info(f"Model decrypted successfully: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to decrypt model {encrypted_path}: {e}")
            self._log_security_event(
                event_type="decryption_failure",
                description=f"Failed to decrypt model: {encrypted_path}",
                severity="ERROR",
                metadata={"encrypted_path": encrypted_path, "error": str(e)},
            )
            return False

    def detect_adversarial_attack(
        self, input_tensor: torch.Tensor, model_output: torch.Tensor, confidence: float
    ) -> AdversarialDetectionResult:
        """
        Detect potential adversarial attacks on model inputs.

        **Validates: Requirements 9.1.3.3**

        Args:
            input_tensor: Input tensor to analyze
            model_output: Model output for the input
            confidence: Model confidence score

        Returns:
            AdversarialDetectionResult with detection results
        """
        threat_indicators = []
        detection_methods = []

        # Statistical anomaly detection
        if self._detect_statistical_anomalies(input_tensor):
            threat_indicators.append("statistical_anomaly")
            detection_methods.append("statistical_analysis")

        # Confidence-based detection
        if self._detect_confidence_anomalies(confidence, model_output):
            threat_indicators.append("confidence_anomaly")
            detection_methods.append("confidence_analysis")

        # Gradient-based detection
        if self._detect_gradient_anomalies(input_tensor):
            threat_indicators.append("gradient_anomaly")
            detection_methods.append("gradient_analysis")

        # Calculate overall threat score
        threat_score = len(threat_indicators) / 3.0  # Normalize to 0-1
        is_adversarial = threat_score > 0.5  # Threshold for adversarial detection

        # Generate recommendations
        recommended_actions = []
        if is_adversarial:
            recommended_actions.extend(
                [
                    "Reject input and request new sample",
                    "Apply input preprocessing/denoising",
                    "Use ensemble prediction for verification",
                    "Log incident for security analysis",
                ]
            )

        result = AdversarialDetectionResult(
            is_adversarial=is_adversarial,
            confidence_score=1.0 - threat_score,  # Invert for confidence
            detection_method=", ".join(detection_methods) if detection_methods else "none",
            threat_indicators=threat_indicators,
            recommended_actions=recommended_actions,
        )

        # Log detection result
        if is_adversarial:
            self._log_security_event(
                event_type="adversarial_attack_detected",
                threat_type=SecurityThreatType.ADVERSARIAL_ATTACK,
                description=f"Adversarial attack detected (score: {threat_score:.3f})",
                severity="WARNING",
                metadata={
                    "threat_score": threat_score,
                    "threat_indicators": threat_indicators,
                    "detection_methods": detection_methods,
                },
            )

        return result

    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA-256 hash of model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _create_digital_signature(self, data_hash: str) -> str:
        """Create digital signature for data hash."""
        # Convert hash to bytes
        hash_bytes = data_hash.encode("utf-8")

        # Sign with RSA-PSS
        signature = self.signing_key.sign(
            hash_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # Encode signature as base64
        return base64.b64encode(signature).decode("utf-8")

    def _verify_digital_signature(self, data_hash: str, signature: str) -> bool:
        """Verify digital signature."""
        try:
            # Decode signature from base64
            signature_bytes = base64.b64decode(signature.encode("utf-8"))

            # Convert hash to bytes
            hash_bytes = data_hash.encode("utf-8")

            # Verify signature
            self.verification_key.verify(
                signature_bytes,
                hash_bytes,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )

            return True

        except Exception:
            return False

    def _get_key_fingerprint(self, public_key: rsa.RSAPublicKey) -> str:
        """Generate fingerprint for public key."""
        # Serialize public key
        key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        # Calculate SHA-256 hash
        fingerprint = hashlib.sha256(key_bytes).hexdigest()

        # Format as fingerprint (colon-separated pairs)
        return ":".join(fingerprint[i : i + 2] for i in range(0, len(fingerprint), 2))

    def _detect_statistical_anomalies(self, input_tensor: torch.Tensor) -> bool:
        """Detect statistical anomalies in input tensor."""
        # Convert to numpy for analysis
        data = input_tensor.detach().cpu().numpy()

        # Check for unusual statistical properties
        mean_val = np.mean(data)
        std_val = np.std(data)

        # Detect extreme values (beyond 3 standard deviations)
        z_scores = np.abs((data - mean_val) / (std_val + 1e-8))
        extreme_values = np.sum(z_scores > 3) / data.size

        # Flag if more than 5% of values are extreme
        return extreme_values > 0.05

    def _detect_confidence_anomalies(self, confidence: float, model_output: torch.Tensor) -> bool:
        """Detect confidence-based anomalies."""
        # Check for suspiciously high confidence with ambiguous output
        output_entropy = self._calculate_entropy(model_output)

        # High confidence but high entropy suggests adversarial input
        return confidence > 0.95 and output_entropy > 0.5

    def _detect_gradient_anomalies(self, input_tensor: torch.Tensor) -> bool:
        """Detect gradient-based anomalies."""
        # Calculate input gradients (simplified detection)
        if input_tensor.requires_grad:
            gradients = torch.autograd.grad(
                outputs=input_tensor.sum(),
                inputs=input_tensor,
                create_graph=False,
                retain_graph=False,
            )[0]

            # Check for unusual gradient patterns
            grad_norm = torch.norm(gradients)
            return grad_norm > 10.0  # Threshold for unusual gradients

        return False

    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate entropy of tensor values."""
        # Convert to probabilities
        probs = torch.softmax(tensor.flatten(), dim=0)

        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()

    def _log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str,
        threat_type: Optional[SecurityThreatType] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log security event to audit trail."""
        event_id = f"sec_{int(time.time())}_{len(self.audit_logs)}"

        audit_log = SecurityAuditLog(
            event_id=event_id,
            event_type=event_type,
            threat_type=threat_type,
            severity=severity,
            description=description,
            user_id=user_id,
            model_version=getattr(self.config, "model_version", "1.0.0"),
            metadata=metadata or {},
        )

        self.audit_logs.append(audit_log)

        # Write to audit log file
        self._write_audit_log(audit_log)

    def _write_audit_log(self, audit_log: SecurityAuditLog) -> None:
        """Write audit log entry to file."""
        try:
            audit_file = Path("security_logs") / f"audit_{datetime.now().strftime('%Y%m%d')}.json"

            # Append to daily audit log file
            with open(audit_file, "a") as f:
                f.write(json.dumps(audit_log.to_dict()) + "\n")

        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and statistics."""
        # Calculate recent threat activity
        recent_threats = [
            log for log in self.audit_logs[-100:] if log.threat_type is not None  # Last 100 events
        ]

        threat_counts = {}
        for threat in recent_threats:
            threat_type = threat.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1

        return {
            "security_level": self.security_config["security_level"].value,
            "model_signing_enabled": self.security_config["enable_model_signing"],
            "adversarial_detection_enabled": self.security_config["enable_adversarial_detection"],
            "signed_models": len(self.model_signatures),
            "total_audit_events": len(self.audit_logs),
            "recent_threats": threat_counts,
            "key_fingerprint": (
                self._get_key_fingerprint(self.verification_key) if self.verification_key else None
            ),
            "last_key_rotation": None,  # Would track actual key rotation
            "security_status": "active",
        }

    def export_audit_logs(self, filepath: str, days: int = 30) -> None:
        """Export audit logs for the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_logs = [log for log in self.audit_logs if log.timestamp >= cutoff_date]

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "export_period_days": days,
            "total_events": len(recent_logs),
            "audit_logs": [log.to_dict() for log in recent_logs],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Audit logs exported to {filepath}")


# Example usage and testing functions
def run_model_security_demo():
    """Run model security demonstration."""
    config = StreamingConfig(
        tile_size=1024,
        batch_size=32,
        memory_budget_gb=2.0,
        target_time=30.0,
        confidence_threshold=0.95,
    )

    security_manager = ModelSecurityManager(config)

    print("Starting Model Security Demo...")

    # Create a dummy model file for testing
    dummy_model_path = "dummy_model.pth"
    with open(dummy_model_path, "wb") as f:
        f.write(b"dummy model data for testing")

    try:
        # 1. Sign model
        print("\n1. Signing model...")
        signature = security_manager.sign_model(dummy_model_path, "security_demo")
        print(f"Model signed successfully")
        print(f"  Hash: {signature.model_hash[:16]}...")
        print(f"  Signer: {signature.signer_id}")
        print(f"  Algorithm: {signature.algorithm}")

        # 2. Verify model integrity
        print("\n2. Verifying model integrity...")
        is_valid = security_manager.verify_model_integrity(dummy_model_path)
        print(f"Model integrity verified: {is_valid}")

        # 3. Encrypt model
        print("\n3. Encrypting model...")
        encrypted_path = "dummy_model_encrypted.pth"
        success = security_manager.encrypt_model(dummy_model_path, encrypted_path)
        print(f"Model encryption successful: {success}")

        # 4. Decrypt model
        print("\n4. Decrypting model...")
        decrypted_path = "dummy_model_decrypted.pth"
        success = security_manager.decrypt_model(encrypted_path, decrypted_path)
        print(f"Model decryption successful: {success}")

        # 5. Test adversarial detection
        print("\n5. Testing adversarial detection...")
        # Create synthetic input tensor
        input_tensor = torch.randn(1, 3, 224, 224)
        model_output = torch.randn(1, 2)
        confidence = 0.95

        detection_result = security_manager.detect_adversarial_attack(
            input_tensor, model_output, confidence
        )
        print(f"Adversarial attack detected: {detection_result.is_adversarial}")
        print(f"Detection confidence: {detection_result.confidence_score:.3f}")
        print(f"Threat indicators: {detection_result.threat_indicators}")

        # 6. Get security status
        print("\n6. Security status...")
        status = security_manager.get_security_status()
        print(f"Security level: {status['security_level']}")
        print(f"Signed models: {status['signed_models']}")
        print(f"Total audit events: {status['total_audit_events']}")
        print(f"Recent threats: {status['recent_threats']}")

        # 7. Export audit logs
        print("\n7. Exporting audit logs...")
        security_manager.export_audit_logs("security_audit_demo.json")
        print("Audit logs exported successfully")

    finally:
        # Clean up test files
        for filepath in [dummy_model_path, encrypted_path, decrypted_path]:
            if os.path.exists(filepath):
                os.remove(filepath)

    print("\nModel Security Demo Complete!")
    return security_manager


if __name__ == "__main__":
    # Run the security demo
    run_model_security_demo()
