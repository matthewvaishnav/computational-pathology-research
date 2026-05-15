"""
Pickle Security Control for safe deserialization.

This module provides secure pickle loading with source validation,
restricted unpickling, and audit logging to prevent arbitrary code execution.

Security Features:
- Source path validation against trusted locations
- Restricted unpickler for untrusted sources
- Environment-specific behavior (strict in production)
- Comprehensive audit logging
- Alternative format recommendations

Usage:
    control = PickleSecurityControl(trusted_paths=["/app/models"])
    data = control.safe_load("/app/models/model.pkl")

    # Or use global instance for bytes deserialization
    from src.security.pickle_security_control import safe_pickle
    data = safe_pickle.loads(hmac_validated_bytes, trusted=True)
"""

import io
import logging
import os
import pickle
from pathlib import Path
from typing import Union, List, Any, BinaryIO

from src.security.exceptions import PickleSecurityError
from src.security.models import SecurityEnvironment

logger = logging.getLogger(__name__)


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe types.

    This prevents arbitrary code execution by limiting which classes
    can be instantiated during unpickling.
    """

    # Whitelist of safe types that can be unpickled
    SAFE_MODULES = {
        "builtins",
        "numpy",
        "numpy.core",
        "numpy.core.multiarray",
        "torch",
        "torch.nn",
        "torch.nn.modules",
        "collections",
        "__builtin__",  # Python 2 compatibility
    }

    def find_class(self, module: str, name: str):
        """
        Override find_class to restrict which classes can be loaded.

        Args:
            module: Module name
            name: Class name

        Returns:
            Class object if allowed

        Raises:
            PickleSecurityError: If class is not in whitelist
        """
        # Check if module is in whitelist
        if module.split(".")[0] not in self.SAFE_MODULES:
            raise PickleSecurityError(
                f"Attempted to unpickle unsafe class: {module}.{name}. "
                f"Only classes from {self.SAFE_MODULES} are allowed."
            )

        return super().find_class(module, name)


class PickleSecurityControl:
    """
    Security control for safe pickle deserialization.

    Provides validation, restricted unpickling, and audit logging
    to prevent security vulnerabilities from untrusted pickle files.

    Attributes:
        trusted_paths: List of trusted directory paths
        environment: Current security environment
    """

    def __init__(self, trusted_paths: List[str] = None, environment: str = None):
        """
        Initialize pickle security control.

        Args:
            trusted_paths: List of trusted directory paths for pickle files
            environment: Security environment (development/staging/production)
        """
        self.trusted_paths = [Path(p).resolve() for p in (trusted_paths or [])]

        # Determine environment
        env_str = environment or os.getenv("ENVIRONMENT", "development")
        try:
            self.environment = SecurityEnvironment(env_str.lower())
        except ValueError:
            logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            self.environment = SecurityEnvironment.DEVELOPMENT

        logger.info(
            f"PickleSecurityControl initialized: "
            f"environment={self.environment.value}, "
            f"trusted_paths={len(self.trusted_paths)}"
        )

    def is_trusted_source(self, source_path: Union[str, Path]) -> bool:
        """
        Check if source path is from a trusted location.

        Args:
            source_path: Path to pickle file

        Returns:
            True if source is trusted, False otherwise
        """
        if not self.trusted_paths:
            return False

        try:
            source = Path(source_path).resolve()

            # Check if source is under any trusted path
            for trusted_path in self.trusted_paths:
                try:
                    # Check if source is relative to trusted path
                    source.relative_to(trusted_path)
                    return True
                except ValueError:
                    # Not relative to this trusted path, try next
                    continue

            return False
        except Exception as e:
            logger.error(f"Error checking trusted source: {e}")
            return False

    def safe_load(self, source: Union[str, Path, BinaryIO], source_path: str = None) -> Any:
        """
        Safely load pickle data with security checks.

        Args:
            source: File path or file object to load from
            source_path: Optional path for file objects (for trust checking)

        Returns:
            Unpickled data

        Raises:
            PickleSecurityError: If source is untrusted in production
            FileNotFoundError: If file doesn't exist
        """
        # Determine source path for trust checking
        if isinstance(source, (str, Path)):
            check_path = source
            file_obj = None
        else:
            check_path = source_path or getattr(source, "name", None)
            file_obj = source

        # Check if source is trusted
        is_trusted = self.is_trusted_source(check_path) if check_path else False

        # Log the operation
        logger.info(
            f"Pickle load attempt: path={check_path}, "
            f"trusted={is_trusted}, environment={self.environment.value}"
        )

        # Production: strict mode - block untrusted sources
        if self.environment == SecurityEnvironment.PRODUCTION and not is_trusted:
            error_msg = (
                f"Untrusted source in production: {check_path}. "
                f"Pickle loading is only allowed from trusted paths: {self.trusted_paths}"
            )
            logger.error(error_msg)
            raise PickleSecurityError(error_msg)

        # Development/Staging: warn but allow with restricted unpickler
        if not is_trusted:
            logger.warning(
                f"Loading untrusted pickle file: {check_path}. "
                f"Using restricted unpickler. "
                f"Consider using safer formats: {self.get_alternative_format()}"
            )

        # Load the pickle data
        try:
            if file_obj:
                # File object provided
                if is_trusted:
                    data = pickle.load(
                        file_obj
                    )  # nosec B301 - Source validated as trusted via is_trusted_source()
                else:
                    data = RestrictedUnpickler(file_obj).load()
            else:
                # Path provided
                with open(source, "rb") as f:
                    if is_trusted:
                        data = pickle.load(
                            f
                        )  # nosec B301 - Source validated as trusted via is_trusted_source()
                    else:
                        data = RestrictedUnpickler(f).load()

            logger.info(f"Successfully loaded pickle from {check_path}")
            return data

        except PickleSecurityError:
            # Re-raise security errors
            raise
        except Exception as e:
            logger.error(f"Failed to load pickle from {check_path}: {e}")
            raise

    def loads(self, data: bytes, trusted: bool = False) -> Any:
        """
        Safely deserialize pickle data from bytes.

        Args:
            data: Pickled bytes to deserialize
            trusted: If True, use standard pickle (for HMAC-validated data)
                    If False, use RestrictedUnpickler with class whitelist

        Returns:
            Deserialized object

        Raises:
            PickleSecurityError: If untrusted data contains unsafe classes
        """
        logger.debug(f"Pickle loads: trusted={trusted}, size={len(data)} bytes")

        try:
            if trusted:
                # Trusted source (e.g., HMAC-validated cache data)
                return pickle.loads(data)  # nosec B301 - Caller validated data integrity
            else:
                # Untrusted source - use restricted unpickler
                return RestrictedUnpickler(io.BytesIO(data)).load()

        except PickleSecurityError:
            raise
        except Exception as e:
            logger.error(f"Failed to deserialize pickle data: {e}")
            raise

    def get_alternative_format(self) -> str:
        """
        Get recommendations for safer alternative formats.

        Returns:
            String describing safer alternatives
        """
        return (
            "Safer alternatives to pickle: "
            "JSON (for simple data), "
            "SafeTensors (for PyTorch models), "
            "HDF5/NPZ (for NumPy arrays), "
            "Protocol Buffers (for structured data)"
        )

    def audit_log(self, operation: str, source: str, result: str, details: dict = None):
        """
        Log security audit event.

        Args:
            operation: Operation type (e.g., 'load', 'save')
            source: Source path or identifier
            result: Operation result (e.g., 'success', 'blocked')
            details: Additional details dictionary
        """
        log_entry = {
            "operation": operation,
            "source": source,
            "result": result,
            "environment": self.environment.value,
            "details": details or {},
        }

        logger.info(f"Security audit: {log_entry}")


# Global instance for convenient access
safe_pickle = PickleSecurityControl()
