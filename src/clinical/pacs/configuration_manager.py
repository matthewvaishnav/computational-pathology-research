"""
Configuration Manager for PACS Integration System.

This module implements the ConfigurationManager class that manages PACS connection
settings, environment profiles, and configuration validation for multi-environment
support with encrypted configuration files.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .data_models import (
    PACSConfiguration,
    PACSEndpoint,
    SecurityConfig,
    PerformanceConfig,
    PACSVendor,
    ValidationResult,
    OperationResult,
)

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Manages PACS connection settings, environment profiles, and configuration validation.

    This class provides comprehensive configuration management including:
    - Loading PACS settings from encrypted configuration files
    - Multi-endpoint configuration support for redundancy
    - Configuration validation with detailed error reporting
    - Environment-specific configuration profiles
    - Fallback to default settings when configuration is corrupted
    """

    def __init__(
        self,
        config_directory: Union[str, Path] = "configs/pacs",
        encryption_key: Optional[bytes] = None,
    ):
        """
        Initialize Configuration Manager.

        Args:
            config_directory: Directory containing configuration files
            encryption_key: Optional encryption key for encrypted configs
        """
        self.config_directory = Path(config_directory)
        self.encryption_key = encryption_key
        self._loaded_configurations: Dict[str, PACSConfiguration] = {}
        self._default_config: Optional[PACSConfiguration] = None

        # Ensure config directory exists
        self.config_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"ConfigurationManager initialized with directory: {self.config_directory}")

    def load_configuration(
        self, profile: str = "default", encrypted: bool = False
    ) -> PACSConfiguration:
        """
        Load PACS configuration for specified profile.

        Args:
            profile: Configuration profile name (e.g., 'production', 'staging', 'development')
            encrypted: Whether the configuration file is encrypted

        Returns:
            PACSConfiguration object

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        logger.info(f"Loading configuration profile: {profile}")

        # Check cache first
        if profile in self._loaded_configurations:
            logger.debug(f"Returning cached configuration for profile: {profile}")
            return self._loaded_configurations[profile]

        # Determine config file path
        config_file = self.config_directory / f"{profile}.yaml"
        if encrypted:
            config_file = self.config_directory / f"{profile}.encrypted.yaml"

        if not config_file.exists():
            # Try fallback to unencrypted if encrypted not found
            if encrypted:
                fallback_file = self.config_directory / f"{profile}.yaml"
                if fallback_file.exists():
                    config_file = fallback_file
                    encrypted = False
                    logger.warning(
                        f"Encrypted config not found, using unencrypted: {fallback_file}"
                    )
                else:
                    raise FileNotFoundError(f"Configuration file not found: {config_file}")
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            # Load and parse configuration
            config_data = self._load_config_file(config_file, encrypted)
            pacs_config = self._parse_configuration(config_data, profile)

            # Validate configuration
            validation = self.validate_configuration(pacs_config)
            if not validation.is_valid:
                raise ValueError(f"Invalid configuration: {'; '.join(validation.errors)}")

            # Refuse insecure configs in production
            env = os.environ.get("ENVIRONMENT", "").lower()
            is_prod = env == "production" or profile == "production"
            if is_prod:
                audit_cfg = getattr(pacs_config, "audit_config", {}) or {}
                if not audit_cfg.get("hipaa_compliant", True):
                    raise ValueError(
                        "HIPAA compliance is disabled in audit_config but ENVIRONMENT=production. "
                        "Set audit_config.hipaa_compliant: true or use a non-production profile."
                    )
                if not audit_cfg.get("encryption_enabled", True):
                    raise ValueError(
                        "Audit encryption is disabled in audit_config but ENVIRONMENT=production. "
                        "Set audit_config.encryption_enabled: true or use a non-production profile."
                    )

            # Cache the configuration
            self._loaded_configurations[profile] = pacs_config

            logger.info(f"Successfully loaded configuration profile: {profile}")
            return pacs_config

        except Exception as e:
            logger.error(f"Failed to load configuration {profile}: {str(e)}")

            # Try to fall back to default configuration
            if profile != "default" and self._default_config:
                logger.warning(f"Falling back to default configuration due to error: {str(e)}")
                return self._default_config

            raise

    def validate_configuration(self, config: PACSConfiguration) -> ValidationResult:
        """
        Validate PACS configuration for correctness and completeness.

        Args:
            config: PACSConfiguration to validate

        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(is_valid=True)

        # Validate basic configuration structure
        config_errors = config.validate()
        for error in config_errors:
            result.add_error(error)

        # Validate each PACS endpoint
        for endpoint_id, endpoint in config.pacs_endpoints.items():
            try:
                # Endpoint validation is done in PACSEndpoint.__post_init__
                # Additional validation can be added here

                # Check for duplicate AE titles
                ae_titles = [ep.ae_title for ep in config.pacs_endpoints.values()]
                if len(ae_titles) != len(set(ae_titles)):
                    result.add_error("Duplicate AE titles found in configuration")

                # Validate security configuration
                sec_errors = endpoint.security_config.validate()
                for error in sec_errors:
                    result.add_error(f"Endpoint {endpoint_id}: {error}")

                # Validate performance configuration
                perf_errors = endpoint.performance_config.validate()
                for error in perf_errors:
                    result.add_error(f"Endpoint {endpoint_id}: {error}")

            except Exception as e:
                result.add_error(f"Endpoint {endpoint_id} validation failed: {str(e)}")

        # Validate storage configuration
        storage_config = config.storage_config
        if storage_config:
            cache_path = storage_config.get("local_cache_path")
            if cache_path:
                cache_dir = Path(cache_path)
                if not cache_dir.parent.exists():
                    result.add_error(f"Cache directory parent does not exist: {cache_dir.parent}")

            max_cache_size = storage_config.get("max_cache_size_gb", 0)
            cleanup_threshold = storage_config.get("cleanup_threshold_gb", 0)
            if cleanup_threshold >= max_cache_size:
                result.add_error("Cleanup threshold must be less than max cache size")

        # Validate notification configuration
        notification_config = config.notification_config
        if notification_config and notification_config.get("enabled", False):
            channels = notification_config.get("channels", {})

            # Check email configuration
            email_config = channels.get("email", {})
            if email_config.get("enabled", False):
                required_email_fields = ["smtp_server", "smtp_port", "username"]
                for field in required_email_fields:
                    if not email_config.get(field):
                        result.add_error(f"Email configuration missing required field: {field}")

            # Check recipients
            recipients = notification_config.get("recipients", {})
            if not recipients:
                result.add_warning("No notification recipients configured")

        return result

    def update_endpoint_settings(
        self, profile: str, endpoint_id: str, settings: Dict[str, Any]
    ) -> OperationResult:
        """
        Update settings for a specific PACS endpoint.

        Args:
            profile: Configuration profile name
            endpoint_id: Endpoint identifier to update
            settings: New settings to apply

        Returns:
            OperationResult with update status
        """
        logger.info(f"Updating endpoint {endpoint_id} in profile {profile}")

        operation_id = f"update_endpoint_{profile}_{endpoint_id}"

        try:
            # Load current configuration
            config = self.load_configuration(profile)

            # Check if endpoint exists
            if endpoint_id not in config.pacs_endpoints:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message=f"Endpoint {endpoint_id} not found in profile {profile}",
                    errors=[f"Unknown endpoint: {endpoint_id}"],
                )

            # Update endpoint settings
            endpoint = config.pacs_endpoints[endpoint_id]

            # Update basic settings
            if "host" in settings:
                endpoint.host = settings["host"]
            if "port" in settings:
                endpoint.port = int(settings["port"])
            if "ae_title" in settings:
                endpoint.ae_title = settings["ae_title"]
            if "vendor" in settings:
                endpoint.vendor = PACSVendor(settings["vendor"])
            if "description" in settings:
                endpoint.description = settings["description"]
            if "is_primary" in settings:
                endpoint.is_primary = bool(settings["is_primary"])

            # Update security settings
            if "security_config" in settings:
                sec_settings = settings["security_config"]
                sec_config = endpoint.security_config

                for key, value in sec_settings.items():
                    if hasattr(sec_config, key):
                        if key.endswith("_path") and value:
                            setattr(sec_config, key, Path(value))
                        else:
                            setattr(sec_config, key, value)

            # Update performance settings
            if "performance_config" in settings:
                perf_settings = settings["performance_config"]
                perf_config = endpoint.performance_config

                for key, value in perf_settings.items():
                    if hasattr(perf_config, key):
                        setattr(perf_config, key, value)

            # Validate updated configuration
            validation = self.validate_configuration(config)
            if not validation.is_valid:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message="Updated configuration is invalid",
                    errors=validation.errors,
                )

            # Update cache
            self._loaded_configurations[profile] = config

            return OperationResult.success_result(
                operation_id=operation_id,
                message=f"Successfully updated endpoint {endpoint_id}",
                data={
                    "profile": profile,
                    "endpoint_id": endpoint_id,
                    "updated_settings": list(settings.keys()),
                },
            )

        except Exception as e:
            logger.error(f"Failed to update endpoint settings: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id, message=f"Update failed: {str(e)}", errors=[str(e)]
            )

    def save_configuration(
        self, config: PACSConfiguration, profile: str, encrypted: bool = False
    ) -> OperationResult:
        """
        Save PACS configuration to file.

        Args:
            config: PACSConfiguration to save
            profile: Profile name for the configuration
            encrypted: Whether to encrypt the configuration file

        Returns:
            OperationResult with save status
        """
        logger.info(f"Saving configuration profile: {profile} (encrypted: {encrypted})")

        operation_id = f"save_config_{profile}"

        try:
            # Validate configuration before saving
            validation = self.validate_configuration(config)
            if not validation.is_valid:
                return OperationResult.error_result(
                    operation_id=operation_id,
                    message="Cannot save invalid configuration",
                    errors=validation.errors,
                )

            # Convert configuration to dictionary
            config_data = self._configuration_to_dict(config)

            # Determine output file
            if encrypted:
                config_file = self.config_directory / f"{profile}.encrypted.yaml"
            else:
                config_file = self.config_directory / f"{profile}.yaml"

            # Save configuration
            self._save_config_file(config_data, config_file, encrypted)

            # Update cache
            self._loaded_configurations[profile] = config

            return OperationResult.success_result(
                operation_id=operation_id,
                message=f"Configuration saved successfully: {config_file}",
                data={"profile": profile, "file_path": str(config_file), "encrypted": encrypted},
            )

        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return OperationResult.error_result(
                operation_id=operation_id, message=f"Save failed: {str(e)}", errors=[str(e)]
            )

    def list_available_profiles(self) -> List[str]:
        """
        List all available configuration profiles.

        Returns:
            List of profile names
        """
        profiles = []

        for config_file in self.config_directory.glob("*.yaml"):
            if not config_file.name.startswith("."):
                profile_name = config_file.stem
                if not profile_name.endswith(".encrypted"):
                    profiles.append(profile_name)

        for config_file in self.config_directory.glob("*.encrypted.yaml"):
            profile_name = config_file.stem.replace(".encrypted", "")
            if profile_name not in profiles:
                profiles.append(profile_name)

        return sorted(profiles)

    def create_default_configuration(self) -> PACSConfiguration:
        """
        Create a default PACS configuration for development/testing.

        Returns:
            Default PACSConfiguration
        """
        logger.info("Creating default PACS configuration")

        # Default security configuration (insecure for development)
        default_security = SecurityConfig(
            tls_enabled=False, verify_certificates=False, mutual_authentication=False
        )

        # Default performance configuration
        default_performance = PerformanceConfig(
            max_concurrent_studies=10,
            connection_pool_size=3,
            query_timeout=30,
            retrieval_timeout=300,
            retry_attempts=3,
        )

        # Default PACS endpoint
        default_endpoint = PACSEndpoint(
            endpoint_id="default_pacs",
            ae_title="HISTOCORE_DEFAULT",
            host="localhost",
            port=11112,
            vendor=PACSVendor.GENERIC,
            security_config=default_security,
            performance_config=default_performance,
            description="Default PACS endpoint for development",
            is_primary=True,
        )

        # Default configuration
        default_config = PACSConfiguration(
            profile_name="default",
            pacs_endpoints={"default": default_endpoint},
            storage_config={
                "local_cache_path": "./data/pacs_cache",
                "max_cache_size_gb": 10,
                "cleanup_threshold_gb": 2,
                "temp_directory": "./tmp/pacs",
                "archive_directory": "./data/pacs_archive",
            },
            notification_config={
                "enabled": False,
                "log_notifications": True,
                "log_path": "./logs/notifications.log",
            },
            audit_config={
                "enabled": True,
                "storage_path": "./logs/pacs_audit",
                "retention_years": 1,
                "encryption_enabled": False,
            },
        )

        self._default_config = default_config
        return default_config

    def _load_config_file(self, config_file: Path, encrypted: bool) -> Dict[str, Any]:
        """Load configuration file (encrypted or plain)."""
        try:
            with open(config_file, "rb" if encrypted else "r") as f:
                if encrypted:
                    # Decrypt the file
                    if not self.encryption_key:
                        raise ValueError("Encryption key required for encrypted configuration")

                    encrypted_data = f.read()
                    fernet = Fernet(self.encryption_key)
                    decrypted_data = fernet.decrypt(encrypted_data)
                    config_data = yaml.safe_load(decrypted_data.decode("utf-8"))
                else:
                    # Load plain YAML
                    config_data = yaml.safe_load(f)

            return config_data

        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {str(e)}")
            raise

    def _save_config_file(self, config_data: Dict[str, Any], config_file: Path, encrypted: bool):
        """Save configuration file (encrypted or plain)."""
        try:
            # Convert to YAML
            yaml_data = yaml.dump(config_data, default_flow_style=False, indent=2)

            if encrypted:
                # Encrypt the data
                if not self.encryption_key:
                    raise ValueError("Encryption key required for encrypted configuration")

                fernet = Fernet(self.encryption_key)
                encrypted_data = fernet.encrypt(yaml_data.encode("utf-8"))

                with open(config_file, "wb") as f:
                    f.write(encrypted_data)
            else:
                # Save plain YAML
                with open(config_file, "w") as f:
                    f.write(yaml_data)

            logger.debug(f"Configuration saved to: {config_file}")

        except Exception as e:
            logger.error(f"Failed to save config file {config_file}: {str(e)}")
            raise

    def _parse_configuration(self, config_data: Dict[str, Any], profile: str) -> PACSConfiguration:
        """Parse configuration data into PACSConfiguration object."""
        try:
            # Parse PACS endpoints
            pacs_endpoints = {}
            endpoints_data = config_data.get("pacs_endpoints", {})

            for endpoint_id, endpoint_data in endpoints_data.items():
                # Parse security config
                sec_data = endpoint_data.get("security_config", {})
                security_config = SecurityConfig(
                    tls_enabled=sec_data.get("tls_enabled", True),
                    tls_version=sec_data.get("tls_version", "1.3"),
                    certificate_path=(
                        Path(sec_data["certificate_path"])
                        if sec_data.get("certificate_path")
                        else None
                    ),
                    client_cert_path=(
                        Path(sec_data["client_cert_path"])
                        if sec_data.get("client_cert_path")
                        else None
                    ),
                    client_key_path=(
                        Path(sec_data["client_key_path"])
                        if sec_data.get("client_key_path")
                        else None
                    ),
                    ca_bundle_path=(
                        Path(sec_data["ca_bundle_path"]) if sec_data.get("ca_bundle_path") else None
                    ),
                    verify_certificates=sec_data.get("verify_certificates", True),
                    mutual_authentication=sec_data.get("mutual_authentication", False),
                )

                # Parse performance config
                perf_data = endpoint_data.get("performance_config", {})
                performance_config = PerformanceConfig(
                    max_concurrent_studies=perf_data.get("max_concurrent_studies", 50),
                    connection_pool_size=perf_data.get("connection_pool_size", 10),
                    query_timeout=perf_data.get("query_timeout", 30),
                    retrieval_timeout=perf_data.get("retrieval_timeout", 300),
                    storage_timeout=perf_data.get("storage_timeout", 120),
                    retry_attempts=perf_data.get("retry_attempts", 3),
                    retry_delay=perf_data.get("retry_delay", 1.0),
                    max_retry_delay=perf_data.get("max_retry_delay", 60.0),
                )

                # Create endpoint
                endpoint = PACSEndpoint(
                    endpoint_id=endpoint_id,
                    ae_title=endpoint_data["ae_title"],
                    host=endpoint_data["host"],
                    port=endpoint_data["port"],
                    vendor=PACSVendor(endpoint_data.get("vendor", "Generic")),
                    security_config=security_config,
                    performance_config=performance_config,
                    description=endpoint_data.get("description"),
                    is_primary=endpoint_data.get("is_primary", False),
                )

                pacs_endpoints[endpoint_id] = endpoint

            # Create configuration
            config = PACSConfiguration(
                profile_name=profile,
                pacs_endpoints=pacs_endpoints,
                storage_config=config_data.get("storage_config", {}),
                notification_config=config_data.get("notification_config", {}),
                audit_config=config_data.get("audit_config", {}),
            )

            return config

        except Exception as e:
            logger.error(f"Failed to parse configuration: {str(e)}")
            raise ValueError(f"Configuration parsing failed: {str(e)}")

    def _configuration_to_dict(self, config: PACSConfiguration) -> Dict[str, Any]:
        """Convert PACSConfiguration to dictionary for serialization."""
        config_dict = {
            "profile_name": config.profile_name,
            "pacs_endpoints": {},
            "storage_config": config.storage_config,
            "notification_config": config.notification_config,
            "audit_config": config.audit_config,
        }

        # Convert endpoints
        for endpoint_id, endpoint in config.pacs_endpoints.items():
            endpoint_dict = {
                "endpoint_id": endpoint.endpoint_id,
                "ae_title": endpoint.ae_title,
                "host": endpoint.host,
                "port": endpoint.port,
                "vendor": endpoint.vendor.value,
                "description": endpoint.description,
                "is_primary": endpoint.is_primary,
                "security_config": {
                    "tls_enabled": endpoint.security_config.tls_enabled,
                    "tls_version": endpoint.security_config.tls_version,
                    "certificate_path": (
                        str(endpoint.security_config.certificate_path)
                        if endpoint.security_config.certificate_path
                        else None
                    ),
                    "client_cert_path": (
                        str(endpoint.security_config.client_cert_path)
                        if endpoint.security_config.client_cert_path
                        else None
                    ),
                    "client_key_path": (
                        str(endpoint.security_config.client_key_path)
                        if endpoint.security_config.client_key_path
                        else None
                    ),
                    "ca_bundle_path": (
                        str(endpoint.security_config.ca_bundle_path)
                        if endpoint.security_config.ca_bundle_path
                        else None
                    ),
                    "verify_certificates": endpoint.security_config.verify_certificates,
                    "mutual_authentication": endpoint.security_config.mutual_authentication,
                },
                "performance_config": {
                    "max_concurrent_studies": endpoint.performance_config.max_concurrent_studies,
                    "connection_pool_size": endpoint.performance_config.connection_pool_size,
                    "query_timeout": endpoint.performance_config.query_timeout,
                    "retrieval_timeout": endpoint.performance_config.retrieval_timeout,
                    "storage_timeout": endpoint.performance_config.storage_timeout,
                    "retry_attempts": endpoint.performance_config.retry_attempts,
                    "retry_delay": endpoint.performance_config.retry_delay,
                    "max_retry_delay": endpoint.performance_config.max_retry_delay,
                },
            }

            config_dict["pacs_endpoints"][endpoint_id] = endpoint_dict

        return config_dict

    def generate_encryption_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """
        Generate encryption key from password for encrypted configurations.

        Args:
            password: Password to derive key from
            salt: Optional salt (generated if not provided)

        Returns:
            Encryption key suitable for Fernet
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def get_configuration_statistics(self) -> Dict[str, Any]:
        """Get configuration manager statistics."""
        return {
            "config_directory": str(self.config_directory),
            "loaded_configurations": len(self._loaded_configurations),
            "available_profiles": len(self.list_available_profiles()),
            "has_default_config": self._default_config is not None,
            "encryption_enabled": self.encryption_key is not None,
        }
